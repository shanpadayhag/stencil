//! `stencil restore` — replace `REDACTED_*` placeholders with their real values.
//!
//! A general find-replace over any text/Markdown file containing placeholders
//! (typically the `.stencil.md`). It reverses censorship only; it does not fill
//! bracketed variables. `.docx` is never written (it is read-only in Stencil).

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::cli::RestoreArgs;
use crate::learn::{self, DecisionRecord, LearnedStore};
use crate::model::{MAPPING_VERSION, Mapping, MappingEntry};

mod interactive;

/// Run the `restore` subcommand.
pub fn run(args: RestoreArgs) -> Result<()> {
    let mapping = load_mapping(&args.map)?;
    let input = fs::read_to_string(&args.input)
        .with_context(|| format!("failed to read input `{}`", args.input.display()))?;

    let selected: Vec<&MappingEntry> = if args.interactive {
        let decisions = interactive::select(&mapping, &input)?;
        learn_from_decisions(
            &decisions,
            &mapping.source,
            &input,
            args.data_dir.as_deref(),
        );
        decisions
            .iter()
            .filter(|decision| decision.allow)
            .map(|decision| decision.entry)
            .collect()
    } else {
        select_entries(&mapping, args.only.as_deref())?
    };
    let (restored, replaced) = apply(&selected, &input);

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| default_restored_path(&args.input));
    super::ensure_writable(&out_path, args.force)?;
    fs::write(&out_path, &restored)
        .with_context(|| format!("failed to write `{}`", out_path.display()))?;

    println!(
        "Restored {replaced} placeholder occurrence(s) to {}",
        out_path.display()
    );
    Ok(())
}

/// Persist the interactive-restore decisions: append each to the JSONL training log and
/// fold it into the per-user learned store. Best-effort — a storage hiccup must never
/// fail the restore the user just completed, so problems are reported, not propagated.
fn learn_from_decisions(
    decisions: &[interactive::Decision<'_>],
    source: &str,
    input: &str,
    data_dir: Option<&Path>,
) {
    if decisions.is_empty() {
        return;
    }
    let (Ok(store_path), Ok(log_path)) = (learn::store_path(data_dir), learn::log_path(data_dir))
    else {
        eprintln!("note: could not locate the data dir; decisions were not learned");
        return;
    };

    let mut store = LearnedStore::load(&store_path).unwrap_or_default();
    for decision in decisions {
        let entry = decision.entry;
        let record = DecisionRecord {
            schema: learn::decision_schema(),
            timestamp: learn::now_epoch_secs(),
            source: source.to_string(),
            placeholder: entry.placeholder.clone(),
            value_type: entry.value_type.clone(),
            value: entry.value.clone(),
            decision: if decision.allow { "allow" } else { "deny" }.to_string(),
            shown_context: learn::sentence_window(input, &entry.placeholder),
            block_context: learn::block_window(input, &entry.placeholder),
        };
        if let Err(err) = learn::append_decision(&log_path, &record) {
            eprintln!("note: could not log a decision: {err}");
        }
        store.record(&entry.value, &entry.value_type, decision.allow);
    }

    match store.save(&store_path) {
        Ok(()) => {
            let safe = decisions.iter().filter(|decision| decision.allow).count();
            println!(
                "Learned {} decision(s) ({safe} marked safe); future --censor runs will use them.",
                decisions.len()
            );
        }
        Err(err) => eprintln!("note: could not save the learned store: {err}"),
    }
}

/// Read, parse, and version-check a `mapping.json`.
fn load_mapping(path: &Path) -> Result<Mapping> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read mapping `{}`", path.display()))?;
    let mapping: Mapping = serde_json::from_str(&text)
        .with_context(|| format!("invalid mapping JSON in `{}`", path.display()))?;
    if mapping.version != MAPPING_VERSION {
        bail!(
            "unsupported mapping version {} (expected {})",
            mapping.version,
            MAPPING_VERSION
        );
    }
    Ok(mapping)
}

/// Select the mapping entries to restore: all of them, or only those whose placeholder
/// is named in `only` (inline comma/newline-separated, or `@file`).
///
/// # Errors
/// Returns an error if an `@file` selection cannot be read.
fn select_entries<'a>(mapping: &'a Mapping, only: Option<&str>) -> Result<Vec<&'a MappingEntry>> {
    let Some(spec) = only else {
        return Ok(mapping.entries.iter().collect());
    };
    let wanted = parse_selection(spec)?;
    Ok(mapping
        .entries
        .iter()
        .filter(|entry| wanted.contains(&entry.placeholder))
        .collect())
}

/// Parse an `--only` selection into a set of placeholder tokens: inline comma/newline
/// separated, or `@path` to read from a file.
fn parse_selection(spec: &str) -> Result<HashSet<String>> {
    let raw = if let Some(path) = spec.strip_prefix('@') {
        fs::read_to_string(path)
            .with_context(|| format!("failed to read selection file `{path}`"))?
    } else {
        spec.to_string()
    };
    Ok(raw
        .split([',', '\n'])
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .collect())
}

/// Substitute each selected placeholder with its value, returning the result and the
/// number of occurrences replaced.
///
/// Placeholders are unique `REDACTED_<TYPE>_<NNN>` tokens and none is a substring of
/// another, so a simple per-entry global replace is unambiguous.
fn apply(entries: &[&MappingEntry], text: &str) -> (String, usize) {
    let mut result = text.to_string();
    let mut replaced = 0usize;
    for entry in entries {
        let count = result.matches(&entry.placeholder).count();
        if count > 0 {
            result = result.replace(&entry.placeholder, &entry.value);
            replaced += count;
        }
    }
    (result, replaced)
}

/// Default restored path: `<stem>.restored.<ext>` beside the input.
fn default_restored_path(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("output");
    let mut path = input.to_path_buf();
    match input.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => path.set_file_name(format!("{stem}.restored.{ext}")),
        None => path.set_file_name(format!("{stem}.restored")),
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::censor::{CensorOptions, censor};
    use crate::model::{Block, Document, MappingEntry};
    use std::path::PathBuf as StdPathBuf;

    fn mapping(entries: Vec<(&str, &str)>) -> Mapping {
        Mapping {
            version: MAPPING_VERSION,
            source: "test.txt".into(),
            entries: entries
                .into_iter()
                .map(|(placeholder, value)| MappingEntry {
                    placeholder: placeholder.into(),
                    value_type: "PERSON".into(),
                    value: value.into(),
                    method: "party-list".into(),
                    occurrences: 1,
                })
                .collect(),
        }
    }

    /// All entries of a mapping as references — the default (no `--only`) selection.
    fn all(map: &Mapping) -> Vec<&MappingEntry> {
        map.entries.iter().collect()
    }

    #[test]
    fn replaces_all_occurrences() {
        let map = mapping(vec![("REDACTED_PERSON_001", "Jane Doe")]);
        let (out, n) = apply(&all(&map), "REDACTED_PERSON_001 met REDACTED_PERSON_001.");
        assert_eq!(out, "Jane Doe met Jane Doe.");
        assert_eq!(n, 2);
    }

    #[test]
    fn leaves_unmapped_tokens_uncounted() {
        let map = mapping(vec![("REDACTED_PERSON_001", "Jane")]);
        let (out, n) = apply(&all(&map), "REDACTED_PERSON_001 and REDACTED_ORG_009");
        assert_eq!(n, 1, "only the mapped placeholder is replaced");
        assert!(
            out.contains("REDACTED_ORG_009"),
            "the unmapped token is left as-is"
        );
    }

    #[test]
    fn only_restores_selected_placeholders() {
        let map = mapping(vec![
            ("REDACTED_PERSON_001", "Jane Doe"),
            ("REDACTED_EMAIL_001", "jane@acme.example"),
        ]);
        let selected = select_entries(&map, Some("REDACTED_PERSON_001")).expect("select");
        let (out, n) = apply(&selected, "REDACTED_PERSON_001 at REDACTED_EMAIL_001");
        assert_eq!(n, 1, "only the selected placeholder is restored");
        assert!(out.contains("Jane Doe"), "selected value restored");
        assert!(
            out.contains("REDACTED_EMAIL_001"),
            "unselected placeholder is left untouched"
        );
    }

    #[test]
    fn no_only_selects_every_entry() {
        let map = mapping(vec![
            ("REDACTED_PERSON_001", "Jane"),
            ("REDACTED_EMAIL_001", "jane@acme.example"),
        ]);
        assert_eq!(select_entries(&map, None).expect("select").len(), 2);
    }

    #[test]
    fn default_restored_path_inserts_restored() {
        assert_eq!(
            default_restored_path(Path::new("contract.txt")),
            StdPathBuf::from("contract.restored.txt")
        );
        assert_eq!(
            default_restored_path(Path::new("c.stencil.md")),
            StdPathBuf::from("c.stencil.restored.md")
        );
    }

    #[test]
    fn round_trip_censor_then_restore_recovers_original() {
        let original = "Pay billing@acme.example for account 0012345678.";
        let document = Document {
            source: StdPathBuf::from("test.txt"),
            blocks: vec![Block::Paragraph {
                text: original.into(),
            }],
        };

        let outcome = censor(&document, &CensorOptions::default());
        let censored_text = match &outcome.document.blocks[0] {
            Block::Paragraph { text } => text.clone(),
            _ => unreachable!(),
        };
        assert!(censored_text.contains("REDACTED_"));

        let (restored, _) = apply(&all(&outcome.mapping), &censored_text);
        assert_eq!(restored, original);
        assert!(
            !restored.contains("REDACTED_"),
            "no placeholders should remain"
        );
    }

    #[test]
    fn unsupported_version_errors() {
        let path = std::env::temp_dir().join(format!("stencil_t9_ver_{}.json", std::process::id()));
        fs::write(&path, r#"{"version": 999, "source": "x", "entries": []}"#).expect("seed");
        let err = load_mapping(&path).unwrap_err();
        assert!(err.to_string().contains("unsupported mapping version"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn invalid_json_errors() {
        let path = std::env::temp_dir().join(format!("stencil_t9_bad_{}.json", std::process::id()));
        fs::write(&path, "{ not json").expect("seed");
        assert!(load_mapping(&path).is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_mapping_errors() {
        let missing = std::env::temp_dir().join("stencil_t9_missing_map.json");
        let _ = fs::remove_file(&missing);
        assert!(load_mapping(&missing).is_err());
    }
}
