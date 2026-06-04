//! `stencil restore` — replace `REDACTED_*` placeholders with their real values.
//!
//! A general find-replace over any text/Markdown file containing placeholders
//! (typically the `.stencil.md`). It reverses censorship only; it does not fill
//! bracketed variables. `.docx` is never written (it is read-only in Stencil).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::cli::RestoreArgs;
use crate::model::{MAPPING_VERSION, Mapping};

/// Run the `restore` subcommand.
pub fn run(args: RestoreArgs) -> Result<()> {
    let mapping = load_mapping(&args.map)?;
    let input = fs::read_to_string(&args.input)
        .with_context(|| format!("failed to read input `{}`", args.input.display()))?;

    let (restored, replaced) = apply(&mapping, &input);

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

/// Substitute every placeholder with its value, returning the result and the number of
/// occurrences replaced.
///
/// Placeholders are unique `REDACTED_<TYPE>_<NNN>` tokens and none is a substring of
/// another, so a simple per-entry global replace is unambiguous.
fn apply(mapping: &Mapping, text: &str) -> (String, usize) {
    let mut result = text.to_string();
    let mut replaced = 0usize;
    for entry in &mapping.entries {
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

    #[test]
    fn replaces_all_occurrences() {
        let map = mapping(vec![("REDACTED_PERSON_001", "Jane Doe")]);
        let (out, n) = apply(&map, "REDACTED_PERSON_001 met REDACTED_PERSON_001.");
        assert_eq!(out, "Jane Doe met Jane Doe.");
        assert_eq!(n, 2);
    }

    #[test]
    fn leaves_unmapped_tokens_uncounted() {
        let map = mapping(vec![("REDACTED_PERSON_001", "Jane")]);
        let (out, n) = apply(&map, "REDACTED_PERSON_001 and REDACTED_ORG_009");
        assert_eq!(n, 1, "only the mapped placeholder is replaced");
        assert!(
            out.contains("REDACTED_ORG_009"),
            "the unmapped token is left as-is"
        );
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

        let (restored, _) = apply(&outcome.mapping, &censored_text);
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
