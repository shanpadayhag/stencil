//! `stencil detect` — scan a template for bracketed variables and write a Markdown
//! snippet file, optionally censoring sensitive values first.
//!
//! Pipeline: extract → (censor) → detect → section → render. When `--censor` is set,
//! censoring runs first so the section context in the Markdown shows `REDACTED_*`
//! placeholders rather than real values, and a `mapping.json` is written for `restore`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::censor::{CensorOptions, PartyList, censor};
use crate::cli::DetectArgs;
use crate::detect::{Detection, detect};
use crate::extract;
use crate::model::{Document, Mapping};
use crate::render::render;
use crate::section::sections;

/// Run the `detect` subcommand.
pub fn run(args: DetectArgs) -> Result<()> {
    if !args.censor && (args.parties.is_some() || args.guess_names) {
        bail!("--parties and --guess-names require --censor");
    }

    let extracted = extract::from_path(&args.input)?;
    let (document, mapping) = maybe_censor(extracted, &args)?;

    let detection = detect(&document);
    let secs = sections(&document, &detection);
    let markdown = render(&document.source, &secs);

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| sibling_path(&args.input, ".stencil.md"));
    let map_path = args
        .map
        .clone()
        .unwrap_or_else(|| sibling_path(&args.input, ".mapping.json"));

    // Pre-check both targets so we never write one then fail on the other.
    super::ensure_writable(&out_path, args.force)?;
    if mapping.is_some() {
        super::ensure_writable(&map_path, args.force)?;
    }

    fs::write(&out_path, &markdown)
        .with_context(|| format!("failed to write `{}`", out_path.display()))?;
    if let Some(mapping) = &mapping {
        write_mapping(&map_path, mapping)?;
    }

    if let Some(mapping) = &mapping {
        report_censorship(mapping);
    }
    report_brackets(&detection);

    let hit_sections = secs.iter().filter(|section| section.has_hits()).count();
    println!(
        "Wrote {} bracket(s) across {} section(s) to {}{}",
        detection.hits.len(),
        hit_sections,
        out_path.display(),
        match &mapping {
            Some(_) => format!(" (mapping: {})", map_path.display()),
            None => String::new(),
        },
    );
    Ok(())
}

/// Apply censoring if requested, returning the (possibly rewritten) document and the
/// mapping to persist.
fn maybe_censor(document: Document, args: &DetectArgs) -> Result<(Document, Option<Mapping>)> {
    if !args.censor {
        return Ok((document, None));
    }

    let parties = match &args.parties {
        Some(spec) => Some(PartyList::parse(spec)?),
        None => None,
    };
    let options = CensorOptions {
        parties: parties.as_ref(),
        guess_names: args.guess_names,
    };
    let outcome = censor(&document, &options);
    Ok((outcome.document, Some(outcome.mapping)))
}

/// Build a sibling path `<stem><suffix>` beside `input` (e.g. `contract.stencil.md`).
fn sibling_path(input: &Path, suffix: &str) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("output");
    let mut path = input.to_path_buf();
    path.set_file_name(format!("{stem}{suffix}"));
    path
}

/// Serialize and write the mapping as pretty JSON.
fn write_mapping(path: &Path, mapping: &Mapping) -> Result<()> {
    let json = serde_json::to_string_pretty(mapping).context("failed to serialize mapping")?;
    fs::write(path, json).with_context(|| format!("failed to write `{}`", path.display()))
}

/// Print the censorship summary (per-type counts; heuristic guesses listed) to stderr.
fn report_censorship(mapping: &Mapping) {
    if mapping.entries.is_empty() {
        eprintln!("censor: no sensitive values found");
        return;
    }

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for entry in &mapping.entries {
        *counts.entry(entry.value_type.as_str()).or_insert(0) += 1;
    }
    eprintln!("censored {} distinct value(s):", mapping.entries.len());
    for (value_type, count) in &counts {
        eprintln!("    {value_type}: {count}");
    }

    let guessed: Vec<_> = mapping
        .entries
        .iter()
        .filter(|entry| entry.method == "heuristic")
        .collect();
    if !guessed.is_empty() {
        eprintln!(
            "⚠ {} heuristic-guessed name(s) — verify these are real before sharing:",
            guessed.len()
        );
        for entry in guessed {
            eprintln!("    {} = {}", entry.placeholder, entry.value);
        }
    }
}

/// Print the bracket-balance diagnostic (and any guessed spans) to stderr.
fn report_brackets(detection: &Detection) {
    let balance = detection.balance;
    if balance.is_balanced() {
        eprintln!(
            "bracket balance OK: {} '[' / {} ']'",
            balance.open, balance.close
        );
    } else {
        eprintln!(
            "⚠ unbalanced brackets: {} '[' vs {} ']'",
            balance.open, balance.close
        );
    }

    let guessed: Vec<_> = detection.guessed().collect();
    if !guessed.is_empty() {
        eprintln!(
            "⚠ {} guessed lone-bracket span(s) to review:",
            guessed.len()
        );
        for hit in guessed {
            eprintln!(
                "    [block {}] {}: {}",
                hit.block,
                hit.kind.label(),
                hit.span_text
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sibling_path_uses_stem_beside_input() {
        assert_eq!(
            sibling_path(Path::new("dir/contract.txt"), ".stencil.md"),
            PathBuf::from("dir/contract.stencil.md")
        );
        assert_eq!(
            sibling_path(Path::new("contract.docx"), ".mapping.json"),
            PathBuf::from("contract.mapping.json")
        );
    }
}
