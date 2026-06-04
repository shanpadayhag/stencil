//! `stencil detect` — scan a template for bracketed variables and write a Markdown
//! snippet file, optionally censoring sensitive values first.
//!
//! Pipeline: extract → (censor) → detect → section → render. When `--censor` is set,
//! censoring runs first so the section context in the Markdown shows `REDACTED_*`
//! placeholders rather than real values, and a `mapping.json` is written for `restore`.
//!
//! Every detected bracket additionally gets its own always-censored snippet file under a
//! `snippets/` folder (cross-paragraph spans in a `cross-paragraph/` subfolder); see
//! [`crate::review`]. On success the command prints a single confirmation line.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use std::collections::BTreeSet;

use crate::censor::{CensorOptions, PartyList, censor};
use crate::cli::DetectArgs;
use crate::detect::detect;
use crate::extract;
use crate::learn::{self, LearnedStore};
use crate::model::{Document, Mapping};
use crate::render::render;
use crate::review::{self, ReviewFile};
use crate::section::sections;

/// Run the `detect` subcommand.
pub fn run(args: DetectArgs) -> Result<()> {
    if !args.censor && args.parties.is_some() {
        bail!("--parties requires --censor");
    }

    let extracted = extract::from_path(&args.input)?;
    let parties = match &args.parties {
        Some(spec) => Some(PartyList::parse(spec)?),
        None => None,
    };
    // Values the user has previously marked safe (via interactive restore) are skipped by
    // censoring. Best-effort: a missing/unreadable store just means nothing is allowed yet.
    let allowed = load_allowed_values();
    let (document, mapping) = maybe_censor(&extracted, &args, parties.as_ref(), &allowed);

    // Cross-paragraph artifacts are ALWAYS censored — using any party list plus the
    // regex patterns — so they are safe to share regardless of the main run's `--censor`.
    let review_options = CensorOptions {
        parties: parties.as_ref(),
        allow: Some(&allowed),
    };

    let mut detection = detect(&document);
    // Censor the cross-paragraph rows' preview text before they reach the inventory, so
    // the main file never dumps a raw multi-paragraph span into the table.
    review::censor_cross_paragraph_previews(&mut detection, &extracted, &review_options);
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

    // Review files are built from the uncensored source so censoring runs cleanly here.
    let review_dir = review_dir(&out_path);
    let review_files =
        review::build_review_files(&extracted, &detection, &review_options, &review_dir);

    // Pre-check every target so we never write some then fail on another.
    super::ensure_writable(&out_path, args.force)?;
    if mapping.is_some() {
        super::ensure_writable(&map_path, args.force)?;
    }
    for file in &review_files {
        super::ensure_writable(&file.path, args.force)?;
    }

    fs::write(&out_path, &markdown)
        .with_context(|| format!("failed to write `{}`", out_path.display()))?;
    if let Some(mapping) = &mapping {
        write_mapping(&map_path, mapping)?;
    }
    write_review_files(&review_files)?;

    let hit_sections = secs.iter().filter(|section| section.has_hits()).count();
    let mut summary = format!(
        "Wrote {} bracket(s) across {} section(s) to {}",
        detection.hits.len(),
        hit_sections,
        out_path.display(),
    );
    if mapping.is_some() {
        summary.push_str(&format!(" (mapping: {})", map_path.display()));
    }
    if !review_files.is_empty() {
        summary.push_str(&format!(
            " [{} snippet file(s) in {}]",
            review_files.len(),
            review_dir.display()
        ));
    }
    println!("{summary}");
    Ok(())
}

/// Apply censoring if requested, returning the (possibly rewritten) document and the
/// mapping to persist. The party list and learned allowlist are passed in by the caller.
fn maybe_censor(
    document: &Document,
    args: &DetectArgs,
    parties: Option<&PartyList>,
    allowed: &BTreeSet<String>,
) -> (Document, Option<Mapping>) {
    if !args.censor {
        return (document.clone(), None);
    }
    let options = CensorOptions {
        parties,
        allow: Some(allowed),
    };
    let outcome = censor(document, &options);
    (outcome.document, Some(outcome.mapping))
}

/// Load the per-user learned allowlist, or an empty set if it cannot be located/read.
fn load_allowed_values() -> BTreeSet<String> {
    learn::store_path()
        .and_then(|path| LearnedStore::load(&path))
        .map(|store| store.allowed_values())
        .unwrap_or_default()
}

/// The directory for per-bracket snippet files: a `snippets/` folder beside the main
/// output file (with a `cross-paragraph/` subfolder for ambiguous spans).
fn review_dir(out_path: &Path) -> PathBuf {
    out_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join("snippets")
}

/// Write each snippet file, creating its parent directory as needed.
fn write_review_files(files: &[ReviewFile]) -> Result<()> {
    for file in files {
        if let Some(parent) = file.path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create `{}`", parent.display()))?;
        }
        fs::write(&file.path, &file.content)
            .with_context(|| format!("failed to write `{}`", file.path.display()))?;
    }
    Ok(())
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
