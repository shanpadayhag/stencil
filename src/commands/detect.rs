//! `stencil detect` — scan a template for bracketed variables and write a Markdown
//! snippet file, optionally censoring sensitive values first.
//!
//! Pipeline: extract → (censor) → detect → section → render. When `--censor` is set,
//! censoring runs first so the section context in the Markdown shows `REDACTED_*`
//! placeholders rather than real values, and a `mapping.json` is written for `restore`.
//!
//! Any cross-paragraph bracket spans additionally get their own always-censored review
//! file under a `cross-paragraph/` subfolder (see [`crate::review`]). On success the
//! command prints a single confirmation line and nothing else.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::censor::{CensorOptions, IgnoreList, PartyList, censor};
use crate::cli::DetectArgs;
use crate::detect::detect;
use crate::extract;
use crate::model::{Document, Mapping};
use crate::render::render;
use crate::review::{self, ReviewFile};
use crate::section::sections;

/// Run the `detect` subcommand.
pub fn run(args: DetectArgs) -> Result<()> {
    if !args.censor && (args.parties.is_some() || args.guess_names || args.ignore.is_some()) {
        bail!("--parties, --guess-names, and --ignore require --censor");
    }

    let extracted = extract::from_path(&args.input)?;
    let parties = match &args.parties {
        Some(spec) => Some(PartyList::parse(spec)?),
        None => None,
    };
    let ignore = match &args.ignore {
        Some(spec) => Some(IgnoreList::parse(spec)?),
        None => None,
    };
    let (document, mapping) = maybe_censor(&extracted, &args, parties.as_ref(), ignore.as_ref());

    // Cross-paragraph artifacts are ALWAYS censored — heuristic names on, plus any party
    // list — so they are safe to share regardless of the main run's `--censor`.
    let review_options = CensorOptions {
        parties: parties.as_ref(),
        guess_names: true,
        ignore: ignore.as_ref(),
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
    write_review_files(&review_dir, &review_files)?;

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
            " [{} cross-paragraph review file(s) in {}]",
            review_files.len(),
            review_dir.display()
        ));
    }
    println!("{summary}");
    Ok(())
}

/// Apply censoring if requested, returning the (possibly rewritten) document and the
/// mapping to persist. Party and ignore lists are parsed by the caller and passed in.
fn maybe_censor(
    document: &Document,
    args: &DetectArgs,
    parties: Option<&PartyList>,
    ignore: Option<&IgnoreList>,
) -> (Document, Option<Mapping>) {
    if !args.censor {
        return (document.clone(), None);
    }
    let options = CensorOptions {
        parties,
        guess_names: args.guess_names,
        ignore,
    };
    let outcome = censor(document, &options);
    (outcome.document, Some(outcome.mapping))
}

/// The directory for per-candidate review files: a `cross-paragraph/` subfolder beside
/// the main output file.
fn review_dir(out_path: &Path) -> PathBuf {
    out_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join("cross-paragraph")
}

/// Create the review subfolder (when there is anything to write) and write each file.
fn write_review_files(dir: &Path, files: &[ReviewFile]) -> Result<()> {
    if files.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dir).with_context(|| format!("failed to create `{}`", dir.display()))?;
    for file in files {
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
