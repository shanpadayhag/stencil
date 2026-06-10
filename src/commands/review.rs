//! `stencil review` — the single interactive pipeline over a template.
//!
//! Runs three stages in order — `censor` → `styling` → `snippet` — each selectable with
//! `--only`/`--skip`. The review stages read single keypresses, so the command requires an
//! interactive terminal (TTY) and refuses to run without one.
//!
//! - **censor** (T26): detect candidates → interactive confirm/reject/re-type → apply only the
//!   confirmed values → append schema-3 decisions and fold them into the learned store.
//! - **styling** (T28–T31): extract each block's formatting → censor its text → interactive
//!   per-block fine/weird review → append `styling.jsonl` records and the profile sidecar.
//! - **snippet**: write the context-rich `.stencil.md` inventory + per-bracket snippet files.

use std::collections::BTreeSet;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::censor::review::review as run_censor_review;
use crate::censor::{self, CensorDecision, CensorOptions, PartyList, Verdict};
use crate::cli::{ReviewArgs, Stage};
use crate::detect::{Detection, Status, detect};
use crate::extract;
use crate::learn::{self, LearnedStore};
use crate::model::{Block, Document, StyledBlock};
use crate::render::{SnippetEntry, render};
use crate::review::{self, ReviewFile};
use crate::section::sections;
use crate::style;
use crate::style::review::{StyleVerdict, review as run_styling_review};

/// Run the `review` subcommand.
pub fn run(args: ReviewArgs) -> Result<()> {
    let stages = active_stages(&args.only, &args.skip);

    // The review stages capture keypresses; without a terminal there is no way to do that, so
    // refuse up front rather than silently collecting nothing.
    if !std::io::stdin().is_terminal() {
        bail!("review needs an interactive terminal (TTY); run it directly in your terminal");
    }

    let extracted = extract::from_path(&args.input)?;
    let parties = match &args.parties {
        Some(spec) => Some(PartyList::parse(spec)?),
        None => None,
    };
    let learned_allowed = load_allowed(&args);

    // ── Censor stage ────────────────────────────────────────────────────────
    // Produces `working` (only confirmed values censored) and the set of values the reviewer
    // rejected this run (so the snippet stage stays consistent with their decisions).
    let (working, rejected) = if stages.contains(&Stage::Censor) {
        run_censor_stage(&args, &extracted, parties.as_ref(), &learned_allowed)?
    } else {
        (extracted.clone(), BTreeSet::new())
    };

    // Both the styling and snippet stages censor for safe sharing: skip only what the user
    // rejected this run plus their learned-safe values, so all outputs agree on what stays.
    let mut censor_allow = learned_allowed.clone();
    censor_allow.extend(rejected);
    let options = CensorOptions {
        parties: parties.as_ref(),
        allow: Some(&censor_allow),
    };

    // ── Styling stage (T28–T31) ─────────────────────────────────────────────
    if stages.contains(&Stage::Styling) {
        run_styling_stage(&args, &options)?;
    }

    // ── Snippet stage ───────────────────────────────────────────────────────
    if stages.contains(&Stage::Snippet) {
        write_snippet_outputs(&args, &extracted, &working, &options)?;
    }

    Ok(())
}

/// Run the interactive styling stage: extract each block's formatting, censor its text, review it
/// block-by-block, then best-effort persist the records and the per-document profile sidecar.
///
/// Styling reads the `.docx` formatting directly, so it is a no-op for other inputs. The block
/// text shown and logged is censored with `options` even when the censor stage was skipped (e.g.
/// `--only styling`), since styling judgment never needs the real values.
fn run_styling_stage(args: &ReviewArgs, options: &CensorOptions<'_>) -> Result<()> {
    if !is_docx(&args.input) {
        println!("Styling: only `.docx` carries styling, skipped.");
        return Ok(());
    }

    let mut blocks = style::extract::from_path(&args.input)?;
    if blocks.is_empty() {
        println!("Styling: no blocks to review, skipped.");
        return Ok(());
    }
    censor_block_text(&mut blocks, &args.input, options);

    let profile = style::profile::build_profile(&blocks);
    let decisions = run_styling_review(&blocks, &profile)?;

    let weird = decisions
        .iter()
        .filter(|decision| matches!(decision.verdict, StyleVerdict::Weird { .. }))
        .count();
    println!(
        "Styling: reviewed {} block(s) — {} fine, {weird} weird.",
        decisions.len(),
        decisions.len() - weird
    );

    persist_styling(args, &blocks, &profile, &decisions);
    Ok(())
}

/// Censor each styled block's text (and so the neighbor context derived from it) in place, using
/// the deterministic "censor everything" pass — a one-paragraph-per-block document keeps the
/// censored text aligned 1:1 with the styled blocks.
fn censor_block_text(blocks: &mut [StyledBlock], input: &Path, options: &CensorOptions<'_>) {
    let text_doc = Document {
        source: input.to_path_buf(),
        blocks: blocks
            .iter()
            .map(|block| Block::Paragraph {
                text: block.text.clone(),
            })
            .collect(),
    };
    let censored = censor::censor(&text_doc, options).document;
    for (block, censored_block) in blocks.iter_mut().zip(censored.blocks) {
        if let Block::Paragraph { text } = censored_block {
            block.text = text;
        }
    }
}

/// Append the styling records and write the profile sidecar. Best-effort: a storage hiccup must
/// never fail the review the user just completed, so problems are reported, not propagated.
fn persist_styling(
    args: &ReviewArgs,
    blocks: &[StyledBlock],
    profile: &crate::model::DocumentStyleProfile,
    decisions: &[style::review::StyleDecision],
) {
    let (Ok(log_path), Ok(profiles_dir)) = (
        learn::styling_log_path(args.data_dir.as_deref(), args.styling_dir.as_deref()),
        learn::styling_profiles_dir(args.data_dir.as_deref(), args.styling_dir.as_deref()),
    ) else {
        eprintln!("note: could not locate the styling data dir; styling was not saved");
        return;
    };
    if let Err(err) = style::record::persist(
        &log_path,
        &profiles_dir,
        blocks,
        profile,
        decisions,
        &args.input,
    ) {
        eprintln!("note: could not save styling records: {err}");
    }
}

/// Whether `path` has a `.docx` extension (case-insensitive).
fn is_docx(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("docx"))
}

/// The stages to run, in pipeline order, given the `--only`/`--skip` selections. `--only` and
/// `--skip` are mutually exclusive at the CLI layer, so at most one is non-empty here.
fn active_stages(only: &[Stage], skip: &[Stage]) -> Vec<Stage> {
    Stage::ALL
        .into_iter()
        .filter(|stage| {
            if !only.is_empty() {
                only.contains(stage)
            } else {
                !skip.contains(stage)
            }
        })
        .collect()
}

/// Run the interactive censor stage: plan → review → apply, then best-effort persist the
/// decisions. Returns the censored document and the set of rejected values.
fn run_censor_stage(
    args: &ReviewArgs,
    extracted: &Document,
    parties: Option<&PartyList>,
    learned_allowed: &BTreeSet<String>,
) -> Result<(Document, BTreeSet<String>)> {
    let options = CensorOptions {
        parties,
        allow: Some(learned_allowed),
    };
    let items = censor::plan_review(extracted, &options);
    let decisions = run_censor_review(extracted, &items)?;
    let working = censor::apply(extracted, &decisions, &options);

    // A value is "rejected" downstream (left in the clear in the snippet/styling stages) only when
    // *no* occurrence of it was confirmed. A split with mixed verdicts keeps the value censored
    // everywhere downstream, so a confirmed occurrence never leaks into an always-censored snippet.
    let confirmed_values: BTreeSet<String> = decisions
        .iter()
        .filter(|decision| matches!(decision.verdict, Verdict::Confirm { .. }))
        .map(|decision| decision.value.clone())
        .collect();
    let rejected: BTreeSet<String> = decisions
        .iter()
        .filter(|decision| matches!(decision.verdict, Verdict::Reject))
        .map(|decision| decision.value.clone())
        .filter(|value| !confirmed_values.contains(value))
        .collect();
    let kept = decisions
        .iter()
        .filter(|decision| matches!(decision.verdict, Verdict::Confirm { .. }))
        .count();
    let rejected_count = decisions.len() - kept;
    println!(
        "Censor: {} value(s) decided — {kept} kept, {rejected_count} rejected.",
        decisions
            .iter()
            .filter(|decision| decision.reviewed)
            .count(),
    );

    let source = extracted.source.display().to_string();
    persist_decisions(args, &decisions, &source);
    Ok((working, rejected))
}

/// Load the per-user learned allowlist (censor store), or an empty set if it cannot be read.
fn load_allowed(args: &ReviewArgs) -> BTreeSet<String> {
    learn::store_path(args.data_dir.as_deref(), args.censor_dir.as_deref())
        .and_then(|path| LearnedStore::load(&path))
        .map(|store| store.allowed_values())
        .unwrap_or_default()
}

/// Append the schema-3 decision records and fold the verdicts into the learned store.
///
/// Best-effort: a storage hiccup must never fail the review the user just completed, so problems
/// are reported, not propagated. Only human-reviewed decisions are persisted.
fn persist_decisions(args: &ReviewArgs, decisions: &[CensorDecision], source: &str) {
    if !decisions.iter().any(|decision| decision.reviewed) {
        return;
    }
    let records = censor::decision_records(decisions, source, learn::now_epoch_secs());
    let (Ok(store_path), Ok(log_path)) = (
        learn::store_path(args.data_dir.as_deref(), args.censor_dir.as_deref()),
        learn::log_path(args.data_dir.as_deref(), args.censor_dir.as_deref()),
    ) else {
        eprintln!("note: could not locate the censor data dir; decisions were not saved");
        return;
    };

    for record in &records {
        if let Err(err) = learn::append_decision(&log_path, record) {
            eprintln!("note: could not log a decision: {err}");
        }
    }
    let mut store = LearnedStore::load(&store_path).unwrap_or_default();
    censor::update_store(&mut store, decisions);
    if let Err(err) = store.save(&store_path) {
        eprintln!("note: could not save the learned store: {err}");
    }
}

/// Build and write the `.stencil.md` inventory plus the per-bracket snippet files.
///
/// `working` is the (possibly censored) document the inventory and section context are rendered
/// from; `extracted` is the uncensored source the always-censored snippet files are built from.
fn write_snippet_outputs(
    args: &ReviewArgs,
    extracted: &Document,
    working: &Document,
    options: &CensorOptions<'_>,
) -> Result<()> {
    let mut detection = detect(working);
    // Censor cross-paragraph preview rows so the inventory never dumps a raw multi-paragraph span.
    review::censor_cross_paragraph_previews(&mut detection, extracted, options);

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| sibling_path(&args.input, ".stencil.md"));
    let review_dir = review_dir(&out_path);
    let main_md_name = out_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("stencil.md");
    let review_files =
        review::build_review_files(extracted, &detection, options, &review_dir, main_md_name);
    let snippet_entries = build_snippet_entries(&review_files, &detection);

    let secs = sections(working, &detection);
    let markdown = render(&working.source, &secs, &snippet_entries);

    // Pre-check every target so we never write some then fail on another.
    super::ensure_writable(&out_path, args.force)?;
    for file in &review_files {
        super::ensure_writable(&file.path, args.force)?;
    }

    fs::write(&out_path, &markdown)
        .with_context(|| format!("failed to write `{}`", out_path.display()))?;
    write_review_files(&review_files)?;

    let mut summary = format!(
        "Wrote {} bracket(s) to {}",
        detection.hits.len(),
        out_path.display()
    );
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

/// Assemble the snippet map the renderer links from: one [`SnippetEntry`] per review file, with
/// its relative link and a review flag set when any bracket in its range is guessed.
fn build_snippet_entries(review_files: &[ReviewFile], detection: &Detection) -> Vec<SnippetEntry> {
    review_files
        .iter()
        .map(|file| SnippetEntry {
            id: file.id.clone(),
            range: file.range,
            rel_path: relative_snippet_link(&file.path),
            needs_review: detection
                .hits
                .iter()
                .filter(|hit| (hit.block, hit.end_block) == file.range)
                .any(|hit| hit.status == Status::Guessed),
        })
        .collect()
}

/// The snippet file's link relative to the main `.stencil.md`: the path from the `snippets/`
/// component onward, with forward slashes so the Markdown link is portable.
fn relative_snippet_link(path: &Path) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut found = false;
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            found |= name == "snippets";
            if found {
                parts.push(name.to_string_lossy().into_owned());
            }
        }
    }
    if parts.is_empty() {
        path.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    } else {
        parts.join("/")
    }
}

/// The directory for per-bracket snippet files: a `snippets/` folder beside the main output.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_selection_runs_all_stages_in_order() {
        assert_eq!(active_stages(&[], &[]), Stage::ALL.to_vec());
    }

    #[test]
    fn only_keeps_just_the_named_stages_in_pipeline_order() {
        let only = [Stage::Snippet, Stage::Censor];
        assert_eq!(
            active_stages(&only, &[]),
            vec![Stage::Censor, Stage::Snippet]
        );
    }

    #[test]
    fn skip_removes_the_named_stages() {
        assert_eq!(
            active_stages(&[], &[Stage::Styling]),
            vec![Stage::Censor, Stage::Snippet]
        );
    }

    #[test]
    fn sibling_path_uses_stem_beside_input() {
        assert_eq!(
            sibling_path(Path::new("dir/contract.docx"), ".stencil.md"),
            PathBuf::from("dir/contract.stencil.md")
        );
    }

    #[test]
    fn is_docx_is_case_insensitive_and_extension_only() {
        assert!(is_docx(Path::new("dir/Contract.docx")));
        assert!(is_docx(Path::new("dir/Contract.DOCX")));
        assert!(!is_docx(Path::new("dir/contract.txt")));
        assert!(!is_docx(Path::new("docx")));
    }

    #[test]
    fn relative_snippet_link_starts_at_snippets_component() {
        assert_eq!(
            relative_snippet_link(Path::new("/tmp/out/snippets/foo.md")),
            "snippets/foo.md"
        );
        assert_eq!(
            relative_snippet_link(Path::new("/tmp/out/snippets/cross-paragraph/bar.md")),
            "snippets/cross-paragraph/bar.md"
        );
    }
}
