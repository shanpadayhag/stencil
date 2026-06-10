//! `stencil review` — the single interactive pipeline over a template.
//!
//! Runs three stages in order — `censor` → `styling` → `snippet` — each selectable with
//! `--only`/`--skip`. The review stages read single keypresses, so the command requires an
//! interactive terminal (TTY) and refuses to run without one.
//!
//! - **censor** (T26): detect candidates → interactive confirm/reject/re-type/edit/split → apply
//!   only the confirmed values → append schema-4 decisions and fold them into the learned store.
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
use crate::model::Document;
use crate::pages::PageSelection;
use crate::render::{SnippetEntry, render};
use crate::review::{self, ReviewFile};
use crate::section::sections;

/// Run the `review` subcommand.
pub fn run(args: ReviewArgs) -> Result<()> {
    let stages = active_stages(&args.only, &args.skip);

    // The review stages capture keypresses; without a terminal there is no way to do that, so
    // refuse up front rather than silently collecting nothing.
    if !std::io::stdin().is_terminal() {
        bail!("review needs an interactive terminal (TTY); run it directly in your terminal");
    }

    let extracted = extract::from_path(&args.input)?;
    // Content-derived id: the key for this document's records and style profile (filenames collide
    // across folders). Computed once so every stage records the same id.
    let doc_id = crate::doc_id::doc_id(&extracted);
    // Page scope for `--pages` (validated against the document's explicit page breaks).
    let page_numbers = extract::page_numbers(&args.input)?;
    let page_selection = parse_page_selection(&args, page_numbers.as_deref())?;
    let page_scope = match (&page_selection, &page_numbers) {
        (Some(selection), Some(pages)) => Some((pages.as_slice(), selection)),
        _ => None,
    };
    let parties = match &args.parties {
        Some(spec) => Some(PartyList::parse(spec)?),
        None => None,
    };
    let learned_allowed = load_allowed(&args);

    // ── Censor stage ────────────────────────────────────────────────────────
    // Produces `working` (only confirmed values censored) and the set of values the reviewer
    // rejected this run (so the snippet stage stays consistent with their decisions).
    let (working, rejected) = if stages.contains(&Stage::Censor) {
        run_censor_stage(
            &args,
            &extracted,
            parties.as_ref(),
            &learned_allowed,
            &doc_id,
            page_scope,
        )?
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

    // ── Snippet stage ───────────────────────────────────────────────────────
    if stages.contains(&Stage::Snippet) {
        write_snippet_outputs(&args, &extracted, &working, &options)?;
    }

    Ok(())
}

/// Parse and validate `--pages` against the document's pages. `None` when the flag is absent.
///
/// Errors when `--pages` is given for a page-less input (`.txt`) or requests a page beyond the
/// document's explicit page breaks (e.g. page 2 of a `.docx` with no breaks).
fn parse_page_selection(
    args: &ReviewArgs,
    page_numbers: Option<&[u32]>,
) -> Result<Option<PageSelection>> {
    let Some(spec) = &args.pages else {
        return Ok(None);
    };
    let selection = PageSelection::parse(spec)?;
    let Some(pages) = page_numbers else {
        bail!("--pages is only supported for .docx input (this document has no pages)");
    };
    let max_page = pages.iter().copied().max().unwrap_or(1);
    if selection.max_page() > max_page {
        bail!(
            "--pages requested page {} but the document has explicit page breaks only up to page \
             {max_page}; cannot scope by page",
            selection.max_page()
        );
    }
    Ok(Some(selection))
}

/// Split items into (reviewed, auto-censored) by the page scope. An item is reviewed when any of
/// its occurrences sits on a selected page; with no scope, every item is reviewed.
fn scope_items(
    items: Vec<censor::ReviewItem>,
    page_scope: Option<(&[u32], &PageSelection)>,
) -> (Vec<censor::ReviewItem>, Vec<censor::ReviewItem>) {
    let Some((pages, selection)) = page_scope else {
        return (items, Vec::new());
    };
    items.into_iter().partition(|item| {
        item.occurrences.iter().any(|occurrence| {
            pages
                .get(occurrence.block_index)
                .is_some_and(|&page| selection.contains(page))
        })
    })
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
    doc_id: &str,
    page_scope: Option<(&[u32], &PageSelection)>,
) -> Result<(Document, BTreeSet<String>)> {
    let options = CensorOptions {
        parties,
        allow: Some(learned_allowed),
    };
    let mut items = censor::plan_review(extracted, &options);
    censor::tag_occurrence_languages(extracted, &mut items, super::lang_override(&args.lang));

    // `--pages`: review only values on the selected pages; the rest are auto-censored (kept
    // confirmed for safety, but `reviewed: false` so they are not logged as human labels).
    let (reviewed_items, out_of_scope) = scope_items(items, page_scope);
    let mut decisions = run_censor_review(extracted, &reviewed_items)?;
    if !out_of_scope.is_empty() {
        println!(
            "Censor: {} value(s) on other pages auto-censored (not reviewed).",
            out_of_scope.len()
        );
        decisions.extend(out_of_scope.iter().map(|item| {
            CensorDecision::from_item(
                item,
                Verdict::Confirm {
                    final_type: item.detected_type.label().to_string(),
                },
                false,
            )
        }));
    }
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
    persist_decisions(args, &decisions, &source, doc_id);
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
fn persist_decisions(args: &ReviewArgs, decisions: &[CensorDecision], source: &str, doc_id: &str) {
    if !decisions.iter().any(|decision| decision.reviewed) {
        return;
    }
    let records = censor::decision_records(decisions, source, doc_id, learn::now_epoch_secs());
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
        assert_eq!(active_stages(&[], &[Stage::Snippet]), vec![Stage::Censor]);
    }

    #[test]
    fn sibling_path_uses_stem_beside_input() {
        assert_eq!(
            sibling_path(Path::new("dir/contract.docx"), ".stencil.md"),
            PathBuf::from("dir/contract.stencil.md")
        );
    }

    fn item_on_block(value: &str, block_index: usize) -> censor::ReviewItem {
        censor::ReviewItem {
            value: value.into(),
            detected_type: crate::censor::ValueType::Entity,
            method: "regex:test".into(),
            occurrences: vec![crate::model::Occurrence {
                block_index,
                ..Default::default()
            }],
        }
    }

    #[test]
    fn scope_items_partitions_reviewed_and_auto_censored_by_page() {
        let pages = [1u32, 2, 3]; // block 0 → p1, block 1 → p2, block 2 → p3
        let selection = PageSelection::parse("2").expect("valid");
        let items = vec![
            item_on_block("a", 0),
            item_on_block("b", 1),
            item_on_block("c", 2),
        ];
        let (reviewed, out_of_scope) = scope_items(items, Some((&pages, &selection)));
        assert_eq!(
            reviewed
                .iter()
                .map(|i| i.value.as_str())
                .collect::<Vec<_>>(),
            vec!["b"],
            "only the page-2 value is reviewed",
        );
        assert_eq!(out_of_scope.len(), 2, "pages 1 and 3 are auto-censored");
    }

    #[test]
    fn scope_items_reviews_everything_without_a_selection() {
        let items = vec![item_on_block("a", 0), item_on_block("b", 1)];
        let (reviewed, out_of_scope) = scope_items(items, None);
        assert_eq!(reviewed.len(), 2);
        assert!(out_of_scope.is_empty());
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
