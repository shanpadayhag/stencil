//! `stencil style` — the standalone styling review (v7).
//!
//! Walks every block's formatting, surfaces oddities for the human to fix in Word, and records
//! each block's `fine`/`weird` verdict to `styling.jsonl` (plus the per-document profile sidecar)
//! as training data for the future styling model. It **never edits the document**.
//!
//! Like the censor review it needs an interactive terminal (TTY). `--pages` scopes the review to
//! part of the document; the style profile is still built over the whole document so the per-role
//! norms stay accurate.

use std::io::IsTerminal;

use anyhow::{Result, bail};

use crate::cli::StyleArgs;
use crate::extract;
use crate::lang;
use crate::learn;
use crate::model::StyledBlock;
use crate::pages::PageSelection;
use crate::style;
use crate::style::review::{StyleVerdict, review as run_styling_review};

use super::{is_docx, lang_override};

/// Run the `style` subcommand.
pub fn run(args: StyleArgs) -> Result<()> {
    // The per-block review reads single keypresses; refuse without a terminal.
    if !std::io::stdin().is_terminal() {
        bail!("style needs an interactive terminal (TTY); run it directly in your terminal");
    }
    if !is_docx(&args.input) {
        bail!(
            "styling review is only available for .docx input (got `{}`)",
            args.input.display()
        );
    }

    let mut blocks = style::extract::from_path(&args.input)?;
    if blocks.is_empty() {
        println!("Styling: no blocks to review.");
        return Ok(());
    }

    // Validate `--pages` against the document's explicit page breaks before any review.
    let selection = parse_page_selection(&args, &blocks)?;

    // The id keying this document's styling records/profile (matches the `review` command's id).
    let doc_id = crate::doc_id::doc_id(&extract::from_path(&args.input)?);

    // Tag each block with its detected language for the records.
    tag_block_languages(&mut blocks, lang_override(&args.lang));

    // Profile over the whole document (accurate norms); review only the in-scope blocks.
    let profile = style::profile::build_profile(&blocks);
    let decisions = match &selection {
        // Scoped to `--pages`: review just those blocks. This is the only path that can be empty,
        // since the whole-document case already bailed above when there were no blocks.
        Some(sel) => {
            let reviewed: Vec<StyledBlock> = blocks
                .iter()
                .filter(|block| sel.contains(block.page))
                .cloned()
                .collect();
            if reviewed.is_empty() {
                println!("Styling: no blocks on the selected pages.");
                return Ok(());
            }
            run_styling_review(&reviewed, &profile)?
        }
        // Whole document: review the blocks in place — no clone needed.
        None => run_styling_review(&blocks, &profile)?,
    };
    let weird = decisions
        .iter()
        .filter(|decision| matches!(decision.verdict, StyleVerdict::Weird { .. }))
        .count();
    println!(
        "Styling: reviewed {} block(s) — {} fine, {weird} weird.",
        decisions.len(),
        decisions.len() - weird
    );

    // The styling model trains locally and the review never edits the document, so the log keeps
    // the real block text — a faithful feature, not a lossy censored copy.
    persist_styling(&args, &blocks, &profile, &decisions, &doc_id);
    Ok(())
}

/// Parse and validate `--pages` against the styled blocks' page range (styling is `.docx`-only, so
/// there is no page-less case here).
fn parse_page_selection(args: &StyleArgs, blocks: &[StyledBlock]) -> Result<Option<PageSelection>> {
    let Some(spec) = &args.pages else {
        return Ok(None);
    };
    let selection = PageSelection::parse(spec)?;
    let max_page = blocks
        .iter()
        .map(|block| block.page)
        .max()
        .unwrap_or(1)
        .max(1);
    if selection.max_page() > max_page {
        bail!(
            "--pages requested page {} but the document has explicit page breaks only up to page \
             {max_page}; cannot scope by page",
            selection.max_page()
        );
    }
    Ok(Some(selection))
}

/// Tag each styled block with its detected language.
fn tag_block_languages(blocks: &mut [StyledBlock], override_lang: Option<&str>) {
    let tags = {
        let texts: Vec<&str> = blocks.iter().map(|block| block.text.as_str()).collect();
        lang::tag_texts(&texts, override_lang)
    };
    for (block, tag) in blocks.iter_mut().zip(tags) {
        block.lang = tag.lang;
        block.lang_confidence = tag.confidence;
    }
}

/// Append the styling records and write the profile sidecar. Best-effort: a storage hiccup must
/// never fail the review the user just completed, so problems are reported, not propagated.
fn persist_styling(
    args: &StyleArgs,
    blocks: &[StyledBlock],
    profile: &crate::model::DocumentStyleProfile,
    decisions: &[style::review::StyleDecision],
    doc_id: &str,
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
        doc_id,
    ) {
        eprintln!("note: could not save styling records: {err}");
    }
}
