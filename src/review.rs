//! Per-bracket snippet files.
//!
//! Every detected bracket gets its own small Markdown file, **always censored**, so it
//! is safe to paste into Claude or show colleagues. A bracket fully inside one block
//! uses the whole block (e.g. the entire paragraph) as its snippet; a cross-paragraph
//! span (a `[` paired with a `]` in a later block) uses the full `[`…`]` span and is
//! additionally flagged as ambiguous (it may be an intended multi-paragraph
//! variable/condition, or two stray brackets that happened to pair).
//!
//! Files are laid out under a `snippets/` directory: single-block snippets sit directly
//! inside it, cross-paragraph spans in a `cross-paragraph/` subfolder.
//!
//! The detection core stays pure: this module only consumes a [`Detection`] and builds
//! file contents in memory; the command layer does the actual writing.

use std::path::{Path, PathBuf};

use crate::censor::{CensorOptions, censor};
use crate::detect::{BracketKind, Detection};
use crate::model::{Block, Document};

/// Maximum length (in bytes/ASCII chars) of a generated filename slug.
const SLUG_MAX_LEN: usize = 50;

/// A review file ready to be written: its target path and full Markdown content,
/// plus the stable ID and block range it shares with the main inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewFile {
    /// Stable snippet ID (`S1`, `S2`, …) — shared with the `.stencil.md` inventory so the
    /// two artifacts cross-reference unambiguously.
    pub id: String,
    /// The block range this snippet covers (`(block, end_block)`), the inventory join key.
    pub range: (usize, usize),
    /// Where the file should be written.
    pub path: PathBuf,
    /// The censored Markdown content.
    pub content: String,
}

/// Build one censored snippet file per distinct bracket snippet in `detection`.
///
/// Each bracket's covering blocks are sliced from `document`, censored with `options`
/// (so the output is always safe to share), rendered to Markdown, and assigned a path
/// under `out_dir`: single-block snippets directly inside it, cross-paragraph spans in a
/// `cross-paragraph/` subfolder. Brackets that cover the same block range (e.g. two
/// brackets in one paragraph) share a single file. Returns an empty vector when there
/// are no brackets.
///
/// `document` should be the **uncensored** source: censoring is applied here exactly
/// once, with `options`, regardless of how the main run was invoked.
///
/// ```
/// use stencil::censor::CensorOptions;
/// use stencil::detect::detect;
/// use stencil::model::{Block, Document};
/// use stencil::review::build_review_files;
/// use std::path::{Path, PathBuf};
///
/// let doc = Document {
///     source: PathBuf::from("c.txt"),
///     blocks: vec![
///         Block::Paragraph { text: "[If the buyer defaults".into() },
///         Block::Paragraph { text: "and fails to cure]".into() },
///     ],
/// };
/// let detection = detect(&doc);
/// let files = build_review_files(&doc, &detection, &CensorOptions::default(), Path::new("snippets"), "c.stencil.md");
/// assert_eq!(files.len(), 1);
/// assert_eq!(files[0].id, "S1");
/// assert_eq!(files[0].path, PathBuf::from("snippets/cross-paragraph/if-the-buyer-defaults.md"));
/// // The file links back up to the main inventory.
/// assert!(files[0].content.contains("../../c.stencil.md"));
/// ```
pub fn build_review_files(
    document: &Document,
    detection: &Detection,
    options: &CensorOptions<'_>,
    out_dir: &Path,
    main_md_name: &str,
) -> Vec<ReviewFile> {
    let mut files = Vec::new();
    let mut seen_ranges: Vec<(usize, usize)> = Vec::new();
    let mut used_slugs: Vec<String> = Vec::new();
    let mut used_cross_slugs: Vec<String> = Vec::new();
    let cross_dir = out_dir.join("cross-paragraph");

    for hit in &detection.hits {
        let range = (hit.block, hit.end_block);
        // One file per distinct block range — brackets sharing a paragraph share a file.
        if seen_ranges.contains(&range) {
            continue;
        }
        seen_ranges.push(range);

        // Whole covering blocks give the human full context; offsets within blocks do
        // not matter here, so block indices (stable across censoring) are all we need.
        let slice = Document {
            source: document.source.clone(),
            blocks: document.blocks[hit.block..=hit.end_block].to_vec(),
        };
        let censored = censor(&slice, options).document;
        let opening = opening_text(&censored);
        let is_cross = hit.kind == BracketKind::PairedCrossParagraph;

        let (dir, used) = if is_cross {
            (cross_dir.as_path(), &mut used_cross_slugs)
        } else {
            (out_dir, &mut used_slugs)
        };
        let slug = unique_slug(opening, used);
        // IDs are assigned in stable hit order; one per distinct range.
        let id = format!("S{}", files.len() + 1);
        // A cross-paragraph file is one level deeper (`snippets/cross-paragraph/`), so it
        // needs an extra `../` to climb back to the main `.stencil.md`.
        let back_link = if is_cross {
            format!("../../{main_md_name}")
        } else {
            format!("../{main_md_name}")
        };
        let content = render_review(
            &censored,
            &id,
            hit.block,
            hit.end_block,
            is_cross,
            &back_link,
        );
        files.push(ReviewFile {
            id,
            range,
            path: dir.join(format!("{slug}.md")),
            content,
        });
    }

    files
}

/// Replace each cross-paragraph hit's `snippet` with a **censored** preview, so the main
/// inventory never displays a raw multi-paragraph span.
///
/// Censoring is done per-block before joining: censoring the joined string directly
/// would not work, since its outer `[`\u{2026}`]` reserves the whole interior from
/// redaction. `source` is the uncensored document; censoring runs with `options`, so the
/// preview is safe regardless of the main run's `--censor` flag. Single-block hits are
/// left untouched — their snippet is the surrounding paragraph, already shown in the
/// section context above the inventory.
pub fn censor_cross_paragraph_previews(
    detection: &mut Detection,
    source: &Document,
    options: &CensorOptions<'_>,
) {
    for hit in &mut detection.hits {
        if hit.kind != BracketKind::PairedCrossParagraph {
            continue;
        }
        let slice = Document {
            source: source.source.clone(),
            blocks: source.blocks[hit.block..=hit.end_block].to_vec(),
        };
        let censored = censor(&slice, options).document;
        hit.snippet = censored_span_preview(&censored);
    }
}

/// Rejoin a censored span from its `[` (in the first block) to its `]` (in the last),
/// whole blocks in between, as a single space-separated line. The inventory renderer
/// collapses whitespace and truncates, so this only needs to be safe and representative.
fn censored_span_preview(censored: &Document) -> String {
    let blocks = &censored.blocks;
    let last = blocks.len().saturating_sub(1);
    let mut parts: Vec<&str> = Vec::with_capacity(blocks.len());
    for (index, block) in blocks.iter().enumerate() {
        let Block::Paragraph { text } = block else {
            continue;
        };
        let slice = if index == 0 {
            text.find('[').map_or(text.as_str(), |at| &text[at..])
        } else if index == last {
            text.rfind(']')
                .map_or(text.as_str(), |at| &text[..at + ']'.len_utf8()])
        } else {
            text.as_str()
        };
        parts.push(slice);
    }
    parts.join(" ")
}

/// The first block's text, used to name the file. `slugify` strips the leading `[` and
/// other punctuation, so a cross-paragraph span beginning with `[` and a single-block
/// paragraph both yield a readable slug. Falls back to `""` for an empty/table-only
/// slice.
fn opening_text(document: &Document) -> &str {
    document
        .blocks
        .iter()
        .find_map(|block| match block {
            Block::Heading { text, .. } | Block::Paragraph { text } => Some(text.as_str()),
            Block::Table { .. } => None,
        })
        .unwrap_or("")
}

/// Render a censored snippet as a standalone Markdown document. Cross-paragraph spans
/// (`is_cross`) get an extra ambiguity warning; single-block snippets get a plain header.
/// Both carry a back-reference line (`id` + `back_link`) so a reader can crawl back to the
/// main `.stencil.md` inventory.
fn render_review(
    document: &Document,
    id: &str,
    start_block: usize,
    end_block: usize,
    is_cross: bool,
    back_link: &str,
) -> String {
    let mut out = String::new();
    let block_label = block_label(start_block, end_block);
    if is_cross {
        out.push_str(
            "<!-- Generated by Stencil — censored cross-paragraph span for review. \
Safe to share; confirm whether this is an intended single variable before relying on it. -->\n",
        );
        out.push_str(&format!(
            "# {id} — cross-paragraph span, blocks {block_label} (\u{26a0} review)\n\n"
        ));
        out.push_str(&format!(
            "> Back to inventory: [`{back_link}`]({back_link}) · kind: paired (cross-paragraph) · status: \u{26a0} GUESSED\n\n"
        ));
        out.push_str(
            "This bracketed span crosses paragraph boundaries, so Stencil flagged it for \
review. Confirm whether the `[`\u{2026}`]` is one intended variable/condition, or a \
stray bracket.\n\n---\n\n",
        );
    } else {
        out.push_str(
            "<!-- Generated by Stencil — censored snippet for review. Safe to share. -->\n",
        );
        out.push_str(&format!("# {id} — snippet, block {block_label}\n\n"));
        out.push_str(&format!(
            "> Back to inventory: [`{back_link}`]({back_link})\n\n---\n\n"
        ));
    }
    for block in &document.blocks {
        match block {
            Block::Paragraph { text } => {
                out.push_str(text);
                out.push_str("\n\n");
            }
            // Cross-paragraph spans only cover paragraphs, but stay total just in case.
            Block::Heading { level, text } => {
                let hashes = "#".repeat((*level).clamp(1, 6) as usize);
                out.push_str(&format!("{hashes} {text}\n\n"));
            }
            Block::Table { .. } => {}
        }
    }
    out
}

/// A human label for a block range: `"4"` for a single block, `"4–6"` for a span.
fn block_label(start_block: usize, end_block: usize) -> String {
    if start_block == end_block {
        start_block.to_string()
    } else {
        format!("{start_block}\u{2013}{end_block}")
    }
}

/// A filesystem-safe, readable slug, unique within the run (collisions get a numeric
/// suffix so files never overwrite each other).
fn unique_slug(text: &str, used: &mut Vec<String>) -> String {
    let base = slugify(text);
    if !used.contains(&base) {
        used.push(base.clone());
        return base;
    }
    let mut suffix = 2;
    loop {
        let candidate = format!("{base}-{suffix}");
        if !used.contains(&candidate) {
            used.push(candidate.clone());
            return candidate;
        }
        suffix += 1;
    }
}

/// Lowercase kebab-case slug of the leading words: ASCII alphanumerics kept, every other
/// run collapsed to a single `-`, capped at [`SLUG_MAX_LEN`]. Empty input yields `span`.
fn slugify(text: &str) -> String {
    let mut slug = String::with_capacity(SLUG_MAX_LEN);
    let mut pending_dash = false;
    for ch in text.chars() {
        if !ch.is_ascii_alphanumeric() {
            pending_dash = true;
            continue;
        }
        let needs_dash = pending_dash && !slug.is_empty();
        // Stop before exceeding the cap (a `-` only precedes an alphanumeric, so the
        // slug never ends on a trailing dash).
        if slug.len() + usize::from(needs_dash) + 1 > SLUG_MAX_LEN {
            break;
        }
        if needs_dash {
            slug.push('-');
        }
        pending_dash = false;
        slug.push(ch.to_ascii_lowercase());
    }
    if slug.is_empty() {
        "span".to_string()
    } else {
        slug
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::detect;
    use std::path::PathBuf;

    fn doc(blocks: Vec<Block>) -> Document {
        Document {
            source: PathBuf::from("test.txt"),
            blocks,
        }
    }

    fn para(text: &str) -> Block {
        Block::Paragraph { text: text.into() }
    }

    #[test]
    fn slugify_makes_readable_kebab_case() {
        assert_eq!(slugify("If the buyer defaults"), "if-the-buyer-defaults");
        assert_eq!(slugify("  multiple   spaces!! "), "multiple-spaces");
        assert_eq!(slugify("***"), "span");
    }

    #[test]
    fn slugify_caps_length() {
        let long = "word ".repeat(40);
        assert!(slugify(&long).len() <= SLUG_MAX_LEN);
    }

    #[test]
    fn unique_slug_suffixes_collisions() {
        let mut used = Vec::new();
        assert_eq!(unique_slug("Clause A", &mut used), "clause-a");
        assert_eq!(unique_slug("Clause A", &mut used), "clause-a-2");
        assert_eq!(unique_slug("Clause A", &mut used), "clause-a-3");
    }

    #[test]
    fn cross_paragraph_file_lands_in_subfolder_and_is_censored() {
        let document = doc(vec![
            para("[Pay billing@acme.example by"),
            para("the end of the month]"),
        ]);
        let detection = detect(&document);
        let files = build_review_files(
            &document,
            &detection,
            &CensorOptions::default(),
            Path::new("snippets"),
            "test.stencil.md",
        );

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].id, "S1");
        assert_eq!(
            files[0].path,
            PathBuf::from("snippets/cross-paragraph/pay-redacted-email-001-by.md")
        );
        // A cross-paragraph file sits one level deeper, so it climbs two levels back.
        assert!(
            files[0]
                .content
                .contains("[`../../test.stencil.md`](../../test.stencil.md)")
        );
        // The real value is censored; the file is safe to share.
        assert!(files[0].content.contains("REDACTED_EMAIL_001"));
        assert!(!files[0].content.contains("billing@acme.example"));
        // Full span context, including the closing paragraph, is present.
        assert!(files[0].content.contains("the end of the month]"));
    }

    #[test]
    fn single_block_bracket_gets_a_whole_paragraph_snippet_file() {
        let document = doc(vec![para("Pay billing@acme.example for [Invoice] now.")]);
        let detection = detect(&document);
        let files = build_review_files(
            &document,
            &detection,
            &CensorOptions::default(),
            Path::new("snippets"),
            "test.stencil.md",
        );

        // One single-block snippet, directly in the snippets dir (not the subfolder).
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].id, "S1");
        assert_eq!(files[0].range, (0, 0));
        assert_eq!(
            files[0].path,
            PathBuf::from("snippets/pay-redacted-email-001-for-invoice-now.md")
        );
        // Links back up one level to the main inventory.
        assert!(
            files[0]
                .content
                .contains("[`../test.stencil.md`](../test.stencil.md)")
        );
        assert!(files[0].content.contains("# S1 — snippet, block 0"));
        // The whole paragraph is the snippet, censored and safe to share.
        assert!(files[0].content.contains("REDACTED_EMAIL_001"));
        assert!(!files[0].content.contains("billing@acme.example"));
        assert!(files[0].content.contains("[Invoice]"));
    }

    #[test]
    fn two_brackets_in_one_paragraph_share_one_file() {
        let document = doc(vec![para("Pay [Amount] to [Buyer] today.")]);
        let detection = detect(&document);
        let files = build_review_files(
            &document,
            &detection,
            &CensorOptions::default(),
            Path::new("snippets"),
            "test.stencil.md",
        );
        assert_eq!(files.len(), 1, "two brackets, same paragraph → one file");
    }

    #[test]
    fn cross_paragraph_preview_snippet_is_censored() {
        let document = doc(vec![
            para("[Pay billing@acme.example by"),
            para("the end of the month]"),
        ]);
        let mut detection = detect(&document);
        censor_cross_paragraph_previews(&mut detection, &document, &CensorOptions::default());

        let hit = detection
            .hits
            .iter()
            .find(|hit| hit.kind == BracketKind::PairedCrossParagraph)
            .expect("cross-paragraph hit");
        assert!(hit.snippet.contains("REDACTED_EMAIL_001"));
        assert!(!hit.snippet.contains("billing@acme.example"));
        // Span runs from the `[` to the `]`.
        assert!(hit.snippet.starts_with('['));
        assert!(hit.snippet.ends_with(']'));
    }

    #[test]
    fn no_brackets_yields_no_files() {
        let document = doc(vec![para("Nothing to fill here."), para("Thanks.")]);
        let detection = detect(&document);
        let files = build_review_files(
            &document,
            &detection,
            &CensorOptions::default(),
            Path::new("snippets"),
            "test.stencil.md",
        );
        assert!(files.is_empty());
    }
}
