//! Per-candidate review files for cross-paragraph bracket spans.
//!
//! A cross-paragraph span (a `[` paired with a `]` in a later block) is genuinely
//! ambiguous — it may be an intended multi-paragraph variable/condition, or two stray
//! brackets that happened to pair. To let a human decide, each such span is written to
//! its own small Markdown file, **always censored**, so it is safe to paste into Claude
//! and to show colleagues before relying on it.
//!
//! The detection core stays pure: this module only consumes a [`Detection`] and builds
//! file contents in memory; the command layer does the actual writing.

use std::path::{Path, PathBuf};

use crate::censor::{CensorOptions, censor};
use crate::detect::{BracketKind, Detection};
use crate::model::{Block, Document};

/// Maximum length (in bytes/ASCII chars) of a generated filename slug.
const SLUG_MAX_LEN: usize = 50;

/// A review file ready to be written: its target path and full Markdown content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewFile {
    /// Where the file should be written.
    pub path: PathBuf,
    /// The censored Markdown content.
    pub content: String,
}

/// Build one censored review file per cross-paragraph span in `detection`.
///
/// Each span's covering blocks are sliced from `document`, censored with `options` (so
/// the output is always safe to share), rendered to Markdown, and assigned a path under
/// `out_dir` named by a readable slug from the span's opening text. Returns an empty
/// vector when there are no cross-paragraph spans.
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
/// let files = build_review_files(&doc, &detection, &CensorOptions::default(), Path::new("out"));
/// assert_eq!(files.len(), 1);
/// assert_eq!(files[0].path, PathBuf::from("out/if-the-buyer-defaults.md"));
/// ```
pub fn build_review_files(
    document: &Document,
    detection: &Detection,
    options: &CensorOptions<'_>,
    out_dir: &Path,
) -> Vec<ReviewFile> {
    let mut files = Vec::new();
    let mut used_slugs: Vec<String> = Vec::new();

    for hit in &detection.hits {
        if hit.kind != BracketKind::PairedCrossParagraph {
            continue;
        }
        // Whole covering blocks give the human full context; offsets within blocks do
        // not matter here, so block indices (stable across censoring) are all we need.
        let slice = Document {
            source: document.source.clone(),
            blocks: document.blocks[hit.block..=hit.end_block].to_vec(),
        };
        let censored = censor(&slice, options).document;

        let opening = opening_text(&censored);
        let slug = unique_slug(opening, &mut used_slugs);
        files.push(ReviewFile {
            path: out_dir.join(format!("{slug}.md")),
            content: render_review(&censored, hit.block, hit.end_block),
        });
    }

    files
}

/// Replace each cross-paragraph hit's `span_text` with a **censored** preview, so the
/// main inventory never displays a raw multi-paragraph span.
///
/// Censoring is done per-block before joining: censoring the joined string directly
/// would not work, since its outer `[`\u{2026}`]` reserves the whole interior from
/// redaction. `source` is the uncensored document; censoring runs with `options`, so the
/// preview is safe regardless of the main run's `--censor` flag. Single-block hits are
/// left untouched.
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
        hit.span_text = censored_span_preview(&censored);
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

/// The first paragraph's text starting just after its `[`, used to name the file. Falls
/// back to the whole first paragraph (or `""`) when there is no bracket.
fn opening_text(document: &Document) -> &str {
    let first = document
        .blocks
        .iter()
        .find_map(|block| match block {
            Block::Paragraph { text } => Some(text.as_str()),
            _ => None,
        })
        .unwrap_or("");
    match first.find('[') {
        Some(index) => &first[index + '['.len_utf8()..],
        None => first,
    }
}

/// Render a censored span as a standalone review document.
fn render_review(document: &Document, start_block: usize, end_block: usize) -> String {
    let mut out = String::new();
    out.push_str(
        "<!-- Generated by Stencil — censored cross-paragraph span for review. \
Safe to share; confirm whether this is an intended single variable before relying on it. -->\n",
    );
    out.push_str(&format!(
        "# Cross-paragraph span — blocks {start_block}\u{2013}{end_block} (\u{26a0} review)\n\n"
    ));
    out.push_str(
        "This bracketed span crosses paragraph boundaries, so Stencil flagged it for \
review. Confirm whether the `[`\u{2026}`]` is one intended variable/condition, or a \
stray bracket.\n\n---\n\n",
    );
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
    fn builds_one_censored_file_per_cross_paragraph_span() {
        let document = doc(vec![
            para("[Pay billing@acme.example by"),
            para("the end of the month]"),
        ]);
        let detection = detect(&document);
        let files = build_review_files(
            &document,
            &detection,
            &CensorOptions::default(),
            Path::new("out"),
        );

        assert_eq!(files.len(), 1);
        assert_eq!(
            files[0].path,
            PathBuf::from("out/pay-redacted-email-001-by.md")
        );
        // The real value is censored; the file is safe to share.
        assert!(files[0].content.contains("REDACTED_EMAIL_001"));
        assert!(!files[0].content.contains("billing@acme.example"));
        // Full span context, including the closing paragraph, is present.
        assert!(files[0].content.contains("the end of the month]"));
    }

    #[test]
    fn previews_are_censored_per_block() {
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
        assert!(hit.span_text.contains("REDACTED_EMAIL_001"));
        assert!(!hit.span_text.contains("billing@acme.example"));
        // Span runs from the `[` to the `]`.
        assert!(hit.span_text.starts_with('['));
        assert!(hit.span_text.ends_with(']'));
    }

    #[test]
    fn no_cross_paragraph_spans_yields_no_files() {
        let document = doc(vec![para("Pay [Amount] now."), para("Thanks.")]);
        let detection = detect(&document);
        let files = build_review_files(
            &document,
            &detection,
            &CensorOptions::default(),
            Path::new("out"),
        );
        assert!(files.is_empty());
    }
}
