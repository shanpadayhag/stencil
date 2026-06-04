//! Plain-text extractor: paragraphs split on blank lines, no headings.
//!
//! The splitting logic ([`blocks_from_str`]) is kept IO-free so it can be tested
//! directly; [`from_path`] is the thin file-reading wrapper.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use crate::model::{Block, Document};

/// Read a `.txt` file into a [`Document`] of [`Block::Paragraph`]s.
///
/// # Errors
/// Returns an error if the file cannot be read.
pub fn from_path(path: &Path) -> Result<Document> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read text file `{}`", path.display()))?;
    Ok(Document {
        source: path.to_path_buf(),
        blocks: blocks_from_str(&text),
    })
}

/// Split plain text into paragraph blocks.
///
/// A paragraph is a run of consecutive non-blank lines; a blank line (empty or
/// whitespace-only) separates paragraphs. Leading and trailing blank lines, and runs
/// of multiple blank lines, are ignored. Line breaks *within* a paragraph are
/// preserved. `\r\n` line endings are handled (the `\r` is dropped).
///
/// ```
/// use stencil::extract::txt::blocks_from_str;
/// use stencil::model::Block;
///
/// let blocks = blocks_from_str("First para.\n\nSecond para.");
/// assert_eq!(blocks, vec![
///     Block::Paragraph { text: "First para.".into() },
///     Block::Paragraph { text: "Second para.".into() },
/// ]);
/// ```
pub fn blocks_from_str(text: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut current: Vec<&str> = Vec::new();

    for line in text.lines() {
        if line.trim().is_empty() {
            flush_paragraph(&mut current, &mut blocks);
        } else {
            current.push(line);
        }
    }
    flush_paragraph(&mut current, &mut blocks);

    blocks
}

/// Join any accumulated lines into a paragraph block and reset the buffer.
fn flush_paragraph(lines: &mut Vec<&str>, blocks: &mut Vec<Block>) {
    if lines.is_empty() {
        return;
    }
    blocks.push(Block::Paragraph {
        text: lines.join("\n"),
    });
    lines.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paragraphs(text: &str) -> Vec<String> {
        blocks_from_str(text)
            .into_iter()
            .map(|block| match block {
                Block::Paragraph { text } => text,
                other => panic!("txt extractor should only produce paragraphs, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn empty_input_yields_no_blocks() {
        assert!(blocks_from_str("").is_empty());
    }

    #[test]
    fn whitespace_only_input_yields_no_blocks() {
        assert!(blocks_from_str("   \n\t\n  \n").is_empty());
    }

    #[test]
    fn single_line_paragraph() {
        assert_eq!(paragraphs("Just one line."), vec!["Just one line."]);
    }

    #[test]
    fn multi_line_paragraph_preserves_internal_breaks() {
        assert_eq!(
            paragraphs("line one\nline two\nline three"),
            vec!["line one\nline two\nline three"]
        );
    }

    #[test]
    fn blank_line_separates_paragraphs() {
        assert_eq!(paragraphs("alpha\n\nbeta"), vec!["alpha", "beta"]);
    }

    #[test]
    fn multiple_blank_lines_collapse_to_one_separator() {
        assert_eq!(paragraphs("alpha\n\n\n\nbeta"), vec!["alpha", "beta"]);
    }

    #[test]
    fn leading_and_trailing_blank_lines_ignored() {
        assert_eq!(paragraphs("\n\nalpha\n\n"), vec!["alpha"]);
    }

    #[test]
    fn whitespace_only_line_acts_as_separator() {
        assert_eq!(paragraphs("alpha\n   \t \nbeta"), vec!["alpha", "beta"]);
    }

    #[test]
    fn crlf_line_endings_are_handled() {
        assert_eq!(
            paragraphs("alpha\r\n\r\nbeta\r\ngamma"),
            vec!["alpha", "beta\ngamma"]
        );
    }

    #[test]
    fn from_path_reads_file_into_document() {
        // Unique-enough temp path without needing randomness/clock.
        let path = std::env::temp_dir().join(format!("stencil_t2_{}.txt", std::process::id()));
        fs::write(&path, "alpha\n\nbeta").expect("write temp file");

        let doc = from_path(&path).expect("read temp file");

        assert_eq!(doc.source, path);
        assert_eq!(
            doc.blocks,
            vec![
                Block::Paragraph {
                    text: "alpha".into()
                },
                Block::Paragraph {
                    text: "beta".into()
                },
            ]
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn from_path_missing_file_errors() {
        let missing = std::env::temp_dir().join("stencil_t2_definitely_missing_file.txt");
        assert!(from_path(&missing).is_err());
    }
}
