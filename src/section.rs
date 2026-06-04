//! Section slicing: group a document's blocks into the context units that the
//! Markdown renderer emits one entry per.
//!
//! ## Rule
//! - A [`Block::Heading`] starts a section that runs until the next heading; every
//!   block in between (the heading plus its body) is the section's context.
//! - Blocks that sit *above the first heading* — including the whole of a heading-less
//!   `.txt` — fall back to **one section per block** (the bracket's own paragraph is
//!   its context).
//!
//! Each section also carries the [`BracketHit`]s found within its blocks. Sections
//! with no hits are still produced; the renderer decides which to emit.

use crate::detect::{BracketHit, Detection};
use crate::model::{Block, Document};

/// A context unit: a run of blocks plus the bracket hits found within them.
#[derive(Debug)]
pub struct Section<'a> {
    /// The blocks forming this section's context, in order. For a heading-governed
    /// section the first block is the [`Block::Heading`].
    pub blocks: Vec<&'a Block>,
    /// The bracket hits located within this section's blocks, in document order.
    pub hits: Vec<&'a BracketHit>,
}

impl<'a> Section<'a> {
    /// The section's heading as `(level, text)`, if it is heading-governed.
    pub fn heading(&self) -> Option<(u8, &'a str)> {
        match self.blocks.first() {
            Some(Block::Heading { level, text }) => Some((*level, text.as_str())),
            _ => None,
        }
    }

    /// Whether the section contains any detected brackets.
    pub fn has_hits(&self) -> bool {
        !self.hits.is_empty()
    }
}

/// Slice a document into sections and attach each section's bracket hits.
///
/// ```
/// use stencil::detect::detect;
/// use stencil::model::{Block, Document};
/// use stencil::section::sections;
/// use std::path::PathBuf;
///
/// let doc = Document {
///     source: PathBuf::from("c.txt"),
///     blocks: vec![
///         Block::Heading { level: 1, text: "Payment".into() },
///         Block::Paragraph { text: "Pay [Amount].".into() },
///     ],
/// };
/// let detection = detect(&doc);
/// let secs = sections(&doc, &detection);
/// assert_eq!(secs.len(), 1);
/// assert_eq!(secs[0].heading(), Some((1, "Payment")));
/// assert_eq!(secs[0].hits.len(), 1);
/// ```
pub fn sections<'a>(document: &'a Document, detection: &'a Detection) -> Vec<Section<'a>> {
    let hits_by_block = group_hits_by_block(document.blocks.len(), detection);

    let mut sections: Vec<Section<'a>> = Vec::new();
    let mut current_heading: Option<Section<'a>> = None;

    for (index, block) in document.blocks.iter().enumerate() {
        let block_hits = || hits_by_block[index].iter().copied();

        if matches!(block, Block::Heading { .. }) {
            if let Some(section) = current_heading.take() {
                sections.push(section);
            }
            current_heading = Some(Section {
                blocks: vec![block],
                hits: block_hits().collect(),
            });
        } else if let Some(section) = current_heading.as_mut() {
            section.blocks.push(block);
            section.hits.extend(block_hits());
        } else {
            // No governing heading yet: this block is its own fallback section.
            sections.push(Section {
                blocks: vec![block],
                hits: block_hits().collect(),
            });
        }
    }

    if let Some(section) = current_heading.take() {
        sections.push(section);
    }

    sections
}

/// Bucket the detection's hits by their block index for O(1) lookup per block.
fn group_hits_by_block<'a>(
    block_count: usize,
    detection: &'a Detection,
) -> Vec<Vec<&'a BracketHit>> {
    let mut by_block: Vec<Vec<&'a BracketHit>> = vec![Vec::new(); block_count];
    for hit in &detection.hits {
        if let Some(bucket) = by_block.get_mut(hit.block) {
            bucket.push(hit);
        }
    }
    by_block
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

    fn heading(level: u8, text: &str) -> Block {
        Block::Heading {
            level,
            text: text.into(),
        }
    }

    fn para(text: &str) -> Block {
        Block::Paragraph { text: text.into() }
    }

    #[test]
    fn heading_groups_following_paragraphs() {
        let document = doc(vec![
            heading(2, "Payment Terms"),
            para("The deposit is [Amount]."),
            para("Due within [days] days."),
        ]);
        let detection = detect(&document);
        let secs = sections(&document, &detection);

        assert_eq!(secs.len(), 1);
        assert_eq!(secs[0].heading(), Some((2, "Payment Terms")));
        assert_eq!(secs[0].blocks.len(), 3);
        assert_eq!(secs[0].hits.len(), 2);
    }

    #[test]
    fn two_headings_make_two_sections() {
        let document = doc(vec![
            heading(1, "One"),
            para("[a]"),
            heading(1, "Two"),
            para("[b]"),
        ]);
        let detection = detect(&document);
        let secs = sections(&document, &detection);

        assert_eq!(secs.len(), 2);
        assert_eq!(secs[0].heading(), Some((1, "One")));
        assert_eq!(secs[1].heading(), Some((1, "Two")));
        assert_eq!(secs[0].hits[0].span_text, "[a]");
        assert_eq!(secs[1].hits[0].span_text, "[b]");
    }

    #[test]
    fn headingless_document_is_one_section_per_paragraph() {
        let document = doc(vec![para("[a]"), para("[b]"), para("no brackets")]);
        let detection = detect(&document);
        let secs = sections(&document, &detection);

        assert_eq!(secs.len(), 3);
        assert!(secs.iter().all(|s| s.heading().is_none()));
        assert!(secs[0].has_hits());
        assert!(secs[1].has_hits());
        assert!(!secs[2].has_hits());
    }

    #[test]
    fn blocks_before_first_heading_fall_back_to_paragraphs() {
        let document = doc(vec![
            para("preamble [x]"),
            heading(1, "Body"),
            para("clause [y]"),
        ]);
        let detection = detect(&document);
        let secs = sections(&document, &detection);

        // One fallback section for the preamble, one heading section for the body.
        assert_eq!(secs.len(), 2);
        assert_eq!(secs[0].heading(), None);
        assert_eq!(secs[0].hits[0].span_text, "[x]");
        assert_eq!(secs[1].heading(), Some((1, "Body")));
        assert_eq!(secs[1].hits[0].span_text, "[y]");
    }

    #[test]
    fn empty_document_has_no_sections() {
        let document = doc(vec![]);
        let detection = detect(&document);
        assert!(sections(&document, &detection).is_empty());
    }

    #[test]
    fn heading_with_no_body_still_a_section() {
        let document = doc(vec![heading(1, "Lonely [heading]")]);
        let detection = detect(&document);
        let secs = sections(&document, &detection);
        assert_eq!(secs.len(), 1);
        assert_eq!(secs[0].hits.len(), 1);
        assert_eq!(secs[0].hits[0].span_text, "[heading]");
    }
}
