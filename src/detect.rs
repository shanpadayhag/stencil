//! Bracket detection: greedy pairing, lone-bracket span guessing (flagged), and the
//! `[`/`]` balance tally.
//!
//! Detection is intentionally syntactic, not semantic: it finds bracketed spans and
//! flags the dubious ones. It never decides whether a bracket is a fill-in value or a
//! show/hide block — that judgement happens downstream (out of scope).
//!
//! ## Pairing rule
//! A left-to-right scan keeps a stack of open `[` positions. A `]` pairs with the most
//! recent unmatched `[` (well-defined even for the rare nested case). Brackets left
//! unmatched are *lone* and get a **guessed** span: a stray `[` runs to the end of its
//! line, a stray `]` runs from the start of its line. Guessed hits are flagged so a
//! human reviews them.

use crate::model::{Block, Document};

/// Character length above which a paired bracket is treated as "long" (likely a block
/// condition). Tunable; advisory only — it never changes detection, just labelling.
pub const LONG_BRACKET_THRESHOLD: usize = 120;

/// Maximum number of blocks a single cross-paragraph span may cover (the opening
/// block, the closing block, and any in between). A `[` only pairs with a `]` in a
/// later block when the span stays within this many blocks; beyond it the brackets
/// stay lone. Bounds the blast radius of a stray `[` — especially in heading-less
/// `.txt`, where the heading-boundary stop does not apply.
pub const MAX_CROSS_BLOCK_SPAN: usize = 5;

/// What kind of bracket a [`BracketHit`] represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BracketKind {
    /// A matched `[...]` pair.
    Paired,
    /// A matched pair whose span exceeds [`LONG_BRACKET_THRESHOLD`].
    PairedLong,
    /// A `[` paired with a `]` in a *later* block (the span crosses paragraph
    /// boundaries). Always flagged for human review — cross-paragraph pairing is
    /// genuinely ambiguous, so it is never treated as confident.
    PairedCrossParagraph,
    /// An unmatched `[` (guessed span to end of line).
    LoneOpen,
    /// An unmatched `]` (guessed span from start of line).
    LoneClose,
}

impl BracketKind {
    /// Human-facing label for the Markdown inventory "Kind" column.
    pub fn label(self) -> &'static str {
        match self {
            BracketKind::Paired => "paired",
            BracketKind::PairedLong => "paired (long)",
            BracketKind::PairedCrossParagraph => "paired (cross-paragraph)",
            BracketKind::LoneOpen => "lone `[`",
            BracketKind::LoneClose => "lone `]`",
        }
    }

    /// Whether this kind is a matched `[...]` pair (single-block or cross-paragraph),
    /// as opposed to a lone, unmatched bracket.
    pub fn is_paired(self) -> bool {
        matches!(
            self,
            BracketKind::Paired | BracketKind::PairedLong | BracketKind::PairedCrossParagraph
        )
    }
}

/// Confidence in a detected bracket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// A matched pair — trusted.
    Confident,
    /// A lone-bracket guess — needs human review.
    Guessed,
}

/// A single detected bracket occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BracketHit {
    /// The kind of bracket.
    pub kind: BracketKind,
    /// Confidence in the hit.
    pub status: Status,
    /// The bracketed text, brackets included (the guessed span for lone brackets). For
    /// a cross-paragraph pair this is the full span joined across blocks (paragraphs
    /// separated by a blank line).
    pub span_text: String,
    /// Index of the block where the span starts (the `[`).
    pub block: usize,
    /// Byte offset of the span start within `block`'s text.
    pub start: usize,
    /// Index of the block where the span ends (the `]`). Equal to `block` for every
    /// hit except a [`BracketKind::PairedCrossParagraph`], which ends in a later block.
    pub end_block: usize,
    /// Byte offset just past the span end within `end_block`'s text.
    pub end: usize,
}

/// Tally of raw `[` and `]` occurrences across a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Balance {
    /// Number of `[` characters seen.
    pub open: usize,
    /// Number of `]` characters seen.
    pub close: usize,
}

impl Balance {
    /// Whether the brackets are balanced (`[` count equals `]` count).
    pub fn is_balanced(self) -> bool {
        self.open == self.close
    }
}

/// The result of scanning a whole document.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Detection {
    /// All detected bracket hits, in document order.
    pub hits: Vec<BracketHit>,
    /// The raw bracket-balance tally.
    pub balance: Balance,
}

impl Detection {
    /// The lone (guessed) hits — the ones a human should review.
    pub fn guessed(&self) -> impl Iterator<Item = &BracketHit> {
        self.hits.iter().filter(|hit| hit.status == Status::Guessed)
    }
}

/// Detect brackets across every text field of a document.
///
/// ```
/// use stencil::detect::{detect, BracketKind};
/// use stencil::model::{Block, Document};
/// use std::path::PathBuf;
///
/// let doc = Document {
///     source: PathBuf::from("c.txt"),
///     blocks: vec![Block::Paragraph { text: "Pay [Buyer Name] now.".into() }],
/// };
/// let result = detect(&doc);
/// assert_eq!(result.hits.len(), 1);
/// assert_eq!(result.hits[0].kind, BracketKind::Paired);
/// assert!(result.balance.is_balanced());
/// ```
pub fn detect(document: &Document) -> Detection {
    let mut detection = Detection::default();
    for (block_index, block) in document.blocks.iter().enumerate() {
        for text in block_texts(block) {
            scan_text(text, block_index, &mut detection);
        }
    }
    pair_across_blocks(document, &mut detection);
    detection
}

/// Second pass: pair a lone `[` with a lone `]` in a *later* block, forming a
/// cross-paragraph span.
///
/// Lone brackets are matched greedily nearest-first, mirroring the single-block rule.
/// A pair is only formed when every block in the inclusive range is a
/// [`Block::Paragraph`] (the heading-boundary stop, which also excludes tables) and the
/// span covers at most [`MAX_CROSS_BLOCK_SPAN`] blocks. Formed pairs are always flagged
/// for review; brackets that cannot pair stay lone.
fn pair_across_blocks(document: &Document, detection: &mut Detection) {
    // `detection.hits` is in document order (blocks ascending, sorted within a block),
    // so a single stack pass pairs each `]` with the nearest unmatched earlier `[`.
    let mut open_stack: Vec<usize> = Vec::new();
    let mut consumed = vec![false; detection.hits.len()];
    let mut pairs: Vec<BracketHit> = Vec::new();

    for index in 0..detection.hits.len() {
        match detection.hits[index].kind {
            BracketKind::LoneOpen => open_stack.push(index),
            BracketKind::LoneClose => {
                // No open to pair with: this `]` stays lone.
                let Some(open_index) = open_stack.pop() else {
                    continue;
                };
                if let Some(span) = cross_block_span(
                    document,
                    &detection.hits[open_index],
                    &detection.hits[index],
                ) {
                    consumed[open_index] = true;
                    consumed[index] = true;
                    pairs.push(span);
                }
                // Not pairable: both stay lone. The popped `[` is not requeued — any
                // earlier `[` is even farther away, so it could not pair here either.
            }
            _ => {}
        }
    }

    if pairs.is_empty() {
        return;
    }

    let mut kept: Vec<BracketHit> = std::mem::take(&mut detection.hits)
        .into_iter()
        .zip(consumed)
        .filter_map(|(hit, gone)| (!gone).then_some(hit))
        .collect();
    kept.append(&mut pairs);
    kept.sort_by_key(|hit| (hit.block, hit.start));
    detection.hits = kept;
}

/// Build a cross-paragraph hit from a lone `[` and a lone `]`, or `None` if the pair is
/// disallowed (closing bracket not in a later block, a non-paragraph block in the way,
/// or the span exceeds [`MAX_CROSS_BLOCK_SPAN`] blocks).
fn cross_block_span(
    document: &Document,
    open: &BracketHit,
    close: &BracketHit,
) -> Option<BracketHit> {
    let (start_block, end_block) = (open.block, close.end_block);
    if end_block <= start_block || end_block - start_block + 1 > MAX_CROSS_BLOCK_SPAN {
        return None;
    }
    let all_paragraphs = (start_block..=end_block)
        .all(|index| matches!(document.blocks.get(index), Some(Block::Paragraph { .. })));
    if !all_paragraphs {
        return None;
    }

    Some(BracketHit {
        kind: BracketKind::PairedCrossParagraph,
        status: Status::Guessed,
        span_text: join_blocks(document, start_block, open.start, end_block, close.end),
        block: start_block,
        start: open.start,
        end_block,
        end: close.end,
    })
}

/// Join a span's text across paragraph blocks: the opening block from `start`, whole
/// middle blocks, and the closing block up to `end`, separated by blank lines.
fn join_blocks(
    document: &Document,
    start_block: usize,
    start: usize,
    end_block: usize,
    end: usize,
) -> String {
    let mut out = String::new();
    for index in start_block..=end_block {
        let text = paragraph_text(document, index);
        let slice = if index == start_block {
            &text[start..]
        } else if index == end_block {
            &text[..end]
        } else {
            text
        };
        if index != start_block {
            out.push_str("\n\n");
        }
        out.push_str(slice);
    }
    out
}

/// The text of a paragraph block, or `""` if absent or not a paragraph.
fn paragraph_text(document: &Document, index: usize) -> &str {
    match document.blocks.get(index) {
        Some(Block::Paragraph { text }) => text.as_str(),
        _ => "",
    }
}

/// Byte ranges of **paired** bracket spans in a single text field.
///
/// Used by censoring to avoid redacting variable labels (a bracket's interior is a
/// blank to send to Claude, not a sensitive baked-in value). Only confident paired
/// spans are reserved — a malformed lone bracket's guessed span can legitimately
/// contain real values that should still be censored.
pub(crate) fn paired_spans(text: &str) -> Vec<(usize, usize)> {
    let mut detection = Detection::default();
    scan_text(text, 0, &mut detection);
    detection
        .hits
        .iter()
        .filter(|hit| hit.kind.is_paired())
        .map(|hit| (hit.start, hit.start + hit.span_text.len()))
        .collect()
}

/// The text fields a block contributes to detection.
fn block_texts(block: &Block) -> Vec<&str> {
    match block {
        Block::Heading { text, .. } | Block::Paragraph { text } => vec![text.as_str()],
        Block::Table { rows } => rows
            .iter()
            .flat_map(|row| row.iter().map(|cell| cell.text.as_str()))
            .collect(),
    }
}

/// Scan one text field, appending hits (in positional order) and updating the tally.
fn scan_text(text: &str, block: usize, detection: &mut Detection) {
    let mut open_stack: Vec<usize> = Vec::new();
    let first_hit = detection.hits.len();

    for (idx, ch) in text.char_indices() {
        match ch {
            '[' => {
                detection.balance.open += 1;
                open_stack.push(idx);
            }
            ']' => {
                detection.balance.close += 1;
                match open_stack.pop() {
                    Some(open) => {
                        let end = idx + ch.len_utf8();
                        detection.hits.push(paired_hit(text, block, open, end));
                    }
                    None => {
                        let end = idx + ch.len_utf8();
                        let start = line_start(text, idx);
                        detection.hits.push(lone_hit(
                            BracketKind::LoneClose,
                            text,
                            block,
                            start,
                            end,
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    // Any `[` still open never matched: lone-open, guessed span to end of line.
    for open in open_stack {
        let end = line_end(text, open);
        detection
            .hits
            .push(lone_hit(BracketKind::LoneOpen, text, block, open, end));
    }

    // Lone-opens were appended last; restore positional order within this text.
    detection.hits[first_hit..].sort_by_key(|hit| hit.start);
}

/// Build a paired hit, classifying long spans.
fn paired_hit(text: &str, block: usize, start: usize, end: usize) -> BracketHit {
    let span = &text[start..end];
    let kind = if span.chars().count() > LONG_BRACKET_THRESHOLD {
        BracketKind::PairedLong
    } else {
        BracketKind::Paired
    };
    BracketHit {
        kind,
        status: Status::Confident,
        span_text: span.to_string(),
        block,
        start,
        end_block: block,
        end,
    }
}

/// Build a lone (guessed) hit from a span.
fn lone_hit(kind: BracketKind, text: &str, block: usize, start: usize, end: usize) -> BracketHit {
    BracketHit {
        kind,
        status: Status::Guessed,
        span_text: text[start..end].to_string(),
        block,
        start,
        end_block: block,
        end,
    }
}

/// Byte offset of the first character on the line containing `idx`.
fn line_start(text: &str, idx: usize) -> usize {
    text[..idx].rfind('\n').map_or(0, |nl| nl + 1)
}

/// Byte offset of the line break at/after `idx`, or the end of `text`.
fn line_end(text: &str, idx: usize) -> usize {
    text[idx..].find('\n').map_or(text.len(), |rel| idx + rel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Cell;
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
    fn paired_bracket_is_confident() {
        let result = detect(&doc(vec![para("Pay [Buyer Name] today.")]));
        assert_eq!(result.hits.len(), 1);
        let hit = &result.hits[0];
        assert_eq!(hit.kind, BracketKind::Paired);
        assert_eq!(hit.status, Status::Confident);
        assert_eq!(hit.span_text, "[Buyer Name]");
        assert_eq!(hit.block, 0);
        assert_eq!(hit.start, 4);
        assert!(result.balance.is_balanced());
    }

    #[test]
    fn long_paired_bracket_is_flagged_long() {
        let inner = "x".repeat(LONG_BRACKET_THRESHOLD + 5);
        let text = format!("[{inner}]");
        let result = detect(&doc(vec![para(&text)]));
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].kind, BracketKind::PairedLong);
        assert_eq!(result.hits[0].status, Status::Confident);
    }

    #[test]
    fn short_paired_at_threshold_stays_paired() {
        // Span length exactly the threshold is not "long" (strictly greater).
        let inner = "y".repeat(LONG_BRACKET_THRESHOLD - 2); // +2 brackets == threshold span
        let text = format!("[{inner}]");
        assert_eq!(text.chars().count(), LONG_BRACKET_THRESHOLD);
        let result = detect(&doc(vec![para(&text)]));
        assert_eq!(result.hits[0].kind, BracketKind::Paired);
    }

    #[test]
    fn lone_open_guesses_to_end_of_line() {
        let result = detect(&doc(vec![para("obligations of [ the seller\nnext line")]));
        assert_eq!(result.hits.len(), 1);
        let hit = &result.hits[0];
        assert_eq!(hit.kind, BracketKind::LoneOpen);
        assert_eq!(hit.status, Status::Guessed);
        assert_eq!(hit.span_text, "[ the seller");
        assert!(!result.balance.is_balanced());
    }

    #[test]
    fn lone_close_guesses_from_start_of_line() {
        let result = detect(&doc(vec![para("prev line\nthe seller ] owes")]));
        assert_eq!(result.hits.len(), 1);
        let hit = &result.hits[0];
        assert_eq!(hit.kind, BracketKind::LoneClose);
        assert_eq!(hit.status, Status::Guessed);
        assert_eq!(hit.span_text, "the seller ]");
    }

    #[test]
    fn multiple_paired_in_one_text() {
        let result = detect(&doc(vec![para("[a] and [b] and [c]")]));
        assert_eq!(result.hits.len(), 3);
        assert!(result.hits.iter().all(|h| h.kind == BracketKind::Paired));
        let spans: Vec<_> = result.hits.iter().map(|h| h.span_text.as_str()).collect();
        assert_eq!(spans, vec!["[a]", "[b]", "[c]"]);
    }

    #[test]
    fn nested_pairs_innermost_then_lone_open() {
        // "[a[b]": inner [b] pairs; outer [ is left lone. Positional order preserved.
        let result = detect(&doc(vec![para("[a[b]")]));
        assert_eq!(result.hits.len(), 2);
        assert_eq!(result.hits[0].kind, BracketKind::LoneOpen);
        assert_eq!(result.hits[0].start, 0);
        assert_eq!(result.hits[1].kind, BracketKind::Paired);
        assert_eq!(result.hits[1].span_text, "[b]");
    }

    #[test]
    fn no_brackets_is_empty_and_balanced() {
        let result = detect(&doc(vec![para("nothing to see here")]));
        assert!(result.hits.is_empty());
        assert_eq!(result.balance, Balance { open: 0, close: 0 });
        assert!(result.balance.is_balanced());
    }

    #[test]
    fn block_indices_track_across_blocks() {
        let result = detect(&doc(vec![
            para("intro, no brackets"),
            para("first [one]"),
            para("second [two]"),
        ]));
        assert_eq!(result.hits.len(), 2);
        assert_eq!(result.hits[0].block, 1);
        assert_eq!(result.hits[1].block, 2);
    }

    #[test]
    fn detects_inside_table_cells() {
        let table = Block::Table {
            rows: vec![vec![Cell::new("[Amount]"), Cell::new("plain")]],
        };
        let result = detect(&doc(vec![table]));
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].span_text, "[Amount]");
        assert_eq!(result.hits[0].block, 0);
    }

    #[test]
    fn guessed_iterator_returns_only_lone_hits() {
        let result = detect(&doc(vec![para("[ok] and [oops and ]stray")]));
        // "[ok]" paired; "[oops and ]" pairs greedily, leaving... let's just count guesses.
        let guessed: Vec<_> = result.guessed().collect();
        assert!(guessed.iter().all(|h| h.status == Status::Guessed));
    }

    #[test]
    fn balance_counts_raw_brackets() {
        let result = detect(&doc(vec![para("[a] [b [c]")]));
        assert_eq!(result.balance.open, 3);
        assert_eq!(result.balance.close, 2);
        assert!(!result.balance.is_balanced());
    }

    #[test]
    fn cross_paragraph_pair_is_one_flagged_hit() {
        let result = detect(&doc(vec![
            para("[If the buyer defaults"),
            para("all deposits are forfeited"),
            para("and the contract ends]"),
        ]));
        assert_eq!(
            result.hits.len(),
            1,
            "the two lone brackets become one span"
        );
        let hit = &result.hits[0];
        assert_eq!(hit.kind, BracketKind::PairedCrossParagraph);
        assert_eq!(hit.status, Status::Guessed);
        assert_eq!(hit.block, 0);
        assert_eq!(hit.end_block, 2);
        // The whole span, including the middle paragraph, is captured.
        assert!(hit.span_text.contains("If the buyer defaults"));
        assert!(hit.span_text.contains("all deposits are forfeited"));
        assert!(hit.span_text.contains("and the contract ends]"));
    }

    #[test]
    fn cross_paragraph_does_not_pair_across_a_heading() {
        let result = detect(&doc(vec![
            para("clause opens [here"),
            Block::Heading {
                level: 1,
                text: "New Section".into(),
            },
            para("and closes ] there"),
        ]));
        // No pairing across the heading: both brackets stay lone guesses.
        assert_eq!(result.hits.len(), 2);
        assert!(result.hits.iter().all(|hit| hit.status == Status::Guessed));
        assert!(
            result
                .hits
                .iter()
                .all(|hit| hit.kind != BracketKind::PairedCrossParagraph)
        );
    }

    #[test]
    fn cross_paragraph_respects_block_span_cap() {
        // Open in block 0, close in block 5 → span of 6 blocks > MAX_CROSS_BLOCK_SPAN.
        let mut blocks = vec![para("opens [here")];
        for _ in 0..4 {
            blocks.push(para("filler paragraph"));
        }
        blocks.push(para("closes ] here"));
        assert_eq!(blocks.len(), MAX_CROSS_BLOCK_SPAN + 1);

        let result = detect(&doc(blocks));
        assert!(
            result
                .hits
                .iter()
                .all(|hit| hit.kind != BracketKind::PairedCrossParagraph),
            "a span beyond the cap must not pair"
        );
    }

    #[test]
    fn cross_paragraph_pairs_at_the_span_cap() {
        // Open in block 0, close in the last allowed block → exactly the cap, pairs.
        let mut blocks = vec![para("opens [here")];
        for _ in 0..(MAX_CROSS_BLOCK_SPAN - 2) {
            blocks.push(para("filler paragraph"));
        }
        blocks.push(para("closes ] here"));
        assert_eq!(blocks.len(), MAX_CROSS_BLOCK_SPAN);

        let result = detect(&doc(blocks));
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].kind, BracketKind::PairedCrossParagraph);
        assert_eq!(result.hits[0].end_block, MAX_CROSS_BLOCK_SPAN - 1);
    }

    #[test]
    fn single_block_pairs_are_unaffected_by_cross_pass() {
        let result = detect(&doc(vec![para("Pay [Buyer Name] now."), para("Done.")]));
        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].kind, BracketKind::Paired);
        assert_eq!(result.hits[0].end_block, 0);
    }

    #[test]
    fn cross_paragraph_does_not_pair_across_a_table() {
        let result = detect(&doc(vec![
            para("opens [here"),
            Block::Table {
                rows: vec![vec![Cell::new("a"), Cell::new("b")]],
            },
            para("closes ] here"),
        ]));
        assert!(
            result
                .hits
                .iter()
                .all(|hit| hit.kind != BracketKind::PairedCrossParagraph)
        );
    }
}
