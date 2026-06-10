//! Shared document model: the block tree that every input format normalizes into
//! and every later stage operates on.
//!
//! A [`Document`] is an ordered list of [`Block`]s. Plain-text inputs produce only
//! [`Block::Paragraph`]s; `.docx` inputs additionally produce [`Block::Heading`]s
//! and [`Block::Table`]s (task T10).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Schema version of the in-memory [`Mapping`].
pub const MAPPING_VERSION: u32 = 1;

/// The placeholder ↔ value record of a censoring run. v6 no longer persists this (the
/// un-censoring round-trip was removed); it remains an internal byproduct of [`crate::censor::censor`],
/// which the snippet builder uses for its always-censored output (only `document` is consumed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mapping {
    /// Schema version (see [`MAPPING_VERSION`]).
    pub version: u32,
    /// The source document the mapping was produced from.
    pub source: String,
    /// One entry per distinct censored value.
    pub entries: Vec<MappingEntry>,
}

/// A single placeholder ↔ value record within a [`Mapping`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappingEntry {
    /// The placeholder token (e.g. `REDACTED_PERSON_001`).
    pub placeholder: String,
    /// The value category label (e.g. `PERSON`).
    #[serde(rename = "type")]
    pub value_type: String,
    /// The real value the placeholder stands for.
    pub value: String,
    /// How the value was detected: `party-list` or `regex:<kind>`.
    pub method: String,
    /// How many times the value occurred in the document.
    pub occurrences: usize,
}

/// A single table cell.
///
/// v1 holds plain text; nested block content inside cells is deferred.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Cell {
    /// The cell's text content.
    pub text: String,
}

impl Cell {
    /// Create a cell from anything string-like.
    ///
    /// ```
    /// use stencil::model::Cell;
    ///
    /// let cell = Cell::new("Buyer");
    /// assert_eq!(cell.text, "Buyer");
    /// ```
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// A single structural block of a document.
///
/// Sections are later derived from the [`Block::Heading`] boundaries; detection and
/// censoring operate on each block's text content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Block {
    /// A heading, with its outline level (1 = top level).
    Heading {
        /// Outline level, 1-based.
        level: u8,
        /// The heading text.
        text: String,
    },
    /// A body paragraph.
    Paragraph {
        /// The paragraph text; internal line breaks are preserved.
        text: String,
    },
    /// A table as a grid of rows, each a row of [`Cell`]s.
    Table {
        /// Rows, each holding its cells left-to-right.
        rows: Vec<Vec<Cell>>,
    },
}

/// The structural role of a [`StyledBlock`].
///
/// `heading` and `list_item` reflect the paragraph's semantic role; `table_cell` is a
/// plain paragraph inside a table cell (a heading or list item inside a table keeps its
/// role and is flagged by [`StyledBlock::in_table`] instead).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    /// A body paragraph.
    #[default]
    Paragraph,
    /// A heading paragraph (carries [`StyledBlock::heading_level`]).
    Heading,
    /// A numbered or bulleted list item.
    ListItem,
    /// A plain paragraph inside a table cell.
    TableCell,
}

impl BlockKind {
    /// The wire/display label for this kind (`paragraph`, `heading`, `list_item`, `table_cell`).
    ///
    /// ```
    /// use stencil::model::BlockKind;
    ///
    /// assert_eq!(BlockKind::ListItem.as_str(), "list_item");
    /// ```
    pub fn as_str(self) -> &'static str {
        match self {
            BlockKind::Paragraph => "paragraph",
            BlockKind::Heading => "heading",
            BlockKind::ListItem => "list_item",
            BlockKind::TableCell => "table_cell",
        }
    }
}

/// One location of a detected value within a document, with the review context captured at
/// that spot.
///
/// A [`crate::censor::ReviewItem`] groups all occurrences of one value; splitting a group in
/// review (v7) decides each `Occurrence` on its own. The byte offsets index into the text of
/// the field located by [`block_index`](Occurrence::block_index) and [`cell`](Occurrence::cell).
///
/// ```
/// use stencil::model::{BlockKind, Occurrence};
///
/// let occ = Occurrence {
///     block_index: 2,
///     cell: None,
///     start: 4,
///     end: 22,
///     block_kind: BlockKind::Paragraph,
///     heading_level: None,
///     shown_context: "Pay REDACTED_MONEY_001 now.".into(),
///     block_context: "The buyer shall pay REDACTED_MONEY_001 now.".into(),
///     ..Default::default()
/// };
/// assert_eq!(&occ.shown_context[occ.start..occ.end], "REDACTED_MONEY_001");
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Occurrence {
    /// Index of the containing [`Block`] in [`Document::blocks`].
    pub block_index: usize,
    /// For a [`BlockKind::TableCell`], the `(row, column)` of the cell; `None` otherwise.
    pub cell: Option<(usize, usize)>,
    /// Byte offset of the value's start within the located field's text.
    pub start: usize,
    /// Byte offset of the value's end (exclusive) within the located field's text.
    pub end: usize,
    /// The structural role of the containing block. Censor detection never emits
    /// [`BlockKind::ListItem`] — list items fold into [`BlockKind::Paragraph`] in v7.
    pub block_kind: BlockKind,
    /// Outline level when [`block_kind`](Occurrence::block_kind) is [`BlockKind::Heading`].
    pub heading_level: Option<u8>,
    /// The sentence-ish window shown to the reviewer / recorded as `shown_context`.
    pub shown_context: String,
    /// The whole-paragraph window recorded as the richer `block_context` feature.
    pub block_context: String,
    /// ISO language code of the containing block (a v7 training feature; empty until tagged).
    pub lang: String,
    /// Language-detection confidence in 0..=1 (`0.0` for a fallback/untagged occurrence).
    pub lang_confidence: f32,
}

/// Paragraph indentation, in twips (1/1440 inch); `None` where the property is unset.
///
/// `hanging` and `first_line` are mutually exclusive in OOXML's `special_indent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct IndentTwips {
    /// Left (start-edge) indent.
    pub left: Option<i32>,
    /// Right (end-edge) indent.
    pub right: Option<i32>,
    /// Hanging indent (first line outdented).
    pub hanging: Option<i32>,
    /// First-line indent (first line indented).
    pub first_line: Option<i32>,
}

/// A list-numbering reference: which list (`num_id`) and which level (`ilvl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Numbering {
    /// The numbering definition id.
    pub num_id: Option<usize>,
    /// The indent level within the list, 0-based.
    pub ilvl: Option<usize>,
}

/// Paragraph spacing, in twips; `None` where the property is unset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Spacing {
    /// Space before the paragraph.
    pub before: Option<u32>,
    /// Space after the paragraph.
    pub after: Option<u32>,
    /// Line spacing (interpretation depends on the paragraph's line rule).
    pub line: Option<i32>,
}

/// Paragraph-level styling read from a paragraph's properties.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ParaStyle {
    /// The paragraph style id (e.g. `Normal`, `Heading1`), if set.
    pub style_name: Option<String>,
    /// The alignment/justification value (e.g. `center`, `both`), if set.
    pub alignment: Option<String>,
    /// Indentation in twips.
    pub indent_twips: IndentTwips,
    /// List-numbering reference.
    pub numbering: Numbering,
    /// Paragraph spacing in twips.
    pub spacing: Spacing,
}

/// Run-level styling, aggregated over a block's text-bearing runs.
///
/// The fields hold the first text-bearing run's values; [`mixed`](RunStyle::mixed) is
/// `true` when later runs in the block disagree on any tracked property.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RunStyle {
    /// The ascii font name, if set.
    pub font: Option<String>,
    /// Font size in half-points (point size = `size_half_pt / 2`), if set.
    pub size_half_pt: Option<u64>,
    /// Bold, if explicitly set.
    pub bold: Option<bool>,
    /// Italic, if explicitly set.
    pub italic: Option<bool>,
    /// Underline style (e.g. `single`), if set.
    pub underline: Option<String>,
    /// Font color as a hex RGB string, if set.
    pub color: Option<String>,
    /// `true` when the block's runs do not all share the same styling.
    pub mixed: bool,
}

/// A single block's styling, captured in document order for the styling-review stage.
///
/// Produced by [`crate::style::extract`]. The [`text`](StyledBlock::text) is the block's
/// visible text; it is censored before being shown or logged (see the styling stage).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct StyledBlock {
    /// Position of this block among the emitted styled blocks, 0-based.
    pub block_index: usize,
    /// The block's structural role.
    pub block_kind: BlockKind,
    /// Outline level when [`block_kind`](StyledBlock::block_kind) is
    /// [`BlockKind::Heading`]; `None` otherwise.
    pub heading_level: Option<u8>,
    /// `true` when the block came from a table cell.
    pub in_table: bool,
    /// The block's visible text.
    pub text: String,
    /// Paragraph-level styling.
    pub para: ParaStyle,
    /// Run-level styling.
    pub run: RunStyle,
    /// ISO language code detected for this block (a v7 training feature; empty until tagged).
    #[serde(default)]
    pub lang: String,
    /// Language-detection confidence in 0..=1 (`0.0` for a fallback/untagged block).
    #[serde(default)]
    pub lang_confidence: f32,
    /// 1-based page (from explicit `.docx` page breaks) for `--pages` scoping; `0` if untracked.
    #[serde(default)]
    pub page: u32,
}

/// How many blocks use a given paragraph style id (`None` = unstyled/inherited).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyleCount {
    /// The paragraph style id, or `None` for blocks with no explicit style.
    pub style_name: Option<String>,
    /// Number of blocks with this style.
    pub count: usize,
}

/// The norm left-indent (twips) observed at a list level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IlvlIndentNorm {
    /// The list level, 0-based.
    pub ilvl: usize,
    /// The most common left indent among blocks at this level.
    pub left_norm: i32,
}

/// The grouping key for per-role style norms: a structural role, refined by heading level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleKey {
    /// The block's structural role.
    pub block_kind: BlockKind,
    /// The outline level for headings; `None` for non-headings.
    pub heading_level: Option<u8>,
}

/// A comparable summary of a block's salient styling, used to find each role's norm.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct StyleSignature {
    /// Paragraph style id.
    pub style_name: Option<String>,
    /// Run font.
    pub font: Option<String>,
    /// Run size in half-points.
    pub size_half_pt: Option<u64>,
    /// Run bold flag.
    pub bold: Option<bool>,
    /// Run italic flag.
    pub italic: Option<bool>,
    /// Paragraph alignment.
    pub alignment: Option<String>,
}

/// The dominant style signature shared by a role's peers, and how many peers it covers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleNorm {
    /// The role these peers share.
    pub role: RoleKey,
    /// The most common style signature among the peers.
    pub signature: StyleSignature,
    /// Number of blocks in this role.
    pub peers: usize,
}

/// A deterministic, descriptive picture of a document's styling: distributions and
/// per-level/per-role norms. It is **not** a detector — it produces no verdicts, only the
/// reference values from which a block's [`RelativeFeatures`] are derived.
///
/// Emitted once per document as a sidecar so any relative feature stays re-derivable from
/// the absolute [`StyledBlock`] rows plus this profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentStyleProfile {
    /// Total number of styled blocks the profile was built from.
    pub total_blocks: usize,
    /// Per-style block counts, most frequent first.
    pub style_counts: Vec<StyleCount>,
    /// The most common explicit run font across the document, if any.
    pub dominant_font: Option<String>,
    /// The most common explicit run size (half-points) across the document, if any.
    pub dominant_size_half_pt: Option<u64>,
    /// The norm left-indent for each observed list level.
    pub ilvl_indent_norms: Vec<IlvlIndentNorm>,
    /// The style norm for each observed role, in first-appearance order.
    pub role_norms: Vec<RoleNorm>,
}

/// A block's styling expressed relative to its [`DocumentStyleProfile`].
///
/// Each field measures *deviation from the norm*, not a judgement: a block that simply
/// inherits (unset font/size) counts as matching. These feed both the ML features and the
/// reviewer's "vs document" panel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelativeFeatures {
    /// Fraction of the document's blocks sharing this block's paragraph style (0..=1).
    pub style_doc_freq: f64,
    /// `true` when the block inherits its font or matches the document's dominant font.
    pub font_matches_doc_dominant: bool,
    /// `true` when the block inherits its size or matches the document's dominant size.
    pub size_matches_doc_dominant: bool,
    /// `true` when the block's style signature matches its role's norm.
    pub matches_role_peers: bool,
    /// For list items, the block's left indent minus its level's norm (twips); else `None`.
    pub indent_vs_ilvl_norm: Option<i32>,
}

/// A document normalized into an ordered list of [`Block`]s.
///
/// ```
/// use std::path::PathBuf;
/// use stencil::model::{Block, Document};
///
/// let doc = Document {
///     source: PathBuf::from("contract.txt"),
///     blocks: vec![Block::Paragraph { text: "Hello [Buyer Name].".into() }],
/// };
/// assert_eq!(doc.blocks.len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    /// The path the document was read from.
    pub source: PathBuf,
    /// The document's blocks, in order.
    pub blocks: Vec<Block>,
}
