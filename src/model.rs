//! Shared document model: the block tree that every input format normalizes into
//! and every later stage operates on.
//!
//! A [`Document`] is an ordered list of [`Block`]s. Plain-text inputs produce only
//! [`Block::Paragraph`]s; `.docx` inputs additionally produce [`Block::Heading`]s
//! and [`Block::Table`]s (task T10).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// The current `mapping.json` schema version.
pub const MAPPING_VERSION: u32 = 1;

/// The reversible record of a censoring run: which placeholder stands for which real
/// value. Serialized to `mapping.json` and consumed by `restore`.
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
