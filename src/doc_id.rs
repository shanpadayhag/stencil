//! A stable, content-derived document id.
//!
//! Training records and the per-document style profile are keyed by this id rather than the
//! filename, because the same filename is reused across folders for different documents (so the
//! name collides). The id is the first 16 hex chars of the SHA-256 of the document's text, tagged
//! with block structure: the same content yields the same id regardless of path, and same-named
//! distinct documents get distinct ids. The filename is kept alongside the id for human traceability.

use sha2::{Digest, Sha256};

use crate::model::{Block, Document};

/// The stable content id for `document`: 16 lowercase hex characters.
///
/// ```
/// use std::path::PathBuf;
/// use stencil::doc_id::doc_id;
/// use stencil::model::{Block, Document};
///
/// let here = Document {
///     source: PathBuf::from("/x/contract.docx"),
///     blocks: vec![Block::Paragraph { text: "Hello".into() }],
/// };
/// let there = Document {
///     source: PathBuf::from("/y/contract.docx"),
///     blocks: vec![Block::Paragraph { text: "Hello".into() }],
/// };
/// // Same content, different paths → same id.
/// assert_eq!(doc_id(&here), doc_id(&there));
/// assert_eq!(doc_id(&here).len(), 16);
/// ```
pub fn doc_id(document: &Document) -> String {
    let mut hasher = Sha256::new();
    for block in &document.blocks {
        match block {
            Block::Heading { level, text } => {
                hasher.update([1u8, *level]);
                hasher.update(text.as_bytes());
            }
            Block::Paragraph { text } => {
                hasher.update([2u8]);
                hasher.update(text.as_bytes());
            }
            Block::Table { rows } => {
                hasher.update([3u8]);
                for row in rows {
                    for cell in row {
                        hasher.update(cell.text.as_bytes());
                        hasher.update([0u8]); // cell boundary
                    }
                    hasher.update([b'\n']); // row boundary
                }
            }
        }
        hasher.update([0xffu8]); // block boundary
    }
    hasher
        .finalize()
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn paragraph(text: &str) -> Document {
        Document {
            source: PathBuf::from("ignored.txt"),
            blocks: vec![Block::Paragraph { text: text.into() }],
        }
    }

    #[test]
    fn same_content_different_path_same_id() {
        let mut a = paragraph("Acme owes 3%");
        let mut b = paragraph("Acme owes 3%");
        a.source = PathBuf::from("/folder-a/contract.docx");
        b.source = PathBuf::from("/folder-b/contract.docx");
        assert_eq!(doc_id(&a), doc_id(&b));
    }

    #[test]
    fn different_content_different_id() {
        assert_ne!(doc_id(&paragraph("one")), doc_id(&paragraph("two")));
    }

    #[test]
    fn structure_changes_the_id() {
        // Same text, different block kind → different id (the tag bytes disambiguate).
        let para = paragraph("Title");
        let heading = Document {
            source: PathBuf::from("ignored.txt"),
            blocks: vec![Block::Heading {
                level: 1,
                text: "Title".into(),
            }],
        };
        assert_ne!(doc_id(&para), doc_id(&heading));
    }

    #[test]
    fn id_is_sixteen_hex_chars() {
        let id = doc_id(&paragraph("anything"));
        assert_eq!(id.len(), 16);
        assert!(id.chars().all(|ch| ch.is_ascii_hexdigit()));
    }
}
