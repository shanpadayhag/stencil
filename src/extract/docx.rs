//! `.docx` extractor (read-only) via `docx-rs`: maps headings, paragraphs, and table
//! grids into the block model.
//!
//! Heading level comes from the paragraph style (`Heading1`, `Heading2`, … or
//! `Title`). Empty body paragraphs (very common in `.docx`) are dropped to keep the
//! block tree clean; headings and tables are always kept.
//!
//! Known v1 limitations: text inside hyperlinks and tracked-change insertions is not
//! extracted, and nested tables inside cells are flattened to their paragraph text.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use docx_rs::{
    Break, DocumentChild, Paragraph, ParagraphChild, Run, RunChild, TableCell, TableCellContent,
    TableChild, TableRowChild, read_docx,
};

use crate::model::{Block, Cell, Document};

/// Read a `.docx` file into the block model.
///
/// # Errors
/// Returns an error if the file cannot be read or is not a valid `.docx`.
pub fn from_path(path: &Path) -> Result<Document> {
    let blocks = paged_blocks(path)?
        .into_iter()
        .map(|(block, _)| block)
        .collect();
    Ok(Document {
        source: path.to_path_buf(),
        blocks,
    })
}

/// The 1-based page number of each block, parallel to [`from_path`]'s blocks.
///
/// Pages are delimited by **explicit** page breaks — a manual page break (`<w:br w:type="page"/>`)
/// or a paragraph's `pageBreakBefore`. Automatic (rendered) pagination is not in the file and is
/// not inferred, so a document with no explicit breaks is entirely page 1.
///
/// # Errors
/// Returns an error if the file cannot be read or is not a valid `.docx`.
pub fn page_numbers(path: &Path) -> Result<Vec<u32>> {
    Ok(paged_blocks(path)?
        .into_iter()
        .map(|(_, page)| page)
        .collect())
}

/// Walk the document once, returning each kept block with its 1-based page number. A page break
/// on a dropped (empty) paragraph still advances the page for the blocks that follow.
fn paged_blocks(path: &Path) -> Result<Vec<(Block, u32)>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read `{}`", path.display()))?;
    let docx = read_docx(&bytes)
        .map_err(|err| anyhow!("failed to parse .docx `{}`: {err:?}", path.display()))?;

    let mut blocks = Vec::new();
    let mut page = 1u32;
    for child in &docx.document.children {
        match child {
            DocumentChild::Paragraph(paragraph) => {
                if starts_new_page(paragraph) {
                    page += 1;
                }
                if let Some(block) = block_from_paragraph(paragraph) {
                    blocks.push((block, page));
                }
                if paragraph_has_page_break(paragraph) {
                    page += 1;
                }
            }
            DocumentChild::Table(table) => {
                blocks.push((block_from_table(&table.rows), page));
            }
            _ => {}
        }
    }
    Ok(blocks)
}

/// Whether a paragraph carries `pageBreakBefore` — it (and what follows) is on the next page.
pub(crate) fn starts_new_page(paragraph: &Paragraph) -> bool {
    paragraph.property.page_break_before == Some(true)
}

/// Whether a paragraph contains a manual page break in one of its runs.
pub(crate) fn paragraph_has_page_break(paragraph: &Paragraph) -> bool {
    paragraph.children.iter().any(|child| match child {
        ParagraphChild::Run(run) => run_has_page_break(run),
        _ => false,
    })
}

/// Whether a run contains a `<w:br w:type="page"/>`.
fn run_has_page_break(run: &Run) -> bool {
    run.children
        .iter()
        .any(|child| matches!(child, RunChild::Break(brk) if is_page_break(brk)))
}

/// Whether a break is a page break. The `break_type` field is private, but `Break` serializes as
/// `{"breakType":"page"}`, so we read it via serde (the same introspection the styling stage uses).
fn is_page_break(brk: &Break) -> bool {
    serde_json::to_value(brk)
        .ok()
        .as_ref()
        .and_then(|value| value.get("breakType"))
        .and_then(|kind| kind.as_str())
        .is_some_and(|kind| kind == "page")
}

/// Convert a paragraph to a heading or paragraph block, dropping empty body paragraphs.
fn block_from_paragraph(paragraph: &Paragraph) -> Option<Block> {
    let text = paragraph_text(paragraph);
    match heading_level(paragraph) {
        Some(level) => Some(Block::Heading { level, text }),
        None if text.trim().is_empty() => None,
        None => Some(Block::Paragraph { text }),
    }
}

/// Concatenate the visible text of a paragraph's runs.
fn paragraph_text(paragraph: &Paragraph) -> String {
    let mut text = String::new();
    for child in &paragraph.children {
        if let ParagraphChild::Run(run) = child {
            for run_child in &run.children {
                match run_child {
                    RunChild::Text(t) => text.push_str(&t.text),
                    RunChild::Tab(_) => text.push('\t'),
                    RunChild::Break(_) => text.push('\n'),
                    _ => {}
                }
            }
        }
    }
    text
}

/// The heading level from a paragraph's style, if it is a heading.
fn heading_level(paragraph: &Paragraph) -> Option<u8> {
    let style = paragraph.property.style.as_ref()?;
    parse_heading_style(&style.val)
}

/// Parse a paragraph-style id into a heading level (`Heading2` → 2, `Title` → 1).
fn parse_heading_style(style_id: &str) -> Option<u8> {
    let lowered = style_id.to_ascii_lowercase();
    if lowered == "title" {
        return Some(1);
    }
    let rest = lowered.strip_prefix("heading")?;
    let digits: String = rest
        .trim()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    if digits.is_empty() {
        Some(1)
    } else {
        digits.parse::<u8>().ok().map(|level| level.clamp(1, 6))
    }
}

/// Build a table block from its rows.
fn block_from_table(rows: &[TableChild]) -> Block {
    let rows = rows
        .iter()
        .map(|TableChild::TableRow(row)| {
            row.cells
                .iter()
                .map(|TableRowChild::TableCell(cell)| Cell {
                    text: cell_text(cell),
                })
                .collect()
        })
        .collect();
    Block::Table { rows }
}

/// Join a cell's paragraph texts with newlines.
fn cell_text(cell: &TableCell) -> String {
    cell.children
        .iter()
        .filter_map(|content| match content {
            TableCellContent::Paragraph(paragraph) => Some(paragraph_text(paragraph)),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use docx_rs::{Docx, Paragraph as DocxParagraph, Run, Table, TableCell, TableRow};

    #[test]
    fn parse_heading_style_levels() {
        assert_eq!(parse_heading_style("Heading1"), Some(1));
        assert_eq!(parse_heading_style("heading3"), Some(3));
        assert_eq!(parse_heading_style("Title"), Some(1));
        assert_eq!(parse_heading_style("Normal"), None);
        assert_eq!(parse_heading_style("Heading"), Some(1));
    }

    /// Build a `.docx` fixture with docx-rs, then read it back through the extractor.
    /// `label` keeps the temp path unique across tests running in parallel.
    fn round_trip(docx: Docx, label: &str) -> Document {
        let path =
            std::env::temp_dir().join(format!("stencil_t10_{}_{label}.docx", std::process::id()));
        let file = fs::File::create(&path).expect("create temp docx");
        docx.build().pack(file).expect("pack docx");

        let document = from_path(&path).expect("read docx");
        let _ = fs::remove_file(&path);
        document
    }

    fn para(text: &str) -> DocxParagraph {
        DocxParagraph::new().add_run(Run::new().add_text(text))
    }

    #[test]
    fn reads_heading_and_paragraph() {
        let docx = Docx::new()
            .add_paragraph(
                DocxParagraph::new()
                    .style("Heading1")
                    .add_run(Run::new().add_text("Payment Terms")),
            )
            .add_paragraph(para("The deposit is [Amount]."));

        let doc = round_trip(docx, "heading");

        assert_eq!(
            doc.blocks,
            vec![
                Block::Heading {
                    level: 1,
                    text: "Payment Terms".into()
                },
                Block::Paragraph {
                    text: "The deposit is [Amount].".into()
                },
            ]
        );
    }

    #[test]
    fn drops_empty_paragraphs() {
        let docx = Docx::new()
            .add_paragraph(para("real content"))
            .add_paragraph(DocxParagraph::new()) // empty
            .add_paragraph(para("more content"));

        let doc = round_trip(docx, "empty");
        assert_eq!(doc.blocks.len(), 2);
    }

    #[test]
    fn page_numbers_track_manual_break_and_page_break_before() {
        use docx_rs::BreakType;
        let docx = Docx::new()
            // page 1: a plain paragraph, then one ending in a manual page break.
            .add_paragraph(para("page one"))
            .add_paragraph(
                DocxParagraph::new()
                    .add_run(Run::new().add_text("still one").add_break(BreakType::Page)),
            )
            // page 2 (after the manual break).
            .add_paragraph(para("page two"))
            // page 3 (this paragraph carries pageBreakBefore).
            .add_paragraph(
                DocxParagraph::new()
                    .page_break_before(true)
                    .add_run(Run::new().add_text("page three")),
            );

        let path =
            std::env::temp_dir().join(format!("stencil_t42_pages_{}.docx", std::process::id()));
        let file = fs::File::create(&path).expect("create temp docx");
        docx.build().pack(file).expect("pack docx");
        let pages = page_numbers(&path).expect("page numbers");
        let doc = from_path(&path).expect("read docx");
        let _ = fs::remove_file(&path);

        assert_eq!(doc.blocks.len(), 4, "all four paragraphs kept");
        assert_eq!(
            pages,
            vec![1, 1, 2, 3],
            "manual break and pageBreakBefore advance the page"
        );
    }

    #[test]
    fn no_breaks_means_everything_is_page_one() {
        let docx = Docx::new()
            .add_paragraph(para("alpha"))
            .add_paragraph(para("beta"));
        let path =
            std::env::temp_dir().join(format!("stencil_t42_nobreak_{}.docx", std::process::id()));
        let file = fs::File::create(&path).expect("create temp docx");
        docx.build().pack(file).expect("pack docx");
        let pages = page_numbers(&path).expect("page numbers");
        let _ = fs::remove_file(&path);
        assert_eq!(pages, vec![1, 1]);
    }

    #[test]
    fn reads_table_into_grid() {
        let table = Table::new(vec![
            TableRow::new(vec![
                TableCell::new().add_paragraph(para("Item")),
                TableCell::new().add_paragraph(para("Value")),
            ]),
            TableRow::new(vec![
                TableCell::new().add_paragraph(para("Fee")),
                TableCell::new().add_paragraph(para("[Fee]")),
            ]),
        ]);
        let docx = Docx::new().add_table(table);

        let doc = round_trip(docx, "table");

        assert_eq!(doc.blocks.len(), 1);
        match &doc.blocks[0] {
            Block::Table { rows } => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0][0].text, "Item");
                assert_eq!(rows[1][1].text, "[Fee]");
            }
            other => panic!("expected a table, got {other:?}"),
        }
    }
}
