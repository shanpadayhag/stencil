//! Walk a `.docx` into [`StyledBlock`]s, in document order.
//!
//! Paragraph-level properties (style id, alignment, indent, numbering) come from the
//! docx-rs public API. Run-level properties (font, size, bold, …) and paragraph spacing
//! live in private fields with no getters, so they are read by serializing the structs
//! to JSON and pulling the keys proven in the T23 spike (`tests/styling_spike.rs`); a
//! docx-rs bump that renames them fails that spike loudly. Effective-style inheritance
//! is deferred: only inline values plus the style id are recorded.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use docx_rs::{
    DocumentChild, Docx, Indent, LineSpacing, NumberingProperty, Paragraph, ParagraphChild, Run,
    RunChild, SpecialIndentType, TableCellContent, TableChild, TableRowChild, read_docx,
};
use serde_json::Value;

use crate::model::{BlockKind, IndentTwips, Numbering, ParaStyle, RunStyle, Spacing, StyledBlock};

/// Read a `.docx` file and extract one [`StyledBlock`] per visible block, in order.
///
/// Empty (whitespace-only) paragraphs are skipped, matching the text extractor; table
/// cells contribute one block per non-empty paragraph.
///
/// # Errors
/// Returns an error if the file cannot be read or is not a valid `.docx`.
pub fn from_path(path: &Path) -> Result<Vec<StyledBlock>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read `{}`", path.display()))?;
    let docx = read_docx(&bytes)
        .map_err(|err| anyhow!("failed to parse .docx `{}`: {err:?}", path.display()))?;
    Ok(styled_blocks(&docx))
}

/// Walk a parsed document into styled blocks, numbering them in emission order.
fn styled_blocks(docx: &Docx) -> Vec<StyledBlock> {
    let mut blocks = Vec::new();
    for child in &docx.document.children {
        match child {
            DocumentChild::Paragraph(paragraph) => {
                push_paragraph(&mut blocks, paragraph, false);
            }
            DocumentChild::Table(table) => {
                for TableChild::TableRow(row) in &table.rows {
                    for TableRowChild::TableCell(cell) in &row.cells {
                        for content in &cell.children {
                            if let TableCellContent::Paragraph(paragraph) = content {
                                push_paragraph(&mut blocks, paragraph, true);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    blocks
}

/// Append the styled block for `paragraph` (unless it is empty), assigning the next index.
fn push_paragraph(blocks: &mut Vec<StyledBlock>, paragraph: &Paragraph, in_table: bool) {
    let text = paragraph_text(paragraph);
    if text.trim().is_empty() {
        return;
    }
    let heading_level = heading_level(paragraph);
    let para = para_style(paragraph);
    let block_kind = classify(in_table, heading_level, &para.numbering);
    blocks.push(StyledBlock {
        block_index: blocks.len(),
        block_kind,
        heading_level,
        in_table,
        text,
        para,
        run: run_style(paragraph),
    });
}

/// Classify a paragraph: role (heading/list) wins over table membership.
fn classify(in_table: bool, heading_level: Option<u8>, numbering: &Numbering) -> BlockKind {
    if heading_level.is_some() {
        BlockKind::Heading
    } else if numbering.num_id.is_some() {
        BlockKind::ListItem
    } else if in_table {
        BlockKind::TableCell
    } else {
        BlockKind::Paragraph
    }
}

/// Read paragraph-level styling from a paragraph's properties.
fn para_style(paragraph: &Paragraph) -> ParaStyle {
    let property = &paragraph.property;
    ParaStyle {
        style_name: property.style.as_ref().map(|style| style.val.clone()),
        alignment: property.alignment.as_ref().map(|just| just.val.clone()),
        indent_twips: property
            .indent
            .as_ref()
            .map(indent_twips)
            .unwrap_or_default(),
        numbering: property
            .numbering_property
            .as_ref()
            .map(numbering)
            .unwrap_or_default(),
        spacing: property
            .line_spacing
            .as_ref()
            .map(spacing)
            .unwrap_or_default(),
    }
}

/// Convert docx-rs's [`Indent`] into [`IndentTwips`].
fn indent_twips(indent: &Indent) -> IndentTwips {
    let (hanging, first_line) = match indent.special_indent {
        Some(SpecialIndentType::Hanging(value)) => (Some(value), None),
        Some(SpecialIndentType::FirstLine(value)) => (None, Some(value)),
        None => (None, None),
    };
    IndentTwips {
        left: indent.start,
        right: indent.end,
        hanging,
        first_line,
    }
}

/// Convert docx-rs's [`NumberingProperty`] into [`Numbering`].
fn numbering(property: &NumberingProperty) -> Numbering {
    Numbering {
        num_id: property.id.as_ref().map(|id| id.id),
        ilvl: property.level.as_ref().map(|level| level.val),
    }
}

/// Read spacing from a [`LineSpacing`] via serde introspection (fields are private).
fn spacing(line_spacing: &LineSpacing) -> Spacing {
    let value = serde_json::to_value(line_spacing).unwrap_or(Value::Null);
    Spacing {
        before: value
            .get("before")
            .and_then(Value::as_u64)
            .map(|n| n as u32),
        after: value.get("after").and_then(Value::as_u64).map(|n| n as u32),
        line: value.get("line").and_then(Value::as_i64).map(|n| n as i32),
    }
}

/// Per-run styling, used to detect disagreement within a block.
#[derive(Clone, PartialEq, Eq)]
struct RunProps {
    font: Option<String>,
    size_half_pt: Option<u64>,
    bold: Option<bool>,
    italic: Option<bool>,
    underline: Option<String>,
    color: Option<String>,
}

/// Read one run's styling via serde introspection, using the T23-proven key paths.
fn run_props(run: &Run) -> RunProps {
    let value = serde_json::to_value(&run.run_property).unwrap_or(Value::Null);
    RunProps {
        font: value
            .pointer("/fonts/ascii")
            .and_then(Value::as_str)
            .map(String::from),
        size_half_pt: value.get("sz").and_then(Value::as_u64),
        bold: value.get("bold").and_then(Value::as_bool),
        italic: value.get("italic").and_then(Value::as_bool),
        underline: value
            .get("underline")
            .and_then(Value::as_str)
            .map(String::from),
        color: value.get("color").and_then(Value::as_str).map(String::from),
    }
}

/// Aggregate a block's run styling: the first text-bearing run's values, with `mixed`
/// set when any other text-bearing run disagrees.
fn run_style(paragraph: &Paragraph) -> RunStyle {
    let mut runs = Vec::with_capacity(paragraph.children.len());
    for child in &paragraph.children {
        if let ParagraphChild::Run(run) = child
            && run_has_text(run)
        {
            runs.push(run_props(run));
        }
    }
    let Some(first) = runs.first() else {
        return RunStyle::default();
    };
    let mixed = runs.iter().any(|props| props != first);
    RunStyle {
        font: first.font.clone(),
        size_half_pt: first.size_half_pt,
        bold: first.bold,
        italic: first.italic,
        underline: first.underline.clone(),
        color: first.color.clone(),
        mixed,
    }
}

/// Whether a run contributes any non-empty visible text.
fn run_has_text(run: &Run) -> bool {
    run.children
        .iter()
        .any(|child| matches!(child, RunChild::Text(text) if !text.text.is_empty()))
}

/// Concatenate the visible text of a paragraph's runs, preserving tabs and breaks.
fn paragraph_text(paragraph: &Paragraph) -> String {
    let mut text = String::new();
    for child in &paragraph.children {
        if let ParagraphChild::Run(run) = child {
            for run_child in &run.children {
                match run_child {
                    RunChild::Text(value) => text.push_str(&value.text),
                    RunChild::Tab(_) => text.push('\t'),
                    RunChild::Break(_) => text.push('\n'),
                    _ => {}
                }
            }
        }
    }
    text
}

/// The heading level from a paragraph's style id, if it is a heading.
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

#[cfg(test)]
mod tests {
    use super::*;
    use docx_rs::{
        AlignmentType, Docx, IndentLevel, NumberingId, Paragraph as DocxParagraph, Run as DocxRun,
        RunFonts, SpecialIndentType, Table, TableCell, TableRow,
    };

    /// Pack a `Docx` to a temp file, read it back through the extractor, and return the
    /// styled blocks. `label` keeps the temp path unique across parallel tests.
    fn round_trip(docx: Docx, label: &str) -> Vec<StyledBlock> {
        let path =
            std::env::temp_dir().join(format!("stencil_t28_{}_{label}.docx", std::process::id()));
        let file = fs::File::create(&path).expect("create temp docx");
        docx.build().pack(file).expect("pack docx");
        let blocks = from_path(&path).expect("extract styled blocks");
        let _ = fs::remove_file(&path);
        blocks
    }

    fn para(text: &str) -> DocxParagraph {
        DocxParagraph::new().add_run(DocxRun::new().add_text(text))
    }

    #[test]
    fn extracts_paragraph_with_run_styling() {
        let docx = Docx::new().add_paragraph(
            DocxParagraph::new().align(AlignmentType::Center).add_run(
                DocxRun::new()
                    .add_text("Hello")
                    .fonts(RunFonts::new().ascii("Courier New"))
                    .size(28)
                    .bold(),
            ),
        );

        let blocks = round_trip(docx, "para");

        assert_eq!(blocks.len(), 1);
        let block = &blocks[0];
        assert_eq!(block.block_index, 0);
        assert_eq!(block.block_kind, BlockKind::Paragraph);
        assert_eq!(block.heading_level, None);
        assert!(!block.in_table);
        assert_eq!(block.text, "Hello");
        assert_eq!(block.para.alignment.as_deref(), Some("center"));
        assert_eq!(block.run.font.as_deref(), Some("Courier New"));
        assert_eq!(block.run.size_half_pt, Some(28));
        assert_eq!(block.run.bold, Some(true));
        assert!(!block.run.mixed);
    }

    #[test]
    fn extracts_heading_with_level() {
        let docx = Docx::new()
            .add_paragraph(
                DocxParagraph::new()
                    .style("Heading2")
                    .add_run(DocxRun::new().add_text("Payment Terms")),
            )
            .add_paragraph(para("Body text."));

        let blocks = round_trip(docx, "heading");

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].block_kind, BlockKind::Heading);
        assert_eq!(blocks[0].heading_level, Some(2));
        assert_eq!(blocks[0].para.style_name.as_deref(), Some("Heading2"));
        assert_eq!(blocks[1].block_kind, BlockKind::Paragraph);
        assert_eq!(blocks[1].block_index, 1);
    }

    #[test]
    fn extracts_list_item_with_numbering_and_indent() {
        let docx = Docx::new().add_paragraph(
            DocxParagraph::new()
                .numbering(NumberingId::new(3), IndentLevel::new(1))
                .indent(Some(720), Some(SpecialIndentType::Hanging(360)), None, None)
                .add_run(DocxRun::new().add_text("First item")),
        );

        let blocks = round_trip(docx, "list");

        assert_eq!(blocks.len(), 1);
        let block = &blocks[0];
        assert_eq!(block.block_kind, BlockKind::ListItem);
        assert_eq!(block.para.numbering.num_id, Some(3));
        assert_eq!(block.para.numbering.ilvl, Some(1));
        assert_eq!(block.para.indent_twips.left, Some(720));
        assert_eq!(block.para.indent_twips.hanging, Some(360));
        assert_eq!(block.para.indent_twips.first_line, None);
    }

    #[test]
    fn extracts_table_cell_paragraphs() {
        let table = Table::new(vec![TableRow::new(vec![
            TableCell::new().add_paragraph(para("Item")),
            TableCell::new().add_paragraph(para("Value")),
        ])]);
        let docx = Docx::new().add_paragraph(para("Intro")).add_table(table);

        let blocks = round_trip(docx, "table");

        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].block_kind, BlockKind::Paragraph);
        assert!(!blocks[0].in_table);
        assert_eq!(blocks[1].block_kind, BlockKind::TableCell);
        assert!(blocks[1].in_table);
        assert_eq!(blocks[1].text, "Item");
        assert_eq!(blocks[2].block_kind, BlockKind::TableCell);
        assert_eq!(blocks[2].text, "Value");
        assert_eq!(blocks[2].block_index, 2);
    }

    #[test]
    fn flags_mixed_runs() {
        let docx = Docx::new().add_paragraph(
            DocxParagraph::new()
                .add_run(DocxRun::new().add_text("plain "))
                .add_run(DocxRun::new().add_text("bold").bold()),
        );

        let blocks = round_trip(docx, "mixed");

        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].run.mixed);
        // The first text-bearing run is unstyled, so its bold is unset.
        assert_eq!(blocks[0].run.bold, None);
    }

    #[test]
    fn uniform_runs_are_not_mixed() {
        let docx = Docx::new().add_paragraph(
            DocxParagraph::new()
                .add_run(DocxRun::new().add_text("one ").bold())
                .add_run(DocxRun::new().add_text("two").bold()),
        );

        let blocks = round_trip(docx, "uniform");

        assert!(!blocks[0].run.mixed);
        assert_eq!(blocks[0].run.bold, Some(true));
    }

    #[test]
    fn skips_empty_paragraphs() {
        let docx = Docx::new()
            .add_paragraph(para("real"))
            .add_paragraph(DocxParagraph::new())
            .add_paragraph(para("more"));

        let blocks = round_trip(docx, "empty");

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].block_index, 0);
        assert_eq!(blocks[1].block_index, 1);
    }

    #[test]
    fn parse_heading_style_levels() {
        assert_eq!(parse_heading_style("Heading1"), Some(1));
        assert_eq!(parse_heading_style("heading4"), Some(4));
        assert_eq!(parse_heading_style("Title"), Some(1));
        assert_eq!(parse_heading_style("Normal"), None);
    }
}
