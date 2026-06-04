//! End-to-end pipeline test on a real `.docx`: extract → detect → section → render.
//!
//! The fixture is generated with docx-rs's builder so no binary asset is checked in.

use std::fs;
use std::path::PathBuf;

use docx_rs::{Docx, Paragraph, Run, Table, TableCell, TableRow};

use stencil::detect::detect;
use stencil::extract;
use stencil::render::render;
use stencil::section::sections;

fn unique_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("stencil_it_{}_{label}.docx", std::process::id()))
}

fn run(text: &str) -> Run {
    Run::new().add_text(text)
}

#[test]
fn docx_detect_pipeline_preserves_headings_and_tables() {
    let docx = Docx::new()
        .add_paragraph(
            Paragraph::new()
                .style("Heading1")
                .add_run(run("Payment Terms")),
        )
        .add_paragraph(Paragraph::new().add_run(run("The deposit is [Amount] due in [days] days.")))
        .add_table(Table::new(vec![
            TableRow::new(vec![
                TableCell::new().add_paragraph(Paragraph::new().add_run(run("Item"))),
                TableCell::new().add_paragraph(Paragraph::new().add_run(run("Value"))),
            ]),
            TableRow::new(vec![
                TableCell::new().add_paragraph(Paragraph::new().add_run(run("Fee"))),
                TableCell::new().add_paragraph(Paragraph::new().add_run(run("[Fee]"))),
            ]),
        ]));

    let path = unique_path("pipeline");
    let file = fs::File::create(&path).expect("create temp docx");
    docx.build().pack(file).expect("pack docx");

    let document = extract::from_path(&path).expect("extract docx");
    let detection = detect(&document);
    let markdown = render(&document.source, &sections(&document, &detection));

    let _ = fs::remove_file(&path);

    // Heading preserved as a Markdown heading.
    assert!(markdown.contains("# Payment Terms"));
    // Paragraph brackets detected; the inventory snippet is the whole paragraph.
    assert!(
        markdown.contains("| `The deposit is [Amount] due in [days] days.` | paired | confident |")
    );
    // Table rendered as a Markdown grid, and its bracket detected (snippet is the cell).
    assert!(markdown.contains("| Item | Value |"));
    assert!(markdown.contains("| Fee | [Fee] |"));
    assert!(markdown.contains("| `[Fee]` | paired | confident |"));
    // Three brackets total across the document.
    assert_eq!(detection.hits.len(), 3);
    assert!(detection.balance.is_balanced());
}
