//! End-to-end pipeline tests for the v6 `review` stages, exercised at the library level on a
//! real `.docx` fixture. The interactive keypress loops need a PTY and so are out of scope here
//! (their pure decision logic is unit-tested); these tests drive the *non-interactive* halves —
//! detect → apply → log for censor, and extract → censor-text → profile → persist for styling —
//! with synthetic decisions standing in for the reviewer, and assert the persisted artifacts.

use std::fs;
use std::path::{Path, PathBuf};

use docx_rs::{Docx, Paragraph, Run, RunFonts};

use stencil::censor::{self, CensorDecision, CensorOptions, Verdict};
use stencil::extract;
use stencil::learn::{self, DecisionRecord, StylingRecord};
use stencil::model::{Block, Document, DocumentStyleProfile};
use stencil::style;
use stencil::style::review::{StyleDecision, StyleVerdict};

/// A unique temp path for a fixture or data dir, namespaced by label and process id.
fn unique(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("stencil_rp_{}_{label}", std::process::id()))
}

/// Pack a `Docx` to `path`.
fn pack(docx: Docx, path: &Path) {
    let file = fs::File::create(path).expect("create temp docx");
    docx.build().pack(file).expect("pack docx");
}

#[test]
fn censor_pipeline_logs_schema_3_and_censors_only_confirmed() {
    // A paragraph with two pattern-detected values: an email (we confirm) and money (we reject).
    let path = unique("censor").with_extension("docx");
    pack(
        Docx::new().add_paragraph(
            Paragraph::new().add_run(Run::new().add_text("Pay $500 to jane@acme.com today.")),
        ),
        &path,
    );

    let document = extract::from_path(&path).expect("extract docx");
    let options = CensorOptions {
        parties: None,
        allow: None,
    };
    let items = censor::plan_review(&document, &options);
    assert!(
        items.iter().any(|item| item.value.contains('@')),
        "the email should be detected; got {items:?}",
    );

    // Synthetic reviewer: confirm the email, reject everything else (e.g. the money).
    let decisions: Vec<CensorDecision> = items
        .iter()
        .map(|item| {
            let verdict = if item.value.contains('@') {
                Verdict::Confirm {
                    final_type: "EMAIL".to_string(),
                }
            } else {
                Verdict::Reject
            };
            CensorDecision::from_item(item, verdict, true)
        })
        .collect();

    // apply censors only confirmed values: the email is gone, the rejected money stays.
    let censored = censor::apply(&document, &decisions, &options);
    let text = match &censored.blocks[0] {
        Block::Paragraph { text } => text.clone(),
        other => panic!("expected a paragraph, got {other:?}"),
    };
    assert!(
        !text.contains("jane@acme.com"),
        "confirmed email censored: {text}"
    );
    assert!(
        text.contains("REDACTED_EMAIL_"),
        "placeholder present: {text}"
    );
    assert!(
        text.contains("$500"),
        "rejected money left in the clear: {text}"
    );

    // The schema-3 decision log: one confirm row (final_type set) and one reject row (null).
    let log_path = unique("censor_log").join("decisions.jsonl");
    let records = censor::decision_records(&decisions, "c.docx", 42);
    for record in &records {
        learn::append_decision(&log_path, record).expect("append decision");
    }
    let logged: Vec<DecisionRecord> = fs::read_to_string(&log_path)
        .expect("read decisions log")
        .lines()
        .map(|line| serde_json::from_str(line).expect("parse decision record"))
        .collect();

    assert!(logged.iter().all(|r| r.schema == learn::decision_schema()));
    let email = logged
        .iter()
        .find(|r| r.value.contains('@'))
        .expect("email row");
    assert_eq!(email.verdict, "confirm");
    assert_eq!(email.final_type.as_deref(), Some("EMAIL"));
    let money = logged
        .iter()
        .find(|r| r.value.contains("500"))
        .expect("money row");
    assert_eq!(money.verdict, "reject");
    assert_eq!(money.final_type, None, "reject carries a null final_type");

    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir_all(unique("censor_log"));
}

#[test]
fn styling_pipeline_writes_censored_jsonl_and_profile_sidecar() {
    // Two styled blocks; the first carries a sensitive value that must be censored in the log.
    let path = unique("styling").with_extension("docx");
    pack(
        Docx::new()
            .add_paragraph(
                Paragraph::new().add_run(
                    Run::new()
                        .add_text("Contact jane@acme.com")
                        .fonts(RunFonts::new().ascii("Arial")),
                ),
            )
            .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Plain clause."))),
        &path,
    );

    let mut blocks = style::extract::from_path(&path).expect("extract styled blocks");
    assert_eq!(blocks.len(), 2);

    // Censor each block's text the same way the command does: a one-paragraph-per-block document
    // through the deterministic censor pass keeps the censored text aligned 1:1.
    let options = CensorOptions {
        parties: None,
        allow: None,
    };
    let text_doc = Document {
        source: path.clone(),
        blocks: blocks
            .iter()
            .map(|block| Block::Paragraph {
                text: block.text.clone(),
            })
            .collect(),
    };
    let censored = censor::censor(&text_doc, &options).document;
    for (block, censored_block) in blocks.iter_mut().zip(censored.blocks) {
        if let Block::Paragraph { text } = censored_block {
            block.text = text;
        }
    }
    assert!(
        !blocks[0].text.contains("jane@acme.com"),
        "block text is censored before logging: {}",
        blocks[0].text
    );

    let profile = style::profile::build_profile(&blocks);

    // Synthetic reviewer: first block fine, second weird with a category + note.
    let decisions = vec![
        StyleDecision {
            block_index: 0,
            verdict: StyleVerdict::Fine,
        },
        StyleDecision {
            block_index: 1,
            verdict: StyleVerdict::Weird {
                category: "wrong-style-for-role".to_string(),
                note: Some("looks off".to_string()),
            },
        },
    ];

    let data = unique("styling_data");
    let log_path = data.join("styling.jsonl");
    let profiles_dir = data.join("profiles");
    let sidecar = style::record::persist(
        &log_path,
        &profiles_dir,
        &blocks,
        &profile,
        &decisions,
        &path,
    )
    .expect("persist styling");

    let logged: Vec<StylingRecord> = fs::read_to_string(&log_path)
        .expect("read styling log")
        .lines()
        .map(|line| serde_json::from_str(line).expect("parse styling record"))
        .collect();

    assert_eq!(logged.len(), 2, "one record per reviewed block");
    assert!(logged.iter().all(|r| r.schema == learn::styling_schema()));
    assert!(
        logged.iter().all(|r| !r.text.contains("jane@acme.com")),
        "logged styling text never carries real values",
    );
    assert_eq!(logged[0].verdict, "fine");
    assert_eq!(logged[0].category, None);
    assert_eq!(logged[1].verdict, "weird");
    assert_eq!(logged[1].category.as_deref(), Some("wrong-style-for-role"));
    assert_eq!(logged[1].note.as_deref(), Some("looks off"));

    // The profile sidecar round-trips back to the in-memory profile.
    let back: DocumentStyleProfile =
        serde_json::from_str(&fs::read_to_string(&sidecar).expect("read sidecar"))
            .expect("parse profile sidecar");
    assert_eq!(back, profile);

    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir_all(&data);
}
