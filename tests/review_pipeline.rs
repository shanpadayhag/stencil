//! End-to-end pipeline tests for the v7 `review` (censor) and `style` (styling) flows, exercised at
//! the library level on a real `.docx` fixture. The interactive keypress loops need a PTY and so are
//! out of scope here (their pure decision logic is unit-tested); these tests drive the
//! *non-interactive* halves — detect → apply → schema-4 log for censor (incl. split + add flows),
//! and extract → censor-text → profile → persist for styling — with synthetic decisions standing in
//! for the reviewer, and assert the persisted artifacts (doc_id, scope, block kinds, language).

use std::fs;
use std::path::{Path, PathBuf};

use docx_rs::{Docx, Paragraph, Run, RunFonts};

use stencil::censor::{self, CensorDecision, CensorOptions, ReviewItem, ValueType, Verdict};
use stencil::extract;
use stencil::learn::{self, DecisionRecord, StylingRecord};
use stencil::model::{Block, Document, DocumentStyleProfile, Occurrence};
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
fn censor_pipeline_logs_schema_4_and_censors_only_confirmed() {
    // A clearly-English paragraph with two pattern-detected values: an email (we confirm) and
    // money (we reject). The English text makes the per-block language detect as `en`.
    let path = unique("censor").with_extension("docx");
    pack(
        Docx::new().add_paragraph(Paragraph::new().add_run(Run::new().add_text(
            "This agreement requires the buyer to pay $500 to jane@acme.com before closing.",
        ))),
        &path,
    );

    let document = extract::from_path(&path).expect("extract docx");
    let doc_id = stencil::doc_id::doc_id(&document);
    let options = CensorOptions {
        parties: None,
        allow: None,
    };
    let mut items = censor::plan_review(&document, &options);
    censor::tag_occurrence_languages(&document, &mut items, None);
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

    // The schema-4 decision log: rows carry the doc_id, scope, block kinds, and language.
    let log_path = unique("censor_log").join("decisions.jsonl");
    let records = censor::decision_records(&decisions, "c.docx", &doc_id, 42);
    for record in &records {
        learn::append_decision(&log_path, record).expect("append decision");
    }
    let logged: Vec<DecisionRecord> = fs::read_to_string(&log_path)
        .expect("read decisions log")
        .lines()
        .map(|line| serde_json::from_str(line).expect("parse decision record"))
        .collect();

    assert!(logged.iter().all(|r| r.schema == learn::decision_schema()));
    assert!(
        logged.iter().all(|r| r.doc_id == doc_id),
        "every row keyed by the content id"
    );
    let email = logged
        .iter()
        .find(|r| r.value.contains('@'))
        .expect("email row");
    assert_eq!(email.verdict, "confirm");
    assert_eq!(email.final_type.as_deref(), Some("EMAIL"));
    assert_eq!(email.scope, "group", "a whole-group decision");
    assert_eq!(email.block_kinds, vec!["paragraph".to_string()]);
    assert_eq!(
        email.langs,
        vec!["en".to_string()],
        "the English block tags as en"
    );
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
fn v7_split_and_added_value_flows_through_apply_and_log() {
    // "3%" appears twice (a split: confirm the first occurrence, reject the second) and "Globex"
    // is a reviewer-*added* value (the detector path is bypassed; it is censored by literal search).
    let text = "Rate is 3% here and 3% there; Globex agrees.";
    let path = unique("v7flows").with_extension("docx");
    pack(
        Docx::new().add_paragraph(Paragraph::new().add_run(Run::new().add_text(text))),
        &path,
    );
    let document = extract::from_path(&path).expect("extract docx");
    let options = CensorOptions {
        parties: None,
        allow: None,
    };

    let first = text.find("3%").expect("first 3%");
    let second = text.rfind("3%").expect("second 3%");
    let occurrence = |start: usize| Occurrence {
        block_index: 0,
        start,
        end: start + 2,
        block_kind: stencil::model::BlockKind::Paragraph,
        ..Default::default()
    };
    let percent = |start, verdict| {
        CensorDecision::from_occurrence(
            "3%",
            ValueType::Percent,
            "regex:percent",
            occurrence(start),
            verdict,
        )
    };
    let added = {
        let item = ReviewItem {
            value: "Globex".into(),
            detected_type: ValueType::Org,
            method: "manual".into(),
            occurrences: vec![occurrence(text.find("Globex").unwrap())],
        };
        let mut decision = CensorDecision::from_item(
            &item,
            Verdict::Confirm {
                final_type: "ORG".into(),
            },
            true,
        );
        decision.user_added = true;
        decision
    };
    let decisions = vec![
        percent(
            first,
            Verdict::Confirm {
                final_type: "PERCENT".into(),
            },
        ),
        percent(second, Verdict::Reject),
        added,
    ];

    // apply: the first 3% is censored at its offset, the second stays literal, Globex is censored.
    let censored = match &censor::apply(&document, &decisions, &options).blocks[0] {
        Block::Paragraph { text } => text.clone(),
        other => panic!("expected a paragraph, got {other:?}"),
    };
    assert_eq!(
        censored, "Rate is REDACTED_PERCENT_001 here and 3% there; REDACTED_ORG_001 agrees.",
        "split offset + literal add applied; rejected occurrence left in the clear",
    );

    // The log: two occurrence-scoped rows for the split + one group row for the added value.
    let records = censor::decision_records(&decisions, "c.docx", "deadbeefcafe0002", 7);
    assert_eq!(records.len(), 3);
    assert!(
        records
            .iter()
            .filter(|r| r.value == "3%")
            .all(|r| r.scope == "occurrence" && r.occurrences == 1),
        "split rows are occurrence-scoped, one occurrence each",
    );
    let globex = records
        .iter()
        .find(|r| r.value == "Globex")
        .expect("added row");
    assert_eq!(globex.scope, "group");
    assert!(globex.user_added, "the added value is flagged user_added");

    let _ = fs::remove_file(&path);
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

    // Tag languages on the original text (as the `style` command does), before censoring.
    let tags = stencil::lang::tag_texts(
        &blocks.iter().map(|b| b.text.as_str()).collect::<Vec<_>>(),
        None,
    );
    for (block, tag) in blocks.iter_mut().zip(tags) {
        block.lang = tag.lang;
        block.lang_confidence = tag.confidence;
    }

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
        "testdocid0000001",
    )
    .expect("persist styling");
    assert_eq!(
        sidecar.file_name().unwrap().to_string_lossy(),
        "testdocid0000001.json",
        "the style profile sidecar is keyed by the content id",
    );

    let logged: Vec<StylingRecord> = fs::read_to_string(&log_path)
        .expect("read styling log")
        .lines()
        .map(|line| serde_json::from_str(line).expect("parse styling record"))
        .collect();

    assert_eq!(logged.len(), 2, "one record per reviewed block");
    assert!(logged.iter().all(|r| r.schema == learn::styling_schema()));
    assert!(
        logged.iter().all(|r| r.doc_id == "testdocid0000001"),
        "every styling row is keyed by the content id",
    );
    assert!(
        logged.iter().all(|r| !r.lang.is_empty()),
        "each styling row records its detected language",
    );
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
