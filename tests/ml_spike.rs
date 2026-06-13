//! T61 spike — the cross-schema regression guard for the v11 ML feature contract.
//!
//! The encoding it originally locked now lives in production (`stencil::ml::features`); this file is
//! the integration check that the production mappers handle **every historical record schema**, plus
//! the home of the cold-start guard decision (which T65's `train` will implement).
//!
//! What it asserts:
//!   1. every live schema (styling 1–5, censor 2–6) deserializes and maps — through the *production*
//!      encoders — to a fixed-width, all-finite feature vector, with forward v11 prediction fields
//!      (schema 5/6) parsed and ignored (no future-field leak);
//!   2. the cold-start / class-collapse guard is decided and encoded ([`trainable`] /
//!      [`reason_trainable`]).
//!
//! Per-field encoding, leakage-by-construction, and golden-vector tests are unit tests of
//! `stencil::ml::features::{styling,censor}` — this file deliberately does not duplicate them.

use stencil::learn::{DecisionRecord, StylingRecord};
use stencil::ml::features::censor::{CENSOR_FEATURE_LEN, censor_features};
use stencil::ml::features::styling::{STYLING_FEATURE_LEN, styling_features};

// ─── cold-start / class-collapse guard (T65 consumes this) ───────────────────────────────────────

/// A verdict head is trainable only when both classes are present. Zero positives (or zero
/// negatives) ⇒ not trainable ⇒ no head is written ⇒ the artifact reads as "no model" rather than a
/// degenerate all-negative classifier.
fn trainable(positives: usize, negatives: usize) -> bool {
    positives >= 1 && negatives >= 1
}

/// A reason head is trainable only with ≥2 distinct reason classes among the positives; otherwise
/// the verdict shows alone and the reason stays "(uncertain)".
fn reason_trainable(distinct_reason_classes: usize) -> bool {
    distinct_reason_classes >= 2
}

// ─── fixtures: one representative line per live schema ───────────────────────────────────────────

/// Styling rows, schema 1 → 5. Each adds the fields that schema introduced; schema 5 carries the
/// (forward) v11 prediction fields, which the feature path must ignore. Only `Option`/`#[serde(
/// default)]` fields may be omitted, so each line includes the required core (incl. `para.indent` /
/// `para.spacing`, which are non-optional sub-objects).
fn styling_rows() -> Vec<&'static str> {
    vec![
        // schema 1 (v6): core only; carries the since-dropped `run.mixed` (must be ignored).
        r#"{"schema":1,"source":"a.docx","block_index":0,"block_kind":"heading","heading_level":1,"in_table":false,"text":"PAYMENT TERMS","para":{"indent":{},"spacing":{}},"run":{"font":"Arial","size_half_pt":28,"bold":true,"italic":false,"mixed":false},"relative":{},"context":{"prev_text":"","next_text":"body"},"verdict":"fine","category":null,"note":null}"#,
        // schema 2 (v7): + doc_id, lang, lang_confidence.
        r#"{"schema":2,"source":"b.docx","doc_id":"id2","lang":"en","lang_confidence":0.9,"block_index":3,"block_kind":"paragraph","heading_level":null,"in_table":false,"text":"The buyer shall pay 1,000.","para":{"alignment":"left","indent":{"left":720,"hanging":360},"spacing":{}},"run":{"font":"Calibri","size_half_pt":22,"bold":false,"italic":false},"relative":{"style_doc_freq":0.5,"font_matches_doc_dominant":false},"context":{"prev_text":"a","next_text":"b"},"verdict":"weird","category":"inconsistent-style","note":null}"#,
        // schema 3 (v8): + segments, numbering_format, *_unresolved.
        r#"{"schema":3,"source":"c.docx","doc_id":"id3","lang":"fr","lang_confidence":0.8,"block_index":7,"block_kind":"list_item","heading_level":null,"in_table":false,"text":"(a) livraison","para":{"indent":{},"numbering":{"num_id":2,"ilvl":1},"spacing":{}},"run":{"bold":false,"italic":false},"segments":[{"text":"(a) ","style":{}},{"text":"livraison","style":{"bold":true}}],"numbering_format":{"kind":"lowerLetter","level_text":"%1."},"style_unresolved":false,"numbering_unresolved":true,"relative":{"size_matches_doc_dominant":true,"indent_vs_ilvl_norm":0.25},"context":{"prev_text":"x","next_text":"y"},"verdict":"weird","category":"bad-indent-level","note":"deep"}"#,
        // schema 4 (v9): + neighbor structure.
        r#"{"schema":4,"source":"d.docx","doc_id":"id4","lang":"en","lang_confidence":0.95,"block_index":2,"block_kind":"table_cell","heading_level":null,"in_table":true,"text":"$1.000.000","para":{"indent":{},"spacing":{}},"run":{"bold":false,"italic":true},"relative":{"matches_role_peers":false},"context":{"prev_text":"p","next_text":"n","prev_kind":"list_item","next_kind":"paragraph","prev_numbering":{"num_id":2,"ilvl":0}},"verdict":"weird","category":"fake-number","note":null}"#,
        // schema 5 (v11, forward): + prediction fields — unknown today, must be ignored (no leak).
        r#"{"schema":5,"source":"e.docx","doc_id":"id5","lang":"en","lang_confidence":0.9,"block_index":9,"block_kind":"paragraph","heading_level":null,"in_table":false,"text":"Closing paragraph.","para":{"indent":{},"spacing":{}},"run":{"bold":false,"italic":false},"relative":{},"context":{"prev_text":"a","next_text":"b"},"verdict":"fine","category":null,"note":null,"predicted_verdict":"fine","predicted_verdict_score":0.12,"predicted_reason":null,"predicted_reason_score":null,"model_trained_at":"2026-06-12"}"#,
    ]
}

/// Censor rows, schema 2 → 6. Schema 6 carries the (forward) v11 prediction fields.
fn censor_rows() -> Vec<&'static str> {
    vec![
        // schema 2 (pre-v6): legacy placeholder/type/decision (now unknown) + the required core.
        r#"{"schema":2,"timestamp":7,"source":"c.txt","placeholder":"REDACTED_PERSON_001","type":"PERSON","value":"Jane Doe","decision":"allow","shown_context":"pay Jane Doe today","block_context":"The buyer pay Jane Doe today within 30 days"}"#,
        // schema 3 (v6): value-based, multi-class label.
        r#"{"schema":3,"timestamp":8,"source":"c.docx","value":"jane@acme.com","method":"regex:email","detected_type":"EMAIL","verdict":"confirm","final_type":"EMAIL","shown_context":"write to jane@acme.com","block_context":"please write to jane@acme.com soon","occurrences":2}"#,
        // schema 4 (v7): + doc_id, scope, block_kinds, heading_level, langs, edit provenance.
        r#"{"schema":4,"timestamp":9,"source":"c.docx","value":"Acme Corp","method":"party-list","detected_type":"ORG","verdict":"confirm","final_type":"ORG","shown_context":"between Acme Corp and","block_context":"agreement between Acme Corp and the buyer","occurrences":5,"doc_id":"id4","scope":"group","block_kinds":["heading","paragraph"],"heading_level":2,"langs":["en"],"span_edited":true,"context_edited":false,"user_added":false}"#,
        // schema 5 (v10): + neighbors.
        r#"{"schema":5,"timestamp":10,"source":"c.docx","value":"123 Main Street","method":"heuristic","detected_type":"ENTITY","verdict":"reject","final_type":null,"shown_context":"at 123 Main Street near","block_context":"located at 123 Main Street near the river","occurrences":1,"doc_id":"id5","scope":"occurrence","block_kinds":["table_cell"],"heading_level":null,"langs":["fr"],"neighbors":{"above":"Mailing Address","col_header":"Address"}}"#,
        // schema 6 (v11, forward): + prediction fields — unknown today, must be ignored (no leak).
        r#"{"schema":6,"timestamp":11,"source":"c.docx","value":"4111 1111 1111 1111","method":"regex:card","detected_type":"CARD","verdict":"confirm","final_type":"CARD","shown_context":"card 4111 1111 1111 1111 on file","block_context":"the card 4111 1111 1111 1111 on file","occurrences":1,"doc_id":"id6","scope":"group","block_kinds":["paragraph"],"langs":["en"],"predicted_verdict":"confirm","predicted_verdict_score":0.88,"predicted_reason":"CARD","predicted_reason_score":0.7,"model_trained_at":"2026-06-12"}"#,
    ]
}

// ─── tests ───────────────────────────────────────────────────────────────────────────────────────

#[test]
fn every_styling_schema_resolves_through_production_encoder() {
    for line in styling_rows() {
        let record: StylingRecord = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("schema row failed to parse: {e}"));
        let features = styling_features(&record);
        assert_eq!(
            features.len(),
            STYLING_FEATURE_LEN,
            "schema {} produced {} features",
            record.schema,
            features.len()
        );
        assert!(
            features.iter().all(|f| f.is_finite()),
            "schema {} produced a non-finite feature",
            record.schema
        );
    }
}

#[test]
fn every_censor_schema_resolves_through_production_encoder() {
    for line in censor_rows() {
        let record: DecisionRecord = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("schema row failed to parse: {e}"));
        let features = censor_features(&record);
        assert_eq!(
            features.len(),
            CENSOR_FEATURE_LEN,
            "schema {} produced {} features",
            record.schema,
            features.len()
        );
        assert!(
            features.iter().all(|f| f.is_finite()),
            "schema {} produced a non-finite feature",
            record.schema
        );
    }
}

#[test]
fn cold_start_guard_requires_both_classes() {
    assert!(!trainable(0, 100), "zero positives ⇒ no head");
    assert!(!trainable(100, 0), "zero negatives ⇒ no head");
    assert!(trainable(1, 1), "one of each ⇒ trainable");

    assert!(!reason_trainable(0), "no reasons ⇒ verdict only");
    assert!(!reason_trainable(1), "a single reason class is degenerate");
    assert!(reason_trainable(2), "≥2 reason classes ⇒ reason head");
}
