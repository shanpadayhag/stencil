//! End-to-end tests for the v11 suggestive-model pipeline: `train` → a (library-simulated) review
//! that stamps predictions → `accuracy`. The interactive review needs a PTY, so the "review" step
//! here computes and logs predictions the same way the wired review does (predict → stamp → append),
//! then the real `stencil accuracy` binary reports the prequential meter over them. Also covers the
//! feature-spec version gate (a stale artifact is ignored).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use stencil::learn::{self, Prediction, StylingRecord};
use stencil::ml::artifact::{self, Head, TaskModel};
use stencil::ml::features::FEATURE_SPEC_VERSION;
use stencil::ml::features::styling::styling_features;
use stencil::ml::logreg::{FitOptions, fit};
use stencil::ml::predict::{STYLING_LABELS, predict, to_prediction};

/// Path to the compiled binary under test (provided by Cargo for integration tests).
const BIN: &str = env!("CARGO_BIN_EXE_stencil");

/// A unique temp data dir for one test, namespaced by label and process id.
fn unique(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("stencil_mlpipe_{}_{label}", std::process::id()))
}

/// Run the binary with `args`, no TTY on stdin, returning (stdout, success).
fn run_bin(args: &[&str]) -> (String, bool) {
    let output = Command::new(BIN)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn stencil");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        output.status.success(),
    )
}

/// A styling row with the given verdict/category/text and an attached prediction.
fn styling_row(
    verdict: &str,
    category: Option<&str>,
    text: &str,
    prediction: Prediction,
) -> StylingRecord {
    StylingRecord {
        schema: learn::styling_schema(),
        source: "fixture.docx".into(),
        block_kind: "paragraph".into(),
        text: text.into(),
        verdict: verdict.into(),
        category: category.map(String::from),
        prediction,
        ..StylingRecord::default()
    }
}

/// Append one styling record to the log.
fn append(log: &Path, record: &StylingRecord) {
    learn::append_styling(log, record).expect("append styling row");
}

#[test]
fn train_simulated_review_then_accuracy() {
    let data = unique("e2e");
    let _ = std::fs::remove_dir_all(&data);
    let styling_dir = data.join("styling");
    std::fs::create_dir_all(&styling_dir).expect("mkdir styling");
    let log = styling_dir.join("styling.jsonl");

    // 1. Seed the training log (no predictions): fine rows are plain text, weird rows are digit-heavy
    //    "fake-number" plus a second "inconsistent-style" category so the reason head can train.
    for _ in 0..8 {
        append(
            &log,
            &styling_row(
                "fine",
                None,
                "the parties agree to the terms herein",
                Prediction::default(),
            ),
        );
    }
    for _ in 0..2 {
        append(
            &log,
            &styling_row(
                "weird",
                Some("fake-number"),
                "1234 5678 9012 3456",
                Prediction::default(),
            ),
        );
    }
    for _ in 0..2 {
        append(
            &log,
            &styling_row(
                "weird",
                Some("inconsistent-style"),
                "MiXeD cApS heading",
                Prediction::default(),
            ),
        );
    }

    // 2. Train via the real binary.
    let (stdout, ok) = run_bin(&["train", "--styling", "--data-dir", data.to_str().unwrap()]);
    assert!(ok, "train should succeed; stdout: {stdout}");
    assert!(
        stdout.contains("styling: trained on 12 record(s)"),
        "summary: {stdout}"
    );

    // 3. The artifact exists, loads, and matches the current feature spec.
    let model = artifact::load(&styling_dir.join("model.json")).expect("a current model on disk");
    assert!(model.is_current());
    assert_eq!(model.n_records, 12);

    // 4. Simulated review: predict each item and stamp the prediction onto a freshly logged row —
    //    exactly what the wired review does, minus the keypress. 95 fine + 5 weird = a full window.
    for index in 0..100 {
        let weird = index < 5;
        let (verdict, category, text) = if weird {
            ("weird", Some("fake-number"), "4321 8765 0001 2222")
        } else {
            ("fine", None, "the contract shall remain in force")
        };
        let mut record = styling_row(verdict, category, text, Prediction::default());
        let suggestion = predict(&model, &styling_features(&record));
        record.prediction = to_prediction(&suggestion, &STYLING_LABELS, &model.trained_at);
        append(&log, &record);
    }

    // 5. A stamped row round-trips with its prediction fields populated.
    let last: StylingRecord = serde_json::from_str(
        std::fs::read_to_string(&log)
            .expect("read log")
            .lines()
            .next_back()
            .expect("a row"),
    )
    .expect("parse row");
    assert!(
        last.prediction.predicted_verdict.is_some(),
        "the simulated review stamped a prediction"
    );
    assert_eq!(
        last.prediction.model_trained_at.as_deref(),
        Some(model.trained_at.as_str())
    );

    // 6. The accuracy meter (real binary) reports a headline over the 100 stamped predictions; the 12
    //    pre-prediction training rows are excluded from the window.
    let (stdout, ok) = run_bin(&["accuracy", "--data-dir", data.to_str().unwrap()]);
    assert!(ok, "accuracy should succeed; stdout: {stdout}");
    assert!(
        stdout.contains("styling model — last 100 prediction(s)"),
        "meter: {stdout}"
    );
    assert!(
        stdout.contains("balanced accuracy:"),
        "window is full ⇒ a headline %: {stdout}"
    );
    assert!(
        stdout.contains("weird 5/5") || stdout.contains("weird"),
        "per-class line present: {stdout}"
    );

    let _ = std::fs::remove_dir_all(&data);
}

#[test]
fn stale_feature_spec_artifact_is_ignored() {
    // An artifact trained against a *different* feature spec must read as "no model" — so the review
    // would compute and show no suggestion. (`load` returning None is exactly that gate.)
    let data = unique("stale");
    let _ = std::fs::remove_dir_all(&data);
    let dir = data.join("styling");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let path = dir.join("model.json");

    let binary = fit(&[vec![-1.0], vec![1.0]], &[0u8, 1], &FitOptions::default());
    let stale = TaskModel {
        feature_spec_version: FEATURE_SPEC_VERSION + 1,
        trained_at: "0".into(),
        n_records: 2,
        verdict: Head {
            model: binary,
            threshold: 0.5,
        },
        reason: None,
        cv_balanced_accuracy: 1.0,
    };
    artifact::save_atomic(&path, &stale).expect("save stale model");

    assert!(
        artifact::load(&path).is_none(),
        "a stale feature-spec artifact must be ignored (review shows no suggestion)"
    );

    let _ = std::fs::remove_dir_all(&data);
}
