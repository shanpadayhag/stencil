//! `stencil accuracy` — the prequential accuracy meters for the suggestive models.
//!
//! Reads each task's log, keeps the rows that carry a prediction (schema-5/6), and reports the
//! [`crate::ml::accuracy`] meter over the most recent window. The same per-model block is reused by
//! the end-of-session summaries in `style` / `review`. Reads only logged predictions, so the metric
//! is leak-free; pre-v11 rows (no prediction) are excluded.

use std::path::Path;

use anyhow::Result;

use crate::cli::AccuracyArgs;
use crate::commands::read_jsonl;
use crate::learn::{self, DecisionRecord, Model, StylingRecord};
use crate::ml::accuracy::{self, ScoredRow};

/// Run the `accuracy` subcommand: print both models' meter blocks.
pub fn run(args: AccuracyArgs) -> Result<()> {
    println!(
        "{}",
        styling_meter_block(args.data_dir.as_deref(), args.styling_dir.as_deref())?
    );
    println!();
    println!(
        "{}",
        censor_meter_block(args.data_dir.as_deref(), args.censor_dir.as_deref())?
    );
    Ok(())
}

/// The styling model's meter block (`fine` vs `weird`).
pub fn styling_meter_block(data_dir: Option<&Path>, styling_dir: Option<&Path>) -> Result<String> {
    let dir = learn::model_dir(Model::Styling, data_dir, styling_dir)?;
    let records: Vec<StylingRecord> = read_jsonl(&dir.join("styling.jsonl"))?;
    let rows = accuracy::tail(styling_rows(&records));
    Ok(accuracy::render(
        &accuracy::summarize(&rows),
        "styling",
        "fine",
        "weird",
    ))
}

/// The censor model's meter block (`reject` vs `confirm`).
pub fn censor_meter_block(data_dir: Option<&Path>, censor_dir: Option<&Path>) -> Result<String> {
    let dir = learn::model_dir(Model::Censor, data_dir, censor_dir)?;
    let records: Vec<DecisionRecord> = read_jsonl(&dir.join("decisions.jsonl"))?;
    let rows = accuracy::tail(censor_rows(&records));
    Ok(accuracy::render(
        &accuracy::summarize(&rows),
        "censor",
        "reject",
        "confirm",
    ))
}

/// Map the predicted styling rows to [`ScoredRow`]s (positive class = `weird`).
fn styling_rows(records: &[StylingRecord]) -> Vec<ScoredRow> {
    records
        .iter()
        .filter(|record| record.prediction.predicted_verdict.is_some())
        .map(|record| ScoredRow {
            predicted_positive: record.prediction.predicted_verdict.as_deref() == Some("weird"),
            actual_positive: record.verdict == "weird",
            predicted_reason: record.prediction.predicted_reason.clone(),
            actual_reason: record.category.clone(),
        })
        .collect()
}

/// Map the predicted censor rows to [`ScoredRow`]s (positive class = `confirm`).
fn censor_rows(records: &[DecisionRecord]) -> Vec<ScoredRow> {
    records
        .iter()
        .filter(|record| record.prediction.predicted_verdict.is_some())
        .map(|record| ScoredRow {
            predicted_positive: record.prediction.predicted_verdict.as_deref() == Some("confirm"),
            actual_positive: record.verdict == "confirm",
            predicted_reason: record.prediction.predicted_reason.clone(),
            actual_reason: record.final_type.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::Prediction;

    fn styling_row(predicted: &str, actual: &str) -> StylingRecord {
        StylingRecord {
            verdict: actual.into(),
            category: (actual == "weird").then(|| "fake-number".to_string()),
            prediction: Prediction {
                predicted_verdict: Some(predicted.into()),
                ..Prediction::default()
            },
            ..StylingRecord::default()
        }
    }

    #[test]
    fn pre_v11_rows_without_a_prediction_are_excluded() {
        let records = vec![
            styling_row("weird", "weird"),
            // A pre-v11 row: no prediction, must not enter the window.
            StylingRecord {
                verdict: "fine".into(),
                ..StylingRecord::default()
            },
        ];
        let rows = styling_rows(&records);
        assert_eq!(rows.len(), 1, "only the predicted row is scored");
        assert!(rows[0].actual_positive);
    }

    #[test]
    fn censor_rows_score_confirm_as_positive() {
        let record = DecisionRecord {
            verdict: "confirm".into(),
            final_type: Some("EMAIL".into()),
            prediction: Prediction {
                predicted_verdict: Some("confirm".into()),
                predicted_reason: Some("EMAIL".into()),
                ..Prediction::default()
            },
            ..DecisionRecord::default()
        };
        let rows = censor_rows(&[record]);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].predicted_positive && rows[0].actual_positive);
        assert_eq!(rows[0].predicted_reason.as_deref(), Some("EMAIL"));
        assert_eq!(rows[0].actual_reason.as_deref(), Some("EMAIL"));
    }
}
