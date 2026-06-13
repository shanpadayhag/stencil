//! Inference: a trained [`TaskModel`] + a record's feature vector → an advisory [`Suggestion`].
//!
//! Pure and task-agnostic — it returns `is_positive` (the "fail" class: `weird` / `censor`) plus a
//! score, and an optional reason. The caller maps `is_positive` to the task's labels
//! (`fine`/`weird` or `reject`/`confirm`) and renders/stamps it. The reason head is consulted only
//! on a positive verdict, and only commits to a label when its top-class score clears
//! [`REASON_CONFIDENCE_FLOOR`]; otherwise the reason is `None` ("(uncertain)").

use crate::learn::Prediction;
use crate::ml::artifact::TaskModel;

/// Minimum top-class score for the reason head to commit to a label. Below it the reason renders
/// "(uncertain)" rather than guessing. A tunable constant; the prequential meter shows if it is set
/// too quiet or too noisy.
pub const REASON_CONFIDENCE_FLOOR: f64 = 0.5;

/// ANSI green for a "pass" suggestion.
const GREEN: &str = "\x1b[32m";
/// ANSI red for a "fail" suggestion.
const RED: &str = "\x1b[31m";
/// ANSI reset.
const RESET: &str = "\x1b[0m";

/// The task-specific labels for turning a task-agnostic [`Suggestion`] into a logged verdict and a
/// rendered line. `record_*` are the verdict strings written to the log (`weird`/`fine`,
/// `confirm`/`reject`); `display_*` are what the reviewer reads on the suggestion line.
pub struct SuggestionLabels {
    /// The positive/"fail"-class verdict as logged (`weird` / `confirm`).
    pub record_positive: &'static str,
    /// The negative/"pass"-class verdict as logged (`fine` / `reject`).
    pub record_negative: &'static str,
    /// The positive-class word shown on the (red) line (`weird` / `censor`).
    pub display_positive: &'static str,
    /// The negative-class word shown on the (green) line (`fine` / `leave in clear`).
    pub display_negative: &'static str,
}

/// Labels for the styling model (`fine` vs `weird`).
pub const STYLING_LABELS: SuggestionLabels = SuggestionLabels {
    record_positive: "weird",
    record_negative: "fine",
    display_positive: "weird",
    display_negative: "fine",
};

/// Labels for the censor model: the positive class is `confirm` (keep censored), shown as "censor".
pub const CENSOR_LABELS: SuggestionLabels = SuggestionLabels {
    record_positive: "confirm",
    record_negative: "reject",
    display_positive: "censor",
    display_negative: "leave in clear",
};

/// Map a [`Suggestion`] to the [`Prediction`] stamped on the logged row, using `labels` for the
/// verdict strings and `model_trained_at` for provenance.
pub fn to_prediction(
    suggestion: &Suggestion,
    labels: &SuggestionLabels,
    model_trained_at: &str,
) -> Prediction {
    let verdict = if suggestion.is_positive {
        labels.record_positive
    } else {
        labels.record_negative
    };
    Prediction {
        predicted_verdict: Some(verdict.to_string()),
        predicted_verdict_score: Some(suggestion.score),
        predicted_reason: suggestion.reason.as_ref().map(|(label, _)| label.clone()),
        predicted_reason_score: suggestion.reason.as_ref().map(|(_, score)| *score),
        model_trained_at: Some(model_trained_at.to_string()),
    }
}

/// Render the single green/red suggestion line for a stamped [`Prediction`], or `None` when no
/// prediction was made (no model). Red (fail class) carries the reason — `(reason uncertain)` when
/// the reason head was unsure; green (pass class) is just the verdict.
pub fn prediction_line(prediction: &Prediction, labels: &SuggestionLabels) -> Option<String> {
    let verdict = prediction.predicted_verdict.as_deref()?;
    let line = if verdict == labels.record_positive {
        let reason = match prediction.predicted_reason.as_deref() {
            Some(reason) => format!(" — {reason}"),
            None => " — (reason uncertain)".to_string(),
        };
        format!(
            "{RED}suggestion: {}{reason}{RESET}",
            labels.display_positive
        )
    } else {
        format!("{GREEN}suggestion: {}{RESET}", labels.display_negative)
    };
    Some(line)
}

/// An advisory suggestion for one reviewed item — what the model thinks the human will decide.
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    /// `true` when the model predicts the positive / "fail" class (`weird` / `censor`).
    pub is_positive: bool,
    /// The verdict head's positive-class score in `0.0..=1.0`.
    pub score: f64,
    /// The predicted reason label and its score — `Some` only on a positive verdict whose reason
    /// head is present and confident.
    pub reason: Option<(String, f64)>,
}

/// Predict the advisory suggestion for `features` (the record's encoded vector) under `model`.
///
/// ```
/// use stencil::ml::logreg::{fit, FitOptions};
/// use stencil::ml::artifact::{Head, TaskModel};
/// use stencil::ml::features::FEATURE_SPEC_VERSION;
/// use stencil::ml::predict::predict;
///
/// let binary = fit(&[vec![-1.0], vec![1.0]], &[0u8, 1], &FitOptions::default());
/// let model = TaskModel {
///     feature_spec_version: FEATURE_SPEC_VERSION,
///     trained_at: "0".into(),
///     n_records: 2,
///     verdict: Head { model: binary, threshold: 0.5 },
///     reason: None,
///     cv_balanced_accuracy: 1.0,
/// };
/// assert!(predict(&model, &[1.0]).is_positive);
/// assert!(!predict(&model, &[-1.0]).is_positive);
/// ```
pub fn predict(model: &TaskModel, features: &[f64]) -> Suggestion {
    let score = model.verdict.model.predict_proba(features);
    let is_positive = score >= model.verdict.threshold;
    let reason = if is_positive {
        predict_reason(model, features)
    } else {
        None
    };
    Suggestion {
        is_positive,
        score,
        reason,
    }
}

/// The reason head's confident top class, or `None` when there is no reason head or its top score is
/// below [`REASON_CONFIDENCE_FLOOR`].
fn predict_reason(model: &TaskModel, features: &[f64]) -> Option<(String, f64)> {
    let reason_head = model.reason.as_ref()?;
    let (index, top_score) = reason_head.best(features)?;
    if top_score < REASON_CONFIDENCE_FLOOR {
        return None;
    }
    reason_head
        .classes
        .get(index)
        .map(|label| (label.clone(), top_score))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::artifact::{Head, TaskModel};
    use crate::ml::features::FEATURE_SPEC_VERSION;
    use crate::ml::logreg::{FitOptions, fit, fit_multiclass};

    /// A verdict head separating negatives (feature 0) from positives (feature 1), with an optional
    /// reason head over the same axis.
    fn model_with_reason(reason: bool) -> TaskModel {
        let x = vec![
            vec![-1.0, 0.0],
            vec![-0.9, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let verdict = fit(&x, &[0u8, 0, 1, 1], &FitOptions::default());
        let reason_head = reason.then(|| {
            // Two reason classes among positives, split on feature[1].
            let px = vec![vec![1.0, 0.0], vec![1.0, 1.0]];
            fit_multiclass(
                &px,
                &[0usize, 1],
                vec!["a".into(), "b".into()],
                &FitOptions::default(),
            )
        });
        TaskModel {
            feature_spec_version: FEATURE_SPEC_VERSION,
            trained_at: "0".into(),
            n_records: 4,
            verdict: Head {
                model: verdict,
                threshold: 0.5,
            },
            reason: reason_head,
            cv_balanced_accuracy: 1.0,
        }
    }

    #[test]
    fn positive_and_negative_verdicts() {
        let model = model_with_reason(false);
        let positive = predict(&model, &[1.0, 0.0]);
        assert!(positive.is_positive);
        assert!(positive.score >= 0.5);

        let negative = predict(&model, &[-1.0, 0.0]);
        assert!(!negative.is_positive);
        assert!(negative.score < 0.5);
        assert_eq!(negative.reason, None, "no reason on a pass verdict");
    }

    #[test]
    fn reason_only_on_positive_verdict() {
        let model = model_with_reason(true);
        // A negative verdict never consults the reason head.
        assert_eq!(predict(&model, &[-1.0, 0.0]).reason, None);
        // A positive verdict surfaces a reason from the head.
        let positive = predict(&model, &[1.0, 1.0]);
        assert!(positive.is_positive);
        let (label, score) = positive.reason.expect("a confident reason");
        assert!(label == "a" || label == "b");
        assert!(score >= REASON_CONFIDENCE_FLOOR);
    }

    #[test]
    fn no_reason_head_yields_no_reason() {
        let model = model_with_reason(false);
        let positive = predict(&model, &[1.0, 0.0]);
        assert!(positive.is_positive);
        assert_eq!(positive.reason, None, "no reason head ⇒ no reason");
    }

    #[test]
    fn prediction_line_renders_green_pass_and_red_fail() {
        let pass = to_prediction(
            &Suggestion {
                is_positive: false,
                score: 0.1,
                reason: None,
            },
            &STYLING_LABELS,
            "stamp",
        );
        let line = prediction_line(&pass, &STYLING_LABELS).expect("a line");
        assert!(line.contains("\x1b[32m"), "pass is green");
        assert!(line.contains("fine"));

        let fail = to_prediction(
            &Suggestion {
                is_positive: true,
                score: 0.9,
                reason: Some(("wrong-style-for-role".into(), 0.8)),
            },
            &STYLING_LABELS,
            "stamp",
        );
        let line = prediction_line(&fail, &STYLING_LABELS).expect("a line");
        assert!(line.contains("\x1b[31m"), "fail is red");
        assert!(line.contains("weird — wrong-style-for-role"));
    }

    #[test]
    fn fail_without_reason_reads_uncertain() {
        let pred = to_prediction(
            &Suggestion {
                is_positive: true,
                score: 0.9,
                reason: None,
            },
            &CENSOR_LABELS,
            "stamp",
        );
        // Censor positive is logged as `confirm` but shown as `censor`.
        assert_eq!(pred.predicted_verdict.as_deref(), Some("confirm"));
        let line = prediction_line(&pred, &CENSOR_LABELS).expect("a line");
        assert!(line.contains("censor — (reason uncertain)"), "got: {line}");
    }

    #[test]
    fn no_prediction_renders_no_line() {
        assert_eq!(
            prediction_line(&Prediction::default(), &STYLING_LABELS),
            None
        );
    }

    #[test]
    fn to_prediction_carries_scores_and_stamp() {
        let pred = to_prediction(
            &Suggestion {
                is_positive: true,
                score: 0.77,
                reason: Some(("EMAIL".into(), 0.6)),
            },
            &CENSOR_LABELS,
            "1781257084",
        );
        assert_eq!(pred.predicted_verdict_score, Some(0.77));
        assert_eq!(pred.predicted_reason.as_deref(), Some("EMAIL"));
        assert_eq!(pred.predicted_reason_score, Some(0.6));
        assert_eq!(pred.model_trained_at.as_deref(), Some("1781257084"));
    }

    #[test]
    fn threshold_shifts_the_decision() {
        let mut model = model_with_reason(false);
        // A borderline point that is positive at 0.5 becomes negative at a strict threshold.
        let features = [0.2, 0.0];
        let score = predict(&model, &features).score;
        model.verdict.threshold = (score + 1.0) / 2.0; // strictly above the score
        assert!(
            !predict(&model, &features).is_positive,
            "raising the threshold above the score flips the verdict to pass"
        );
    }
}
