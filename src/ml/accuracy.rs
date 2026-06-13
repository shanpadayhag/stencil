//! The prequential accuracy meter: how well the model has *recently* predicted what the human then
//! decided.
//!
//! Pure and task-agnostic. The caller maps its logged rows (the schema-5/6 records that carry a
//! prediction) into [`ScoredRow`]s — each pairing the predicted class with the actual one — and this
//! module computes a [`Meter`]: **balanced accuracy** over the last [`WINDOW_SIZE`] predictions, the
//! per-class hit counts, and a separate reason figure. Because each row's prediction was logged
//! *before* the human decided (T67), the measurement is leak-free.
//!
//! Rows without a prediction (pre-v11, or reviewed with no model) are never mapped to a [`ScoredRow`]
//! by the caller, so they fall out of the window naturally.

/// The accuracy window: the most recent N predictions. Below this the headline percentage is
/// withheld in favor of honest counts (the estimate would be too noisy to trust).
pub const WINDOW_SIZE: usize = 100;

/// Minimum positive-class examples in the window for the balanced-accuracy headline to be shown.
/// The positive class is rare (~2–3% styling, ~9% censor), so a thin positive count makes the
/// number meaningless.
pub const MIN_POSITIVES: usize = 5;

/// One logged prediction paired with the eventual human decision. Task-agnostic: the "positive"
/// class is the fail class (`weird` / `confirm`); `*_reason` is the category/type, present on the
/// positive case.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredRow {
    /// The model predicted the positive class.
    pub predicted_positive: bool,
    /// The human decided the positive class.
    pub actual_positive: bool,
    /// The predicted reason label, when the reason head committed to one.
    pub predicted_reason: Option<String>,
    /// The actual reason label (the human's category/type), present on a positive decision.
    pub actual_reason: Option<String>,
}

/// The computed meter over a window of [`ScoredRow`]s.
#[derive(Debug, Clone, PartialEq)]
pub struct Meter {
    /// Rows in the window (`<= WINDOW_SIZE`).
    pub window: usize,
    /// Actual-positive rows in the window.
    pub positives: usize,
    /// Actual-negative rows in the window.
    pub negatives: usize,
    /// Correctly-predicted positives (true positives).
    pub pos_correct: usize,
    /// Correctly-predicted negatives (true negatives).
    pub neg_correct: usize,
    /// Rows where both a reason was predicted and an actual reason exists.
    pub reason_total: usize,
    /// Of [`Self::reason_total`], how many matched.
    pub reason_correct: usize,
    /// Balanced accuracy (mean of per-class hit rates) — `None` when the window is too thin to
    /// report a trustworthy number (see [`Self::low_sample`]).
    pub balanced_accuracy: Option<f64>,
    /// The window is too thin for a headline percentage (fewer than [`WINDOW_SIZE`] rows, fewer than
    /// [`MIN_POSITIVES`] positives, or no negatives).
    pub low_sample: bool,
}

/// Keep only the most recent [`WINDOW_SIZE`] rows (the prequential tail).
pub fn tail(mut rows: Vec<ScoredRow>) -> Vec<ScoredRow> {
    if rows.len() > WINDOW_SIZE {
        rows.split_off(rows.len() - WINDOW_SIZE)
    } else {
        rows
    }
}

/// Compute the [`Meter`] over `rows` (assumed already trimmed to the window via [`tail`]).
pub fn summarize(rows: &[ScoredRow]) -> Meter {
    let window = rows.len();
    let positives = rows.iter().filter(|row| row.actual_positive).count();
    let negatives = window - positives;
    let pos_correct = rows
        .iter()
        .filter(|row| row.actual_positive && row.predicted_positive)
        .count();
    let neg_correct = rows
        .iter()
        .filter(|row| !row.actual_positive && !row.predicted_positive)
        .count();
    let reason_total = rows
        .iter()
        .filter(|row| row.predicted_reason.is_some() && row.actual_reason.is_some())
        .count();
    let reason_correct = rows
        .iter()
        .filter(|row| row.actual_reason.is_some() && row.predicted_reason == row.actual_reason)
        .count();

    let low_sample = window < WINDOW_SIZE || positives < MIN_POSITIVES || negatives == 0;
    let balanced_accuracy = (!low_sample).then(|| {
        let true_positive_rate = pos_correct as f64 / positives as f64;
        let true_negative_rate = neg_correct as f64 / negatives as f64;
        (true_positive_rate + true_negative_rate) / 2.0
    });

    Meter {
        window,
        positives,
        negatives,
        pos_correct,
        neg_correct,
        reason_total,
        reason_correct,
        balanced_accuracy,
        low_sample,
    }
}

/// Render a multi-line meter block for display. `model` names the model; `negative`/`positive` are
/// the per-class words (`fine`/`weird`, `reject`/`confirm`).
pub fn render(meter: &Meter, model: &str, negative: &str, positive: &str) -> String {
    let mut lines = vec![format!(
        "{model} model — last {} prediction(s)",
        meter.window
    )];

    if meter.window == 0 {
        lines.push("  no predictions logged yet — review some items first".to_string());
        return lines.join("\n");
    }

    let per_class = format!(
        "  {negative} {}/{} · {positive} {}/{} (positives: {})",
        meter.neg_correct, meter.negatives, meter.pos_correct, meter.positives, meter.positives
    );

    match meter.balanced_accuracy {
        Some(balanced) => {
            lines.push(format!("  balanced accuracy: {:.1}%", balanced * 100.0));
            lines.push(per_class);
        }
        None => {
            lines.push(format!(
                "  not enough data yet (need {WINDOW_SIZE} predictions and {MIN_POSITIVES} \
                 positives; have {}, {} positive)",
                meter.window, meter.positives
            ));
            lines.push(per_class);
        }
    }

    lines.push(match meter.reason_total {
        0 => "  reason: (no confident reason predictions yet)".to_string(),
        total => format!("  reason: {}/{total} correct", meter.reason_correct),
    });

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(predicted_positive: bool, actual_positive: bool) -> ScoredRow {
        ScoredRow {
            predicted_positive,
            actual_positive,
            predicted_reason: None,
            actual_reason: None,
        }
    }

    /// A window of `negatives` true-negatives + `positives` rows, of which `caught` are predicted
    /// positive — enough rows to clear the low-sample gate.
    fn window(negatives: usize, positives: usize, caught: usize) -> Vec<ScoredRow> {
        let mut rows = Vec::new();
        for _ in 0..negatives {
            rows.push(row(false, false));
        }
        for index in 0..positives {
            rows.push(row(index < caught, true));
        }
        rows
    }

    #[test]
    fn balanced_accuracy_is_mean_of_per_class_rates() {
        // 95 negatives all correct, 5 positives of which 3 caught: TNR 1.0, TPR 0.6 ⇒ 0.8.
        let meter = summarize(&window(95, 5, 3));
        assert_eq!(meter.window, 100);
        assert!((meter.balanced_accuracy.unwrap() - 0.8).abs() < 1e-12);
        assert_eq!((meter.neg_correct, meter.negatives), (95, 95));
        assert_eq!((meter.pos_correct, meter.positives), (3, 5));
    }

    #[test]
    fn majority_only_classifier_scores_half_not_the_base_rate() {
        // 99 negatives (all correct), 1 positive missed — but that is low-sample (1 positive).
        // Bump to 100 rows with 5 positives, none caught: TNR 1.0, TPR 0.0 ⇒ 0.5.
        let meter = summarize(&window(95, 5, 0));
        assert!((meter.balanced_accuracy.unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn thin_window_withholds_the_percentage() {
        let meter = summarize(&window(40, 5, 5)); // only 45 rows
        assert!(meter.low_sample);
        assert_eq!(meter.balanced_accuracy, None);
        let text = render(&meter, "styling", "fine", "weird");
        assert!(text.contains("not enough data yet"));
        assert!(!text.contains('%'));
    }

    #[test]
    fn too_few_positives_withholds_the_percentage() {
        let meter = summarize(&window(98, 2, 2)); // 100 rows but only 2 positives
        assert!(meter.low_sample);
        assert_eq!(meter.balanced_accuracy, None);
    }

    #[test]
    fn reason_accuracy_counts_only_predicted_and_actual_reasons() {
        let rows = vec![
            ScoredRow {
                predicted_positive: true,
                actual_positive: true,
                predicted_reason: Some("EMAIL".into()),
                actual_reason: Some("EMAIL".into()),
            },
            ScoredRow {
                predicted_positive: true,
                actual_positive: true,
                predicted_reason: Some("ORG".into()),
                actual_reason: Some("PERSON".into()),
            },
            // Reason uncertain (none predicted) — excluded from the reason figure.
            ScoredRow {
                predicted_positive: true,
                actual_positive: true,
                predicted_reason: None,
                actual_reason: Some("DATE".into()),
            },
        ];
        let meter = summarize(&rows);
        assert_eq!(meter.reason_total, 2);
        assert_eq!(meter.reason_correct, 1);
    }

    #[test]
    fn tail_keeps_only_the_last_window() {
        let rows: Vec<ScoredRow> = (0..150).map(|_| row(false, false)).collect();
        assert_eq!(tail(rows).len(), WINDOW_SIZE);
    }

    #[test]
    fn empty_window_renders_a_no_data_note() {
        let text = render(&summarize(&[]), "censor", "reject", "confirm");
        assert!(text.contains("no predictions logged yet"));
    }
}
