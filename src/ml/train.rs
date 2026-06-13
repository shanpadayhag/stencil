//! Full-batch training: turn a task's logged records (already mapped to features + labels) into a
//! fitted [`TaskModel`] with a cross-validated decision threshold.
//!
//! This module is pure — it takes [`TrainingData`] (built by the caller from the T63 mappers) and
//! returns a model, with no file or clock access. The verdict head is fit on all rows; its
//! threshold and the reported balanced accuracy come from a **k-fold CV** pass (held-out, so the
//! number is honest rather than optimistic train-fit). The optional reason head is one-vs-rest over
//! the positive rows. The cold-start guard refuses to train a degenerate single-class model.

use crate::ml::artifact::{Head, TaskModel};
use crate::ml::features::FEATURE_SPEC_VERSION;
use crate::ml::logreg::{BinaryModel, FitOptions, MulticlassModel, fit, fit_multiclass};

/// Folds for the CV threshold/metric pass, clamped down for thin data (see [`choose_k`]).
const CV_FOLDS: usize = 5;

/// A task's training inputs, mapped out of the raw records by the caller. Keeping this decoupled
/// from the record types and feature mappers lets the trainer stay pure and task-agnostic.
///
/// All three vectors are parallel (one entry per record): `features[i]` is the encoded vector,
/// `verdict_labels[i]` is `1` for the positive class (`weird`/`confirm`) else `0`, and `reasons[i]`
/// is the reason label (weird-category / value-type) — `Some` only on positive rows.
pub struct TrainingData {
    /// One encoded feature vector per record.
    pub features: Vec<Vec<f64>>,
    /// Verdict label per record: `1` = positive, `0` = negative.
    pub verdict_labels: Vec<u8>,
    /// Reason label per record; `Some` only on positives.
    pub reasons: Vec<Option<String>>,
}

/// The result of a training attempt.
pub enum TrainOutcome {
    /// A model was trained.
    Trained(Box<TaskModel>),
    /// The cold-start guard tripped (a class was absent) — no model written.
    NotEnoughData {
        /// Total records seen.
        n_records: usize,
        /// Positive-class records seen.
        positives: usize,
    },
}

/// Train both heads from `data`, stamping `trained_at` on the artifact.
///
/// Returns [`TrainOutcome::NotEnoughData`] (and trains nothing) when either class is absent — the
/// verdict head would otherwise be degenerate, and the artifact must read as "no model".
pub fn train(data: &TrainingData, trained_at: String, opts: &FitOptions) -> TrainOutcome {
    let n_records = data.features.len();
    let positives = data
        .verdict_labels
        .iter()
        .filter(|&&label| label == 1)
        .count();
    let negatives = n_records - positives;
    if positives < 1 || negatives < 1 {
        return TrainOutcome::NotEnoughData {
            n_records,
            positives,
        };
    }

    let verdict_model = fit(&data.features, &data.verdict_labels, opts);
    let (threshold, cv_balanced_accuracy) =
        cross_validate(&data.features, &data.verdict_labels, opts);
    let reason = train_reason_head(&data.features, &data.reasons, opts);

    TrainOutcome::Trained(Box::new(TaskModel {
        feature_spec_version: FEATURE_SPEC_VERSION,
        trained_at,
        n_records,
        verdict: Head {
            model: verdict_model,
            threshold,
        },
        reason,
        cv_balanced_accuracy,
    }))
}

/// k-fold CV over the verdict head: returns the threshold that maximizes held-out balanced accuracy
/// and that accuracy. Falls back to a single full-fit self-estimate when the data is too thin to
/// split (fewer than two of either class per fold).
fn cross_validate(features: &[Vec<f64>], labels: &[u8], opts: &FitOptions) -> (f64, f64) {
    let positives = labels.iter().filter(|&&label| label == 1).count();
    let negatives = labels.len() - positives;
    let k = choose_k(positives, negatives);

    if k < 2 {
        // Too few to cross-validate: tune on the full-fit's own predictions (a weak, optimistic
        // estimate, flagged as noisy upstream; the live prequential meter is the real signal).
        let model = fit(features, labels, opts);
        let predictions = predict_all(&model, features, labels);
        return tune_threshold(&predictions);
    }

    let folds = fold_assignments(labels, k);
    let mut out_of_fold: Vec<(f64, u8)> = Vec::with_capacity(features.len());
    for held_out in 0..k {
        let mut train_features = Vec::new();
        let mut train_labels = Vec::new();
        for (index, (row, &label)) in features.iter().zip(labels).enumerate() {
            if folds[index] != held_out {
                train_features.push(row.clone());
                train_labels.push(label);
            }
        }
        let model = fit(&train_features, &train_labels, opts);
        for (index, (row, &label)) in features.iter().zip(labels).enumerate() {
            if folds[index] == held_out {
                out_of_fold.push((model.predict_proba(row), label));
            }
        }
    }
    tune_threshold(&out_of_fold)
}

/// Folds to use: at most [`CV_FOLDS`], but never more than the rarer class has members (so every
/// fold can hold at least one of each).
fn choose_k(positives: usize, negatives: usize) -> usize {
    CV_FOLDS.min(positives).min(negatives)
}

/// Stratified round-robin fold assignment: positives and negatives are each spread evenly across
/// the `k` folds, so leaving one out still leaves both classes in the training set. Deterministic
/// (no RNG), so a retrain on the same log is reproducible.
fn fold_assignments(labels: &[u8], k: usize) -> Vec<usize> {
    let mut folds = vec![0usize; labels.len()];
    let (mut positive_seen, mut negative_seen) = (0usize, 0usize);
    for (index, &label) in labels.iter().enumerate() {
        if label == 1 {
            folds[index] = positive_seen % k;
            positive_seen += 1;
        } else {
            folds[index] = negative_seen % k;
            negative_seen += 1;
        }
    }
    folds
}

/// Score every row, pairing the probability with its true label.
fn predict_all(model: &BinaryModel, features: &[Vec<f64>], labels: &[u8]) -> Vec<(f64, u8)> {
    features
        .iter()
        .zip(labels)
        .map(|(row, &label)| (model.predict_proba(row), label))
        .collect()
}

/// Pick the threshold maximizing balanced accuracy over `predictions`, breaking ties toward `0.5`.
/// Candidates are the observed probabilities (predict positive when `proba >= threshold`), plus the
/// `0.5` baseline.
fn tune_threshold(predictions: &[(f64, u8)]) -> (f64, f64) {
    let mut best_threshold: f64 = 0.5;
    let mut best_accuracy = balanced_accuracy(predictions, 0.5);
    for &(candidate, _) in predictions {
        let accuracy = balanced_accuracy(predictions, candidate);
        let improves = accuracy > best_accuracy + 1e-12;
        let ties_closer_to_half = (accuracy - best_accuracy).abs() <= 1e-12
            && (candidate - 0.5).abs() < (best_threshold - 0.5).abs();
        if improves || ties_closer_to_half {
            best_threshold = candidate;
            best_accuracy = accuracy;
        }
    }
    (best_threshold, best_accuracy)
}

/// Balanced accuracy — the mean of the per-class hit rates — at `threshold`. Robust to imbalance: a
/// majority-class-only classifier scores `0.5`, not the majority fraction.
fn balanced_accuracy(predictions: &[(f64, u8)], threshold: f64) -> f64 {
    let (mut true_pos, mut false_neg, mut true_neg, mut false_pos) =
        (0usize, 0usize, 0usize, 0usize);
    for &(proba, label) in predictions {
        match (label == 1, proba >= threshold) {
            (true, true) => true_pos += 1,
            (true, false) => false_neg += 1,
            (false, false) => true_neg += 1,
            (false, true) => false_pos += 1,
        }
    }
    let true_positive_rate = ratio(true_pos, true_pos + false_neg);
    let true_negative_rate = ratio(true_neg, true_neg + false_pos);
    (true_positive_rate + true_negative_rate) / 2.0
}

/// `numerator / denominator`, or `0.0` when the denominator is zero.
fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Train the one-vs-rest reason head over the positive rows, or `None` when fewer than two distinct
/// reason classes are present (a single class is degenerate — the reason stays "(uncertain)").
fn train_reason_head(
    features: &[Vec<f64>],
    reasons: &[Option<String>],
    opts: &FitOptions,
) -> Option<MulticlassModel> {
    let mut positive_features: Vec<Vec<f64>> = Vec::new();
    let mut positive_reasons: Vec<&String> = Vec::new();
    for (row, reason) in features.iter().zip(reasons) {
        if let Some(label) = reason {
            positive_features.push(row.clone());
            positive_reasons.push(label);
        }
    }

    let mut classes: Vec<String> = positive_reasons
        .iter()
        .map(|reason| (*reason).clone())
        .collect();
    classes.sort();
    classes.dedup();
    if classes.len() < 2 {
        return None;
    }

    let indices: Vec<usize> = positive_reasons
        .iter()
        .map(|reason| {
            // Every reason was just used to build `classes`, so the position always exists.
            classes
                .iter()
                .position(|class| class == *reason)
                .expect("reason class present in the derived class set")
        })
        .collect();
    Some(fit_multiclass(&positive_features, &indices, classes, opts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_accuracy_matches_known_confusion() {
        let predictions = [(0.9, 1u8), (0.8, 1), (0.2, 0), (0.1, 0)];
        assert!((balanced_accuracy(&predictions, 0.5) - 1.0).abs() < 1e-12);
        // Threshold 0.85: one positive missed (TPR 0.5), negatives all correct (TNR 1.0) ⇒ 0.75.
        assert!((balanced_accuracy(&predictions, 0.85) - 0.75).abs() < 1e-12);
        // A majority-only classifier (everything negative) scores 0.5, not the 0.5 base rate trap.
        let imbalanced = [(0.1, 0u8), (0.1, 0), (0.1, 0), (0.1, 1)];
        assert!((balanced_accuracy(&imbalanced, 0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn tune_threshold_finds_a_separating_cut() {
        let predictions = [(0.9, 1u8), (0.8, 1), (0.2, 0), (0.1, 0)];
        let (_, accuracy) = tune_threshold(&predictions);
        assert!(
            (accuracy - 1.0).abs() < 1e-12,
            "separable ⇒ balanced acc 1.0"
        );
    }

    #[test]
    fn fold_assignments_are_stratified() {
        let labels = [1u8, 1, 1, 1, 0, 0, 0, 0];
        let folds = fold_assignments(&labels, 2);
        // Each fold holds two positives and two negatives.
        for fold in 0..2 {
            let positives = labels
                .iter()
                .zip(&folds)
                .filter(|(label, f)| **label == 1 && **f == fold)
                .count();
            let negatives = labels
                .iter()
                .zip(&folds)
                .filter(|(label, f)| **label == 0 && **f == fold)
                .count();
            assert_eq!((positives, negatives), (2, 2), "fold {fold} is balanced");
        }
    }

    /// Separable data: negatives at feature[0]=0, positives at feature[0]=1; feature[1] splits the
    /// two reason classes among the positives.
    fn separable_data() -> TrainingData {
        let mut features = Vec::new();
        let mut verdict_labels = Vec::new();
        let mut reasons = Vec::new();
        for _ in 0..20 {
            features.push(vec![0.0, 0.0]);
            verdict_labels.push(0);
            reasons.push(None);
        }
        for i in 0..10 {
            let second = if i % 2 == 0 { 0.0 } else { 1.0 };
            features.push(vec![1.0, second]);
            verdict_labels.push(1);
            reasons.push(Some(if i % 2 == 0 { "a" } else { "b" }.to_string()));
        }
        TrainingData {
            features,
            verdict_labels,
            reasons,
        }
    }

    #[test]
    fn trains_both_heads_on_separable_data() {
        let data = separable_data();
        let TrainOutcome::Trained(model) = train(&data, "stamp".into(), &FitOptions::default())
        else {
            panic!("separable data should train");
        };
        assert_eq!(model.feature_spec_version, FEATURE_SPEC_VERSION);
        assert_eq!(model.n_records, 30);
        assert!(
            model.cv_balanced_accuracy > 0.9,
            "held-out balanced accuracy should be high on separable data: {}",
            model.cv_balanced_accuracy
        );
        assert!((0.0..=1.0).contains(&model.verdict.threshold));
        let reason = model.reason.expect("two reason classes ⇒ a reason head");
        assert_eq!(reason.classes, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn cold_start_with_no_positives_trains_nothing() {
        let data = TrainingData {
            features: vec![vec![0.0], vec![0.0], vec![0.0]],
            verdict_labels: vec![0, 0, 0],
            reasons: vec![None, None, None],
        };
        match train(&data, "stamp".into(), &FitOptions::default()) {
            TrainOutcome::NotEnoughData {
                n_records,
                positives,
            } => {
                assert_eq!(n_records, 3);
                assert_eq!(positives, 0);
            }
            TrainOutcome::Trained(_) => panic!("a single-class log must not train a model"),
        }
    }

    #[test]
    fn reason_head_absent_with_a_single_reason_class() {
        // Positives all share one reason ⇒ no reason head, but the verdict head still trains.
        let mut data = separable_data();
        for reason in data.reasons.iter_mut().flatten() {
            *reason = "only".to_string();
        }
        let TrainOutcome::Trained(model) = train(&data, "stamp".into(), &FitOptions::default())
        else {
            panic!("verdict head should still train");
        };
        assert!(
            model.reason.is_none(),
            "a single reason class is degenerate ⇒ no reason head"
        );
    }
}
