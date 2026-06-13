//! Pure L2-regularized logistic regression, trained by batch gradient descent.
//!
//! This is the shared learner behind both v11 heads. It is deliberately small, allocation-light,
//! and free of `unsafe` (the crate root forbids it). Two design choices make it fit the project's
//! constraints:
//!
//! - **Z-score standardization is built in.** [`fit`] computes each feature's mean/std over the
//!   training set, stores them in the returned [`BinaryModel`], and re-applies them in
//!   [`BinaryModel::predict_proba`]. Inputs therefore need no pre-scaling, and — because every
//!   feature is on the same scale — the fitted weights double as *signed feature importance*.
//! - **Class weighting handles imbalance.** With [`FitOptions::class_weighting`] on, each class's
//!   gradient is up-weighted by inverse frequency, so the rare positive (~2–3% styling, ~9%
//!   censor) is actually learned instead of ignored. The decision threshold is a separate lever
//!   applied at [`BinaryModel::predict`] time (tuned downstream in `train`).
//!
//! The multiclass reason head is a thin one-vs-rest wrapper, [`MulticlassModel`].
//!
//! ```
//! use stencil::ml::logreg::{fit, FitOptions};
//!
//! // Linearly separable 1-D data: negatives near -1, positives near +1.
//! let x = vec![vec![-1.0], vec![-0.8], vec![1.0], vec![0.9]];
//! let y = [0u8, 0, 1, 1];
//! let model = fit(&x, &y, &FitOptions::default());
//!
//! assert!(model.predict_proba(&[1.0]) > 0.5);
//! assert!(model.predict_proba(&[-1.0]) < 0.5);
//! ```

use serde::{Deserialize, Serialize};

/// Default L2 regularization strength — modest, so weights stay interpretable without overfitting
/// the small logs. A tunable constant, not a magic number sprinkled through the code.
pub const DEFAULT_LAMBDA: f64 = 1e-2;

/// Default gradient-descent step size. Safe on standardized (well-conditioned) features.
pub const DEFAULT_LEARNING_RATE: f64 = 0.5;

/// Default number of full-batch gradient-descent iterations. Training sets are tiny (≤10k rows,
/// ≤~45 features), so this stays in the milliseconds.
pub const DEFAULT_ITERATIONS: usize = 1000;

/// Hyperparameters for a single [`fit`].
#[derive(Debug, Clone, PartialEq)]
pub struct FitOptions {
    /// L2 regularization strength (the bias term is never regularized). For stable convergence the
    /// weight-decay factor `learning_rate * lambda` must stay below `1.0`; the defaults leave a wide
    /// margin (`0.5 * 1e-2`).
    pub lambda: f64,
    /// Gradient-descent step size.
    pub learning_rate: f64,
    /// Number of full-batch gradient-descent iterations.
    pub iterations: usize,
    /// Up-weight each class by inverse frequency, so the rare class is not ignored.
    pub class_weighting: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            lambda: DEFAULT_LAMBDA,
            learning_rate: DEFAULT_LEARNING_RATE,
            iterations: DEFAULT_ITERATIONS,
            class_weighting: true,
        }
    }
}

/// Per-feature z-score normalization, learned from the training set and re-applied at predict time.
///
/// A feature with zero variance keeps a std of `1.0`, so standardizing it yields `0.0` rather than a
/// `NaN`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Normalizer {
    /// Per-feature mean over the training set.
    pub means: Vec<f64>,
    /// Per-feature standard deviation (clamped to `>= 1.0`-safe: zero variance ⇒ `1.0`).
    pub stds: Vec<f64>,
}

impl Normalizer {
    /// Learn per-feature mean/std from `rows`. Returns empty stats for an empty training set.
    pub fn fit(rows: &[Vec<f64>]) -> Self {
        let n_features = rows.first().map_or(0, Vec::len);
        let count = rows.len();
        if count == 0 || n_features == 0 {
            return Self {
                means: vec![0.0; n_features],
                stds: vec![1.0; n_features],
            };
        }

        let n = count as f64;
        let mut means = vec![0.0; n_features];
        for row in rows {
            for (mean, value) in means.iter_mut().zip(row) {
                *mean += value;
            }
        }
        for mean in &mut means {
            *mean /= n;
        }

        let mut variances = vec![0.0; n_features];
        for row in rows {
            for (variance, (value, mean)) in variances.iter_mut().zip(row.iter().zip(&means)) {
                let delta = value - mean;
                *variance += delta * delta;
            }
        }
        let stds = variances
            .iter()
            .map(|variance| {
                let std = (variance / n).sqrt();
                if std > 0.0 { std } else { 1.0 }
            })
            .collect();

        Self { means, stds }
    }

    /// Standardize one raw feature vector into z-scores. Pairs element-wise with the stored stats,
    /// stopping at the shorter length (so a dimension mismatch degrades rather than panics).
    pub fn standardize(&self, raw: &[f64]) -> Vec<f64> {
        self.means
            .iter()
            .zip(&self.stds)
            .zip(raw)
            .map(|((mean, std), value)| (value - mean) / std)
            .collect()
    }
}

/// A fitted binary logistic-regression model: standardized-space weights, a bias, and the
/// normalizer needed to map raw inputs into that space.
///
/// Because the weights live in standardized space, their magnitude is comparable across features —
/// the largest-magnitude entries are the strongest drivers of a positive prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryModel {
    /// One weight per feature, in standardized space (signed feature importance).
    pub weights: Vec<f64>,
    /// The intercept term (never regularized).
    pub bias: f64,
    /// The standardization applied to raw inputs before scoring.
    pub normalizer: Normalizer,
}

impl BinaryModel {
    /// Probability that `raw` belongs to the positive class, in `0.0..=1.0`.
    ///
    /// ```
    /// use stencil::ml::logreg::{fit, FitOptions};
    /// let x = vec![vec![0.0], vec![1.0]];
    /// let model = fit(&x, &[0u8, 1], &FitOptions::default());
    /// let p = model.predict_proba(&[1.0]);
    /// assert!((0.0..=1.0).contains(&p));
    /// ```
    pub fn predict_proba(&self, raw: &[f64]) -> f64 {
        let standardized = self.normalizer.standardize(raw);
        sigmoid(dot(&self.weights, &standardized) + self.bias)
    }

    /// Positive-class decision at `threshold` (e.g. `0.5`, or a tuned value from `train`).
    pub fn predict(&self, raw: &[f64], threshold: f64) -> bool {
        self.predict_proba(raw) >= threshold
    }
}

/// Fit a binary logistic-regression model. `y` is `1` for the positive class, `0` otherwise; rows
/// of `x` and entries of `y` correspond by index.
///
/// Degenerate input (empty `x`, or a single class) never panics: it yields a valid model whose
/// predictions are simply uninformative. Guarding against a single-class log is the caller's job
/// (the cold-start guard in `train`).
pub fn fit(x: &[Vec<f64>], y: &[u8], opts: &FitOptions) -> BinaryModel {
    let normalizer = Normalizer::fit(x);
    let standardized: Vec<Vec<f64>> = x.iter().map(|row| normalizer.standardize(row)).collect();
    let (weights, bias) = train_weights(&standardized, y, opts);
    BinaryModel {
        weights,
        bias,
        normalizer,
    }
}

/// A one-vs-rest multiclass model: one [`BinaryModel`] per class, sharing the prediction-time
/// normalization of each binary head. Used for the reason heads (weird-category / value-type).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MulticlassModel {
    /// Class labels, indexed in parallel with [`Self::models`].
    pub classes: Vec<String>,
    /// One binary "this class vs the rest" model per label.
    pub models: Vec<BinaryModel>,
}

impl MulticlassModel {
    /// Per-class scores for `raw`, normalized to sum to `1.0`. An empty model yields an empty
    /// vector; an all-zero raw score collapses to a uniform distribution.
    pub fn predict_scores(&self, raw: &[f64]) -> Vec<f64> {
        let raw_scores: Vec<f64> = self
            .models
            .iter()
            .map(|model| model.predict_proba(raw))
            .collect();
        let sum: f64 = raw_scores.iter().sum();
        if sum > 0.0 {
            raw_scores.iter().map(|score| score / sum).collect()
        } else if raw_scores.is_empty() {
            Vec::new()
        } else {
            let uniform = 1.0 / raw_scores.len() as f64;
            vec![uniform; raw_scores.len()]
        }
    }

    /// The best class index and its normalized score, or `None` when the model has no classes.
    ///
    /// ```
    /// use stencil::ml::logreg::{fit_multiclass, FitOptions};
    /// // Two classes separated on a single feature.
    /// let x = vec![vec![-1.0], vec![-0.9], vec![1.0], vec![0.8]];
    /// let y = [0usize, 0, 1, 1];
    /// let model = fit_multiclass(&x, &y, vec!["a".into(), "b".into()], &FitOptions::default());
    /// let (best, score) = model.best(&[1.0]).expect("two classes");
    /// assert_eq!(model.classes[best], "b");
    /// assert!(score > 0.5);
    /// ```
    pub fn best(&self, raw: &[f64]) -> Option<(usize, f64)> {
        self.predict_scores(raw)
            .into_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
    }
}

/// Fit a one-vs-rest [`MulticlassModel`]. `y` holds the class *index* of each row (`0..classes.len()`);
/// out-of-range indices simply never set a positive target.
pub fn fit_multiclass(
    x: &[Vec<f64>],
    y: &[usize],
    classes: Vec<String>,
    opts: &FitOptions,
) -> MulticlassModel {
    let models = (0..classes.len())
        .map(|class| {
            let binary_y: Vec<u8> = y.iter().map(|&label| u8::from(label == class)).collect();
            fit(x, &binary_y, opts)
        })
        .collect();
    MulticlassModel { classes, models }
}

/// Run gradient descent over already-standardized rows, returning `(weights, bias)`.
fn train_weights(standardized: &[Vec<f64>], y: &[u8], opts: &FitOptions) -> (Vec<f64>, f64) {
    let n_features = standardized.first().map_or(0, Vec::len);
    let count = standardized.len();
    if count == 0 {
        return (vec![0.0; n_features], 0.0);
    }

    let n = count as f64;
    let (weight_pos, weight_neg) = class_weights(y, opts.class_weighting);

    let mut weights = vec![0.0; n_features];
    let mut bias = 0.0;
    for _ in 0..opts.iterations {
        let mut grad_w = vec![0.0; n_features];
        let mut grad_b = 0.0;
        for (row, &label) in standardized.iter().zip(y) {
            let probability = sigmoid(dot(&weights, row) + bias);
            let class_weight = if label == 1 { weight_pos } else { weight_neg };
            let error = class_weight * (probability - f64::from(label));
            for (gradient, value) in grad_w.iter_mut().zip(row) {
                *gradient += error * value;
            }
            grad_b += error;
        }
        for (weight, gradient) in weights.iter_mut().zip(&grad_w) {
            let regularized = gradient / n + opts.lambda * *weight;
            *weight -= opts.learning_rate * regularized;
        }
        bias -= opts.learning_rate * (grad_b / n);
    }
    (weights, bias)
}

/// Inverse-frequency class weights `(positive, negative)`, balanced so each class contributes
/// equally. Falls back to `(1.0, 1.0)` when weighting is off or a class is absent.
fn class_weights(y: &[u8], enabled: bool) -> (f64, f64) {
    let n = y.len() as f64;
    let n_pos = y.iter().filter(|&&label| label == 1).count() as f64;
    let n_neg = n - n_pos;
    if enabled && n_pos > 0.0 && n_neg > 0.0 {
        (n / (2.0 * n_pos), n / (2.0 * n_neg))
    } else {
        (1.0, 1.0)
    }
}

/// Numerically stable logistic sigmoid; never overflows for large-magnitude `z`.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let exp = z.exp();
        exp / (1.0 + exp)
    }
}

/// Dot product of two equal-length slices (stops at the shorter, so it never indexes out of range).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fraction of positive-class rows the model predicts positive, at `threshold`.
    fn positive_recall(model: &BinaryModel, x: &[Vec<f64>], y: &[u8], threshold: f64) -> f64 {
        let positives: Vec<&Vec<f64>> = x
            .iter()
            .zip(y)
            .filter(|(_, label)| **label == 1)
            .map(|(row, _)| row)
            .collect();
        if positives.is_empty() {
            return 0.0;
        }
        let caught = positives
            .iter()
            .filter(|row| model.predict(row, threshold))
            .count();
        caught as f64 / positives.len() as f64
    }

    fn l2_norm(weights: &[f64]) -> f64 {
        weights.iter().map(|w| w * w).sum::<f64>().sqrt()
    }

    #[test]
    fn sigmoid_is_stable_and_monotonic() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(40.0) > 0.999);
        assert!(sigmoid(-40.0) < 0.001);
        // No overflow / NaN at extreme magnitudes.
        assert!(sigmoid(1e6).is_finite());
        assert!(sigmoid(-1e6).is_finite());
        assert!(sigmoid(1.0) > sigmoid(-1.0));
    }

    #[test]
    fn recovers_separable_boundary() {
        // Negatives at -1, positives at +1 along one axis — perfectly separable.
        let x = vec![
            vec![-1.0],
            vec![-0.9],
            vec![-1.1],
            vec![1.0],
            vec![0.9],
            vec![1.1],
        ];
        let y = [0u8, 0, 0, 1, 1, 1];
        let model = fit(&x, &y, &FitOptions::default());
        for (row, &label) in x.iter().zip(&y) {
            assert_eq!(
                model.predict(row, 0.5),
                label == 1,
                "misclassified {row:?} (label {label})"
            );
        }
    }

    #[test]
    fn weights_shrink_as_lambda_grows() {
        let x = vec![vec![-1.0], vec![-0.8], vec![1.0], vec![0.9]];
        let y = [0u8, 0, 1, 1];
        let weak = fit(
            &x,
            &y,
            &FitOptions {
                lambda: 1e-4,
                ..FitOptions::default()
            },
        );
        // A strong-but-stable lambda (keeps learning_rate * lambda < 1).
        let strong = fit(
            &x,
            &y,
            &FitOptions {
                lambda: 1.0,
                ..FitOptions::default()
            },
        );
        assert!(
            l2_norm(&strong.weights) < l2_norm(&weak.weights),
            "stronger L2 must shrink the weights: weak={:.4} strong={:.4}",
            l2_norm(&weak.weights),
            l2_norm(&strong.weights)
        );
    }

    #[test]
    fn class_weighting_improves_minority_recall() {
        // Heavy imbalance with overlap at x=+0.5: 100 clean negatives at -1, then an ambiguous
        // cluster at +0.5 that is mostly negative (20) but holds the 15 positives. Unweighted, the
        // majority pushes the boundary so the cluster reads negative; weighting rebalances it.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..100 {
            x.push(vec![-1.0]);
            y.push(0u8);
        }
        for _ in 0..20 {
            x.push(vec![0.5]);
            y.push(0u8);
        }
        for _ in 0..15 {
            x.push(vec![0.5]);
            y.push(1u8);
        }

        let unweighted = fit(
            &x,
            &y,
            &FitOptions {
                class_weighting: false,
                ..FitOptions::default()
            },
        );
        let weighted = fit(
            &x,
            &y,
            &FitOptions {
                class_weighting: true,
                ..FitOptions::default()
            },
        );

        let recall_unweighted = positive_recall(&unweighted, &x, &y, 0.5);
        let recall_weighted = positive_recall(&weighted, &x, &y, 0.5);
        assert!(
            recall_weighted > recall_unweighted,
            "class weighting must raise minority recall: unweighted={recall_unweighted:.2} weighted={recall_weighted:.2}"
        );
    }

    #[test]
    fn standardization_makes_predict_scale_invariant() {
        // The same data with one feature scaled by 100×; standardization should absorb the scale,
        // so the predicted probability at the correspondingly-scaled query point matches.
        let base_x = vec![
            vec![-1.0, 0.2],
            vec![-0.5, 0.1],
            vec![1.0, -0.2],
            vec![0.7, -0.1],
        ];
        let y = [0u8, 0, 1, 1];
        let scaled_x: Vec<Vec<f64>> = base_x
            .iter()
            .map(|row| vec![row[0] * 100.0, row[1]])
            .collect();

        let base = fit(&base_x, &y, &FitOptions::default());
        let scaled = fit(&scaled_x, &y, &FitOptions::default());

        let query = [1.0, -0.2];
        let scaled_query = [100.0, -0.2];
        let p_base = base.predict_proba(&query);
        let p_scaled = scaled.predict_proba(&scaled_query);
        assert!(
            (p_base - p_scaled).abs() < 1e-9,
            "standardized models must agree under feature scaling: base={p_base:.6} scaled={p_scaled:.6}"
        );
    }

    #[test]
    fn multiclass_predicts_argmax_and_normalized_scores() {
        // Three classes separated along one axis: low, mid, high.
        let x = vec![
            vec![-2.0],
            vec![-1.8],
            vec![0.0],
            vec![0.1],
            vec![2.0],
            vec![1.9],
        ];
        let y = [0usize, 0, 1, 1, 2, 2];
        let classes = vec!["low".to_string(), "mid".to_string(), "high".to_string()];
        let model = fit_multiclass(&x, &y, classes, &FitOptions::default());

        let scores = model.predict_scores(&[2.0]);
        assert_eq!(scores.len(), 3);
        let sum: f64 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "scores must be normalized");

        let (best, score) = model.best(&[2.0]).expect("three classes");
        assert_eq!(model.classes[best], "high");
        assert!(score >= scores[0] && score >= scores[1]);
    }

    #[test]
    fn fit_on_empty_is_panic_free() {
        let model = fit(&[], &[], &FitOptions::default());
        assert!(model.weights.is_empty());
        // Predicting on an empty model is the bias-only sigmoid — finite, no panic.
        assert!(model.predict_proba(&[]).is_finite());
    }

    #[test]
    fn constant_feature_does_not_produce_nan() {
        // Feature 1 is constant (zero variance) → std clamped to 1.0, no NaN.
        let x = vec![vec![0.0, 5.0], vec![1.0, 5.0], vec![2.0, 5.0]];
        let y = [0u8, 1, 1];
        let model = fit(&x, &y, &FitOptions::default());
        assert!(model.weights.iter().all(|w| w.is_finite()));
        assert!(model.predict_proba(&[1.0, 5.0]).is_finite());
    }

    #[test]
    fn empty_multiclass_best_is_none() {
        let model = fit_multiclass(&[vec![1.0]], &[0], Vec::new(), &FitOptions::default());
        assert_eq!(model.best(&[1.0]), None);
        assert!(model.predict_scores(&[1.0]).is_empty());
    }
}
