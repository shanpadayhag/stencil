//! The on-disk model artifact: one inspectable JSON file per task, version-gated and atomically
//! swapped.
//!
//! A [`TaskModel`] bundles both heads (the binary verdict head and the optional multiclass reason
//! head) plus the metadata a reviewer or a future retrain needs. It is plain `serde_json`, so the
//! file is human-readable — the standardized weights are legible as signed feature importances.
//!
//! Two safety properties live here:
//! - **Version gate.** [`load`] returns `None` for a missing file, a parse error, *or* a
//!   `feature_spec_version` that does not match the current [`FEATURE_SPEC_VERSION`]. A stale model
//!   is therefore never used to mispredict against a changed feature encoding — it reads as "no
//!   model" (the review then shows no suggestion; `train` rebuilds it).
//! - **Atomic swap.** [`save_atomic`] writes a sibling `*.tmp` and `rename`s it over the target, so
//!   a concurrent reader always sees either the old complete file or the new one — never a partial
//!   write.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ml::features::FEATURE_SPEC_VERSION;
use crate::ml::logreg::{BinaryModel, MulticlassModel};

/// The binary verdict head: a fitted logistic-regression model plus its tuned decision threshold.
///
/// The model carries its own weights, bias, and standardization (see [`BinaryModel`]); the
/// `threshold` is the per-head cutoff chosen by `train` (cross-validated), not a fixed `0.5`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Head {
    /// The fitted binary model (standardized weights + bias + normalizer).
    pub model: BinaryModel,
    /// The positive-class decision threshold on `model.predict_proba`.
    pub threshold: f64,
}

/// One task's complete trained artifact (styling or censor): both heads + metadata.
///
/// The reason head is `Option` because it is only trained when the positives carry ≥2 distinct
/// reason classes; otherwise the verdict shows alone and the reason renders "(uncertain)".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskModel {
    /// The feature-spec version this model was trained against; checked at [`load`].
    pub feature_spec_version: u32,
    /// When the model was trained — a caller-supplied stamp (Unix epoch seconds as a string in v11),
    /// echoed onto each logged prediction so a row can be tied back to the model that produced it.
    pub trained_at: String,
    /// How many records the model was trained on.
    pub n_records: usize,
    /// The binary verdict head (`fine`/`weird` or `reject`/`confirm`).
    pub verdict: Head,
    /// The one-vs-rest reason head (weird-category / value-type), when one could be trained.
    pub reason: Option<MulticlassModel>,
    /// Held-out (k-fold CV) balanced accuracy of the verdict head — the honest train-time estimate.
    pub cv_balanced_accuracy: f64,
}

impl TaskModel {
    /// Whether this artifact matches the current feature encoding. A mismatch means the on-disk
    /// model predates a feature change and must not be used.
    pub fn is_current(&self) -> bool {
        self.feature_spec_version == FEATURE_SPEC_VERSION
    }
}

/// Load a [`TaskModel`] from `path`, or `None` when there is no usable model.
///
/// By design this collapses every failure mode — missing file, unreadable/corrupt JSON, and a
/// stale `feature_spec_version` — into `None`. The suggestive model is advisory, so "no usable
/// model" is always a safe, silent fallback (no suggestion) rather than an error to surface.
pub fn load(path: &Path) -> Option<TaskModel> {
    let text = fs::read_to_string(path).ok()?;
    let model: TaskModel = serde_json::from_str(&text).ok()?;
    model.is_current().then_some(model)
}

/// Atomically write `model` to `path`: serialize to a sibling `<path>.tmp`, then `rename` it over
/// `path`. On success no temp file remains and the live file is never left partially written.
///
/// # Errors
/// Returns an error if the parent directory cannot be created or the temp file cannot be written or
/// renamed.
///
/// ```
/// use stencil::ml::artifact::{load, save_atomic, Head, TaskModel};
/// use stencil::ml::features::FEATURE_SPEC_VERSION;
/// use stencil::ml::logreg::{fit, FitOptions};
///
/// let model = fit(&[vec![0.0], vec![1.0]], &[0u8, 1], &FitOptions::default());
/// let task = TaskModel {
///     feature_spec_version: FEATURE_SPEC_VERSION,
///     trained_at: "2026-06-12T00:00:00Z".into(),
///     n_records: 2,
///     verdict: Head { model, threshold: 0.5 },
///     reason: None,
///     cv_balanced_accuracy: 0.5,
/// };
///
/// let path = std::env::temp_dir().join("stencil_artifact_doctest.json");
/// save_atomic(&path, &task)?;
/// let loaded = load(&path).expect("a current-version model round-trips");
/// assert_eq!(loaded, task);
/// # std::fs::remove_file(&path).ok();
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn save_atomic(path: &Path, model: &TaskModel) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create `{}`", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(model).context("failed to serialize model")?;
    let tmp = temp_sibling(path);
    fs::write(&tmp, json).with_context(|| format!("failed to write `{}`", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| {
        format!(
            "failed to swap `{}` into `{}`",
            tmp.display(),
            path.display()
        )
    })
}

/// `path` with `.tmp` appended (e.g. `model.json` → `model.json.tmp`), keeping the temp file in the
/// same directory so the `rename` stays on one filesystem and is therefore atomic.
fn temp_sibling(path: &Path) -> PathBuf {
    let mut name: OsString = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::logreg::{FitOptions, fit};

    fn sample_model() -> TaskModel {
        let model = fit(&[vec![-1.0], vec![1.0]], &[0u8, 1], &FitOptions::default());
        TaskModel {
            feature_spec_version: FEATURE_SPEC_VERSION,
            trained_at: "2026-06-12T00:00:00Z".into(),
            n_records: 2,
            verdict: Head {
                model,
                threshold: 0.42,
            },
            reason: None,
            cv_balanced_accuracy: 0.75,
        }
    }

    fn temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("stencil_artifact_{}_{}", label, std::process::id()))
    }

    #[test]
    fn round_trips_through_json() {
        let model = sample_model();
        let json = serde_json::to_string_pretty(&model).expect("serialize");
        // Human-readable: the threshold and weights are visible in the text.
        assert!(json.contains("\"threshold\": 0.42"));
        assert!(json.contains("\"weights\""));
        let back: TaskModel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, model);
    }

    #[test]
    fn save_atomic_round_trips_and_leaves_no_temp() {
        let dir = temp_dir("save");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("model.json");
        let model = sample_model();

        save_atomic(&path, &model).expect("save");
        assert_eq!(load(&path), Some(model), "loads back identical");
        assert!(
            !temp_sibling(&path).exists(),
            "no .tmp file remains after a successful swap"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_atomic_overwrites_existing_without_partial_state() {
        let dir = temp_dir("overwrite");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("model.json");

        let mut first = sample_model();
        first.n_records = 10;
        save_atomic(&path, &first).expect("save first");

        let mut second = sample_model();
        second.n_records = 99;
        save_atomic(&path, &second).expect("save second");

        let loaded = load(&path).expect("loads the second model");
        assert_eq!(loaded.n_records, 99, "the swap replaced the old file");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_missing_file_is_none() {
        let path = temp_dir("missing").join("nope.json");
        let _ = fs::remove_file(&path);
        assert_eq!(load(&path), None);
    }

    #[test]
    fn load_corrupt_json_is_none() {
        let dir = temp_dir("corrupt");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("model.json");
        fs::write(&path, "{ not valid json").expect("write");
        assert_eq!(load(&path), None, "a corrupt artifact reads as no model");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_stale_feature_spec_version_is_none() {
        let dir = temp_dir("stale");
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("model.json");
        let mut model = sample_model();
        model.feature_spec_version = FEATURE_SPEC_VERSION + 1;
        save_atomic(&path, &model).expect("save");

        assert!(!model.is_current());
        assert_eq!(
            load(&path),
            None,
            "a model from a different feature spec must not be used"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
