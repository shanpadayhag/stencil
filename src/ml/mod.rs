//! Machine-learning core for the v11 *suggestive* models (styling + censor).
//!
//! The learner is a classical, interpretable, class-aware **logistic regression**, hand-rolled in
//! pure Rust with no new dependencies. Everything here is decoupled from detection, censoring, and
//! restyling: nothing in this module is called by those paths — it only ever reads the logged
//! records and produces an advisory prediction.
//!
//! - [`logreg`] — the pure L2-regularized logistic-regression core (fit / predict / one-vs-rest).
//! - [`features`] — record → feature vector mappers + `FEATURE_SPEC_VERSION` (the locked encoding).
//! - [`artifact`] — the on-disk `TaskModel` (version-gated load + atomic save).
//! - [`train`] — full-batch training: records → fitted `TaskModel` with a CV-tuned threshold.
//! - [`predict`] — inference: a `TaskModel` + a feature vector → an advisory `Suggestion`.
//! - [`accuracy`] — the prequential meter over logged predictions (balanced accuracy + reason).

pub mod accuracy;
pub mod artifact;
pub mod features;
pub mod logreg;
pub mod predict;
pub mod train;
