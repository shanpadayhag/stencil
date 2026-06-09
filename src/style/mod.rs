//! Styling stage: read each document block's formatting into a [`StyledBlock`], build a
//! descriptive document-wide style profile, and drive the manual per-block review.
//!
//! Detection here is deliberately absent — the stage measures styling and records the
//! reviewer's verdicts; it never labels a block "weird" on its own.
//!
//! - [`extract`] — walk a `.docx` into [`crate::model::StyledBlock`]s (task T28).
//! - [`profile`] — descriptive document style profile + per-block relative features (T29).
//! - [`review`] — interactive two-step per-block verdict loop (T30).
//! - [`record`] — assemble and persist [`crate::learn::StylingRecord`]s + profile sidecar (T30).

pub mod extract;
pub mod profile;
pub mod record;
pub mod review;
