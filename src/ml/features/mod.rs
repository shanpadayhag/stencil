//! Record → feature-vector mappers for the v11 models, and the shared encoding primitives.
//!
//! This is the production home of the contract the T61 spike (`tests/ml_spike.rs`) locked. Two hard
//! rules govern everything here:
//!
//! - **Leakage rule.** A feature vector reads only fields known *before* the human decides. The
//!   label fields and post-decision provenance are never touched by the encoders (styling:
//!   `verdict`/`category`/`note`; censor: `verdict`/`final_type`/`span_edited`/`context_edited`/
//!   `user_added`). The label extractors ([`styling::is_weird`], [`censor::is_confirm`], and the
//!   `reason_*` helpers) are kept strictly separate from the feature path.
//! - **Versioned encoding.** [`FEATURE_SPEC_VERSION`] stamps the exact encoding. Any change to a
//!   feature vector (a field added/removed/reordered, a bucket boundary moved) MUST bump it;
//!   [`artifact`](crate::ml) load then rejects a stale on-disk model rather than mispredicting.
//!
//! Encoding conventions, applied uniformly:
//! - a numeric `Option<_>` → `[value (0.0 when absent), known-bit]` ([`opt_num`]);
//! - a tri-state `Option<bool>` → `[is_true, is_false]`, both `0.0` when unknown ([`tri_state`]);
//! - a low-cardinality string → [`one_hot`] (callers append an explicit `unknown` bit where the
//!   vocabulary can miss);
//! - a text field → six cheap [`surface_stats`] only — never the raw string.

pub mod censor;
pub mod styling;

pub use censor::censor_features;
pub use styling::styling_features;

/// The feature-spec version every encoder in this module conforms to. Bump on any encoding change.
pub const FEATURE_SPEC_VERSION: u32 = 1;

/// The four structural block kinds, in the one-hot order shared by both encoders.
pub(crate) const BLOCK_KINDS: [&str; 4] = ["paragraph", "heading", "list_item", "table_cell"];

/// A numeric `Option`: its value (`0.0` when absent) followed by a `known` bit.
pub(crate) fn opt_num(value: Option<f64>) -> [f64; 2] {
    match value {
        Some(v) => [v, 1.0],
        None => [0.0, 0.0],
    }
}

/// A tri-state `Option<bool>` as `[is_true, is_false]`; `None` ⇒ both `0.0` ("unknown" ≠ "match").
pub(crate) fn tri_state(value: Option<bool>) -> [f64; 2] {
    match value {
        Some(true) => [1.0, 0.0],
        Some(false) => [0.0, 1.0],
        None => [0.0, 0.0],
    }
}

/// One-hot over `vocab`; an unrecognized value yields all-zeros (the caller adds an `unknown` bit
/// where one is specified).
pub(crate) fn one_hot(value: &str, vocab: &[&str]) -> Vec<f64> {
    vocab
        .iter()
        .map(|known| f64::from(*known == value))
        .collect()
}

/// Multi-hot over `vocab`: each known label set if it appears anywhere in `values`.
pub(crate) fn multi_hot(values: &[String], vocab: &[&str]) -> Vec<f64> {
    vocab
        .iter()
        .map(|known| bit(values.iter().any(|value| value == known)))
        .collect()
}

/// Heading-level bucket `[h1, h2, h3, h4plus]`; `None` (not a heading) ⇒ all-zeros.
pub(crate) fn heading_bucket(level: Option<u8>) -> [f64; 4] {
    match level {
        Some(1) => [1.0, 0.0, 0.0, 0.0],
        Some(2) => [0.0, 1.0, 0.0, 0.0],
        Some(3) => [0.0, 0.0, 1.0, 0.0],
        Some(_) => [0.0, 0.0, 0.0, 1.0],
        None => [0.0, 0.0, 0.0, 0.0],
    }
}

/// `bool` → `f64` (`1.0` / `0.0`).
pub(crate) fn bit(value: bool) -> f64 {
    f64::from(value)
}

/// The six cheap surface stats of a text field, in order: length bucket, digit ratio, all-caps,
/// starts-with-number, normalized word count, punctuation-present. Reduces a raw string to shape
/// without any language model.
pub(crate) fn surface_stats(text: &str) -> [f64; 6] {
    let chars = text.chars().count();
    let len_bucket = (chars.min(200) as f64) / 200.0;
    let digit_ratio = if chars == 0 {
        0.0
    } else {
        text.chars().filter(|c| c.is_ascii_digit()).count() as f64 / chars as f64
    };
    let has_alpha = text.chars().any(char::is_alphabetic);
    let all_caps = has_alpha
        && !text
            .chars()
            .filter(|c| c.is_alphabetic())
            .any(char::is_lowercase);
    let starts_with_number = text
        .trim_start()
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_digit());
    let word_count = (text.split_whitespace().count().min(20) as f64) / 20.0;
    let has_punct = text.chars().any(|c| c.is_ascii_punctuation());
    [
        len_bucket,
        digit_ratio,
        bit(all_caps),
        bit(starts_with_number),
        word_count,
        bit(has_punct),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::features::{censor::CENSOR_FEATURE_LEN, styling::STYLING_FEATURE_LEN};

    #[test]
    fn feature_spec_version_pins_the_contract() {
        // If you change either encoder's output, update the LEN const AND bump FEATURE_SPEC_VERSION,
        // then update this pin. A stale on-disk artifact is rejected by version mismatch (T64).
        assert_eq!(
            (
                FEATURE_SPEC_VERSION,
                STYLING_FEATURE_LEN,
                CENSOR_FEATURE_LEN
            ),
            (1, 43, 45)
        );
    }

    #[test]
    fn primitives_encode_as_documented() {
        assert_eq!(opt_num(Some(0.5)), [0.5, 1.0]);
        assert_eq!(opt_num(None), [0.0, 0.0]);
        assert_eq!(tri_state(Some(true)), [1.0, 0.0]);
        assert_eq!(tri_state(Some(false)), [0.0, 1.0]);
        assert_eq!(tri_state(None), [0.0, 0.0]);
        assert_eq!(one_hot("heading", &BLOCK_KINDS), vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(one_hot("nope", &BLOCK_KINDS), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(heading_bucket(Some(2)), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(heading_bucket(Some(9)), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(heading_bucket(None), [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn surface_stats_capture_shape() {
        let stats = surface_stats("ACME 12");
        assert_eq!(stats[0], 7.0 / 200.0, "length bucket");
        assert!(stats[1] > 0.0, "has digits");
        assert_eq!(stats[2], 1.0, "all-caps letters");
        assert_eq!(stats[5], 0.0, "no punctuation");
        // A lowercase, punctuated value is not all-caps and flags punctuation.
        let email = surface_stats("a@b.com");
        assert_eq!(email[2], 0.0);
        assert_eq!(email[5], 1.0);
    }
}
