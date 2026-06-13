//! Decision-record → feature vector, plus the censor labels (verdict + value-type reason).
//!
//! Reads only pre-decision fields of a [`DecisionRecord`]; the label extractors ([`is_confirm`],
//! [`reason`], [`reason_index`]) are separate so the feature path can never see the answer. See the
//! [module contract](super) for the encoding conventions.

use crate::learn::DecisionRecord;

use super::{BLOCK_KINDS, bit, heading_bucket, multi_hot, one_hot, surface_stats};

/// Width of the censor feature vector (categorical 32 + numeric 2 + neighbor-presence 4 + value-
/// surface 7).
pub const CENSOR_FEATURE_LEN: usize = 45;

/// The 13 censor value types, in the one-hot order shared by the verdict features and the reason
/// head's class order.
pub const CENSOR_TYPES: [&str; 13] = [
    "PERSON", "ORG", "IBAN", "CARD", "ACCOUNT", "PHONE", "DATE", "MONEY", "PERCENT", "EMAIL",
    "LOCATION", "ADDRESS", "ENTITY",
];

/// The detection methods, in one-hot order. `regex:<kind>` collapses to `regex` (the `<kind>` is
/// already carried by `detected_type`), keeping this low-cardinality.
const METHODS: [&str; 4] = ["party-list", "regex", "heuristic", "manual"];

/// Languages the corpus uses, in multi-hot order; anything else falls in the `other` bucket.
const LANGS: [&str; 2] = ["en", "fr"];

/// Map a [`DecisionRecord`] to its verdict-head feature vector ([`CENSOR_FEATURE_LEN`] features).
///
/// ```
/// use stencil::learn::DecisionRecord;
/// use stencil::ml::features::censor::{censor_features, CENSOR_FEATURE_LEN};
///
/// let record = DecisionRecord {
///     detected_type: "EMAIL".into(),
///     method: "regex:email".into(),
///     value: "jane@acme.com".into(),
///     ..Default::default()
/// };
/// let features = censor_features(&record);
/// assert_eq!(features.len(), CENSOR_FEATURE_LEN);
/// assert!(features.iter().all(|f| f.is_finite()));
/// ```
pub fn censor_features(record: &DecisionRecord) -> Vec<f64> {
    let mut features = Vec::with_capacity(CENSOR_FEATURE_LEN);

    // categorical (32)
    features.extend(one_hot(&record.detected_type, &CENSOR_TYPES));
    features.push(bit(!CENSOR_TYPES.contains(&record.detected_type.as_str())));
    let method = collapse_method(&record.method);
    features.extend(one_hot(method, &METHODS));
    features.push(bit(!METHODS.contains(&method)));
    features.extend(multi_hot(&record.block_kinds, &BLOCK_KINDS));
    features.push(bit(record.scope == "occurrence"));
    features.push(bit(record.heading_level.is_some()));
    features.extend(heading_bucket(record.heading_level));
    features.extend(multi_hot(&record.langs, &LANGS));
    features.push(bit(record
        .langs
        .iter()
        .any(|lang| !LANGS.contains(&lang.as_str()))));

    // numeric (2)
    features.push((record.occurrences.min(50) as f64) / 50.0);
    features.push((record.block_kinds.len().min(4) as f64) / 4.0);

    // neighbor presence (4)
    let neighbors = &record.neighbors;
    features.push(bit(neighbors.above.is_some()));
    features.push(bit(neighbors.below.is_some()));
    features.push(bit(neighbors.col_header.is_some()));
    features.push(bit(neighbors.row_label.is_some()));

    // value surface (7) — `value` reduced to stats; +has_at for email / false-positive shape
    features.extend(surface_stats(&record.value));
    features.push(bit(record.value.contains('@')));

    features
}

/// Whether this value is the positive (`confirm`) class — the verdict-head label.
pub fn is_confirm(record: &DecisionRecord) -> bool {
    record.verdict == "confirm"
}

/// The confirmed value-type reason, when the value is `confirm`; `None` otherwise (the reason-head
/// label). A `reject` carries no type.
pub fn reason(record: &DecisionRecord) -> Option<&str> {
    record.final_type.as_deref()
}

/// The reason class index into [`CENSOR_TYPES`], when present and recognized.
pub fn reason_index(record: &DecisionRecord) -> Option<usize> {
    let final_type = reason(record)?;
    CENSOR_TYPES.iter().position(|known| *known == final_type)
}

/// Collapse `regex:<kind>` to the low-cardinality `regex`; pass other methods through unchanged.
fn collapse_method(method: &str) -> &str {
    if method.starts_with("regex:") {
        "regex"
    } else {
        method
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::DecisionRecord;
    use crate::model::CensorNeighbors;

    fn vectors_equal(left: &[f64], right: &[f64]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    #[test]
    fn golden_vector_matches_by_hand_encoding() {
        let record = DecisionRecord {
            detected_type: "EMAIL".into(),
            method: "regex:email".into(),
            block_kinds: vec!["paragraph".into()],
            langs: vec!["en".into()],
            occurrences: 2,
            value: "jane@acme.com".into(),
            ..DecisionRecord::default()
        };

        let expected = vec![
            // detected_type one-hot (13): EMAIL is index 9
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0, // detected unknown
            0.0,
            1.0,
            0.0,
            0.0, // method = regex
            0.0, // method unknown
            1.0,
            0.0,
            0.0,
            0.0, // block_kinds multi-hot = paragraph
            0.0, // scope_is_occurrence
            0.0, // heading present
            0.0,
            0.0,
            0.0,
            0.0, // heading bucket
            1.0,
            0.0, // langs = en
            0.0, // langs other
            // numeric (2)
            2.0 / 50.0,
            1.0 / 4.0,
            // neighbor presence (4)
            0.0,
            0.0,
            0.0,
            0.0,
            // value surface (7) of "jane@acme.com" (13 chars, lowercase, has @ and .)
            13.0 / 200.0,
            0.0,
            0.0,
            0.0,
            1.0 / 20.0,
            1.0,
            1.0,
        ];

        assert_eq!(expected.len(), CENSOR_FEATURE_LEN);
        assert!(
            vectors_equal(&censor_features(&record), &expected),
            "censor encoder drifted from the golden vector"
        );
    }

    #[test]
    fn features_ignore_label_and_post_decision_fields() {
        let base = DecisionRecord {
            detected_type: "ORG".into(),
            method: "party-list".into(),
            block_kinds: vec!["heading".into(), "paragraph".into()],
            heading_level: Some(2),
            langs: vec!["en".into()],
            occurrences: 5,
            value: "Acme Corp".into(),
            verdict: "confirm".into(),
            final_type: Some("ORG".into()),
            span_edited: false,
            context_edited: false,
            user_added: false,
            ..DecisionRecord::default()
        };
        let mut flipped = base.clone();
        flipped.verdict = "reject".into();
        flipped.final_type = Some("PERSON".into());
        flipped.span_edited = true;
        flipped.context_edited = true;
        flipped.user_added = true;

        assert!(
            vectors_equal(&censor_features(&base), &censor_features(&flipped)),
            "features must not depend on verdict/final_type/edit-provenance (leakage)"
        );
    }

    #[test]
    fn unknown_categoricals_set_their_unknown_bit() {
        let record = DecisionRecord {
            detected_type: "WIDGET".into(),
            method: "telepathy".into(),
            ..DecisionRecord::default()
        };
        let features = censor_features(&record);
        assert_eq!(
            features[..13].iter().sum::<f64>(),
            0.0,
            "no known type matched"
        );
        assert_eq!(features[13], 1.0, "detected_type unknown bit set");
        // method one-hot is indices 14..18, its unknown bit is 18.
        assert_eq!(features[14..18].iter().sum::<f64>(), 0.0, "no known method");
        assert_eq!(features[18], 1.0, "method unknown bit set");
    }

    #[test]
    fn default_record_maps_to_finite_full_width_vector() {
        let features = censor_features(&DecisionRecord::default());
        assert_eq!(features.len(), CENSOR_FEATURE_LEN);
        assert!(features.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn neighbor_presence_and_scope_encode() {
        let record = DecisionRecord {
            scope: "occurrence".into(),
            neighbors: CensorNeighbors {
                above: Some("Mailing Address".into()),
                col_header: Some("Address".into()),
                ..CensorNeighbors::default()
            },
            ..DecisionRecord::default()
        };
        let features = censor_features(&record);
        // scope_is_occurrence bit (index 18 is method-unknown; scope is after block_kinds multi-hot).
        // Recompute positions: 13+1+4+1 = 19 (block_kinds start), +4 = 23 (scope).
        assert_eq!(features[23], 1.0, "scope = occurrence");
        // neighbor presence is the 4 features before the 7-wide value surface.
        let base = CENSOR_FEATURE_LEN - 7 - 4;
        assert_eq!(features[base], 1.0, "has_above");
        assert_eq!(features[base + 1], 0.0, "has_below");
        assert_eq!(features[base + 2], 1.0, "has_col_header");
        assert_eq!(features[base + 3], 0.0, "has_row_label");
    }

    #[test]
    fn reason_maps_to_type_index() {
        let confirm = DecisionRecord {
            verdict: "confirm".into(),
            final_type: Some("EMAIL".into()),
            ..DecisionRecord::default()
        };
        assert!(is_confirm(&confirm));
        assert_eq!(reason(&confirm), Some("EMAIL"));
        assert_eq!(reason_index(&confirm), Some(9));

        let reject = DecisionRecord {
            verdict: "reject".into(),
            final_type: None,
            ..DecisionRecord::default()
        };
        assert!(!is_confirm(&reject));
        assert_eq!(reason(&reject), None);
        assert_eq!(reason_index(&reject), None);
    }
}
