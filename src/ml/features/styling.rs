//! Styling-record → feature vector, plus the styling labels (verdict + weird-category reason).
//!
//! Reads only pre-decision fields of a [`StylingRecord`]; the label extractors ([`is_weird`],
//! [`reason`], [`reason_index`]) are deliberately separate so the feature path can never see the
//! answer. See the [module contract](super) for the encoding conventions.

use crate::learn::StylingRecord;

use super::{BLOCK_KINDS, bit, heading_bucket, one_hot, opt_num, surface_stats, tri_state};

/// Width of the styling feature vector (relative 10 + structural 27 + text-surface 6).
pub const STYLING_FEATURE_LEN: usize = 43;

/// The five weird-categories, in the one-vs-rest class order of the styling reason head. Mirrors the
/// review's category menu.
pub const WEIRD_CATEGORIES: [&str; 5] = [
    "fake-number",
    "wrong-style-for-role",
    "inconsistent-style",
    "bad-indent-level",
    "other",
];

/// The alignment one-hot vocabulary (low-cardinality; `None`/other ⇒ all-zeros).
const ALIGNMENTS: [&str; 4] = ["left", "center", "right", "justify"];

/// Map a [`StylingRecord`] to its verdict-head feature vector ([`STYLING_FEATURE_LEN`] features).
///
/// ```
/// use stencil::learn::StylingRecord;
/// use stencil::ml::features::styling::{styling_features, STYLING_FEATURE_LEN};
///
/// let record = StylingRecord {
///     block_kind: "paragraph".into(),
///     text: "Hello world".into(),
///     ..Default::default()
/// };
/// let features = styling_features(&record);
/// assert_eq!(features.len(), STYLING_FEATURE_LEN);
/// assert!(features.iter().all(|f| f.is_finite()));
/// ```
pub fn styling_features(record: &StylingRecord) -> Vec<f64> {
    let mut features = Vec::with_capacity(STYLING_FEATURE_LEN);

    // relative (10)
    let relative = &record.relative;
    features.extend(opt_num(relative.style_doc_freq.map(f64::from)));
    features.extend(tri_state(relative.font_matches_doc_dominant));
    features.extend(tri_state(relative.size_matches_doc_dominant));
    features.extend(tri_state(relative.matches_role_peers));
    features.extend(opt_num(relative.indent_vs_ilvl_norm.map(f64::from)));

    // structural (27)
    features.extend(one_hot(&record.block_kind, &BLOCK_KINDS));
    features.push(bit(record.in_table));
    features.extend(heading_bucket(record.heading_level));
    features.push(bit(record.style_unresolved));
    features.push(bit(record.numbering_unresolved));
    features.push((record.segments.len().min(20) as f64) / 20.0);
    features.push(bit(record.segments.len() >= 2));
    features.push(bit(record.numbering_format.is_some()));
    let has_para_numbering = record
        .para
        .numbering
        .as_ref()
        .is_some_and(|numbering| numbering.num_id.is_some());
    features.push(bit(has_para_numbering));
    let indent_left = record
        .para
        .indent
        .left
        .map_or(0.0, |twips| twips as f64 / 1440.0);
    features.push(indent_left);
    features.push(bit(record.para.indent.hanging.is_some()));
    features.extend(one_hot(
        record.para.alignment.as_deref().unwrap_or(""),
        &ALIGNMENTS,
    ));
    features.push(bit(record.run.bold));
    features.push(bit(record.run.italic));
    features.push(bit(record.run.underline.is_some()));
    features.extend(opt_num(record.run.size_half_pt.map(|size| size as f64)));
    features.push(bit(record.run.font.is_some()));

    // text surface (6) — `text` reduced to stats, never the raw string
    features.extend(surface_stats(&record.text));

    features
}

/// Whether this block is the positive (`weird`) class — the verdict-head label.
pub fn is_weird(record: &StylingRecord) -> bool {
    record.verdict == "weird"
}

/// The weird-category reason, when the block is `weird`; `None` otherwise (the reason-head label).
pub fn reason(record: &StylingRecord) -> Option<&str> {
    record.category.as_deref()
}

/// The reason class index into [`WEIRD_CATEGORIES`], when present and recognized.
pub fn reason_index(record: &StylingRecord) -> Option<usize> {
    let category = reason(record)?;
    WEIRD_CATEGORIES.iter().position(|known| *known == category)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::{Indent, ParaStyle, RelativeStyle, RunStyle, StylingRecord};

    /// Bit-exact vector equality (avoids the `float_cmp` lint).
    fn vectors_equal(left: &[f64], right: &[f64]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    #[test]
    fn golden_vector_matches_by_hand_encoding() {
        let record = StylingRecord {
            block_kind: "heading".into(),
            heading_level: Some(2),
            text: "Hello".into(),
            relative: RelativeStyle {
                style_doc_freq: Some(0.5),
                font_matches_doc_dominant: Some(false),
                matches_role_peers: Some(true),
                ..RelativeStyle::default()
            },
            ..StylingRecord::default()
        };

        // Built in encoder order; arithmetic mirrors the encoder so the comparison is bit-exact.
        let expected = vec![
            // relative (10): style_doc_freq{0.5,known}; font[F]; size[unknown]; role[T]; indent[unknown]
            0.5,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            // structural (27)
            0.0,
            1.0,
            0.0,
            0.0, // block_kind = heading
            0.0, // in_table
            0.0,
            1.0,
            0.0,
            0.0, // heading bucket = h2
            0.0, // style_unresolved
            0.0, // numbering_unresolved
            0.0, // segment_count
            0.0, // is_mixed
            0.0, // has_numbering_format
            0.0, // has_para_numbering
            0.0, // indent_left
            0.0, // has_hanging
            0.0,
            0.0,
            0.0,
            0.0, // alignment
            0.0, // bold
            0.0, // italic
            0.0, // underline
            0.0,
            0.0, // run_size{val,known}
            0.0, // has_font
            // surface (6) of "Hello"
            5.0 / 200.0,
            0.0,
            0.0,
            0.0,
            1.0 / 20.0,
            0.0,
        ];

        assert_eq!(expected.len(), STYLING_FEATURE_LEN);
        assert!(
            vectors_equal(&styling_features(&record), &expected),
            "styling encoder drifted from the golden vector"
        );
    }

    #[test]
    fn features_ignore_label_and_post_decision_fields() {
        let base = StylingRecord {
            block_kind: "table_cell".into(),
            in_table: true,
            text: "$1.000.000".into(),
            relative: RelativeStyle {
                matches_role_peers: Some(false),
                ..RelativeStyle::default()
            },
            verdict: "weird".into(),
            category: Some("fake-number".into()),
            note: Some("looks off".into()),
            ..StylingRecord::default()
        };
        let mut flipped = base.clone();
        flipped.verdict = "fine".into();
        flipped.category = None;
        flipped.note = None;

        assert!(
            vectors_equal(&styling_features(&base), &styling_features(&flipped)),
            "features must not depend on verdict/category/note (leakage)"
        );
    }

    #[test]
    fn default_record_maps_to_finite_unknown_marked_vector() {
        // A record with every optional field absent (an old-schema row after serde defaults) maps
        // cleanly: tri-states unknown (both bits 0), full width, all finite.
        let record = StylingRecord::default();
        let features = styling_features(&record);
        assert_eq!(features.len(), STYLING_FEATURE_LEN);
        assert!(features.iter().all(|f| f.is_finite()));
        // The font tri-state (indices 2..4) is "unknown" ⇒ both bits 0.
        assert_eq!(&features[2..4], &[0.0, 0.0]);
    }

    #[test]
    fn fake_number_block_exposes_digit_shape() {
        let record = StylingRecord {
            text: "$1.000.000".into(),
            ..StylingRecord::default()
        };
        let features = styling_features(&record);
        // surface block is the last six: [len, digit, caps, num, words, punct]
        assert!(
            features[STYLING_FEATURE_LEN - 5] > 0.0,
            "non-zero digit ratio"
        );
    }

    #[test]
    fn reason_maps_to_class_index() {
        let weird = StylingRecord {
            verdict: "weird".into(),
            category: Some("inconsistent-style".into()),
            ..StylingRecord::default()
        };
        assert!(is_weird(&weird));
        assert_eq!(reason(&weird), Some("inconsistent-style"));
        assert_eq!(reason_index(&weird), Some(2));

        let fine = StylingRecord {
            verdict: "fine".into(),
            ..StylingRecord::default()
        };
        assert!(!is_weird(&fine));
        assert_eq!(reason(&fine), None);
        assert_eq!(reason_index(&fine), None);
    }

    #[test]
    fn alignment_and_indent_encode() {
        let record = StylingRecord {
            block_kind: "paragraph".into(),
            para: ParaStyle {
                alignment: Some("center".into()),
                indent: Indent {
                    left: Some(1440),
                    hanging: Some(360),
                    ..Indent::default()
                },
                ..ParaStyle::default()
            },
            run: RunStyle {
                bold: true,
                ..RunStyle::default()
            },
            ..StylingRecord::default()
        };
        let features = styling_features(&record);
        assert_eq!(features.len(), STYLING_FEATURE_LEN);
        assert!(features.iter().all(|f| f.is_finite()));
    }
}
