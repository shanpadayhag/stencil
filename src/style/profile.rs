//! Build a [`DocumentStyleProfile`] from a document's [`StyledBlock`]s and stamp each
//! block's [`RelativeFeatures`].
//!
//! Everything here is **descriptive and deterministic**: it measures how each block sits
//! relative to the document's norms (style frequency, dominant font/size, per-level indent,
//! per-role signature) and never labels a block "weird". Ties when picking a norm are broken
//! toward the smallest value, so the same input always yields the same profile.

use std::collections::BTreeMap;

use crate::model::{
    DocumentStyleProfile, IlvlIndentNorm, RelativeFeatures, RoleKey, RoleNorm, StyleCount,
    StyleSignature, StyledBlock,
};

/// Build the descriptive style profile for a document's blocks.
///
/// ```
/// use stencil::model::{BlockKind, ParaStyle, RunStyle, StyledBlock};
/// use stencil::style::profile::build_profile;
///
/// let block = |i, font: &str| StyledBlock {
///     block_index: i,
///     block_kind: BlockKind::Paragraph,
///     heading_level: None,
///     in_table: false,
///     text: "x".into(),
///     para: ParaStyle::default(),
///     run: RunStyle { font: Some(font.into()), ..RunStyle::default() },
///     ..Default::default()
/// };
/// let profile = build_profile(&[block(0, "Arial"), block(1, "Arial"), block(2, "Times")]);
/// assert_eq!(profile.total_blocks, 3);
/// assert_eq!(profile.dominant_font.as_deref(), Some("Arial"));
/// ```
pub fn build_profile(blocks: &[StyledBlock]) -> DocumentStyleProfile {
    DocumentStyleProfile {
        total_blocks: blocks.len(),
        style_counts: style_counts(blocks),
        dominant_font: mode(blocks.iter().filter_map(|block| block.run.font.clone())),
        dominant_size_half_pt: mode(blocks.iter().filter_map(|block| block.run.size_half_pt)),
        ilvl_indent_norms: ilvl_indent_norms(blocks),
        role_norms: role_norms(blocks),
    }
}

/// Express a single block's styling relative to the document `profile`.
///
/// ```
/// use stencil::model::{BlockKind, ParaStyle, RunStyle, StyledBlock};
/// use stencil::style::profile::{build_profile, relative_features};
///
/// let block = |i, font: &str| StyledBlock {
///     block_index: i,
///     block_kind: BlockKind::Paragraph,
///     heading_level: None,
///     in_table: false,
///     text: "x".into(),
///     para: ParaStyle::default(),
///     run: RunStyle { font: Some(font.into()), ..RunStyle::default() },
///     ..Default::default()
/// };
/// let blocks = [block(0, "Arial"), block(1, "Arial"), block(2, "Times")];
/// let profile = build_profile(&blocks);
/// assert!(relative_features(&blocks[0], &profile).font_matches_doc_dominant);
/// assert!(!relative_features(&blocks[2], &profile).font_matches_doc_dominant);
/// ```
pub fn relative_features(block: &StyledBlock, profile: &DocumentStyleProfile) -> RelativeFeatures {
    RelativeFeatures {
        style_doc_freq: style_doc_freq(block, profile),
        font_matches_doc_dominant: block.run.font.is_none()
            || block.run.font == profile.dominant_font,
        size_matches_doc_dominant: block.run.size_half_pt.is_none()
            || block.run.size_half_pt == profile.dominant_size_half_pt,
        matches_role_peers: matches_role_peers(block, profile),
        indent_vs_ilvl_norm: indent_vs_ilvl_norm(block, profile),
    }
}

/// Fraction of the document's blocks that share this block's paragraph style.
fn style_doc_freq(block: &StyledBlock, profile: &DocumentStyleProfile) -> f64 {
    if profile.total_blocks == 0 {
        return 0.0;
    }
    let count = profile
        .style_counts
        .iter()
        .find(|entry| entry.style_name == block.para.style_name)
        .map_or(0, |entry| entry.count);
    count as f64 / profile.total_blocks as f64
}

/// Whether the block's style signature matches its role's norm (unknown role ⇒ matches).
fn matches_role_peers(block: &StyledBlock, profile: &DocumentStyleProfile) -> bool {
    let role = role_key(block);
    profile
        .role_norms
        .iter()
        .find(|norm| norm.role == role)
        .is_none_or(|norm| norm.signature == signature(block))
}

/// For a list item, its left indent minus the norm for its level; `None` otherwise.
fn indent_vs_ilvl_norm(block: &StyledBlock, profile: &DocumentStyleProfile) -> Option<i32> {
    let ilvl = block.para.numbering.ilvl?;
    let norm = profile
        .ilvl_indent_norms
        .iter()
        .find(|entry| entry.ilvl == ilvl)?;
    Some(block.para.indent_twips.left.unwrap_or(0) - norm.left_norm)
}

/// Count blocks per paragraph style, most frequent first (ties by style id ascending).
fn style_counts(blocks: &[StyledBlock]) -> Vec<StyleCount> {
    let mut counts: BTreeMap<Option<String>, usize> = BTreeMap::new();
    for block in blocks {
        *counts.entry(block.para.style_name.clone()).or_default() += 1;
    }
    let mut out: Vec<StyleCount> = counts
        .into_iter()
        .map(|(style_name, count)| StyleCount { style_name, count })
        .collect();
    out.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.style_name.cmp(&right.style_name))
    });
    out
}

/// The norm left-indent for each list level present in the document.
fn ilvl_indent_norms(blocks: &[StyledBlock]) -> Vec<IlvlIndentNorm> {
    let mut by_level: BTreeMap<usize, Vec<i32>> = BTreeMap::new();
    for block in blocks {
        if let Some(ilvl) = block.para.numbering.ilvl {
            by_level
                .entry(ilvl)
                .or_default()
                .push(block.para.indent_twips.left.unwrap_or(0));
        }
    }
    by_level
        .into_iter()
        .filter_map(|(ilvl, lefts)| mode(lefts).map(|left_norm| IlvlIndentNorm { ilvl, left_norm }))
        .collect()
}

/// The dominant style signature for each role, in first-appearance order.
fn role_norms(blocks: &[StyledBlock]) -> Vec<RoleNorm> {
    let mut roles: Vec<RoleKey> = Vec::new();
    for block in blocks {
        let role = role_key(block);
        if !roles.contains(&role) {
            roles.push(role);
        }
    }
    roles
        .into_iter()
        .map(|role| {
            let signatures: Vec<StyleSignature> = blocks
                .iter()
                .filter(|block| role_key(block) == role)
                .map(signature)
                .collect();
            let peers = signatures.len();
            RoleNorm {
                role,
                signature: mode(signatures).unwrap_or_default(),
                peers,
            }
        })
        .collect()
}

/// The role grouping key for a block.
fn role_key(block: &StyledBlock) -> RoleKey {
    RoleKey {
        block_kind: block.block_kind,
        heading_level: block.heading_level,
    }
}

/// The comparable style signature for a block.
fn signature(block: &StyledBlock) -> StyleSignature {
    StyleSignature {
        style_name: block.para.style_name.clone(),
        font: block.run.font.clone(),
        size_half_pt: block.run.size_half_pt,
        bold: block.run.bold,
        italic: block.run.italic,
        alignment: block.para.alignment.clone(),
    }
}

/// The most frequent item, breaking ties toward the smallest [`Ord`] value; `None` if empty.
fn mode<T: Ord>(items: impl IntoIterator<Item = T>) -> Option<T> {
    let mut counts: BTreeMap<T, usize> = BTreeMap::new();
    for item in items {
        *counts.entry(item).or_default() += 1;
    }
    let mut best: Option<(T, usize)> = None;
    for (key, count) in counts {
        // Ascending key order + replace only on a strictly greater count ⇒ the smallest
        // key among the most frequent wins, deterministically.
        if best
            .as_ref()
            .is_none_or(|(_, best_count)| count > *best_count)
        {
            best = Some((key, count));
        }
    }
    best.map(|(key, _)| key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BlockKind, IndentTwips, Numbering, ParaStyle, RunStyle};

    /// A plain paragraph block carrying only the fields a test cares about.
    fn block(index: usize, para: ParaStyle, run: RunStyle) -> StyledBlock {
        StyledBlock {
            block_index: index,
            block_kind: BlockKind::Paragraph,
            heading_level: None,
            in_table: false,
            text: "x".into(),
            para,
            run,
            ..Default::default()
        }
    }

    fn styled(name: &str) -> ParaStyle {
        ParaStyle {
            style_name: Some(name.into()),
            ..ParaStyle::default()
        }
    }

    fn font(name: &str) -> RunStyle {
        RunStyle {
            font: Some(name.into()),
            ..RunStyle::default()
        }
    }

    #[test]
    fn style_counts_rank_by_frequency() {
        let blocks = [
            block(0, styled("Normal"), RunStyle::default()),
            block(1, styled("Normal"), RunStyle::default()),
            block(2, styled("Quote"), RunStyle::default()),
        ];
        let profile = build_profile(&blocks);

        assert_eq!(profile.total_blocks, 3);
        assert_eq!(
            profile.style_counts[0].style_name.as_deref(),
            Some("Normal")
        );
        assert_eq!(profile.style_counts[0].count, 2);
        assert_eq!(profile.style_counts[1].style_name.as_deref(), Some("Quote"));
    }

    #[test]
    fn style_doc_freq_reflects_share() {
        let blocks = [
            block(0, styled("Normal"), RunStyle::default()),
            block(1, styled("Normal"), RunStyle::default()),
            block(2, styled("Quote"), RunStyle::default()),
        ];
        let profile = build_profile(&blocks);

        let common = relative_features(&blocks[0], &profile);
        let rare = relative_features(&blocks[2], &profile);
        assert!((common.style_doc_freq - 2.0 / 3.0).abs() < 1e-9);
        assert!((rare.style_doc_freq - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn dominant_font_and_size_ignore_inherited() {
        let sized = |pt| RunStyle {
            font: Some("Arial".into()),
            size_half_pt: Some(pt),
            ..RunStyle::default()
        };
        let blocks = [
            block(0, ParaStyle::default(), sized(24)),
            block(1, ParaStyle::default(), sized(24)),
            block(2, ParaStyle::default(), RunStyle::default()), // inherits both
            block(3, ParaStyle::default(), font("Times")),
        ];
        let profile = build_profile(&blocks);

        assert_eq!(profile.dominant_font.as_deref(), Some("Arial"));
        assert_eq!(profile.dominant_size_half_pt, Some(24));

        // Inheriting block counts as matching (no deviation).
        let inherited = relative_features(&blocks[2], &profile);
        assert!(inherited.font_matches_doc_dominant);
        assert!(inherited.size_matches_doc_dominant);
        // Explicit odd-one-out font deviates.
        assert!(!relative_features(&blocks[3], &profile).font_matches_doc_dominant);
    }

    #[test]
    fn odd_run_out_breaks_role_peers_but_uniform_does_not() {
        let bold = RunStyle {
            bold: Some(true),
            ..RunStyle::default()
        };
        let blocks = [
            block(0, ParaStyle::default(), RunStyle::default()),
            block(1, ParaStyle::default(), RunStyle::default()),
            block(2, ParaStyle::default(), bold), // the odd one out
        ];
        let profile = build_profile(&blocks);

        assert!(relative_features(&blocks[0], &profile).matches_role_peers);
        assert!(relative_features(&blocks[1], &profile).matches_role_peers);
        assert!(!relative_features(&blocks[2], &profile).matches_role_peers);
    }

    #[test]
    fn headings_form_their_own_role_group() {
        let heading = |index, run| StyledBlock {
            block_index: index,
            block_kind: BlockKind::Heading,
            heading_level: Some(1),
            ..block(index, ParaStyle::default(), run)
        };
        let bold = RunStyle {
            bold: Some(true),
            ..RunStyle::default()
        };
        let blocks = [
            heading(0, bold.clone()),
            heading(1, bold),
            // A plain paragraph: different role, so it never compares against the headings.
            block(2, ParaStyle::default(), RunStyle::default()),
        ];
        let profile = build_profile(&blocks);

        assert_eq!(profile.role_norms.len(), 2);
        assert!(relative_features(&blocks[0], &profile).matches_role_peers);
        assert!(relative_features(&blocks[2], &profile).matches_role_peers);
    }

    #[test]
    fn indent_deviation_is_relative_to_ilvl_norm() {
        let listed = |index, left| StyledBlock {
            block_index: index,
            para: ParaStyle {
                numbering: Numbering {
                    num_id: Some(1),
                    ilvl: Some(0),
                },
                indent_twips: IndentTwips {
                    left: Some(left),
                    ..IndentTwips::default()
                },
                ..ParaStyle::default()
            },
            ..block(index, ParaStyle::default(), RunStyle::default())
        };
        let blocks = [listed(0, 720), listed(1, 720), listed(2, 1080)];
        let profile = build_profile(&blocks);

        assert_eq!(profile.ilvl_indent_norms[0].ilvl, 0);
        assert_eq!(profile.ilvl_indent_norms[0].left_norm, 720);
        assert_eq!(
            relative_features(&blocks[1], &profile).indent_vs_ilvl_norm,
            Some(0)
        );
        assert_eq!(
            relative_features(&blocks[2], &profile).indent_vs_ilvl_norm,
            Some(360)
        );
    }

    #[test]
    fn non_list_blocks_have_no_indent_deviation() {
        let blocks = [block(0, ParaStyle::default(), RunStyle::default())];
        let profile = build_profile(&blocks);
        assert_eq!(
            relative_features(&blocks[0], &profile).indent_vs_ilvl_norm,
            None
        );
    }

    #[test]
    fn mode_breaks_ties_toward_smallest() {
        // 10 and 20 each appear once; the smaller wins deterministically.
        assert_eq!(mode([20, 10]), Some(10));
        assert_eq!(mode([20, 10, 10]), Some(10));
        assert_eq!(mode::<i32>([]), None);
    }
}
