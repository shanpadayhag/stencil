//! Build a [`DocumentStyleProfile`] from a document's [`StyledBlock`]s and stamp each
//! block's [`RelativeFeatures`].
//!
//! Everything here is **descriptive and deterministic**: it measures how each block sits
//! relative to the document's norms (style frequency, dominant font/size, per-level indent,
//! per-role signature) and never labels a block "weird". Ties when picking a norm are broken
//! toward the smallest value, so the same input always yields the same profile.

use std::collections::BTreeMap;

use crate::model::{
    BlockKind, DocumentStyleProfile, IlvlIndentNorm, RelativeFeatures, RoleKey, RoleNorm,
    StyleCount, StyleSignature, StyledBlock,
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
/// assert_eq!(relative_features(&blocks[0], &profile).font_matches_doc_dominant, Some(true));
/// assert_eq!(relative_features(&blocks[2], &profile).font_matches_doc_dominant, Some(false));
/// ```
pub fn relative_features(block: &StyledBlock, profile: &DocumentStyleProfile) -> RelativeFeatures {
    // A block whose style chain could not be resolved is "unknown" on every match axis — never a
    // silent match. An unset (post-resolution) font/size is likewise unknown, not a match.
    let known = !block.style_unresolved;
    RelativeFeatures {
        style_doc_freq: style_doc_freq(block, profile),
        font_matches_doc_dominant: (known && block.run.font.is_some())
            .then(|| block.run.font == profile.dominant_font),
        size_matches_doc_dominant: (known && block.run.size_half_pt.is_some())
            .then(|| block.run.size_half_pt == profile.dominant_size_half_pt),
        matches_role_peers: known.then(|| matches_role_peers(block, profile)),
        indent_vs_ilvl_norm: indent_vs_ilvl_norm(block, profile),
    }
}

/// Factual, non-judgmental "vs peers" notes for a block: each names a fact and the peer context with
/// counts (e.g. `font Calibri — 3 of 4 other H2 use Arial`), never a verdict. Empty when the block
/// agrees with its role peers on every measured axis. `blocks` is the whole document; `block` must
/// be one of them.
pub fn deviation_notes(
    block: &StyledBlock,
    blocks: &[StyledBlock],
    _profile: &DocumentStyleProfile,
) -> Vec<String> {
    let role = role_key(block);
    let label = role_label(&role);
    let peers: Vec<&StyledBlock> = blocks
        .iter()
        .filter(|other| other.block_index != block.block_index && role_key(other) == role)
        .collect();
    let peer_count = peers.len();

    let mut notes = Vec::new();

    if let (Some(font), Some((common, count))) = (
        &block.run.font,
        majority(peers.iter().filter_map(|peer| peer.run.font.clone())),
    ) && &common != font
    {
        notes.push(format!(
            "font {font} — {count} of {peer_count} other {label} use {common}"
        ));
    }

    if let (Some(size), Some((common, count))) = (
        block.run.size_half_pt,
        majority(peers.iter().filter_map(|peer| peer.run.size_half_pt)),
    ) && common != size
    {
        notes.push(format!(
            "size {}pt — {count} of {peer_count} other {label} use {}pt",
            points(size),
            points(common),
        ));
    }

    if block.is_mixed() {
        let bold = block
            .segments
            .iter()
            .filter(|segment| segment.style.bold == Some(true))
            .count();
        notes.push(format!("{bold} of {} segments bold", block.segments.len()));
    }

    notes
}

/// Factual, non-judgmental "vs neighbors" notes for a block: each names a structural fact about how
/// the block sits relative to its *immediate neighbors* in document order — a different axis from
/// [`deviation_notes`], which compares a block against its role peers. Never a verdict; empty when
/// nothing positional stands out. `blocks` is the document (or the in-scope slice) in order; `block`
/// must be one of them. Neighbors are the blocks adjacent to `block` within `blocks` (located by
/// `block_index`, so a `--pages` subset still works — adjacency is then within that subset).
///
/// Three high-precision anomalies fire (design v9); each is a case where the neighbors make the
/// block's structure self-contradictory:
/// - a **paragraph** wedged between two list items of the same list,
/// - a **heading** wedged between two list items of the same list,
/// - a list item whose nesting **level jumps by more than one** from the previous same-list item.
///
/// Deliberately silent (too often legitimate): a lone list item among paragraphs, and a list ending
/// where a different list begins.
///
/// ```
/// use stencil::model::{BlockKind, Numbering, ParaStyle, StyledBlock};
/// use stencil::style::profile::positional_notes;
///
/// let list = |index, num_id| StyledBlock {
///     block_index: index,
///     block_kind: BlockKind::ListItem,
///     para: ParaStyle {
///         numbering: Numbering { num_id: Some(num_id), ilvl: Some(0) },
///         ..ParaStyle::default()
///     },
///     ..StyledBlock::default()
/// };
/// let orphan = StyledBlock { block_index: 1, ..StyledBlock::default() }; // a plain paragraph
/// let blocks = [list(0, 3), orphan, list(2, 3)];
/// assert_eq!(
///     positional_notes(&blocks[1], &blocks),
///     vec!["paragraph interrupts list 3 (between two list items)".to_string()],
/// );
/// ```
pub fn positional_notes(block: &StyledBlock, blocks: &[StyledBlock]) -> Vec<String> {
    let Some(pos) = blocks
        .iter()
        .position(|other| other.block_index == block.block_index)
    else {
        return Vec::new();
    };
    let prev = pos.checked_sub(1).and_then(|index| blocks.get(index));
    let next = blocks.get(pos + 1);

    let mut notes = Vec::new();

    // Orphan breaking a list run: a paragraph or heading sitting between two list items of the same
    // list. The block's own kind picks the wording; the two are mutually exclusive.
    if matches!(block.block_kind, BlockKind::Paragraph | BlockKind::Heading)
        && let (Some(prev), Some(next)) = (prev, next)
        && let (Some(prev_num), Some(next_num)) = (list_num_id(prev), list_num_id(next))
        && prev_num == next_num
    {
        let lead = if block.block_kind == BlockKind::Heading {
            "heading inside"
        } else {
            "paragraph interrupts"
        };
        notes.push(format!("{lead} list {prev_num} (between two list items)"));
    }

    // Nesting-level jump: a list item whose level is more than one deeper than the previous item of
    // the same list. Only deeper jumps are flagged (per requirements); a missing ilvl reads as 0.
    if let (Some(level), Some(prev)) = (list_level(block), prev)
        && list_num_id(prev) == list_num_id(block)
        && let Some(prev_level) = list_level(prev)
        && level > prev_level + 1
    {
        notes.push(format!("list level jumps {prev_level}→{level}"));
    }

    notes
}

/// The numbering id of a block when it is a list item; `None` for any other kind (a numbered
/// heading is a [`BlockKind::Heading`], not a list item, so it is excluded here).
fn list_num_id(block: &StyledBlock) -> Option<usize> {
    match block.block_kind {
        BlockKind::ListItem => block.para.numbering.num_id,
        _ => None,
    }
}

/// The nesting level of a list-item block, treating an unset `ilvl` as level 0 (matching the
/// extractor's `ilvl.unwrap_or(0)`); `None` for any non-list block.
fn list_level(block: &StyledBlock) -> Option<usize> {
    match block.block_kind {
        BlockKind::ListItem => Some(block.para.numbering.ilvl.unwrap_or(0)),
        _ => None,
    }
}

/// A short human label for a role, used in deviation notes (`H2`, `body`, `list items`, …).
fn role_label(role: &RoleKey) -> String {
    match role.block_kind {
        crate::model::BlockKind::Heading => role
            .heading_level
            .map(|level| format!("H{level}"))
            .unwrap_or_else(|| "headings".to_string()),
        crate::model::BlockKind::Paragraph => "body".to_string(),
        crate::model::BlockKind::ListItem => "list items".to_string(),
        crate::model::BlockKind::TableCell => "table cells".to_string(),
    }
}

/// Point size as a compact string (`13`, `13.5`) from a half-point value.
fn points(size_half_pt: u64) -> String {
    format!("{}", size_half_pt as f64 / 2.0)
}

/// The most frequent item and its count, ties broken toward the smallest value; `None` if empty.
fn majority<T: Ord>(items: impl IntoIterator<Item = T>) -> Option<(T, usize)> {
    let mut counts: BTreeMap<T, usize> = BTreeMap::new();
    for item in items {
        *counts.entry(item).or_default() += 1;
    }
    let mut best: Option<(T, usize)> = None;
    for (key, count) in counts {
        if best
            .as_ref()
            .is_none_or(|(_, best_count)| count > *best_count)
        {
            best = Some((key, count));
        }
    }
    best
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
    use crate::model::{
        BlockKind, EffectiveRun, IndentTwips, Numbering, ParaStyle, RunStyle, StyleSegment,
    };

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
    fn unset_font_and_size_are_unknown_not_a_match() {
        let sized = |pt| RunStyle {
            font: Some("Arial".into()),
            size_half_pt: Some(pt),
            ..RunStyle::default()
        };
        let blocks = [
            block(0, ParaStyle::default(), sized(24)),
            block(1, ParaStyle::default(), sized(24)),
            block(2, ParaStyle::default(), RunStyle::default()), // unset font/size
            block(3, ParaStyle::default(), font("Times")),
        ];
        let profile = build_profile(&blocks);

        assert_eq!(profile.dominant_font.as_deref(), Some("Arial"));
        assert_eq!(profile.dominant_size_half_pt, Some(24));

        // An unset (post-resolution) font/size is *unknown* — not silently a match (the v7 bug).
        let unset = relative_features(&blocks[2], &profile);
        assert_eq!(unset.font_matches_doc_dominant, None);
        assert_eq!(unset.size_matches_doc_dominant, None);
        // A resolved matching font matches; a resolved different font deviates.
        assert_eq!(
            relative_features(&blocks[0], &profile).font_matches_doc_dominant,
            Some(true)
        );
        assert_eq!(
            relative_features(&blocks[3], &profile).font_matches_doc_dominant,
            Some(false)
        );
    }

    #[test]
    fn unresolved_block_is_unknown_on_every_axis() {
        let mut block = block(0, ParaStyle::default(), font("Arial"));
        block.style_unresolved = true;
        let blocks = [block];
        let profile = build_profile(&blocks);
        let relative = relative_features(&blocks[0], &profile);
        assert_eq!(relative.font_matches_doc_dominant, None);
        assert_eq!(relative.matches_role_peers, None);
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

        assert_eq!(
            relative_features(&blocks[0], &profile).matches_role_peers,
            Some(true)
        );
        assert_eq!(
            relative_features(&blocks[1], &profile).matches_role_peers,
            Some(true)
        );
        assert_eq!(
            relative_features(&blocks[2], &profile).matches_role_peers,
            Some(false)
        );
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
        assert_eq!(
            relative_features(&blocks[0], &profile).matches_role_peers,
            Some(true)
        );
        assert_eq!(
            relative_features(&blocks[2], &profile).matches_role_peers,
            Some(true)
        );
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

    #[test]
    fn deviation_notes_flag_off_font_against_role_peers() {
        let heading = |index, font_name: &str| StyledBlock {
            block_index: index,
            block_kind: BlockKind::Heading,
            heading_level: Some(2),
            ..block(index, ParaStyle::default(), font(font_name))
        };
        let blocks = [
            heading(0, "Arial"),
            heading(1, "Arial"),
            heading(2, "Arial"),
            heading(3, "Calibri"), // the odd one out
        ];
        let profile = build_profile(&blocks);

        let notes = deviation_notes(&blocks[3], &blocks, &profile);
        assert!(
            notes.iter().any(|note| note.contains("font Calibri")
                && note.contains("use Arial")
                && note.contains("H2")),
            "expected an off-font note vs H2 peers, got: {notes:?}"
        );
        // A peer that matches the majority font gets no note.
        assert!(deviation_notes(&blocks[0], &blocks, &profile).is_empty());
    }

    #[test]
    fn deviation_notes_summarize_mixed_segments() {
        let mut mixed = block(0, ParaStyle::default(), RunStyle::default());
        mixed.segments = vec![
            StyleSegment {
                text: "plain ".into(),
                style: EffectiveRun::default(),
            },
            StyleSegment {
                text: "bold".into(),
                style: EffectiveRun {
                    bold: Some(true),
                    ..EffectiveRun::default()
                },
            },
        ];
        let blocks = [mixed];
        let profile = build_profile(&blocks);
        let notes = deviation_notes(&blocks[0], &blocks, &profile);
        assert!(
            notes.iter().any(|note| note.contains("segments bold")),
            "expected a mixed-segments note, got: {notes:?}"
        );
    }

    fn list_item(index: usize, num_id: usize, ilvl: usize) -> StyledBlock {
        StyledBlock {
            block_index: index,
            block_kind: BlockKind::ListItem,
            para: ParaStyle {
                numbering: Numbering {
                    num_id: Some(num_id),
                    ilvl: Some(ilvl),
                },
                ..ParaStyle::default()
            },
            ..StyledBlock::default()
        }
    }

    fn heading_block(index: usize) -> StyledBlock {
        StyledBlock {
            block_index: index,
            block_kind: BlockKind::Heading,
            heading_level: Some(1),
            ..StyledBlock::default()
        }
    }

    fn para_block(index: usize) -> StyledBlock {
        StyledBlock {
            block_index: index,
            block_kind: BlockKind::Paragraph,
            ..StyledBlock::default()
        }
    }

    #[test]
    fn positional_notes_flags_paragraph_orphan_in_list() {
        let blocks = [list_item(0, 2, 0), para_block(1), list_item(2, 2, 0)];
        assert_eq!(
            positional_notes(&blocks[1], &blocks),
            vec!["paragraph interrupts list 2 (between two list items)".to_string()]
        );
    }

    #[test]
    fn positional_notes_flags_heading_inside_list() {
        let blocks = [list_item(0, 2, 0), heading_block(1), list_item(2, 2, 0)];
        assert_eq!(
            positional_notes(&blocks[1], &blocks),
            vec!["heading inside list 2 (between two list items)".to_string()]
        );
    }

    #[test]
    fn positional_notes_flags_nesting_level_jump() {
        // 0 → 2 skips a level; the second item is flagged, the first (no prior item) is not.
        let blocks = [list_item(0, 2, 0), list_item(1, 2, 2)];
        assert_eq!(
            positional_notes(&blocks[1], &blocks),
            vec!["list level jumps 0→2".to_string()]
        );
        assert!(positional_notes(&blocks[0], &blocks).is_empty());
    }

    #[test]
    fn positional_notes_silent_for_single_level_step() {
        let blocks = [list_item(0, 2, 0), list_item(1, 2, 1)];
        assert!(positional_notes(&blocks[1], &blocks).is_empty());
    }

    #[test]
    fn positional_notes_silent_across_different_lists() {
        // A paragraph between two *different* lists is a normal section break, not an orphan.
        let blocks = [list_item(0, 2, 0), para_block(1), list_item(2, 5, 0)];
        assert!(positional_notes(&blocks[1], &blocks).is_empty());
    }

    #[test]
    fn positional_notes_silent_for_lone_list_item() {
        // A single list item among paragraphs is held back (one-item lists are often legitimate).
        let blocks = [para_block(0), list_item(1, 2, 0), para_block(2)];
        assert!(positional_notes(&blocks[1], &blocks).is_empty());
    }

    #[test]
    fn positional_notes_silent_at_document_boundaries() {
        // A block with no neighbor on one side cannot interrupt a run.
        let blocks = [para_block(0), list_item(1, 2, 0)];
        assert!(positional_notes(&blocks[0], &blocks).is_empty());
        assert!(positional_notes(&blocks[1], &blocks).is_empty());
    }

    #[test]
    fn positional_notes_locate_neighbors_within_a_subset() {
        // Under `--pages` the slice is a subset whose positions differ from `block_index`;
        // neighbors are still resolved correctly by locating the block within the slice.
        let blocks = [list_item(7, 2, 0), para_block(8), list_item(9, 2, 0)];
        assert_eq!(
            positional_notes(&blocks[1], &blocks),
            vec!["paragraph interrupts list 2 (between two list items)".to_string()]
        );
    }
}
