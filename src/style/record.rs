//! Assemble [`StylingRecord`]s from the extracted blocks, the document profile, and the
//! reviewer's verdicts, then persist them: one JSONL row per reviewed block plus a per-document
//! profile sidecar.
//!
//! This is the seam between the in-memory styling model ([`crate::model`]) and the on-disk
//! training schema ([`crate::learn`]): it translates [`StyledBlock`] + [`RelativeFeatures`] +
//! [`StyleVerdict`] into a flat [`StylingRecord`]. The block `text` and neighbor context are
//! stored as-is (real text): the styling model trains locally, so it keeps the faithful feature.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::learn::{
    self, Indent, NeighborContext, Numbering, ParaStyle, RelativeStyle, RunStyle, StylingRecord,
};
use crate::model::{BlockKind, DocumentStyleProfile, RelativeFeatures, StyledBlock};
use crate::style::review::{StyleDecision, StyleVerdict};

/// Build the persisted [`StylingRecord`] for one reviewed block.
///
/// `relative` is the block's [`RelativeFeatures`] against the document profile; `context` is the
/// neighboring blocks' text and structure (see [`neighbor_context`]); `source` is the document path.
pub fn build_record(
    block: &StyledBlock,
    relative: &RelativeFeatures,
    context: NeighborContext,
    source: &str,
    doc_id: &str,
    verdict: &StyleVerdict,
) -> StylingRecord {
    let (verdict_label, category, note) = match verdict {
        StyleVerdict::Fine => ("fine", None, None),
        StyleVerdict::Weird { category, note } => ("weird", Some(category.clone()), note.clone()),
    };

    StylingRecord {
        schema: learn::styling_schema(),
        source: source.to_string(),
        doc_id: doc_id.to_string(),
        lang: block.lang.clone(),
        lang_confidence: block.lang_confidence,
        block_index: block.block_index,
        block_kind: block.block_kind.as_str().to_string(),
        heading_level: block.heading_level,
        in_table: block.in_table,
        text: block.text.clone(),
        para: para_style(block),
        run: run_style(block),
        segments: block.segments.clone(),
        numbering_format: block.numbering_format.clone(),
        style_unresolved: block.style_unresolved,
        numbering_unresolved: block.numbering_unresolved,
        relative: relative_style(relative),
        context,
        verdict: verdict_label.to_string(),
        category,
        note,
    }
}

/// The text and structure of the blocks immediately before and after `index`, in document order.
///
/// `*_text` is empty and `*_kind`/`*_numbering` are `None` at a document edge (no neighbor on that
/// side). The numbering is captured only when the neighbor is a list item — the raw facts behind
/// [`crate::style::profile::positional_notes`], so a future anomaly definition stays re-derivable.
pub fn neighbor_context(blocks: &[StyledBlock], index: usize) -> NeighborContext {
    let prev = index
        .checked_sub(1)
        .and_then(|position| blocks.get(position));
    let next = blocks.get(index + 1);
    NeighborContext {
        prev_text: neighbor_text(prev),
        next_text: neighbor_text(next),
        prev_kind: prev.map(|block| block.block_kind.as_str().to_string()),
        next_kind: next.map(|block| block.block_kind.as_str().to_string()),
        prev_numbering: prev.and_then(neighbor_numbering),
        next_numbering: next.and_then(neighbor_numbering),
    }
}

/// The trimmed text of a neighbor, or empty when there is no neighbor on that side.
fn neighbor_text(block: Option<&StyledBlock>) -> String {
    block
        .map(|block| block.text.trim().to_string())
        .unwrap_or_default()
}

/// A neighbor's numbering when it is a list item; `None` otherwise. Mirrors the `positional_notes`
/// rule that a numbered heading is a heading, not a list item.
fn neighbor_numbering(block: &StyledBlock) -> Option<Numbering> {
    match block.block_kind {
        BlockKind::ListItem => Some(Numbering {
            num_id: block.para.numbering.num_id,
            ilvl: block.para.numbering.ilvl,
        }),
        _ => None,
    }
}

/// Append the records for `decisions` to `log_path` and write the profile sidecar under
/// `profiles_dir`, returning the sidecar's path.
///
/// Each decision references a block by `block_index`; only blocks present in `blocks` are
/// written. The records carry the same document `source`.
///
/// # Errors
/// Returns an error if a record cannot be appended or the sidecar cannot be written.
pub fn persist(
    log_path: &Path,
    profiles_dir: &Path,
    blocks: &[StyledBlock],
    profile: &DocumentStyleProfile,
    decisions: &[StyleDecision],
    source: &Path,
    doc_id: &str,
) -> Result<PathBuf> {
    let source_label = source.to_string_lossy();
    for decision in decisions {
        let Some(block) = blocks
            .iter()
            .find(|block| block.block_index == decision.block_index)
        else {
            continue;
        };
        let relative = crate::style::profile::relative_features(block, profile);
        let context = neighbor_context(blocks, decision.block_index);
        let record = build_record(
            block,
            &relative,
            context,
            &source_label,
            doc_id,
            &decision.verdict,
        );
        learn::append_styling(log_path, &record)?;
    }
    write_profile_sidecar(profiles_dir, doc_id, profile)
}

/// Write `profile` as a pretty-printed JSON sidecar named by the content-derived `doc_id`,
/// returning its path. Keying by id (not filename) avoids clobbering when same-named documents
/// from different folders are processed.
///
/// # Errors
/// Returns an error if the directory or file cannot be written.
pub fn write_profile_sidecar(
    profiles_dir: &Path,
    doc_id: &str,
    profile: &DocumentStyleProfile,
) -> Result<PathBuf> {
    std::fs::create_dir_all(profiles_dir)
        .with_context(|| format!("failed to create `{}`", profiles_dir.display()))?;
    let path = profiles_dir.join(format!("{doc_id}.json"));
    let json =
        serde_json::to_string_pretty(profile).context("failed to serialize style profile")?;
    std::fs::write(&path, json).with_context(|| format!("failed to write `{}`", path.display()))?;
    Ok(path)
}

/// Map the in-memory paragraph styling onto the persisted [`ParaStyle`].
fn para_style(block: &StyledBlock) -> ParaStyle {
    let para = &block.para;
    let numbering = match (para.numbering.num_id, para.numbering.ilvl) {
        (None, None) => None,
        (num_id, ilvl) => Some(Numbering { num_id, ilvl }),
    };
    ParaStyle {
        style_name: para.style_name.clone(),
        alignment: para.alignment.clone(),
        indent: Indent {
            left: para.indent_twips.left,
            right: para.indent_twips.right,
            hanging: para.indent_twips.hanging,
            first_line: para.indent_twips.first_line,
        },
        numbering,
        spacing: learn::Spacing {
            before: para.spacing.before.map(|v| v as i32),
            after: para.spacing.after.map(|v| v as i32),
            line: para.spacing.line,
        },
    }
}

/// Map the in-memory run styling onto the persisted [`RunStyle`] (Option flags collapse to bool).
fn run_style(block: &StyledBlock) -> RunStyle {
    let run = &block.run;
    RunStyle {
        font: run.font.clone(),
        size_half_pt: run.size_half_pt,
        bold: run.bold.unwrap_or(false),
        italic: run.italic.unwrap_or(false),
        underline: run.underline.clone(),
        color: run.color.clone(),
    }
}

/// Map the in-memory relative features onto the persisted [`RelativeStyle`] (all fields optional).
fn relative_style(relative: &RelativeFeatures) -> RelativeStyle {
    RelativeStyle {
        style_doc_freq: Some(relative.style_doc_freq as f32),
        font_matches_doc_dominant: relative.font_matches_doc_dominant,
        size_matches_doc_dominant: relative.size_matches_doc_dominant,
        matches_role_peers: relative.matches_role_peers,
        indent_vs_ilvl_norm: relative.indent_vs_ilvl_norm.map(|v| v as f32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        BlockKind, EffectiveRun, IndentTwips, Numbering as ModelNumbering,
        ParaStyle as ModelParaStyle, RunStyle as ModelRunStyle, StyleSegment, StyledBlock,
    };
    use crate::style::profile::{build_profile, relative_features};

    fn block(index: usize, text: &str, para: ModelParaStyle, run: ModelRunStyle) -> StyledBlock {
        StyledBlock {
            block_index: index,
            block_kind: BlockKind::Paragraph,
            heading_level: None,
            in_table: false,
            text: text.into(),
            para,
            run,
            ..Default::default()
        }
    }

    #[test]
    fn fine_verdict_has_no_category_or_note() {
        let blocks = [block(
            0,
            "x",
            ModelParaStyle::default(),
            ModelRunStyle::default(),
        )];
        let profile = build_profile(&blocks);
        let relative = relative_features(&blocks[0], &profile);
        let record = build_record(
            &blocks[0],
            &relative,
            NeighborContext::default(),
            "c.docx",
            "doc-id-test",
            &StyleVerdict::Fine,
        );
        assert_eq!(record.verdict, "fine");
        assert_eq!(record.category, None);
        assert_eq!(record.note, None);
        assert_eq!(record.schema, learn::styling_schema());
    }

    #[test]
    fn weird_verdict_carries_category_and_note() {
        let run = ModelRunStyle {
            bold: Some(true),
            size_half_pt: Some(28),
            ..ModelRunStyle::default()
        };
        let para = ModelParaStyle {
            style_name: Some("Heading1".into()),
            numbering: ModelNumbering {
                num_id: Some(2),
                ilvl: Some(1),
            },
            indent_twips: IndentTwips {
                left: Some(720),
                hanging: Some(360),
                ..IndentTwips::default()
            },
            ..ModelParaStyle::default()
        };
        // Two distinct segments → the block is mixed (the record derives `mixed` from this).
        let mut styled = block(3, "Section text", para, run);
        styled.segments = vec![
            StyleSegment {
                text: "Section ".into(),
                style: EffectiveRun::default(),
            },
            StyleSegment {
                text: "text".into(),
                style: EffectiveRun {
                    bold: Some(true),
                    ..EffectiveRun::default()
                },
            },
        ];
        let blocks = [styled];
        let profile = build_profile(&blocks);
        let relative = relative_features(&blocks[0], &profile);
        let record = build_record(
            &blocks[0],
            &relative,
            NeighborContext::default(),
            "c.docx",
            "doc-id-test",
            &StyleVerdict::Weird {
                category: "wrong-style-for-role".into(),
                note: Some("title as paragraph".into()),
            },
        );

        assert_eq!(record.verdict, "weird");
        assert_eq!(record.category.as_deref(), Some("wrong-style-for-role"));
        assert_eq!(record.note.as_deref(), Some("title as paragraph"));
        // Option flags collapsed; numbering and indent carried across.
        assert!(record.run.bold);
        // `mixed` is now derivable: the block's two segments are persisted on the record.
        assert_eq!(record.segments.len(), 2);
        assert_eq!(record.run.size_half_pt, Some(28));
        assert_eq!(
            record.para.numbering,
            Some(Numbering {
                num_id: Some(2),
                ilvl: Some(1)
            })
        );
        assert_eq!(record.para.indent.hanging, Some(360));
    }

    #[test]
    fn numbering_is_none_when_unset() {
        let blocks = [block(
            0,
            "x",
            ModelParaStyle::default(),
            ModelRunStyle::default(),
        )];
        let profile = build_profile(&blocks);
        let relative = relative_features(&blocks[0], &profile);
        let record = build_record(
            &blocks[0],
            &relative,
            NeighborContext::default(),
            "c.docx",
            "doc-id-test",
            &StyleVerdict::Fine,
        );
        assert_eq!(record.para.numbering, None);
    }

    #[test]
    fn neighbor_context_picks_adjacent_blocks() {
        let blocks = [
            block(
                0,
                "  first  ",
                ModelParaStyle::default(),
                ModelRunStyle::default(),
            ),
            block(
                1,
                "second",
                ModelParaStyle::default(),
                ModelRunStyle::default(),
            ),
            block(
                2,
                "third",
                ModelParaStyle::default(),
                ModelRunStyle::default(),
            ),
        ];
        let middle = neighbor_context(&blocks, 1);
        assert_eq!(middle.prev_text, "first");
        assert_eq!(middle.next_text, "third");
        assert_eq!(middle.prev_kind.as_deref(), Some("paragraph"));
        assert_eq!(middle.next_kind.as_deref(), Some("paragraph"));

        let first = neighbor_context(&blocks, 0);
        assert_eq!(first.prev_text, "");
        assert_eq!(first.next_text, "second");
        assert_eq!(first.prev_kind, None, "no neighbor before the first block");

        let last = neighbor_context(&blocks, 2);
        assert_eq!(last.next_text, "");
        assert_eq!(last.next_kind, None, "no neighbor after the last block");
    }

    #[test]
    fn neighbor_context_captures_kind_and_list_numbering() {
        let mut item = block(
            0,
            "first item",
            ModelParaStyle {
                numbering: ModelNumbering {
                    num_id: Some(4),
                    ilvl: Some(1),
                },
                ..ModelParaStyle::default()
            },
            ModelRunStyle::default(),
        );
        item.block_kind = BlockKind::ListItem;
        let para = block(
            1,
            "body",
            ModelParaStyle::default(),
            ModelRunStyle::default(),
        );
        let blocks = [item, para];

        // The list-item neighbor contributes its kind *and* numbering.
        let context = neighbor_context(&blocks, 1);
        assert_eq!(context.prev_kind.as_deref(), Some("list_item"));
        assert_eq!(
            context.prev_numbering,
            Some(Numbering {
                num_id: Some(4),
                ilvl: Some(1)
            })
        );
        assert_eq!(context.next_kind, None);

        // A non-list neighbor carries its kind but no numbering.
        let context0 = neighbor_context(&blocks, 0);
        assert_eq!(context0.next_kind.as_deref(), Some("paragraph"));
        assert_eq!(context0.next_numbering, None);
    }

    #[test]
    fn persist_writes_log_and_sidecar_round_trip() {
        let dir = std::env::temp_dir().join(format!("stencil_t30_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let log_path = dir.join("styling.jsonl");
        let profiles_dir = dir.join("profiles");

        let blocks = [
            block(
                0,
                "alpha",
                ModelParaStyle::default(),
                ModelRunStyle::default(),
            ),
            block(
                1,
                "beta",
                ModelParaStyle::default(),
                ModelRunStyle::default(),
            ),
        ];
        let profile = build_profile(&blocks);
        let decisions = vec![
            StyleDecision {
                block_index: 0,
                verdict: StyleVerdict::Fine,
            },
            StyleDecision {
                block_index: 1,
                verdict: StyleVerdict::Weird {
                    category: "other".into(),
                    note: None,
                },
            },
        ];

        let sidecar = persist(
            &log_path,
            &profiles_dir,
            &blocks,
            &profile,
            &decisions,
            Path::new("c.docx"),
            "deadbeefcafe0001",
        )
        .expect("persist");
        assert_eq!(
            sidecar.file_name().unwrap().to_string_lossy(),
            "deadbeefcafe0001.json",
            "sidecar is keyed by the content id, not the filename"
        );

        let log = std::fs::read_to_string(&log_path).expect("read log");
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 2, "one record per decision");
        let first: StylingRecord = serde_json::from_str(lines[0]).expect("parse record");
        assert_eq!(first.verdict, "fine");
        assert!(lines[1].contains("\"verdict\":\"weird\""));

        let sidecar_json = std::fs::read_to_string(&sidecar).expect("read sidecar");
        let back: DocumentStyleProfile =
            serde_json::from_str(&sidecar_json).expect("parse profile");
        assert_eq!(back, profile);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
