//! Assemble [`StylingRecord`]s from the extracted blocks, the document profile, and the
//! reviewer's verdicts, then persist them: one JSONL row per reviewed block plus a per-document
//! profile sidecar.
//!
//! This is the seam between the in-memory styling model ([`crate::model`]) and the on-disk
//! training schema ([`crate::learn`]): it translates [`StyledBlock`] + [`RelativeFeatures`] +
//! [`StyleVerdict`] into a flat [`StylingRecord`]. The block `text` and neighbor context are
//! stored as-is here; censoring them is the styling stage's responsibility when it is wired up.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::learn::{
    self, Indent, NeighborContext, Numbering, ParaStyle, RelativeStyle, RunStyle, StylingRecord,
};
use crate::model::{DocumentStyleProfile, RelativeFeatures, StyledBlock};
use crate::style::review::{StyleDecision, StyleVerdict};

/// Build the persisted [`StylingRecord`] for one reviewed block.
///
/// `relative` is the block's [`RelativeFeatures`] against the document profile; `context` is the
/// neighboring blocks' text (see [`neighbor_context`]); `source` is the document path.
pub fn build_record(
    block: &StyledBlock,
    relative: &RelativeFeatures,
    context: NeighborContext,
    source: &str,
    verdict: &StyleVerdict,
) -> StylingRecord {
    let (verdict_label, category, note) = match verdict {
        StyleVerdict::Fine => ("fine", None, None),
        StyleVerdict::Weird { category, note } => ("weird", Some(category.clone()), note.clone()),
    };

    StylingRecord {
        schema: learn::styling_schema(),
        source: source.to_string(),
        block_index: block.block_index,
        block_kind: block.block_kind.as_str().to_string(),
        heading_level: block.heading_level,
        in_table: block.in_table,
        text: block.text.clone(),
        para: para_style(block),
        run: run_style(block),
        relative: relative_style(relative),
        context,
        verdict: verdict_label.to_string(),
        category,
        note,
    }
}

/// The censored text of the blocks immediately before and after `index`; empty at the ends.
pub fn neighbor_context(blocks: &[StyledBlock], index: usize) -> NeighborContext {
    let text_at = |position: usize| blocks.get(position).map(|b| b.text.trim().to_string());
    NeighborContext {
        prev_text: index.checked_sub(1).and_then(text_at).unwrap_or_default(),
        next_text: text_at(index + 1).unwrap_or_default(),
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
        let record = build_record(block, &relative, context, &source_label, &decision.verdict);
        learn::append_styling(log_path, &record)?;
    }
    write_profile_sidecar(profiles_dir, source, profile)
}

/// Write `profile` as a pretty-printed JSON sidecar named after `source`, returning its path.
///
/// # Errors
/// Returns an error if the directory or file cannot be written.
pub fn write_profile_sidecar(
    profiles_dir: &Path,
    source: &Path,
    profile: &DocumentStyleProfile,
) -> Result<PathBuf> {
    std::fs::create_dir_all(profiles_dir)
        .with_context(|| format!("failed to create `{}`", profiles_dir.display()))?;
    let path = profiles_dir.join(format!("{}.json", source_key(source)));
    let json =
        serde_json::to_string_pretty(profile).context("failed to serialize style profile")?;
    std::fs::write(&path, json).with_context(|| format!("failed to write `{}`", path.display()))?;
    Ok(path)
}

/// A filesystem-safe sidecar key for a source path: its file name with every non-alphanumeric
/// character folded to `_`. Empty or odd paths fall back to `profile`.
fn source_key(source: &Path) -> String {
    let stem = source
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    let key: String = stem
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    let trimmed = key.trim_matches('_');
    if trimmed.is_empty() {
        "profile".to_string()
    } else {
        trimmed.to_string()
    }
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
        mixed: run.mixed,
    }
}

/// Map the in-memory relative features onto the persisted [`RelativeStyle`] (all fields optional).
fn relative_style(relative: &RelativeFeatures) -> RelativeStyle {
    RelativeStyle {
        style_doc_freq: Some(relative.style_doc_freq as f32),
        font_matches_doc_dominant: Some(relative.font_matches_doc_dominant),
        size_matches_doc_dominant: Some(relative.size_matches_doc_dominant),
        matches_role_peers: Some(relative.matches_role_peers),
        indent_vs_ilvl_norm: relative.indent_vs_ilvl_norm.map(|v| v as f32),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        BlockKind, IndentTwips, Numbering as ModelNumbering, ParaStyle as ModelParaStyle,
        RunStyle as ModelRunStyle, StyledBlock,
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
            mixed: true,
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
        let blocks = [block(3, "Section text", para, run)];
        let profile = build_profile(&blocks);
        let relative = relative_features(&blocks[0], &profile);
        let record = build_record(
            &blocks[0],
            &relative,
            NeighborContext::default(),
            "c.docx",
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
        assert!(record.run.mixed);
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

        let first = neighbor_context(&blocks, 0);
        assert_eq!(first.prev_text, "");
        assert_eq!(first.next_text, "second");

        let last = neighbor_context(&blocks, 2);
        assert_eq!(last.next_text, "");
    }

    #[test]
    fn source_key_is_filesystem_safe() {
        assert_eq!(
            source_key(Path::new("/a/b/Contract v2.docx")),
            "Contract_v2_docx"
        );
        assert_eq!(source_key(Path::new("plain.txt")), "plain_txt");
        assert_eq!(source_key(Path::new("/")), "profile");
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
        )
        .expect("persist");

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
