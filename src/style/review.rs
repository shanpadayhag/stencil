//! Interactive styling review: walk each block one at a time and judge its formatting with a
//! two-step keypress — `space` = fine (the cheap default), `w` = weird → pick a category
//! (`f`/`r`/`i`/`b`/`o`) and an optional note, `b` back, `q`/`esc` quit & save.
//!
//! There is no detector here: the screen shows the block's text, its *effective* styling — a
//! per-segment breakdown when the block is mixed, else a single style line — and factual
//! "vs peers" notes derived from the [`DocumentStyleProfile`]; the reviewer supplies every
//! verdict. Every block the reviewer reaches is recorded (a `fine` verdict is the negative
//! class). The terminal I/O lives here; the decision rules ([`key_action`], [`category_for_key`])
//! are pure functions, unit-tested without a TTY.

use std::io::{IsTerminal, Write};

use anyhow::{Result, bail};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, read};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

use crate::model::{DocumentStyleProfile, EffectiveRun, StyledBlock};
use crate::style::profile::deviation_notes;

/// The weird-category menu: a key per category the reviewer can assign.
const CATEGORY_MENU: &[(char, &str)] = &[
    ('f', "fake-number"),
    ('r', "wrong-style-for-role"),
    ('i', "inconsistent-style"),
    ('b', "bad-indent-level"),
    ('o', "other"),
];

/// A block's styling verdict from the review.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StyleVerdict {
    /// The block's styling looks fine (the negative class).
    Fine,
    /// The block's styling looks weird, with a category and an optional note.
    Weird {
        /// The category label (one of [`CATEGORY_MENU`]).
        category: String,
        /// An optional free-text note.
        note: Option<String>,
    },
}

/// One reviewed block: its document-order index and the reviewer's verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyleDecision {
    /// Position of the block in document order.
    pub block_index: usize,
    /// The reviewer's verdict.
    pub verdict: StyleVerdict,
}

/// What a step-one keypress means during the review loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Mark the block's styling fine.
    Fine,
    /// Flag the block as weird (opens the category menu next).
    Weird,
    /// Skip the block — advance without recording any verdict (the unsure case).
    Skip,
    /// Step back to re-decide the previous block.
    Back,
    /// Stop reviewing and save what was decided.
    Quit,
    /// Cancel the whole run (Ctrl-C).
    Abort,
    /// An unrecognized key — keep waiting.
    Ignore,
}

/// Map a step-one key event to its [`Action`]. Pure, so it is unit-testable.
fn key_action(key: KeyEvent) -> Action {
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        return Action::Abort;
    }
    match key.code {
        KeyCode::Char(' ') => Action::Fine,
        KeyCode::Char('w' | 'W') => Action::Weird,
        KeyCode::Tab => Action::Skip,
        KeyCode::Char('b' | 'B') => Action::Back,
        KeyCode::Char('q' | 'Q') | KeyCode::Esc => Action::Quit,
        _ => Action::Ignore,
    }
}

/// The category label for a menu key, if any. Pure, so it is unit-testable.
fn category_for_key(ch: char) -> Option<&'static str> {
    CATEGORY_MENU
        .iter()
        .find(|(key, _)| *key == ch.to_ascii_lowercase())
        .map(|(_, label)| *label)
}

/// Review every block in `blocks`, returning one [`StyleDecision`] per block the reviewer
/// reached (in document order). Quitting early stops the walk; unreached blocks get no record.
///
/// # Errors
/// Returns an error if stdin is not a terminal, if raw mode cannot be toggled, or on Ctrl-C.
pub fn review(
    blocks: &[StyledBlock],
    profile: &DocumentStyleProfile,
) -> Result<Vec<StyleDecision>> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }
    if !std::io::stdin().is_terminal() {
        bail!("styling review needs a terminal (TTY)");
    }

    let mut out = std::io::stdout();
    let total = blocks.len();
    let mut decided: Vec<Option<StyleVerdict>> = vec![None; total];

    enable_raw_mode()?;
    let _guard = RawModeGuard;

    write_line(
        &mut out,
        "Review each block — [space] fine · [w] weird · [tab] skip · [b] back · [q] quit & save",
    )?;

    let mut index = 0;
    while index < total {
        let block = &blocks[index];
        let notes = deviation_notes(block, blocks, profile);
        prompt(&mut out, index + 1, total, block, &notes)?;
        match read_action()? {
            Action::Fine => {
                decided[index] = Some(StyleVerdict::Fine);
                write_line(&mut out, "  → fine")?;
                index += 1;
            }
            Action::Weird => match choose_category(&mut out)? {
                Some(category) => {
                    let note = read_note(&mut out)?;
                    write_line(&mut out, &format!("  → weird: {category}"))?;
                    decided[index] = Some(StyleVerdict::Weird {
                        category: category.to_string(),
                        note,
                    });
                    index += 1;
                }
                None => write_line(&mut out, "  (weird cancelled)")?,
            },
            Action::Skip => {
                // Leave `decided[index]` as `None`; it is filtered out, so nothing is recorded.
                write_line(&mut out, "  → skipped (not recorded)")?;
                index += 1;
            }
            Action::Back => {
                if index > 0 {
                    index -= 1;
                    decided[index] = None;
                    write_line(&mut out, "  ← back")?;
                }
            }
            Action::Quit => {
                write_line(&mut out, "  → stopping; unreviewed blocks are not recorded")?;
                break;
            }
            // The guard restores cooked mode as this scope unwinds on bail.
            Action::Abort => bail!("styling review aborted"),
            Action::Ignore => unreachable!("read_action only returns a decided action"),
        }
    }

    Ok(blocks
        .iter()
        .zip(decided)
        .filter_map(|(block, verdict)| {
            verdict.map(|verdict| StyleDecision {
                block_index: block.block_index,
                verdict,
            })
        })
        .collect())
}

/// Block until the user presses a key that decides the current block; ignore the rest.
fn read_action() -> Result<Action> {
    loop {
        let Event::Key(key) = read()? else { continue };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        match key_action(key) {
            Action::Ignore => continue,
            decided => return Ok(decided),
        }
    }
}

/// Show the category menu and read one choice; `None` if the user cancels with `q`/`esc`.
fn choose_category(out: &mut impl Write) -> Result<Option<&'static str>> {
    let legend = CATEGORY_MENU
        .iter()
        .map(|(key, label)| format!("[{key}] {label}"))
        .collect::<Vec<_>>()
        .join("  ");
    write_line(out, &format!("  weird: {legend}  ([q] cancel)"))?;
    loop {
        let Event::Key(key) = read()? else { continue };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            bail!("styling review aborted");
        }
        match key.code {
            KeyCode::Char('q' | 'Q') | KeyCode::Esc => return Ok(None),
            KeyCode::Char(ch) => {
                if let Some(label) = category_for_key(ch) {
                    return Ok(Some(label));
                }
            }
            _ => {}
        }
    }
}

/// Read an optional one-line note in raw mode: characters until Enter; Enter on an empty line
/// (or Esc) means "no note". Backspace edits.
fn read_note(out: &mut impl Write) -> Result<Option<String>> {
    write!(out, "  note (Enter to skip): ")?;
    out.flush()?;
    let mut note = String::new();
    loop {
        let Event::Key(key) = read()? else { continue };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            bail!("styling review aborted");
        }
        match key.code {
            KeyCode::Enter => break,
            KeyCode::Esc => {
                note.clear();
                break;
            }
            KeyCode::Backspace => {
                note.pop();
                write!(out, "\u{8} \u{8}")?;
                out.flush()?;
            }
            KeyCode::Char(ch) => {
                note.push(ch);
                write!(out, "{ch}")?;
                out.flush()?;
            }
            _ => {}
        }
    }
    write!(out, "\r\n")?;
    out.flush()?;
    let note = note.trim();
    Ok((!note.is_empty()).then(|| note.to_string()))
}

/// Print the prompt for one block: its position, kind, a text preview, its effective styling (a
/// per-segment breakdown when mixed, else a single style line), and the factual "vs peers" notes.
fn prompt(
    out: &mut impl Write,
    index: usize,
    total: usize,
    block: &StyledBlock,
    notes: &[String],
) -> Result<()> {
    write_line(out, "")?;
    write_line(
        out,
        &format!(
            "[{index}/{total}] {}{}",
            block.block_kind.as_str(),
            block
                .heading_level
                .map(|level| format!(" L{level}"))
                .unwrap_or_default(),
        ),
    )?;
    write_line(out, &format!("   text: {}", preview(&block.text)))?;
    for line in style_lines(block) {
        write_line(out, &line)?;
    }
    for note in notes {
        write_line(out, &format!("   vs peers: {note}"))?;
    }
    Ok(())
}

/// A single-line, length-capped preview of text.
fn preview(text: &str) -> String {
    const MAX: usize = 100;
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() > MAX {
        let kept: String = one_line.chars().take(MAX).collect();
        format!("{kept}\u{2026}")
    } else {
        one_line
    }
}

/// The styling display lines for a block: a `segments:` bullet list when the block has ≥2 distinct
/// segments, otherwise a single `style:` line. An unresolved block reads "unknown".
fn style_lines(block: &StyledBlock) -> Vec<String> {
    if block.style_unresolved {
        return vec!["   style: unknown (could not resolve the style)".to_string()];
    }
    if block.is_mixed() {
        let mut lines = vec!["   segments:".to_string()];
        for segment in &block.segments {
            lines.push(format!(
                "     • \"{}\" — {}",
                preview(&segment.text),
                effective_summary(&segment.style),
            ));
        }
        lines
    } else {
        let summary = block
            .segments
            .first()
            .map(|segment| effective_summary(&segment.style))
            .unwrap_or_else(|| "(inherited)".to_string());
        let line = match block.para.style_name.as_deref() {
            Some(name) => format!("   style: {name} → {summary}"),
            None => format!("   style: {summary}"),
        };
        vec![line]
    }
}

/// A compact, human-readable summary of a resolved run's styling; "(inherited)" when nothing is set.
fn effective_summary(run: &EffectiveRun) -> String {
    let mut parts = Vec::new();
    if let Some(font) = &run.font {
        parts.push(font.clone());
    }
    if let Some(size) = run.size_half_pt {
        parts.push(format!("{}pt", size as f64 / 2.0));
    }
    if run.bold == Some(true) {
        parts.push("bold".to_string());
    }
    if run.italic == Some(true) {
        parts.push("italic".to_string());
    }
    if run.underline.is_some() {
        parts.push("underline".to_string());
    }
    if run.strike == Some(true) {
        parts.push("strike".to_string());
    }
    if run.caps == Some(true) {
        parts.push("caps".to_string());
    }
    if let Some(spacing) = run.char_spacing.filter(|value| *value != 0) {
        parts.push(format!("spacing {spacing:+}"));
    }
    if parts.is_empty() {
        "(inherited)".to_string()
    } else {
        parts.join(", ")
    }
}

/// Write a single line followed by a CR+LF (raw mode does not translate `\n`).
fn write_line(out: &mut impl Write, line: &str) -> Result<()> {
    write!(out, "{line}\r\n")?;
    out.flush()?;
    Ok(())
}

/// Restores cooked terminal mode on drop, even on early return or panic.
struct RawModeGuard;

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BlockKind, ParaStyle, StyleSegment};

    fn segment(text: &str, style: EffectiveRun) -> StyleSegment {
        StyleSegment {
            text: text.into(),
            style,
        }
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn keys_map_to_actions() {
        assert_eq!(key_action(key(KeyCode::Char(' '))), Action::Fine);
        assert_eq!(key_action(key(KeyCode::Char('w'))), Action::Weird);
        assert_eq!(key_action(key(KeyCode::Tab)), Action::Skip);
        assert_eq!(key_action(key(KeyCode::Char('b'))), Action::Back);
        assert_eq!(key_action(key(KeyCode::Char('q'))), Action::Quit);
        assert_eq!(key_action(key(KeyCode::Esc)), Action::Quit);
    }

    #[test]
    fn ctrl_c_aborts_even_on_c() {
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(key_action(ctrl_c), Action::Abort);
    }

    #[test]
    fn unknown_keys_are_ignored() {
        assert_eq!(key_action(key(KeyCode::Char('z'))), Action::Ignore);
        assert_eq!(key_action(key(KeyCode::Enter)), Action::Ignore);
    }

    #[test]
    fn category_menu_maps_keys_to_labels() {
        assert_eq!(category_for_key('f'), Some("fake-number"));
        assert_eq!(category_for_key('r'), Some("wrong-style-for-role"));
        assert_eq!(category_for_key('I'), Some("inconsistent-style"));
        assert_eq!(category_for_key('b'), Some("bad-indent-level"));
        assert_eq!(category_for_key('o'), Some("other"));
        assert_eq!(category_for_key('z'), None);
    }

    #[test]
    fn empty_blocks_need_no_terminal() {
        let profile = DocumentStyleProfile {
            total_blocks: 0,
            style_counts: Vec::new(),
            dominant_font: None,
            dominant_size_half_pt: None,
            ilvl_indent_norms: Vec::new(),
            role_norms: Vec::new(),
        };
        assert!(review(&[], &profile).expect("no blocks").is_empty());
    }

    #[test]
    fn effective_summary_lists_set_properties_else_inherited() {
        assert_eq!(effective_summary(&EffectiveRun::default()), "(inherited)");
        let run = EffectiveRun {
            font: Some("Arial".into()),
            size_half_pt: Some(26),
            bold: Some(true),
            strike: Some(true),
            char_spacing: Some(-3),
            ..EffectiveRun::default()
        };
        assert_eq!(
            effective_summary(&run),
            "Arial, 13pt, bold, strike, spacing -3"
        );
    }

    #[test]
    fn uniform_block_shows_one_style_line_with_its_style_name() {
        let block = StyledBlock {
            block_kind: BlockKind::Heading,
            heading_level: Some(2),
            para: ParaStyle {
                style_name: Some("Heading2".into()),
                ..ParaStyle::default()
            },
            segments: vec![segment(
                "Payment Terms",
                EffectiveRun {
                    font: Some("Arial".into()),
                    size_half_pt: Some(26),
                    bold: Some(true),
                    ..EffectiveRun::default()
                },
            )],
            ..Default::default()
        };
        let lines = style_lines(&block);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "   style: Heading2 → Arial, 13pt, bold");
    }

    #[test]
    fn mixed_block_shows_a_segment_bullet_list() {
        let block = StyledBlock {
            segments: vec![
                segment("Plain ", EffectiveRun::default()),
                segment(
                    "bold",
                    EffectiveRun {
                        bold: Some(true),
                        ..EffectiveRun::default()
                    },
                ),
            ],
            ..Default::default()
        };
        let lines = style_lines(&block);
        assert_eq!(lines[0], "   segments:");
        assert_eq!(lines.len(), 3);
        assert!(lines[1].contains("\"Plain\"") && lines[1].contains("(inherited)"));
        assert!(lines[2].contains("\"bold\"") && lines[2].contains("bold"));
    }

    #[test]
    fn unresolved_block_reads_unknown() {
        let block = StyledBlock {
            style_unresolved: true,
            segments: vec![segment("x", EffectiveRun::default())],
            ..Default::default()
        };
        let lines = style_lines(&block);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("unknown"));
    }
}
