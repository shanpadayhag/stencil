//! Interactive censor review: walk each distinct detected value one at a time and decide it
//! with a single keypress — `c` confirm (keep censored), `t` re-type (confirm but pick the
//! correct type), `x` reject (false positive), `b` back, `q`/`esc` quit & save.
//!
//! Unlike the snippet censoring (which censors everything), this is the recall-first stage's
//! human filter: only confirmed values are censored in the output, and every explicit decision
//! is a labeled training example. The terminal I/O lives here; the decision rules
//! ([`key_action`], [`retype_label`]) are pure functions, unit-tested without a TTY.

use std::io::{IsTerminal, Write};

use anyhow::{Result, bail};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, read};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

use super::{CensorDecision, ReviewItem, Verdict};

/// The re-type menu: a key per final type the reviewer can assign. `ID` is coarse (the precise
/// subtype stays in the record's `method`); `other` is the long-tail escape hatch.
const RETYPE_MENU: &[(char, &str)] = &[
    ('a', "PERSON"),
    ('b', "ORG"),
    ('c', "LOCATION"),
    ('d', "ADDRESS"),
    ('e', "MONEY"),
    ('f', "PERCENT"),
    ('g', "DATE"),
    ('h', "EMAIL"),
    ('i', "PHONE"),
    ('j', "ID"),
    ('k', "ENTITY"),
    ('l', "other"),
];

/// What a keypress means during the review loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Confirm: keep censored with the detector's type.
    Confirm,
    /// Re-type: confirm but choose the correct type first.
    Retype,
    /// Reject: a false positive, leave it in the clear.
    Reject,
    /// Step back to re-decide the previous value.
    Back,
    /// Stop reviewing and keep the rest censored by default.
    Quit,
    /// Cancel the whole run (Ctrl-C).
    Abort,
    /// An unrecognized key — keep waiting.
    Ignore,
}

/// Map a key event to its [`Action`]. Pure, so it is unit-testable.
fn key_action(key: KeyEvent) -> Action {
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        return Action::Abort;
    }
    match key.code {
        KeyCode::Char('c' | 'C') => Action::Confirm,
        KeyCode::Char('t' | 'T') => Action::Retype,
        KeyCode::Char('x' | 'X') => Action::Reject,
        KeyCode::Char('b' | 'B') => Action::Back,
        KeyCode::Char('q' | 'Q') | KeyCode::Esc => Action::Quit,
        _ => Action::Ignore,
    }
}

/// The final-type label for a re-type menu key, if any. Pure, so it is unit-testable.
fn retype_label(ch: char) -> Option<&'static str> {
    RETYPE_MENU
        .iter()
        .find(|(key, _)| *key == ch.to_ascii_lowercase())
        .map(|(_, label)| *label)
}

/// Review every distinct value in `items`, returning one [`CensorDecision`] per item (same
/// order). Quitting early keeps the remaining values censored by default (marked unreviewed, so
/// they are not logged as human labels).
///
/// # Errors
/// Returns an error if stdin is not a terminal, if raw mode cannot be toggled, or on Ctrl-C.
pub fn review(items: &[ReviewItem]) -> Result<Vec<CensorDecision>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    if !std::io::stdin().is_terminal() {
        bail!("censor review needs a terminal (TTY)");
    }

    let mut out = std::io::stdout();
    let total = items.len();
    let mut decided: Vec<Option<CensorDecision>> = vec![None; total];

    enable_raw_mode()?;
    let _guard = RawModeGuard;

    write_line(
        &mut out,
        "Review each value — [c] confirm · [t] re-type · [x] reject · [b] back · [q] quit & save",
    )?;

    let mut index = 0;
    while index < total {
        let item = &items[index];
        prompt(&mut out, index + 1, total, item)?;
        match review_one()? {
            Action::Confirm => {
                decided[index] = Some(confirmed(item, item.detected_type.label().to_string()));
                write_line(&mut out, "  → kept censored")?;
                index += 1;
            }
            Action::Retype => match choose_type(&mut out)? {
                Some(label) => {
                    decided[index] = Some(confirmed(item, label.to_string()));
                    write_line(&mut out, &format!("  → kept censored as {label}"))?;
                    index += 1;
                }
                None => write_line(&mut out, "  (re-type cancelled)")?,
            },
            Action::Reject => {
                decided[index] = Some(rejected(item));
                write_line(&mut out, "  → left in the clear")?;
                index += 1;
            }
            Action::Back => {
                if index > 0 {
                    index -= 1;
                    write_line(&mut out, "  ← back")?;
                }
            }
            Action::Quit => {
                write_line(&mut out, "  → stopping; remaining values stay censored")?;
                break;
            }
            // The guard restores cooked mode as this scope unwinds on bail.
            Action::Abort => bail!("censor review aborted"),
            Action::Ignore => unreachable!("review_one only returns a decided action"),
        }
    }

    // Fill any value the user did not reach (quit early): keep it censored, but mark it
    // unreviewed so it is excluded from the decision log and the learned store.
    Ok(items
        .iter()
        .zip(decided)
        .map(|(item, decision)| {
            decision.unwrap_or_else(|| CensorDecision {
                value: item.value.clone(),
                detected_type: item.detected_type,
                verdict: Verdict::Confirm {
                    final_type: item.detected_type.label().to_string(),
                },
                reviewed: false,
            })
        })
        .collect())
}

/// A reviewed confirm decision for `item`, typed with `final_type`.
fn confirmed(item: &ReviewItem, final_type: String) -> CensorDecision {
    CensorDecision {
        value: item.value.clone(),
        detected_type: item.detected_type,
        verdict: Verdict::Confirm { final_type },
        reviewed: true,
    }
}

/// A reviewed reject decision for `item`.
fn rejected(item: &ReviewItem) -> CensorDecision {
    CensorDecision {
        value: item.value.clone(),
        detected_type: item.detected_type,
        verdict: Verdict::Reject,
        reviewed: true,
    }
}

/// Block until the user presses a key that decides the current item; ignore the rest.
fn review_one() -> Result<Action> {
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

/// Show the re-type menu and read one choice; `None` if the user cancels with `q`/`esc`.
fn choose_type(out: &mut impl Write) -> Result<Option<&'static str>> {
    let legend = RETYPE_MENU
        .iter()
        .map(|(key, label)| format!("[{key}] {label}"))
        .collect::<Vec<_>>()
        .join("  ");
    write_line(out, &format!("  re-type: {legend}  ([q] cancel)"))?;
    loop {
        let Event::Key(key) = read()? else { continue };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            bail!("censor review aborted");
        }
        match key.code {
            KeyCode::Char('q' | 'Q') | KeyCode::Esc => return Ok(None),
            KeyCode::Char(ch) => {
                if let Some(label) = retype_label(ch) {
                    return Ok(Some(label));
                }
            }
            _ => {}
        }
    }
}

/// Print the prompt block for one value, including its detected type, occurrence count, and the
/// surrounding sentence so the reviewer can judge whether it is sensitive *here*.
fn prompt(out: &mut impl Write, index: usize, total: usize, item: &ReviewItem) -> Result<()> {
    write_line(out, "")?;
    write_line(
        out,
        &format!(
            "[{index}/{total}] {}  ({}, {}\u{00d7}, via {})",
            item.value,
            item.detected_type.label(),
            item.occurrences,
            item.method
        ),
    )?;
    if !item.shown_context.is_empty() {
        write_line(out, &format!("   context: {}", item.shown_context))?;
    }
    Ok(())
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

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn keys_map_to_actions() {
        assert_eq!(key_action(key(KeyCode::Char('c'))), Action::Confirm);
        assert_eq!(key_action(key(KeyCode::Char('t'))), Action::Retype);
        assert_eq!(key_action(key(KeyCode::Char('x'))), Action::Reject);
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
        assert_eq!(key_action(key(KeyCode::Backspace)), Action::Ignore);
    }

    #[test]
    fn retype_menu_maps_keys_to_labels() {
        // Keys are index letters a–l; lookup is case-insensitive.
        assert_eq!(retype_label('b'), Some("ORG"));
        assert_eq!(retype_label('c'), Some("LOCATION"));
        assert_eq!(retype_label('C'), Some("LOCATION"));
        assert_eq!(retype_label('l'), Some("other"));
        assert_eq!(retype_label('z'), None);
    }

    #[test]
    fn empty_items_need_no_terminal() {
        // The empty fast-path returns before the TTY check, so it is safe in tests.
        assert!(review(&[]).expect("no items").is_empty());
    }
}
