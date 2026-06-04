//! Interactive ("Tinder-style") restore: walk the placeholders present in the input one
//! at a time and let the user decide each with a single keypress — [space] to skip
//! (leave it redacted), [enter] to restore the real value, [q]/[esc] to stop and save
//! what was chosen so far.
//!
//! The terminal I/O lives here; the decision rule ([`key_action`]) is a pure function so
//! it can be unit-tested without a TTY.

use std::io::{IsTerminal, Write};

use anyhow::{Result, bail};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, read};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

use crate::model::{Mapping, MappingEntry};

/// One reviewed placeholder and the user's verdict. `allow` means restore (the value is
/// not critical to censor); otherwise it was kept redacted.
#[derive(Debug, Clone, Copy)]
pub struct Decision<'a> {
    pub entry: &'a MappingEntry,
    pub allow: bool,
}

/// What a keypress means during the review loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Leave this placeholder redacted.
    Skip,
    /// Restore this placeholder's real value.
    Restore,
    /// Stop reviewing and save what was chosen so far.
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
        KeyCode::Char(' ') => Action::Skip,
        KeyCode::Enter => Action::Restore,
        KeyCode::Char('q' | 'Q') | KeyCode::Esc => Action::Quit,
        _ => Action::Ignore,
    }
}

/// Review every mapping entry that actually appears in `input`, returning every decision
/// (both restore and skip). Entries not present in the input are never shown.
///
/// # Errors
/// Returns an error if stdin is not a terminal, if raw mode cannot be toggled, or if the
/// user aborts with Ctrl-C.
pub fn select<'a>(mapping: &'a Mapping, input: &str) -> Result<Vec<Decision<'a>>> {
    let present: Vec<&MappingEntry> = mapping
        .entries
        .iter()
        .filter(|entry| input.contains(&entry.placeholder))
        .collect();

    if present.is_empty() {
        eprintln!("No restorable placeholders found in the input.");
        return Ok(Vec::new());
    }
    if !std::io::stdin().is_terminal() {
        bail!("interactive restore needs a terminal (TTY); use `--only` for non-interactive runs");
    }

    let mut out = std::io::stdout();
    let total = present.len();
    let mut decisions = Vec::new();

    enable_raw_mode()?;
    let _guard = RawModeGuard;

    write_line(
        &mut out,
        "Review each value — [space] skip · [enter] restore · [q] quit & save",
    )?;
    for (index, entry) in present.iter().enumerate() {
        prompt(&mut out, index + 1, total, entry)?;
        match review_one()? {
            Action::Restore => {
                decisions.push(Decision { entry, allow: true });
                write_line(&mut out, "  → restored")?;
            }
            Action::Skip => {
                decisions.push(Decision {
                    entry,
                    allow: false,
                });
                write_line(&mut out, "  → skipped (left redacted)")?;
            }
            Action::Quit => {
                write_line(&mut out, "  → stopping; saving choices so far")?;
                break;
            }
            // The guard restores cooked mode as this scope unwinds on bail.
            Action::Abort => bail!("interactive restore aborted"),
            Action::Ignore => unreachable!("review_one only returns a decided action"),
        }
    }

    Ok(decisions)
}

/// Block until the user presses a key that decides the current item (skip / restore /
/// quit / abort); unrecognized keys are ignored.
fn review_one() -> Result<Action> {
    loop {
        let Event::Key(key) = read()? else { continue };
        // Windows reports both press and release; act only on the press.
        if key.kind == KeyEventKind::Release {
            continue;
        }
        match key_action(key) {
            Action::Ignore => continue,
            decided => return Ok(decided),
        }
    }
}

/// Print the prompt block for one placeholder.
fn prompt(out: &mut impl Write, index: usize, total: usize, entry: &MappingEntry) -> Result<()> {
    write_line(out, "")?;
    write_line(
        out,
        &format!(
            "[{index}/{total}] {}  ({}, {}\u{00d7})",
            entry.placeholder, entry.value_type, entry.occurrences
        ),
    )?;
    write_line(out, &format!("   value: {}", entry.value))?;
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
    fn space_skips_and_enter_restores() {
        assert_eq!(key_action(key(KeyCode::Char(' '))), Action::Skip);
        assert_eq!(key_action(key(KeyCode::Enter)), Action::Restore);
    }

    #[test]
    fn q_and_esc_quit() {
        assert_eq!(key_action(key(KeyCode::Char('q'))), Action::Quit);
        assert_eq!(key_action(key(KeyCode::Char('Q'))), Action::Quit);
        assert_eq!(key_action(key(KeyCode::Esc)), Action::Quit);
    }

    #[test]
    fn ctrl_c_aborts() {
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(key_action(ctrl_c), Action::Abort);
    }

    #[test]
    fn other_keys_are_ignored() {
        assert_eq!(key_action(key(KeyCode::Char('x'))), Action::Ignore);
        assert_eq!(key_action(key(KeyCode::Backspace)), Action::Ignore);
    }
}
