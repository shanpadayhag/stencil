//! A minimal single-line text editor for the censor review (v7): used to edit a detected
//! span, type a detector-missed value, or correct a context window.
//!
//! The core is a pure state machine — [`edit_action`] maps a key to an [`EditAction`], and
//! [`LineEditor::apply`] folds that action into the buffer — so it is unit-tested without a
//! TTY, the same discipline as `review::key_action`. Only [`read_line`] touches the
//! terminal, and it assumes the surrounding review loop already holds raw mode (it does no
//! mode toggling of its own). Editing is append + backspace only; no mid-string cursor.

use std::io::Write;

use anyhow::{Result, bail};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, read};
use crossterm::terminal::{Clear, ClearType};
use crossterm::{cursor, execute};

/// What a keypress means while editing a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditAction {
    /// Append a character to the buffer.
    Insert(char),
    /// Delete the last character (no-op on an empty buffer).
    Backspace,
    /// Finish editing, keeping the current buffer.
    Commit,
    /// Abandon the edit, discarding the buffer.
    Cancel,
    /// An unrecognized key — keep waiting.
    Ignore,
}

/// How an edit ended: the committed text, or a cancellation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditResult {
    /// The user committed this text.
    Committed(String),
    /// The user cancelled; no text.
    Cancelled,
}

impl EditResult {
    /// `Some(text)` when committed, `None` when cancelled.
    pub fn into_option(self) -> Option<String> {
        match self {
            EditResult::Committed(text) => Some(text),
            EditResult::Cancelled => None,
        }
    }
}

/// Map a key event to its [`EditAction`]. Pure, so it is unit-testable.
///
/// `Enter` commits, `Esc` cancels, `Backspace` deletes; any plain printable character is
/// inserted. Characters chorded with Ctrl/Alt (including Ctrl-C, which the driver treats as an
/// abort) are not inserted.
///
/// ```
/// use crossterm::event::{KeyCode, KeyEvent};
/// use stencil::censor::edit::{edit_action, EditAction};
///
/// assert_eq!(edit_action(KeyEvent::from(KeyCode::Char('x'))), EditAction::Insert('x'));
/// assert_eq!(edit_action(KeyEvent::from(KeyCode::Enter)), EditAction::Commit);
/// assert_eq!(edit_action(KeyEvent::from(KeyCode::Esc)), EditAction::Cancel);
/// ```
pub fn edit_action(key: KeyEvent) -> EditAction {
    match key.code {
        KeyCode::Enter => EditAction::Commit,
        KeyCode::Esc => EditAction::Cancel,
        KeyCode::Backspace => EditAction::Backspace,
        KeyCode::Char(ch)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            EditAction::Insert(ch)
        }
        _ => EditAction::Ignore,
    }
}

/// A growing single-line buffer, optionally pre-filled with text to edit.
#[derive(Debug, Clone, Default)]
pub struct LineEditor {
    buffer: String,
}

impl LineEditor {
    /// Start editing, pre-filled with `initial` (empty for a fresh value).
    pub fn new(initial: &str) -> Self {
        Self {
            buffer: initial.to_string(),
        }
    }

    /// The current text.
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Fold one action into the buffer. Returns `Some(result)` when the edit ends
    /// (commit/cancel), or `None` while still editing.
    ///
    /// ```
    /// use stencil::censor::edit::{EditAction, EditResult, LineEditor};
    ///
    /// let mut editor = LineEditor::new("Jane");
    /// assert_eq!(editor.apply(EditAction::Insert(' ')), None);
    /// assert_eq!(editor.apply(EditAction::Insert('D')), None);
    /// assert_eq!(editor.buffer(), "Jane D");
    /// assert_eq!(
    ///     editor.apply(EditAction::Commit),
    ///     Some(EditResult::Committed("Jane D".to_string())),
    /// );
    /// ```
    pub fn apply(&mut self, action: EditAction) -> Option<EditResult> {
        match action {
            EditAction::Insert(ch) => {
                self.buffer.push(ch);
                None
            }
            EditAction::Backspace => {
                self.buffer.pop();
                None
            }
            EditAction::Commit => Some(EditResult::Committed(self.buffer.clone())),
            EditAction::Cancel => Some(EditResult::Cancelled),
            EditAction::Ignore => None,
        }
    }
}

/// Edit a line interactively, returning `Some(text)` on commit or `None` on cancel.
///
/// Assumes the caller already enabled raw mode (the review loop owns it). Redraws the
/// `prompt` + buffer in place on each keystroke. Ctrl-C aborts with an error.
///
/// # Errors
/// Propagates terminal read/write errors, and errors on Ctrl-C.
pub fn read_line(out: &mut impl Write, prompt: &str, initial: &str) -> Result<Option<String>> {
    let mut editor = LineEditor::new(initial);
    redraw(out, prompt, editor.buffer())?;
    loop {
        let Event::Key(key) = read()? else { continue };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        if is_ctrl_c(key) {
            bail!("edit aborted");
        }
        match editor.apply(edit_action(key)) {
            Some(result) => {
                write!(out, "\r\n")?;
                out.flush()?;
                return Ok(result.into_option());
            }
            None => redraw(out, prompt, editor.buffer())?,
        }
    }
}

/// True for Ctrl-C, the abort chord.
fn is_ctrl_c(key: KeyEvent) -> bool {
    key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)
}

/// Repaint the current line as `prompt` followed by the buffer.
fn redraw(out: &mut impl Write, prompt: &str, buffer: &str) -> Result<()> {
    execute!(out, cursor::MoveToColumn(0), Clear(ClearType::CurrentLine))?;
    write!(out, "{prompt}{buffer}")?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::from(code)
    }

    /// Drive an editor through a sequence of actions; returns the final result (if any).
    fn run(initial: &str, actions: &[EditAction]) -> (LineEditor, Option<EditResult>) {
        let mut editor = LineEditor::new(initial);
        let mut result = None;
        for &action in actions {
            result = editor.apply(action);
            if result.is_some() {
                break;
            }
        }
        (editor, result)
    }

    #[test]
    fn keys_map_to_actions() {
        assert_eq!(
            edit_action(key(KeyCode::Char('Z'))),
            EditAction::Insert('Z')
        );
        assert_eq!(
            edit_action(key(KeyCode::Char(' '))),
            EditAction::Insert(' ')
        );
        assert_eq!(edit_action(key(KeyCode::Backspace)), EditAction::Backspace);
        assert_eq!(edit_action(key(KeyCode::Enter)), EditAction::Commit);
        assert_eq!(edit_action(key(KeyCode::Esc)), EditAction::Cancel);
        assert_eq!(edit_action(key(KeyCode::Tab)), EditAction::Ignore);
    }

    #[test]
    fn ctrl_and_alt_chords_are_not_inserted() {
        let ctrl_a = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL);
        let alt_x = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::ALT);
        assert_eq!(edit_action(ctrl_a), EditAction::Ignore);
        assert_eq!(edit_action(alt_x), EditAction::Ignore);
        // Shift is fine — the char already arrives cased.
        let shift_a = KeyEvent::new(KeyCode::Char('A'), KeyModifiers::SHIFT);
        assert_eq!(edit_action(shift_a), EditAction::Insert('A'));
    }

    #[test]
    fn typing_builds_the_buffer() {
        let (editor, result) = run(
            "",
            &[
                EditAction::Insert('a'),
                EditAction::Insert('b'),
                EditAction::Insert('c'),
            ],
        );
        assert_eq!(editor.buffer(), "abc");
        assert_eq!(result, None, "still editing");
    }

    #[test]
    fn backspace_deletes_and_is_a_noop_past_empty() {
        let (editor, _) = run(
            "hi",
            &[
                EditAction::Backspace,
                EditAction::Backspace,
                EditAction::Backspace, // past empty: stays empty, no panic
            ],
        );
        assert_eq!(editor.buffer(), "");
    }

    #[test]
    fn backspace_respects_utf8_boundaries() {
        // One backspace must remove a whole multi-byte char, not a byte.
        let (editor, _) = run("café", &[EditAction::Backspace]);
        assert_eq!(editor.buffer(), "caf");
    }

    #[test]
    fn commit_returns_the_buffer() {
        let (_, result) = run("Acme", &[EditAction::Insert(' '), EditAction::Commit]);
        assert_eq!(result, Some(EditResult::Committed("Acme ".to_string())));
    }

    #[test]
    fn cancel_returns_none() {
        let (_, result) = run("anything", &[EditAction::Cancel]);
        assert_eq!(result, Some(EditResult::Cancelled));
        assert_eq!(result.unwrap().into_option(), None);
    }

    #[test]
    fn initial_prefills_the_buffer() {
        let editor = LineEditor::new("prefilled");
        assert_eq!(editor.buffer(), "prefilled");
    }
}
