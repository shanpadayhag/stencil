//! A minimal single-line text editor for the censor review (v7): used to edit a detected
//! span, type a detector-missed value, or correct a context window.
//!
//! The core is a pure state machine — [`edit_action`] maps a key to an [`EditAction`], and
//! [`LineEditor::apply`] folds that action into the buffer — so it is unit-tested without a
//! TTY, the same discipline as `review::key_action`. Only [`read_line`] touches the
//! terminal, and it assumes the surrounding review loop already holds raw mode (it does no
//! mode toggling of its own).
//!
//! v10 adds a real **mid-string cursor**: `←`/`→`/`Home`/`End` move it, printable characters
//! insert at it, `Backspace`/`Delete` delete around it, and `Ctrl-←`/`Ctrl-→`/`Ctrl-Backspace`
//! act by word. The cursor is a **character** index, so every motion and deletion respects
//! UTF-8 boundaries.

use std::io::Write;

use anyhow::{Result, bail};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, read};
use crossterm::terminal::{Clear, ClearType};
use crossterm::{cursor, execute};

/// What a keypress means while editing a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditAction {
    /// Insert a character at the cursor.
    Insert(char),
    /// Delete the character before the cursor (no-op at the start).
    Backspace,
    /// Delete the character at the cursor (no-op at the end).
    Delete,
    /// Move the cursor one character left.
    Left,
    /// Move the cursor one character right.
    Right,
    /// Move the cursor to the start of the line.
    Home,
    /// Move the cursor to the end of the line.
    End,
    /// Move the cursor to the start of the previous word.
    WordLeft,
    /// Move the cursor to the end of the next word.
    WordRight,
    /// Delete from the start of the previous word up to the cursor.
    DeleteWord,
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
/// `Enter` commits, `Esc` cancels. `←`/`→`/`Home`/`End` move the cursor; `Backspace`/`Delete`
/// delete around it; `Ctrl-←`/`Ctrl-→` move by word and `Ctrl-Backspace` deletes the previous
/// word. Any plain printable character is inserted. Characters chorded with Ctrl/Alt (including
/// Ctrl-C, which the driver treats as an abort) are not inserted.
///
/// ```
/// use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
/// use stencil::censor::edit::{edit_action, EditAction};
///
/// assert_eq!(edit_action(KeyEvent::from(KeyCode::Char('x'))), EditAction::Insert('x'));
/// assert_eq!(edit_action(KeyEvent::from(KeyCode::Left)), EditAction::Left);
/// assert_eq!(
///     edit_action(KeyEvent::new(KeyCode::Right, KeyModifiers::CONTROL)),
///     EditAction::WordRight,
/// );
/// assert_eq!(edit_action(KeyEvent::from(KeyCode::Enter)), EditAction::Commit);
/// ```
pub fn edit_action(key: KeyEvent) -> EditAction {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Enter => EditAction::Commit,
        KeyCode::Esc => EditAction::Cancel,
        KeyCode::Left if ctrl => EditAction::WordLeft,
        KeyCode::Left => EditAction::Left,
        KeyCode::Right if ctrl => EditAction::WordRight,
        KeyCode::Right => EditAction::Right,
        KeyCode::Home => EditAction::Home,
        KeyCode::End => EditAction::End,
        KeyCode::Delete => EditAction::Delete,
        KeyCode::Backspace if ctrl => EditAction::DeleteWord,
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

/// A single-line buffer with a character-indexed cursor, optionally pre-filled with text to edit.
///
/// The cursor is a character index in `0..=chars`; a fresh editor starts with the cursor at the
/// end of `initial`, so typing appends as before unless the cursor is moved first.
#[derive(Debug, Clone, Default)]
pub struct LineEditor {
    buffer: String,
    /// Character index of the cursor, in `0..=self.char_count()`.
    cursor: usize,
}

impl LineEditor {
    /// Start editing, pre-filled with `initial` (empty for a fresh value), cursor at the end.
    pub fn new(initial: &str) -> Self {
        Self {
            buffer: initial.to_string(),
            cursor: initial.chars().count(),
        }
    }

    /// The current text.
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// The cursor's character index, in `0..=`[`char_count`](Self::char_count).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The buffer length in characters (not bytes).
    fn char_count(&self) -> usize {
        self.buffer.chars().count()
    }

    /// The byte offset of character index `char_idx`, or the buffer length at/after the end.
    fn byte_at(&self, char_idx: usize) -> usize {
        self.buffer
            .char_indices()
            .nth(char_idx)
            .map_or(self.buffer.len(), |(byte, _)| byte)
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
    /// // Move into the middle and fix a character.
    /// assert_eq!(editor.apply(EditAction::Home), None);
    /// assert_eq!(editor.apply(EditAction::Insert('"')), None);
    /// assert_eq!(editor.buffer(), "\"Jane D");
    /// assert_eq!(
    ///     editor.apply(EditAction::Commit),
    ///     Some(EditResult::Committed("\"Jane D".to_string())),
    /// );
    /// ```
    pub fn apply(&mut self, action: EditAction) -> Option<EditResult> {
        match action {
            EditAction::Insert(ch) => {
                let at = self.byte_at(self.cursor);
                self.buffer.insert(at, ch);
                self.cursor += 1;
                None
            }
            EditAction::Backspace => {
                if self.cursor > 0 {
                    let start = self.byte_at(self.cursor - 1);
                    self.buffer.remove(start);
                    self.cursor -= 1;
                }
                None
            }
            EditAction::Delete => {
                if self.cursor < self.char_count() {
                    let at = self.byte_at(self.cursor);
                    self.buffer.remove(at);
                }
                None
            }
            EditAction::Left => {
                self.cursor = self.cursor.saturating_sub(1);
                None
            }
            EditAction::Right => {
                if self.cursor < self.char_count() {
                    self.cursor += 1;
                }
                None
            }
            EditAction::Home => {
                self.cursor = 0;
                None
            }
            EditAction::End => {
                self.cursor = self.char_count();
                None
            }
            EditAction::WordLeft => {
                let chars: Vec<char> = self.buffer.chars().collect();
                self.cursor = prev_word_boundary(&chars, self.cursor);
                None
            }
            EditAction::WordRight => {
                let chars: Vec<char> = self.buffer.chars().collect();
                self.cursor = next_word_boundary(&chars, self.cursor);
                None
            }
            EditAction::DeleteWord => {
                let chars: Vec<char> = self.buffer.chars().collect();
                let target = prev_word_boundary(&chars, self.cursor);
                if target < self.cursor {
                    let start = self.byte_at(target);
                    let end = self.byte_at(self.cursor);
                    self.buffer.drain(start..end);
                    self.cursor = target;
                }
                None
            }
            EditAction::Commit => Some(EditResult::Committed(self.buffer.clone())),
            EditAction::Cancel => Some(EditResult::Cancelled),
            EditAction::Ignore => None,
        }
    }
}

/// The character index at the start of the word before `cursor`: skip whitespace left, then the
/// word's characters. Whitespace and non-whitespace runs are the word boundaries.
fn prev_word_boundary(chars: &[char], cursor: usize) -> usize {
    let mut index = cursor.min(chars.len());
    while index > 0 && chars[index - 1].is_whitespace() {
        index -= 1;
    }
    while index > 0 && !chars[index - 1].is_whitespace() {
        index -= 1;
    }
    index
}

/// The character index just past the word after `cursor`: skip whitespace right, then the word's
/// characters.
fn next_word_boundary(chars: &[char], cursor: usize) -> usize {
    let mut index = cursor.min(chars.len());
    while index < chars.len() && chars[index].is_whitespace() {
        index += 1;
    }
    while index < chars.len() && !chars[index].is_whitespace() {
        index += 1;
    }
    index
}

/// Edit a line interactively, returning `Some(text)` on commit or `None` on cancel.
///
/// Assumes the caller already enabled raw mode (the review loop owns it). Redraws the
/// `prompt` + buffer in place on each keystroke and positions the terminal cursor to match the
/// editor's. Ctrl-C aborts with an error.
///
/// # Errors
/// Propagates terminal read/write errors, and errors on Ctrl-C.
pub fn read_line(out: &mut impl Write, prompt: &str, initial: &str) -> Result<Option<String>> {
    let mut editor = LineEditor::new(initial);
    redraw(out, prompt, editor.buffer(), editor.cursor())?;
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
            None => redraw(out, prompt, editor.buffer(), editor.cursor())?,
        }
    }
}

/// True for Ctrl-C, the abort chord.
fn is_ctrl_c(key: KeyEvent) -> bool {
    key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)
}

/// Repaint the current line as `prompt` followed by the buffer, then place the terminal cursor at
/// the editor's character position.
///
/// Column math counts one terminal column per character (the corpus is Latin EN/FR); wide/CJK
/// glyphs would drift the cursor — a noted v10 limitation, not handled here.
fn redraw(out: &mut impl Write, prompt: &str, buffer: &str, cursor: usize) -> Result<()> {
    execute!(out, cursor::MoveToColumn(0), Clear(ClearType::CurrentLine))?;
    write!(out, "{prompt}{buffer}")?;
    let column = prompt.chars().count() + cursor;
    execute!(
        out,
        cursor::MoveToColumn(u16::try_from(column).unwrap_or(u16::MAX))
    )?;
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
    fn cursor_and_word_keys_map_to_actions() {
        assert_eq!(edit_action(key(KeyCode::Left)), EditAction::Left);
        assert_eq!(edit_action(key(KeyCode::Right)), EditAction::Right);
        assert_eq!(edit_action(key(KeyCode::Home)), EditAction::Home);
        assert_eq!(edit_action(key(KeyCode::End)), EditAction::End);
        assert_eq!(edit_action(key(KeyCode::Delete)), EditAction::Delete);

        let ctrl_left = KeyEvent::new(KeyCode::Left, KeyModifiers::CONTROL);
        let ctrl_right = KeyEvent::new(KeyCode::Right, KeyModifiers::CONTROL);
        let ctrl_back = KeyEvent::new(KeyCode::Backspace, KeyModifiers::CONTROL);
        assert_eq!(edit_action(ctrl_left), EditAction::WordLeft);
        assert_eq!(edit_action(ctrl_right), EditAction::WordRight);
        assert_eq!(edit_action(ctrl_back), EditAction::DeleteWord);
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
    fn new_places_cursor_at_end() {
        let editor = LineEditor::new("Acme");
        assert_eq!(editor.cursor(), 4);
        assert_eq!(LineEditor::new("").cursor(), 0);
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
        assert_eq!(editor.cursor(), 3);
        assert_eq!(result, None, "still editing");
    }

    #[test]
    fn insert_happens_at_the_cursor() {
        // Move to the start, then type: the new text lands before the existing buffer.
        let (editor, _) = run("bc", &[EditAction::Home, EditAction::Insert('a')]);
        assert_eq!(editor.buffer(), "abc");
        assert_eq!(editor.cursor(), 1);
    }

    #[test]
    fn left_right_move_and_clamp() {
        let (editor, _) = run(
            "hi",
            &[
                EditAction::Left,
                EditAction::Left,
                EditAction::Left, // clamps at 0
            ],
        );
        assert_eq!(editor.cursor(), 0);

        let (editor, _) = run(
            "hi",
            &[
                EditAction::Home,
                EditAction::Right,
                EditAction::Right,
                EditAction::Right, // clamps at char_count
            ],
        );
        assert_eq!(editor.cursor(), 2);
    }

    #[test]
    fn home_and_end_jump() {
        let (editor, _) = run("hello", &[EditAction::Home]);
        assert_eq!(editor.cursor(), 0);
        let (editor, _) = run("hello", &[EditAction::Home, EditAction::End]);
        assert_eq!(editor.cursor(), 5);
    }

    #[test]
    fn backspace_deletes_before_cursor_and_is_a_noop_at_start() {
        // Backspace in the middle removes the char to the left of the cursor.
        let (editor, _) = run(
            "abc",
            &[EditAction::Left, EditAction::Backspace], // cursor between b and c → removes b
        );
        assert_eq!(editor.buffer(), "ac");
        assert_eq!(editor.cursor(), 1);

        // At the start it is a no-op.
        let (editor, _) = run("hi", &[EditAction::Home, EditAction::Backspace]);
        assert_eq!(editor.buffer(), "hi");
        assert_eq!(editor.cursor(), 0);
    }

    #[test]
    fn delete_removes_at_cursor_and_is_a_noop_at_end() {
        let (editor, _) = run(
            "abc",
            &[EditAction::Home, EditAction::Delete], // removes 'a'
        );
        assert_eq!(editor.buffer(), "bc");
        assert_eq!(editor.cursor(), 0);

        // At the end it is a no-op.
        let (editor, _) = run("hi", &[EditAction::Delete]);
        assert_eq!(editor.buffer(), "hi");
        assert_eq!(editor.cursor(), 2);
    }

    #[test]
    fn word_left_and_right_jump_by_word() {
        // From the end, WordLeft lands at the start of the last word.
        let (editor, _) = run("foo bar baz", &[EditAction::WordLeft]);
        assert_eq!(editor.cursor(), 8, "start of 'baz'");
        let (editor, _) = run("foo bar baz", &[EditAction::WordLeft, EditAction::WordLeft]);
        assert_eq!(editor.cursor(), 4, "start of 'bar'");

        // From the start, WordRight lands just past the first word.
        let (editor, _) = run("foo bar", &[EditAction::Home, EditAction::WordRight]);
        assert_eq!(editor.cursor(), 3, "just past 'foo'");
        let (editor, _) = run(
            "foo bar",
            &[
                EditAction::Home,
                EditAction::WordRight,
                EditAction::WordRight,
            ],
        );
        assert_eq!(editor.cursor(), 7, "just past 'bar'");
    }

    #[test]
    fn delete_word_removes_the_previous_word() {
        let (editor, _) = run("foo bar", &[EditAction::DeleteWord]);
        assert_eq!(editor.buffer(), "foo ");
        assert_eq!(editor.cursor(), 4);

        // Mid-string: only the word left of the cursor goes.
        let (editor, _) = run(
            "alpha beta gamma",
            &[EditAction::WordLeft, EditAction::DeleteWord], // cursor at 'gamma'; delete 'beta '
        );
        assert_eq!(editor.buffer(), "alpha gamma");
        assert_eq!(editor.cursor(), 6);
    }

    #[test]
    fn backspace_respects_utf8_boundaries() {
        // One backspace must remove a whole multi-byte char, not a byte.
        let (editor, _) = run("café", &[EditAction::Backspace]);
        assert_eq!(editor.buffer(), "caf");
        assert_eq!(editor.cursor(), 3);
    }

    #[test]
    fn editing_around_multibyte_chars_is_char_aware() {
        // Insert before the 'é', then delete it forward — offsets must stay on char boundaries.
        let (editor, _) = run(
            "café",
            &[EditAction::Left, EditAction::Insert('X')], // "cafXé"
        );
        assert_eq!(editor.buffer(), "cafXé");
        assert_eq!(editor.cursor(), 4);

        let (editor, _) = run("café", &[EditAction::Left, EditAction::Delete]);
        assert_eq!(editor.buffer(), "caf");
        assert_eq!(editor.cursor(), 3);
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
        assert_eq!(editor.cursor(), 9);
    }
}
