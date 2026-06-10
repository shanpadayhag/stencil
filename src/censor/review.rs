//! Interactive censor review: walk each distinct detected value one at a time and decide it
//! with a single keypress — `c` confirm (keep censored), `t` re-type (confirm but pick the
//! correct type), `x` reject (false positive), `e` edit the censored span, `n` add a value the
//! detector missed, `w` correct the context window, `b` back, `q`/`esc` quit & save (v7).
//!
//! Unlike the snippet censoring (which censors everything), this is the recall-first stage's
//! human filter: only confirmed values are censored in the output, and every explicit decision
//! is a labeled training example. The terminal I/O lives here; the decision rules
//! ([`key_action`], [`retype_label`]) are pure functions, unit-tested without a TTY. The edit
//! flows reuse the pure line editor in [`super::edit`].

use std::io::{IsTerminal, Write};

use anyhow::{Result, bail};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, read};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};

use super::{CensorDecision, ReviewItem, ValueType, Verdict, edit, locate_value};
use crate::model::Document;

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
    /// Edit the censored span (retarget the value), then keep reviewing it.
    EditSpan,
    /// Add a value the detector missed.
    AddValue,
    /// Correct the recorded context window, then keep reviewing the value.
    EditContext,
    /// Split a value-group into its occurrences and decide each on its own.
    Split,
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
        KeyCode::Char('e' | 'E') => Action::EditSpan,
        KeyCode::Char('n' | 'N') => Action::AddValue,
        KeyCode::Char('w' | 'W') => Action::EditContext,
        KeyCode::Char('s' | 'S') => Action::Split,
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

/// Per-value edit provenance accumulated during review.
#[derive(Debug, Clone, Copy, Default)]
struct EditFlags {
    span_edited: bool,
    context_edited: bool,
}

/// Review every detected value in `items`, returning one [`CensorDecision`] per reviewed value
/// plus any values the reviewer added. Quitting early keeps the remaining detected values
/// censored by default (marked unreviewed, so they are not logged as human labels).
///
/// `document` is needed to validate edited/added values (they must occur in the text) and to
/// recompute their occurrences and context.
///
/// # Errors
/// Returns an error if stdin is not a terminal, if raw mode cannot be toggled, or on Ctrl-C.
pub fn review(document: &Document, items: &[ReviewItem]) -> Result<Vec<CensorDecision>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    if !std::io::stdin().is_terminal() {
        bail!("censor review needs a terminal (TTY)");
    }

    let mut out = std::io::stdout();
    let total = items.len();
    // Working copies: span/context edits mutate these in place (length stays `total`).
    let mut work = items.to_vec();
    let mut flags = vec![EditFlags::default(); total];
    let mut decided: Vec<Option<CensorDecision>> = vec![None; total];
    // Items the reviewer split into per-occurrence decisions: skipped in the final group pass.
    let mut split_done = vec![false; total];
    // Decisions with no group slot: reviewer-added values and split-out occurrences.
    let mut extra: Vec<CensorDecision> = Vec::new();

    enable_raw_mode()?;
    let _guard = RawModeGuard;

    write_line(
        &mut out,
        "Review each value — [c] confirm · [t] re-type · [x] reject · [e] edit span · \
         [n] add value · [w] edit context · [s] split · [b] back · [q] quit & save",
    )?;

    let mut index = 0;
    while index < total {
        prompt(&mut out, index + 1, total, &work[index])?;
        match review_one()? {
            Action::Confirm => {
                let label = work[index].detected_type.label().to_string();
                decided[index] = Some(build_decision(
                    &work[index],
                    flags[index],
                    Verdict::Confirm { final_type: label },
                ));
                write_line(&mut out, "  → kept censored")?;
                index += 1;
            }
            Action::Retype => match choose_type(&mut out)? {
                Some(label) => {
                    decided[index] = Some(build_decision(
                        &work[index],
                        flags[index],
                        Verdict::Confirm {
                            final_type: label.to_string(),
                        },
                    ));
                    write_line(&mut out, &format!("  → kept censored as {label}"))?;
                    index += 1;
                }
                None => write_line(&mut out, "  (re-type cancelled)")?,
            },
            Action::Reject => {
                decided[index] = Some(build_decision(&work[index], flags[index], Verdict::Reject));
                write_line(&mut out, "  → left in the clear")?;
                index += 1;
            }
            Action::EditSpan => {
                // cancelled or not found leaves the value unchanged (message already printed).
                if let Some(retargeted) = edit_span(&mut out, document, &work[index])? {
                    write_line(&mut out, &format!("  → span set to “{}”", retargeted.value))?;
                    work[index] = retargeted;
                    flags[index].span_edited = true;
                }
            }
            Action::EditContext => {
                if let Some(context) = edit_context(&mut out, &work[index])? {
                    set_shown_context(&mut work[index], context);
                    flags[index].context_edited = true;
                    write_line(&mut out, "  → context updated")?;
                }
            }
            Action::AddValue => {
                if let Some(decision) = add_value(&mut out, document)? {
                    write_line(
                        &mut out,
                        &format!("  → added “{}” for censoring", decision.value),
                    )?;
                    extra.push(decision);
                }
            }
            Action::Split => {
                if work[index].occurrence_count() < 2 {
                    write_line(&mut out, "  (only one occurrence — nothing to split)")?;
                } else if let Some(occurrence_decisions) = split_review(&mut out, &work[index])? {
                    extra.extend(occurrence_decisions);
                    split_done[index] = true;
                    index += 1;
                }
            }
            Action::Back => {
                // Step back to the previous still-grouped value (skip ones already split out).
                if let Some(prev) = previous_groupable(index, &split_done) {
                    index = prev;
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

    // Group decisions, skipping items split into per-occurrence decisions. Any value the user
    // did not reach (quit early) is kept censored but marked unreviewed, so it is excluded from
    // the decision log and the learned store.
    let mut decisions: Vec<CensorDecision> = Vec::new();
    for (index, (item, decision)) in work.iter().zip(decided).enumerate() {
        if split_done[index] {
            continue;
        }
        decisions.push(decision.unwrap_or_else(|| {
            CensorDecision::from_item(
                item,
                Verdict::Confirm {
                    final_type: item.detected_type.label().to_string(),
                },
                false,
            )
        }));
    }
    decisions.extend(extra);
    Ok(decisions)
}

/// The previous index whose value is still a group (not split out), stepping over split items.
/// `None` when there is no earlier groupable value.
fn previous_groupable(index: usize, split_done: &[bool]) -> Option<usize> {
    (0..index).rev().find(|&candidate| !split_done[candidate])
}

/// Build a self-contained decision from a (possibly edited) working item, stamping edit
/// provenance. A span edit has already rebuilt the item (fresh value/occurrences/context via
/// [`locate_value`]), so the item's own fields are authoritative here.
fn build_decision(item: &ReviewItem, flags: EditFlags, verdict: Verdict) -> CensorDecision {
    let mut decision = CensorDecision::from_item(item, verdict, true);
    decision.span_edited = flags.span_edited;
    decision.context_edited = flags.context_edited;
    decision
}

/// Edit the censored span: prefill the line editor with the current value, validate the result
/// occurs in the document, and return the retargeted item (with fresh occurrences/context).
/// `None` on cancel or when the edited value is not found (a message is printed either way).
fn edit_span(
    out: &mut impl Write,
    document: &Document,
    item: &ReviewItem,
) -> Result<Option<ReviewItem>> {
    let Some(value) = edit::read_line(out, "  edit span: ", &item.value)? else {
        write_line(out, "  (edit cancelled)")?;
        return Ok(None);
    };
    match locate_value(document, &value, item.detected_type, &item.method) {
        Some(retargeted) => Ok(Some(retargeted)),
        None => {
            write_line(
                out,
                "  ⚠ that value is not in the document — span unchanged",
            )?;
            Ok(None)
        }
    }
}

/// Correct the context window: prefill with the current window, return the edited text.
fn edit_context(out: &mut impl Write, item: &ReviewItem) -> Result<Option<String>> {
    edit::read_line(out, "  edit context: ", item.first_shown_context())
}

/// Add a value the detector missed: read it, validate it occurs, pick its type, and return a
/// confirmed, `user_added` decision. `None` on cancel or when the value is not found.
fn add_value(out: &mut impl Write, document: &Document) -> Result<Option<CensorDecision>> {
    let Some(value) = edit::read_line(out, "  add value: ", "")? else {
        write_line(out, "  (add cancelled)")?;
        return Ok(None);
    };
    let Some(item) = locate_value(document, &value, ValueType::Entity, "manual") else {
        write_line(out, "  ⚠ that value is not in the document — nothing added")?;
        return Ok(None);
    };
    let Some(label) = choose_type(out)? else {
        write_line(out, "  (add cancelled)")?;
        return Ok(None);
    };
    let mut decision = CensorDecision::from_item(
        &item,
        Verdict::Confirm {
            final_type: label.to_string(),
        },
        true,
    );
    decision.user_added = true;
    Ok(Some(decision))
}

/// Replace the working item's first-occurrence context window (what the whole-group decision
/// records). A no-op if the item somehow has no occurrences.
fn set_shown_context(item: &mut ReviewItem, context: String) {
    if let Some(first) = item.occurrences.first_mut() {
        first.shown_context = context;
    }
}

/// Review each occurrence of a split value on its own, returning the occurrence-scoped decisions,
/// or `None` if the reviewer cancels the split (the value stays a group). Only `c`/`t`/`x` decide
/// an occurrence here; edit/add/split/back are not offered inside a split.
fn split_review(out: &mut impl Write, item: &ReviewItem) -> Result<Option<Vec<CensorDecision>>> {
    let total = item.occurrences.len();
    write_line(
        out,
        &format!(
            "  → splitting “{}” into {total} occurrence(s) — [c]/[t]/[x] each · [q] cancel",
            item.value
        ),
    )?;
    let mut verdicts = Vec::with_capacity(total);
    for (n, occurrence) in item.occurrences.iter().enumerate() {
        prompt_occurrence(out, n + 1, total, item, occurrence)?;
        let verdict = loop {
            match review_one()? {
                Action::Confirm => {
                    break Verdict::Confirm {
                        final_type: item.detected_type.label().to_string(),
                    };
                }
                Action::Retype => match choose_type(out)? {
                    Some(label) => {
                        break Verdict::Confirm {
                            final_type: label.to_string(),
                        };
                    }
                    None => write_line(out, "  (re-type cancelled)")?,
                },
                Action::Reject => break Verdict::Reject,
                Action::Quit => {
                    write_line(out, "  (split cancelled — value kept as a group)")?;
                    return Ok(None);
                }
                Action::Abort => bail!("censor review aborted"),
                _ => {} // edit/add/split/back are not offered inside a split
            }
        };
        verdicts.push(verdict);
        write_line(out, "  ✓")?;
    }
    Ok(Some(build_split_decisions(item, verdicts)))
}

/// Pair each of `item`'s occurrences with its verdict into an occurrence-scoped decision. Pure,
/// so the split's group-vs-occurrence outcome is unit-testable without a terminal.
fn build_split_decisions(item: &ReviewItem, verdicts: Vec<Verdict>) -> Vec<CensorDecision> {
    item.occurrences
        .iter()
        .cloned()
        .zip(verdicts)
        .map(|(occurrence, verdict)| {
            CensorDecision::from_occurrence(
                &item.value,
                item.detected_type,
                &item.method,
                occurrence,
                verdict,
            )
        })
        .collect()
}

/// Prompt for one occurrence during a split: its position, block kind (+ heading level), and its
/// own context window.
fn prompt_occurrence(
    out: &mut impl Write,
    n: usize,
    total: usize,
    item: &ReviewItem,
    occurrence: &crate::model::Occurrence,
) -> Result<()> {
    write_line(out, "")?;
    let where_at = match occurrence.heading_level {
        Some(level) => format!("{} h{level}", occurrence.block_kind.as_str()),
        None => occurrence.block_kind.as_str().to_string(),
    };
    write_line(
        out,
        &format!("  [{n}/{total}] {} (in {where_at})", item.value),
    )?;
    if !occurrence.shown_context.is_empty() {
        write_line(out, &format!("     context: {}", occurrence.shown_context))?;
    }
    Ok(())
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
            item.occurrence_count(),
            item.method
        ),
    )?;
    let shown_context = item.first_shown_context();
    if !shown_context.is_empty() {
        write_line(out, &format!("   context: {shown_context}"))?;
    }
    if let Some(kinds) = mixed_kind_note(item) {
        // A value straddling block kinds is often context-dependent — nudge to split (advisory).
        write_line(
            out,
            &format!("   \u{26a0} appears in: {kinds} — [s] to split"),
        )?;
    }
    Ok(())
}

/// A comma-joined list of the distinct block kinds a value appears in, but only when it spans
/// more than one (the mixed-context split hint). `None` for a single-kind value. Pure.
fn mixed_kind_note(item: &ReviewItem) -> Option<String> {
    let kinds = item.block_kinds();
    if kinds.len() < 2 {
        return None;
    }
    Some(
        kinds
            .iter()
            .map(|kind| kind.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    )
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
        assert_eq!(key_action(key(KeyCode::Char('e'))), Action::EditSpan);
        assert_eq!(key_action(key(KeyCode::Char('n'))), Action::AddValue);
        assert_eq!(key_action(key(KeyCode::Char('w'))), Action::EditContext);
        assert_eq!(key_action(key(KeyCode::Char('s'))), Action::Split);
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
        let doc = Document {
            source: std::path::PathBuf::from("t.txt"),
            blocks: Vec::new(),
        };
        assert!(review(&doc, &[]).expect("no items").is_empty());
    }

    fn one_occurrence_item(value: &str) -> ReviewItem {
        use crate::model::{BlockKind, Occurrence};
        ReviewItem {
            value: value.into(),
            detected_type: ValueType::Entity,
            method: "regex:test".into(),
            occurrences: vec![Occurrence {
                block_index: 0,
                cell: None,
                start: 0,
                end: value.len(),
                block_kind: BlockKind::Paragraph,
                heading_level: None,
                shown_context: format!("ctx {value}"),
                block_context: format!("blk {value}"),
                ..Default::default()
            }],
        }
    }

    #[test]
    fn build_decision_stamps_edit_flags_and_carries_context() {
        let item = one_occurrence_item("Jane Doe");
        let flags = EditFlags {
            span_edited: true,
            context_edited: false,
        };
        let decision = build_decision(
            &item,
            flags,
            Verdict::Confirm {
                final_type: "PERSON".into(),
            },
        );
        assert_eq!(decision.value, "Jane Doe");
        assert!(decision.span_edited);
        assert!(!decision.context_edited);
        assert!(!decision.user_added);
        assert!(decision.reviewed);
        assert_eq!(decision.shown_context, "ctx Jane Doe");
    }

    #[test]
    fn set_shown_context_replaces_first_window() {
        let mut item = one_occurrence_item("Acme");
        set_shown_context(&mut item, "corrected window".into());
        assert_eq!(item.first_shown_context(), "corrected window");
    }

    fn multi_occurrence_item(value: &str, count: usize) -> ReviewItem {
        use crate::model::{BlockKind, Occurrence};
        let occurrences = (0..count)
            .map(|i| Occurrence {
                block_index: i,
                cell: None,
                start: 0,
                end: value.len(),
                // First in a heading, the rest in paragraphs → a mixed-kind group.
                block_kind: if i == 0 {
                    BlockKind::Heading
                } else {
                    BlockKind::Paragraph
                },
                heading_level: (i == 0).then_some(1),
                shown_context: format!("ctx{i} {value}"),
                block_context: format!("blk{i}"),
                ..Default::default()
            })
            .collect();
        ReviewItem {
            value: value.into(),
            detected_type: ValueType::Percent,
            method: "regex:percent".into(),
            occurrences,
        }
    }

    #[test]
    fn build_split_decisions_makes_one_occurrence_decision_each() {
        let item = multi_occurrence_item("3%", 3);
        let verdicts = vec![
            Verdict::Confirm {
                final_type: "PERCENT".into(),
            },
            Verdict::Confirm {
                final_type: "PERCENT".into(),
            },
            Verdict::Reject,
        ];
        let decisions = build_split_decisions(&item, verdicts);
        assert_eq!(decisions.len(), 3, "one decision per occurrence");
        assert!(
            decisions
                .iter()
                .all(|d| matches!(d.scope, crate::censor::DecisionScope::Occurrence(_))),
            "each is occurrence-scoped",
        );
        assert!(decisions.iter().all(|d| d.occurrences() == 1));
        assert_eq!(decisions[0].value, "3%");
        assert!(
            matches!(decisions[2].verdict, Verdict::Reject),
            "third rejected"
        );
    }

    #[test]
    fn mixed_kind_note_only_when_spanning_kinds() {
        // BlockKind orders Paragraph before Heading, so the set renders in that order.
        assert_eq!(
            mixed_kind_note(&multi_occurrence_item("x", 2)).as_deref(),
            Some("paragraph, heading"),
        );
        assert_eq!(mixed_kind_note(&one_occurrence_item("y")), None);
    }

    #[test]
    fn previous_groupable_steps_over_split_items() {
        let split = [false, true, false];
        assert_eq!(
            previous_groupable(2, &split),
            Some(0),
            "skip the split index 1"
        );
        assert_eq!(previous_groupable(1, &split), Some(0));
        assert_eq!(previous_groupable(0, &split), None);
        assert_eq!(previous_groupable(2, &[true, true]), None);
    }
}
