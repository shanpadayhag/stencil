//! Learning store: turns the user's interactive-restore decisions into persistent,
//! context-aware memory that future `--censor` runs consult.
//!
//! Two artifacts, both per-user under the config dir:
//! - **`learned.json`** — a compact tally keyed on `(value, type)`. A value seen only as
//!   *allowed* (restored — "not critical to censor") becomes an auto-skip. A value seen
//!   both allowed **and** denied is *conflicted*: the app refuses to guess and keeps
//!   censoring it, so context-dependent values stay safe.
//! - **`decisions.jsonl`** — an append-only log of every decision with its surrounding
//!   context (both the sentence the user saw and the whole paragraph). This is the labeled
//!   dataset a future ML model would train on; nothing reads it today. Because the log is
//!   never re-enriched, context is captured generously now and narrowed at training time.
//!
//! No machine learning here — just a deterministic, auditable feedback loop. The model
//! comes later, once the log holds enough labeled examples.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Current on-disk schema version for `learned.json`.
const STORE_VERSION: u32 = 1;

/// Schema version stamped on each `decisions.jsonl` record. Bumped to 2 when the record
/// gained `shown_context` + `block_context` (dropping the old flat `context` field).
const DECISION_SCHEMA: u32 = 2;

/// Max chars kept on each side of the placeholder when growing the sentence window — a
/// safety net so a terminator-less run can't capture an unbounded span.
const SENTENCE_MAX_RADIUS: usize = 160;

/// Max chars kept on each side of the placeholder for the paragraph (`block`) window —
/// guards against a pathological input with no blank lines.
const BLOCK_MAX_RADIUS: usize = 1200;

/// The per-user Stencil config directory (`$XDG_CONFIG_HOME/stencil` or `~/.config/stencil`).
fn config_dir() -> Result<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(xdg).join("stencil"));
    }
    let home = std::env::var_os("HOME").context("cannot locate config dir: $HOME is not set")?;
    Ok(PathBuf::from(home).join(".config").join("stencil"))
}

/// Resolve the directory holding the learning artifacts. Precedence: an explicit
/// `--data-dir` override, then the `STENCIL_DATA_DIR` env var, then the config dir. Keeping
/// the read (`detect --censor`) and write (`restore -i`) sides on this one resolver ensures
/// they always agree on where the files live.
pub fn data_dir(override_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = override_dir {
        return Ok(dir.to_path_buf());
    }
    if let Some(env) = std::env::var_os("STENCIL_DATA_DIR").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(env));
    }
    config_dir()
}

/// Path to the compact learned-store file within the resolved [`data_dir`].
pub fn store_path(override_dir: Option<&Path>) -> Result<PathBuf> {
    Ok(data_dir(override_dir)?.join("learned.json"))
}

/// Path to the append-only decision log (the future training set) within [`data_dir`].
pub fn log_path(override_dir: Option<&Path>) -> Result<PathBuf> {
    Ok(data_dir(override_dir)?.join("decisions.jsonl"))
}

/// One `(value, type)` tally in the learned store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct LearnedEntry {
    value: String,
    #[serde(rename = "type")]
    value_type: String,
    /// Times the user marked this value safe (restored it).
    allow: u32,
    /// Times the user kept it redacted (skipped it).
    deny: u32,
}

/// The compact, conflict-aware learned store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedStore {
    version: u32,
    entries: Vec<LearnedEntry>,
}

impl Default for LearnedStore {
    fn default() -> Self {
        Self {
            version: STORE_VERSION,
            entries: Vec::new(),
        }
    }
}

impl LearnedStore {
    /// Load the store from `path`, or an empty store if the file does not exist.
    ///
    /// # Errors
    /// Returns an error if the file exists but cannot be read or parsed.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        match fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text)
                .with_context(|| format!("invalid learned store in `{}`", path.display())),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(err) => Err(err).with_context(|| format!("failed to read `{}`", path.display())),
        }
    }

    /// Write the store to `path`, creating the parent directory as needed.
    ///
    /// # Errors
    /// Returns an error if the directory or file cannot be written.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create `{}`", parent.display()))?;
        }
        let json =
            serde_json::to_string_pretty(self).context("failed to serialize learned store")?;
        fs::write(path, json).with_context(|| format!("failed to write `{}`", path.display()))
    }

    /// Record one decision: `allow` means the value was restored (deemed not critical),
    /// otherwise it was kept redacted.
    pub fn record(&mut self, value: &str, value_type: &str, allow: bool) {
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.value == value && entry.value_type == value_type);
        let entry = match entry {
            Some(entry) => entry,
            None => {
                self.entries.push(LearnedEntry {
                    value: value.to_string(),
                    value_type: value_type.to_string(),
                    allow: 0,
                    deny: 0,
                });
                self.entries.last_mut().expect("just pushed")
            }
        };
        if allow {
            entry.allow += 1;
        } else {
            entry.deny += 1;
        }
    }

    /// Values the app may safely auto-skip: seen as allowed at least once and **never**
    /// denied. Conflicted values (both allowed and denied) are deliberately excluded, so a
    /// value that is sometimes sensitive stays censored.
    pub fn allowed_values(&self) -> BTreeSet<String> {
        self.entries
            .iter()
            .filter(|entry| entry.allow > 0 && entry.deny == 0)
            .map(|entry| entry.value.clone())
            .collect()
    }
}

/// One row of the append-only decision log — a labeled training example.
///
/// Two context fields are captured deliberately: `shown_context` is exactly what the user
/// saw when they decided (label provenance), while `block_context` is the richer paragraph
/// the placeholder sat in. The log is append-only and never re-enriched, so both are
/// captured up front and a future model narrows them at training time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionRecord {
    /// Record schema version (see [`DECISION_SCHEMA`]).
    pub schema: u32,
    /// Unix epoch seconds when the decision was made.
    pub timestamp: u64,
    /// The source document the mapping came from.
    pub source: String,
    /// The placeholder token decided on.
    pub placeholder: String,
    /// The value category (`PERSON`, `EMAIL`, …).
    #[serde(rename = "type")]
    pub value_type: String,
    /// The real value behind the placeholder.
    pub value: String,
    /// `"allow"` (restored / not critical) or `"deny"` (kept redacted).
    pub decision: String,
    /// The sentence-ish window shown to the user — the exact basis for their decision.
    pub shown_context: String,
    /// The whole paragraph the placeholder sat in — the richer feature for future ML.
    pub block_context: String,
}

/// The current decision-record schema version stamped on freshly written rows.
pub const fn decision_schema() -> u32 {
    DECISION_SCHEMA
}

/// Append one decision record to the JSONL log at `path`, creating it as needed.
///
/// # Errors
/// Returns an error if the directory or file cannot be written.
pub fn append_decision(path: &std::path::Path, record: &DecisionRecord) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create `{}`", parent.display()))?;
    }
    let line = serde_json::to_string(record).context("failed to serialize decision record")?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open `{}`", path.display()))?;
    writeln!(file, "{line}").with_context(|| format!("failed to append to `{}`", path.display()))
}

/// Unix-epoch seconds now, or `0` if the clock is before the epoch.
pub fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A sentence-ish, whitespace-collapsed window of `text` around the first occurrence of
/// `needle`: grown outward to the nearest sentence terminator (`.`/`!`/`?`/newline) on each
/// side, bounded by [`SENTENCE_MAX_RADIUS`] chars per side. Empty if `needle` is absent.
///
/// This is what the interactive prompt shows the user and what `shown_context` records.
pub fn sentence_window(text: &str, needle: &str) -> String {
    let Some(pos) = text.find(needle) else {
        return String::new();
    };
    let after = pos + needle.len();

    // Left: from the radius floor, take everything after the last terminator (if any).
    let left_floor = floor_boundary(text, pos.saturating_sub(SENTENCE_MAX_RADIUS));
    let start = match text[left_floor..pos].rfind(is_terminator) {
        // Terminators are ASCII, so +1 lands on a char boundary just past them.
        Some(offset) => left_floor + offset + 1,
        None => left_floor,
    };

    // Right: from the end of the needle, include up to and including the next terminator.
    let right_ceil = ceil_boundary(text, (after + SENTENCE_MAX_RADIUS).min(text.len()));
    let end = match text[after..right_ceil].find(is_terminator) {
        Some(offset) => after + offset + 1,
        None => right_ceil,
    };

    collapse_ws(&text[start..end])
}

/// The whole blank-line-delimited paragraph (`block`) containing the first occurrence of
/// `needle`, whitespace-collapsed and bounded by [`BLOCK_MAX_RADIUS`] chars per side as a
/// safety net for inputs with no blank lines. Empty if `needle` is absent.
///
/// This is the richer `block_context` feature stored for future ML.
pub fn block_window(text: &str, needle: &str) -> String {
    let Some(pos) = text.find(needle) else {
        return String::new();
    };
    let after = pos + needle.len();

    let left_floor = floor_boundary(text, pos.saturating_sub(BLOCK_MAX_RADIUS));
    // A blank line ("\n\n") separates paragraphs; start just past the last one before us.
    let start = match text[left_floor..pos].rfind("\n\n") {
        Some(offset) => left_floor + offset + 2,
        None => left_floor,
    };

    let right_ceil = ceil_boundary(text, (after + BLOCK_MAX_RADIUS).min(text.len()));
    let end = match text[after..right_ceil].find("\n\n") {
        Some(offset) => after + offset,
        None => right_ceil,
    };

    collapse_ws(&text[start..end])
}

/// True for the sentence terminators the window grows toward.
fn is_terminator(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '\n')
}

/// Collapse all runs of whitespace in `text` to single spaces, trimming the ends.
fn collapse_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The largest char boundary `<= index`.
fn floor_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// The smallest char boundary `>= index`.
fn ceil_boundary(text: &str, mut index: usize) -> usize {
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_only_value_is_auto_skippable() {
        let mut store = LearnedStore::default();
        store.record("Acme Corp", "ORG", true);
        store.record("Acme Corp", "ORG", true);
        assert!(store.allowed_values().contains("Acme Corp"));
    }

    #[test]
    fn conflicted_value_is_not_auto_skippable() {
        let mut store = LearnedStore::default();
        store.record("5 Main Street", "ACCOUNT", true);
        store.record("5 Main Street", "ACCOUNT", false);
        assert!(
            !store.allowed_values().contains("5 Main Street"),
            "a value allowed once and denied once must stay censored"
        );
    }

    #[test]
    fn deny_only_value_is_not_in_allowlist() {
        let mut store = LearnedStore::default();
        store.record("Jane Doe", "PERSON", false);
        assert!(store.allowed_values().is_empty());
    }

    #[test]
    fn record_tallies_by_value_and_type() {
        let mut store = LearnedStore::default();
        store.record("X", "EMAIL", true);
        store.record("X", "PHONE", true);
        // Same value, different types are tracked separately.
        assert_eq!(store.entries.len(), 2);
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = std::env::temp_dir().join(format!("stencil_learn_{}", std::process::id()));
        let path = dir.join("learned.json");
        let mut store = LearnedStore::default();
        store.record("Acme", "ORG", true);
        store.save(&path).expect("save");

        let loaded = LearnedStore::load(&path).expect("load");
        assert!(loaded.allowed_values().contains("Acme"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_missing_file_is_empty() {
        let missing = std::env::temp_dir().join("stencil_learn_missing_xyz.json");
        let _ = fs::remove_file(&missing);
        assert!(
            LearnedStore::load(&missing)
                .expect("load")
                .allowed_values()
                .is_empty()
        );
    }

    #[test]
    fn sentence_window_stops_at_terminators_and_collapses_whitespace() {
        let text = "First sentence. Pay the   sum to REDACTED_EMAIL_001 today! Next bit.";
        let window = sentence_window(text, "REDACTED_EMAIL_001");
        assert!(window.contains("REDACTED_EMAIL_001"));
        assert!(
            window.starts_with("Pay the sum"),
            "trims the prior sentence"
        );
        assert!(
            window.ends_with("today!"),
            "includes the closing terminator"
        );
        assert!(
            !window.contains("First sentence"),
            "prior sentence excluded"
        );
        assert!(!window.contains("Next bit"), "following sentence excluded");
        assert!(!window.contains("  "), "whitespace collapsed");
    }

    #[test]
    fn sentence_window_grows_to_bounds_without_terminators() {
        // No terminators at all: falls back to the radius-bounded slice, not a panic.
        let text = format!("{} REDACTED_X_001 tail", "word ".repeat(10));
        let window = sentence_window(&text, "REDACTED_X_001");
        assert!(window.contains("REDACTED_X_001"));
        assert!(window.ends_with("tail"));
    }

    #[test]
    fn sentence_window_empty_when_absent() {
        assert_eq!(sentence_window("no token here", "REDACTED_X_001"), "");
    }

    #[test]
    fn block_window_captures_whole_paragraph() {
        let text = "Intro paragraph.\n\nThe Buyer, REDACTED_PERSON_001, shall pay\nwithin 30 days.\n\nClosing.";
        let window = block_window(text, "REDACTED_PERSON_001");
        assert_eq!(
            window,
            "The Buyer, REDACTED_PERSON_001, shall pay within 30 days."
        );
        assert!(!window.contains("Intro"), "prior paragraph excluded");
        assert!(!window.contains("Closing"), "next paragraph excluded");
    }

    #[test]
    fn block_window_empty_when_absent() {
        assert_eq!(block_window("no token here", "REDACTED_X_001"), "");
    }

    #[test]
    fn decision_record_round_trips_with_schema_and_contexts() {
        let record = DecisionRecord {
            schema: decision_schema(),
            timestamp: 1,
            source: "c.txt".into(),
            placeholder: "REDACTED_PERSON_001".into(),
            value_type: "PERSON".into(),
            value: "Jane Doe".into(),
            decision: "allow".into(),
            shown_context: "pay REDACTED_PERSON_001 today".into(),
            block_context: "The buyer pay REDACTED_PERSON_001 today within 30 days".into(),
        };
        let json = serde_json::to_string(&record).expect("serialize");
        assert!(json.contains("\"schema\":2"));
        assert!(json.contains("\"shown_context\""));
        assert!(json.contains("\"block_context\""));
        let back: DecisionRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, record);
    }

    #[test]
    fn explicit_data_dir_override_wins_and_locates_files() {
        // An explicit override short-circuits the env/config lookup entirely, so this is
        // safe to assert without touching process-global env vars.
        let dir = Path::new("/tmp/stencil-data");
        assert_eq!(data_dir(Some(dir)).expect("resolve"), dir);
        assert_eq!(
            store_path(Some(dir)).expect("store"),
            dir.join("learned.json")
        );
        assert_eq!(
            log_path(Some(dir)).expect("log"),
            dir.join("decisions.jsonl")
        );
    }
}
