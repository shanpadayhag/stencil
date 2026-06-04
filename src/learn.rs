//! Learning store: turns the user's interactive-restore decisions into persistent,
//! context-aware memory that future `--censor` runs consult.
//!
//! Two artifacts, both per-user under the config dir:
//! - **`learned.json`** — a compact tally keyed on `(value, type)`. A value seen only as
//!   *allowed* (restored — "not critical to censor") becomes an auto-skip. A value seen
//!   both allowed **and** denied is *conflicted*: the app refuses to guess and keeps
//!   censoring it, so context-dependent values stay safe.
//! - **`decisions.jsonl`** — an append-only log of every decision with its surrounding
//!   context. This is the labeled dataset a future ML model would train on; nothing reads
//!   it today.
//!
//! No machine learning here — just a deterministic, auditable feedback loop. The model
//! comes later, once the log holds enough labeled examples.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Current on-disk schema version for `learned.json`.
const STORE_VERSION: u32 = 1;

/// Characters of context captured on each side of a placeholder for the decision log.
const CONTEXT_RADIUS: usize = 60;

/// The per-user Stencil config directory (`$XDG_CONFIG_HOME/stencil` or `~/.config/stencil`).
fn config_dir() -> Result<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(xdg).join("stencil"));
    }
    let home = std::env::var_os("HOME").context("cannot locate config dir: $HOME is not set")?;
    Ok(PathBuf::from(home).join(".config").join("stencil"))
}

/// Path to the compact learned-store file.
pub fn store_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("learned.json"))
}

/// Path to the append-only decision log (the future training set).
pub fn log_path() -> Result<PathBuf> {
    Ok(config_dir()?.join("decisions.jsonl"))
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionRecord {
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
    /// Whitespace-collapsed text surrounding the placeholder, the feature for future ML.
    pub context: String,
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

/// A whitespace-collapsed window of `text` around the first occurrence of `needle`, with
/// up to [`CONTEXT_RADIUS`] characters on each side. Empty if `needle` is absent.
pub fn context_window(text: &str, needle: &str) -> String {
    let Some(pos) = text.find(needle) else {
        return String::new();
    };
    let start = floor_boundary(text, pos.saturating_sub(CONTEXT_RADIUS));
    let raw_end = (pos + needle.len() + CONTEXT_RADIUS).min(text.len());
    let end = ceil_boundary(text, raw_end);
    text[start..end]
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
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
    fn context_window_collapses_whitespace_around_match() {
        let text = "Pay the   sum to REDACTED_EMAIL_001 before\nthe end of the month.";
        let window = context_window(text, "REDACTED_EMAIL_001");
        assert!(window.contains("REDACTED_EMAIL_001"));
        assert!(!window.contains('\n'));
        assert!(!window.contains("  "), "whitespace collapsed");
    }

    #[test]
    fn context_window_empty_when_absent() {
        assert_eq!(context_window("no token here", "REDACTED_X_001"), "");
    }
}
