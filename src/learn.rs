//! Learning stores: turn the user's `review` decisions into persistent, context-aware memory
//! and into labeled training sets for the two future ML models.
//!
//! Artifacts live per-user under a root data dir, split by model into `censor/` and `styling/`
//! subdirectories (see [`model_dir`]):
//! - **`censor/learned.json`** — a compact tally keyed on `(value, type)`. A value seen only as
//!   *allowed* (a reviewer-rejected false positive) becomes an auto-skip; a value seen both
//!   allowed **and** denied is *conflicted* and stays censored, so context-dependent values
//!   stay safe.
//! - **`censor/decisions.jsonl`** — an append-only [`DecisionRecord`] log: every reviewed value
//!   with its detected/confirmed type and surrounding context. The censor model's training set.
//! - **`styling/styling.jsonl`** — an append-only [`StylingRecord`] log: every reviewed block's
//!   styling features and verdict. The styling model's training set.
//!
//! Because the logs are append-only and never re-enriched, context is captured generously now
//! and narrowed at training time. No machine learning here yet — just a deterministic,
//! auditable feedback loop; the models come later, once the logs hold enough labeled examples.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Current on-disk schema version for `learned.json`.
const STORE_VERSION: u32 = 1;

/// Schema version stamped on each `decisions.jsonl` record. v6 bumped it to 3: the record is
/// now keyed on the reviewed value with `method`/`detected_type`/`verdict`/`final_type` (the
/// multi-class label), replacing the schema-2 `placeholder`/`type`/`decision` fields.
const DECISION_SCHEMA: u32 = 3;

/// Schema version stamped on each `styling.jsonl` record.
const STYLING_SCHEMA: u32 = 1;

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

/// Resolve the **root** learning directory. Precedence: an explicit `--data-dir` override, then
/// the `STENCIL_DATA_DIR` env var, then the config dir. Per-model stores live in subdirectories
/// of this root (see [`model_dir`]).
pub fn data_dir(override_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = override_dir {
        return Ok(dir.to_path_buf());
    }
    if let Some(env) = std::env::var_os("STENCIL_DATA_DIR").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(env));
    }
    config_dir()
}

/// A model whose learning artifacts live in their own subdirectory of the root data dir.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// The censor detector store (`learned.json` + `decisions.jsonl`).
    Censor,
    /// The styling detector store (`styling.jsonl` + `profiles/`).
    Styling,
}

impl Model {
    /// The subdirectory name under the root data dir.
    fn subdir(self) -> &'static str {
        match self {
            Model::Censor => "censor",
            Model::Styling => "styling",
        }
    }

    /// The environment variable that overrides this model's directory outright.
    fn env_var(self) -> &'static str {
        match self {
            Model::Censor => "STENCIL_CENSOR_DIR",
            Model::Styling => "STENCIL_STYLING_DIR",
        }
    }
}

/// Resolve a model's store directory. Precedence: the model-specific flag override, then the
/// model-specific env var (`STENCIL_CENSOR_DIR` / `STENCIL_STYLING_DIR`), then `<root>/<model>`
/// where the root comes from [`data_dir`]. v6 reads only these subdirs, so any pre-v6 files at
/// the root of `STENCIL_DATA_DIR` are ignored — a clean, fresh start.
pub fn model_dir(
    model: Model,
    root_override: Option<&Path>,
    model_override: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(dir) = model_override {
        return Ok(dir.to_path_buf());
    }
    if let Some(env) = std::env::var_os(model.env_var()).filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(env));
    }
    Ok(data_dir(root_override)?.join(model.subdir()))
}

/// Path to the compact learned-store file within the censor model dir.
pub fn store_path(root_override: Option<&Path>, censor_override: Option<&Path>) -> Result<PathBuf> {
    Ok(model_dir(Model::Censor, root_override, censor_override)?.join("learned.json"))
}

/// Path to the append-only censor decision log within the censor model dir.
pub fn log_path(root_override: Option<&Path>, censor_override: Option<&Path>) -> Result<PathBuf> {
    Ok(model_dir(Model::Censor, root_override, censor_override)?.join("decisions.jsonl"))
}

/// Path to the append-only styling decision log within the styling model dir.
pub fn styling_log_path(
    root_override: Option<&Path>,
    styling_override: Option<&Path>,
) -> Result<PathBuf> {
    Ok(model_dir(Model::Styling, root_override, styling_override)?.join("styling.jsonl"))
}

/// Path to the per-document style-profile sidecar directory within the styling model dir.
pub fn styling_profiles_dir(
    root_override: Option<&Path>,
    styling_override: Option<&Path>,
) -> Result<PathBuf> {
    Ok(model_dir(Model::Styling, root_override, styling_override)?.join("profiles"))
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

/// One row of the append-only censor decision log — a labeled training example for the future
/// multi-class censor model.
///
/// `detected_type` is what the detector guessed (possibly the neutral `ENTITY`); `final_type`
/// is the reviewer's confirmed/corrected type — the clean classification label — and is `None`
/// on a `reject` (a false positive, which is simply "not sensitive"). Two context fields are
/// kept: `shown_context` (exactly what the reviewer saw — label provenance) and `block_context`
/// (the richer paragraph). The log is append-only and never re-enriched. New fields carry
/// `#[serde(default)]` so older schema-2 lines still deserialize.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionRecord {
    /// Record schema version (see [`decision_schema`]).
    pub schema: u32,
    /// Unix epoch seconds when the decision was made.
    pub timestamp: u64,
    /// The source document the value came from.
    pub source: String,
    /// The real value reviewed.
    pub value: String,
    /// How the value was detected (`party-list`, `regex:<kind>`, `heuristic`).
    #[serde(default)]
    pub method: String,
    /// The detector's guessed type (e.g. `PERSON`, or the neutral `ENTITY`).
    #[serde(default)]
    pub detected_type: String,
    /// The verdict: `confirm` (sensitive, keep censored) or `reject` (false positive).
    #[serde(default)]
    pub verdict: String,
    /// The reviewer's confirmed/corrected type — the classification label; `None` on `reject`.
    #[serde(default)]
    pub final_type: Option<String>,
    /// The sentence-ish window shown to the reviewer — the basis for the decision.
    pub shown_context: String,
    /// The whole paragraph the value sat in — the richer feature for future ML.
    pub block_context: String,
    /// How many times the value occurred in the document.
    #[serde(default)]
    pub occurrences: u32,
}

/// The current decision-record schema version stamped on freshly written rows.
pub const fn decision_schema() -> u32 {
    DECISION_SCHEMA
}

/// The current styling-record schema version stamped on freshly written rows.
pub const fn styling_schema() -> u32 {
    STYLING_SCHEMA
}

/// Paragraph indentation in twips (1/1440 inch). Absent fields mean "not set".
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Indent {
    pub left: Option<i32>,
    pub right: Option<i32>,
    pub hanging: Option<i32>,
    pub first_line: Option<i32>,
}

/// List membership: numbering-definition id and indent level (`ilvl`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Numbering {
    pub num_id: Option<usize>,
    pub ilvl: Option<usize>,
}

/// Paragraph spacing in twips.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Spacing {
    pub before: Option<i32>,
    pub after: Option<i32>,
    pub line: Option<i32>,
}

/// Paragraph-level styling (docx `pPr`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParaStyle {
    pub style_name: Option<String>,
    pub alignment: Option<String>,
    pub indent: Indent,
    pub numbering: Option<Numbering>,
    pub spacing: Spacing,
}

/// Run-level styling (docx `rPr`) for the block's dominant run, with `mixed` set when runs
/// within the block disagree.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunStyle {
    pub font: Option<String>,
    pub size_half_pt: Option<u64>,
    pub bold: bool,
    pub italic: bool,
    pub underline: Option<String>,
    pub color: Option<String>,
    pub mixed: bool,
}

/// Document-relative styling features — how this block deviates from the document's norms.
/// Populated from the deterministic style profile (T29); descriptive, never a verdict.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RelativeStyle {
    pub style_doc_freq: Option<f32>,
    pub font_matches_doc_dominant: Option<bool>,
    pub size_matches_doc_dominant: Option<bool>,
    pub matches_role_peers: Option<bool>,
    pub indent_vs_ilvl_norm: Option<f32>,
}

/// The censored text of the neighboring blocks, for peer judgment at review time.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NeighborContext {
    pub prev_text: String,
    pub next_text: String,
}

/// One row of the append-only styling log — a labeled training example for the future styling
/// model. Every reviewed block produces one (a `fine` verdict is the negative class). The
/// `text` stored is censored (styling judgment needs no real values). Populated by the styling
/// extraction/profile/review stages (T28–T30); defined here so the schema and its paths live
/// alongside the censor record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StylingRecord {
    /// Record schema version (see [`styling_schema`]).
    pub schema: u32,
    /// The source document.
    pub source: String,
    /// Position of the block in document order.
    pub block_index: usize,
    /// `paragraph` | `heading` | `list_item` | `table_cell`.
    pub block_kind: String,
    /// Heading level, when the block is a heading.
    pub heading_level: Option<u8>,
    /// Whether the block is inside a table cell.
    pub in_table: bool,
    /// The block's censored text.
    pub text: String,
    /// Paragraph-level styling.
    pub para: ParaStyle,
    /// Run-level styling (dominant run + `mixed`).
    pub run: RunStyle,
    /// Document-relative deviation features.
    pub relative: RelativeStyle,
    /// Neighboring blocks' censored text.
    pub context: NeighborContext,
    /// `fine` or `weird`.
    pub verdict: String,
    /// The weirdness category when `weird`; `None` when `fine`.
    pub category: Option<String>,
    /// Optional free-text note.
    pub note: Option<String>,
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

/// Append one styling record to the JSONL log at `path`, creating it as needed.
///
/// # Errors
/// Returns an error if the directory or file cannot be written.
pub fn append_styling(path: &std::path::Path, record: &StylingRecord) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create `{}`", parent.display()))?;
    }
    let line = serde_json::to_string(record).context("failed to serialize styling record")?;
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
    fn decision_record_round_trips_with_schema_3_fields() {
        let record = DecisionRecord {
            schema: decision_schema(),
            timestamp: 1,
            source: "c.txt".into(),
            value: "Jane Doe".into(),
            method: "heuristic".into(),
            detected_type: "ENTITY".into(),
            verdict: "confirm".into(),
            final_type: Some("PERSON".into()),
            shown_context: "pay REDACTED_ENTITY_001 today".into(),
            block_context: "The buyer pay REDACTED_ENTITY_001 today within 30 days".into(),
            occurrences: 2,
        };
        let json = serde_json::to_string(&record).expect("serialize");
        assert!(json.contains("\"schema\":3"));
        assert!(json.contains("\"detected_type\":\"ENTITY\""));
        assert!(json.contains("\"final_type\":\"PERSON\""));
        let back: DecisionRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, record);
    }

    #[test]
    fn reject_record_serializes_null_final_type() {
        let record = DecisionRecord {
            schema: decision_schema(),
            timestamp: 0,
            source: "c.txt".into(),
            value: "Reach".into(),
            method: "heuristic".into(),
            detected_type: "ENTITY".into(),
            verdict: "reject".into(),
            final_type: None,
            shown_context: String::new(),
            block_context: String::new(),
            occurrences: 1,
        };
        let json = serde_json::to_string(&record).expect("serialize");
        assert!(json.contains("\"final_type\":null"));
        let back: DecisionRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.final_type, None);
    }

    #[test]
    fn legacy_schema2_decision_line_still_parses() {
        // A pre-v6 schema-2 line lacks the new fields and carries now-unknown ones
        // (placeholder/type/decision); it must still deserialize — defaults for the new fields,
        // unknown fields ignored — rather than erroring.
        let schema2 = r#"{"schema":2,"timestamp":7,"source":"c.txt","placeholder":"REDACTED_PERSON_001","type":"PERSON","value":"Jane","decision":"allow","shown_context":"x","block_context":"y"}"#;
        let rec: DecisionRecord = serde_json::from_str(schema2).expect("schema-2 still parses");
        assert_eq!(rec.value, "Jane");
        assert_eq!(rec.schema, 2);
        assert_eq!(rec.method, "", "new fields fall back to defaults");
        assert_eq!(rec.final_type, None);
    }

    #[test]
    fn styling_record_round_trips() {
        let record = StylingRecord {
            schema: styling_schema(),
            source: "c.docx".into(),
            block_index: 12,
            block_kind: "list_item".into(),
            heading_level: None,
            in_table: false,
            text: "(a) the seller shall deliver".into(),
            para: ParaStyle {
                style_name: Some("ListParagraph".into()),
                alignment: Some("left".into()),
                indent: Indent {
                    left: Some(720),
                    hanging: Some(360),
                    ..Default::default()
                },
                numbering: Some(Numbering {
                    num_id: Some(2),
                    ilvl: Some(1),
                }),
                spacing: Spacing::default(),
            },
            run: RunStyle {
                font: Some("Arial".into()),
                size_half_pt: Some(22),
                mixed: true,
                ..Default::default()
            },
            relative: RelativeStyle {
                style_doc_freq: Some(0.5),
                font_matches_doc_dominant: Some(false),
                ..Default::default()
            },
            context: NeighborContext {
                prev_text: "prev".into(),
                next_text: "next".into(),
            },
            verdict: "weird".into(),
            category: Some("wrong-style-for-role".into()),
            note: Some("title as paragraph".into()),
        };
        let json = serde_json::to_string(&record).expect("serialize");
        assert!(json.contains("\"schema\":1"));
        let back: StylingRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, record);
    }

    #[test]
    fn data_dir_override_wins() {
        let dir = Path::new("/tmp/stencil-data");
        assert_eq!(data_dir(Some(dir)).expect("resolve"), dir);
    }

    #[test]
    fn model_override_wins_outright() {
        let custom = Path::new("/tmp/custom-censor");
        assert_eq!(
            model_dir(Model::Censor, Some(Path::new("/root")), Some(custom)).expect("resolve"),
            custom
        );
    }

    #[test]
    fn per_model_paths_resolve_under_their_subdirs() {
        // A model env override would change the default subdir path asserted here, so skip when
        // one is set (keeps the test deterministic without mutating process-global env vars).
        if std::env::var_os("STENCIL_CENSOR_DIR").is_some()
            || std::env::var_os("STENCIL_STYLING_DIR").is_some()
        {
            return;
        }
        let root = Path::new("/root");
        assert_eq!(
            store_path(Some(root), None).expect("store"),
            root.join("censor").join("learned.json")
        );
        assert_eq!(
            log_path(Some(root), None).expect("log"),
            root.join("censor").join("decisions.jsonl")
        );
        assert_eq!(
            styling_log_path(Some(root), None).expect("styling log"),
            root.join("styling").join("styling.jsonl")
        );
        assert_eq!(
            styling_profiles_dir(Some(root), None).expect("profiles"),
            root.join("styling").join("profiles")
        );
    }
}
