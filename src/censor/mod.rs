//! Censoring: replace sensitive values with `REDACTED_<TYPE>_<NNN>` placeholders.
//!
//! The pipeline gathers candidate spans from [`patterns`] (structured values) and
//! [`names`] (the party list), resolves overlaps by a **fixed
//! precedence** with **longest-match-wins** so every character is replaced at most
//! once, and dedups by exact value (one value → one stable placeholder).

pub mod names;
pub mod patterns;
pub mod review;

use std::collections::HashMap;

use crate::detect::paired_spans;
use crate::learn::{DecisionRecord, LearnedStore, block_window, decision_schema, sentence_window};
use crate::model::{Block, Cell, Document, MAPPING_VERSION, Mapping, MappingEntry};

pub use names::PartyList;

/// The category of a censored value. Determines the placeholder prefix and the
/// detector precedence used to resolve overlaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    /// A person's name (from the party list).
    Person,
    /// An organization's name (from the party list).
    Org,
    /// An IBAN.
    Iban,
    /// A Luhn-valid payment card number.
    Card,
    /// A generic account number.
    Account,
    /// A telephone number.
    Phone,
    /// A calendar date.
    Date,
    /// A monetary amount (numeric like `$1,200` or spelled out like `two thousand
    /// dollars`).
    Money,
    /// A percentage (numeric like `10%` or spelled out like `ten percent`, including the
    /// combined `ten percent (10%)` form).
    Percent,
    /// An email address.
    Email,
    /// A place or jurisdiction (city, country, region).
    Location,
    /// A full postal/mailing address.
    Address,
    /// A proper noun of uncertain subtype — a recall-first guess the reviewer re-types.
    Entity,
}

impl ValueType {
    /// The uppercase label used in placeholders (`REDACTED_<LABEL>_<NNN>`).
    pub fn label(self) -> &'static str {
        match self {
            ValueType::Person => "PERSON",
            ValueType::Org => "ORG",
            ValueType::Iban => "IBAN",
            ValueType::Card => "CARD",
            ValueType::Account => "ACCOUNT",
            ValueType::Phone => "PHONE",
            ValueType::Date => "DATE",
            ValueType::Money => "MONEY",
            ValueType::Percent => "PERCENT",
            ValueType::Email => "EMAIL",
            ValueType::Location => "LOCATION",
            ValueType::Address => "ADDRESS",
            ValueType::Entity => "ENTITY",
        }
    }
}

/// How a candidate value was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DetectSource {
    /// Matched an explicit party-name list entry (authoritative).
    PartyList,
    /// Matched a structured-value regex.
    Pattern,
    /// Guessed by the recall-first proper-noun heuristic (uncertain subtype → `ENTITY`).
    Heuristic,
}

/// A candidate span to consider censoring. Byte offsets index into a single text field.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Candidate {
    pub start: usize,
    pub end: usize,
    pub value_type: ValueType,
    pub source: DetectSource,
}

/// Options controlling a censoring run.
#[derive(Debug, Default)]
pub struct CensorOptions<'a> {
    /// The party-name list to always censor, if any.
    pub parties: Option<&'a PartyList>,
    /// Values the user has learned are safe to leave in the clear (see [`crate::learn`]).
    /// Any candidate whose exact text appears here is **not** censored.
    pub allow: Option<&'a std::collections::BTreeSet<String>>,
}

/// The result of censoring: the placeholder-bearing document plus the reversal mapping.
#[derive(Debug)]
pub struct CensorOutcome {
    /// The document with sensitive values replaced by placeholders.
    pub document: Document,
    /// The placeholder ↔ value mapping for `restore`.
    pub mapping: Mapping,
}

/// Censor a document, returning the placeholder-bearing copy and the mapping.
///
/// ```
/// use stencil::censor::{censor, CensorOptions};
/// use stencil::model::{Block, Document};
/// use std::path::PathBuf;
///
/// let doc = Document {
///     source: PathBuf::from("c.txt"),
///     blocks: vec![Block::Paragraph { text: "email a@b.com".into() }],
/// };
/// let out = censor(&doc, &CensorOptions::default());
/// assert_eq!(out.mapping.entries.len(), 1);
/// assert!(matches!(&out.document.blocks[0], Block::Paragraph { text } if text.contains("REDACTED_EMAIL_001")));
/// ```
pub fn censor(document: &Document, options: &CensorOptions<'_>) -> CensorOutcome {
    let mut allocator = Allocator::default();
    let blocks = document
        .blocks
        .iter()
        .map(|block| censor_block(block, options, &mut allocator))
        .collect();

    CensorOutcome {
        document: Document {
            source: document.source.clone(),
            blocks,
        },
        mapping: Mapping {
            version: MAPPING_VERSION,
            source: document.source.display().to_string(),
            entries: allocator.entries,
        },
    }
}

/// Censor every text field of a block.
fn censor_block(block: &Block, options: &CensorOptions<'_>, allocator: &mut Allocator) -> Block {
    match block {
        Block::Heading { level, text } => Block::Heading {
            level: *level,
            text: censor_text(text, options, allocator),
        },
        Block::Paragraph { text } => Block::Paragraph {
            text: censor_text(text, options, allocator),
        },
        Block::Table { rows } => Block::Table {
            rows: rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| Cell {
                            text: censor_text(&cell.text, options, allocator),
                        })
                        .collect()
                })
                .collect(),
        },
    }
}

/// Censor a single text field: gather candidates, resolve overlaps, substitute.
///
/// Paired bracket interiors are reserved so variable labels (e.g. `[Client Name]`)
/// are never redacted — they are blanks meant for Claude, not sensitive values.
fn censor_text(text: &str, options: &CensorOptions<'_>, allocator: &mut Allocator) -> String {
    let reserved = paired_spans(text);
    let claimed = resolve_overlaps(gather_candidates(text, options), &reserved);
    if claimed.is_empty() {
        return text.to_string();
    }

    let mut result = text.to_string();
    // Replace right-to-left so earlier byte offsets stay valid.
    for span in claimed.iter().rev() {
        let value = &text[span.start..span.end];
        let placeholder = allocator.placeholder_for(value, span.value_type, span.source);
        result.replace_range(span.start..span.end, &placeholder);
    }
    result
}

/// Collect all candidate spans for a text from every detector.
fn gather_candidates(text: &str, options: &CensorOptions<'_>) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = patterns::find_candidates(text)
        .into_iter()
        .map(|m| Candidate {
            start: m.start,
            end: m.end,
            value_type: m.value_type,
            source: DetectSource::Pattern,
        })
        .collect();

    if let Some(list) = options.parties {
        candidates.extend(list.find(text));
    }
    // Recall-first: over-detect capitalized proper nouns as ENTITY candidates. Noise is
    // expected here — false positives are rejected at review and become negative training data.
    candidates.extend(names::guess_entities(text));
    // Drop anything the user has learned is safe to leave in the clear.
    if let Some(allow) = options.allow {
        candidates.retain(|candidate| !allow.contains(&text[candidate.start..candidate.end]));
    }
    candidates
}

/// Resolve overlapping candidates into a non-overlapping, start-sorted set.
///
/// Ordering: precedence first, then longer span, then earlier start. Greedy claiming
/// guarantees no two kept spans overlap, so every character is replaced at most once.
/// Candidates overlapping any `reserved` range (paired bracket interiors) are dropped.
fn resolve_overlaps(mut candidates: Vec<Candidate>, reserved: &[(usize, usize)]) -> Vec<Candidate> {
    candidates.sort_by(|a, b| {
        precedence(a)
            .cmp(&precedence(b))
            .then((b.end - b.start).cmp(&(a.end - a.start)))
            .then(a.start.cmp(&b.start))
    });

    let mut claimed: Vec<Candidate> = Vec::new();
    for candidate in candidates {
        let in_reserved = reserved
            .iter()
            .any(|&(start, end)| candidate.start < end && start < candidate.end);
        if in_reserved {
            continue;
        }
        let overlaps = claimed
            .iter()
            .any(|kept| candidate.start < kept.end && kept.start < candidate.end);
        if !overlaps {
            claimed.push(candidate);
        }
    }

    claimed.sort_by_key(|candidate| candidate.start);
    claimed
}

/// Precedence rank (lower wins): party names first, then the structured order from the
/// design.
fn precedence(candidate: &Candidate) -> u8 {
    match candidate.source {
        DetectSource::PartyList => 0,
        DetectSource::Pattern => match candidate.value_type {
            ValueType::Iban => 1,
            ValueType::Card => 2,
            ValueType::Account => 3,
            ValueType::Phone => 4,
            ValueType::Date => 5,
            ValueType::Money => 6,
            ValueType::Percent => 7,
            ValueType::Email => 8,
            ValueType::Person
            | ValueType::Org
            | ValueType::Location
            | ValueType::Address
            | ValueType::Entity => 9,
        },
        DetectSource::Heuristic => 10,
    }
}

/// Allocates stable placeholders, deduping by exact value and recording mapping entries.
#[derive(Default)]
struct Allocator {
    by_value: HashMap<String, usize>,
    counters: HashMap<&'static str, usize>,
    entries: Vec<MappingEntry>,
}

impl Allocator {
    /// Return the placeholder for `value`, creating a new entry or bumping the
    /// occurrence count of an existing one.
    fn placeholder_for(
        &mut self,
        value: &str,
        value_type: ValueType,
        source: DetectSource,
    ) -> String {
        if let Some(&index) = self.by_value.get(value) {
            self.entries[index].occurrences += 1;
            return self.entries[index].placeholder.clone();
        }

        let label = value_type.label();
        let counter = self.counters.entry(label).or_insert(0);
        *counter += 1;
        let placeholder = format!("REDACTED_{label}_{counter:03}");

        let index = self.entries.len();
        self.entries.push(MappingEntry {
            placeholder: placeholder.clone(),
            value_type: label.to_string(),
            value: value.to_string(),
            method: method_label(source, value_type),
            occurrences: 1,
        });
        self.by_value.insert(value.to_string(), index);
        placeholder
    }
}

/// The `method` label recorded for a detection source.
fn method_label(source: DetectSource, value_type: ValueType) -> String {
    match source {
        DetectSource::PartyList => "party-list".to_string(),
        DetectSource::Pattern => format!("regex:{}", value_type.label().to_ascii_lowercase()),
        DetectSource::Heuristic => "heuristic".to_string(),
    }
}

// ── Interactive censor stage (v6): detect → review → apply ──────────────────────
//
// `censor` (above) censors everything detectable in one pass — used by the snippet builder.
// The interactive stage instead surfaces each distinct value for a confirm/reject/re-type
// decision and applies only the confirmed ones, typed by the reviewer's chosen label.

/// A distinct value surfaced for interactive review, with the metadata the reviewer sees and
/// the decision log records.
#[derive(Debug, Clone)]
pub struct ReviewItem {
    /// The real value to decide on.
    pub value: String,
    /// The detector's guessed type (possibly the neutral `ENTITY`).
    pub detected_type: ValueType,
    /// How it was detected (`party-list` / `regex:<kind>` / `heuristic`).
    pub method: String,
    /// How many times the value occurs across the document.
    pub occurrences: usize,
    /// Sentence-ish window of the first occurrence — what the reviewer sees (label provenance).
    pub shown_context: String,
    /// The paragraph of the first occurrence — the richer logged feature.
    pub block_context: String,
}

/// The reviewer's verdict for a value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Keep censored; the placeholder is typed with this (possibly re-typed) label.
    Confirm { final_type: String },
    /// A false positive — leave the value in the clear.
    Reject,
}

/// One value's resolved outcome. `reviewed` is false for items auto-confirmed when the user
/// quits early: kept censored for safety, but not a human label (excluded from log/store).
#[derive(Debug, Clone)]
pub struct CensorDecision {
    /// The value decided on.
    pub value: String,
    /// The detector's guessed type, carried for logging.
    pub detected_type: ValueType,
    /// The verdict.
    pub verdict: Verdict,
    /// Whether a human explicitly decided this (vs auto-defaulted on quit).
    pub reviewed: bool,
}

/// Every text field of a document, in order (headings, paragraphs, then table cells).
fn text_fields(document: &Document) -> Vec<&str> {
    let mut fields = Vec::new();
    for block in &document.blocks {
        match block {
            Block::Heading { text, .. } | Block::Paragraph { text } => fields.push(text.as_str()),
            Block::Table { rows } => {
                for row in rows {
                    for cell in row {
                        fields.push(cell.text.as_str());
                    }
                }
            }
        }
    }
    fields
}

/// Plan the review: detect across the whole document, resolve overlaps per field, drop
/// allow-listed values, and return one [`ReviewItem`] per distinct value in first-seen order.
pub fn plan_review(document: &Document, options: &CensorOptions<'_>) -> Vec<ReviewItem> {
    let mut order: Vec<String> = Vec::new();
    let mut items: HashMap<String, ReviewItem> = HashMap::new();
    for text in text_fields(document) {
        let reserved = paired_spans(text);
        for candidate in resolve_overlaps(gather_candidates(text, options), &reserved) {
            let value = text[candidate.start..candidate.end].to_string();
            if let Some(item) = items.get_mut(&value) {
                item.occurrences += 1;
            } else {
                order.push(value.clone());
                items.insert(
                    value.clone(),
                    ReviewItem {
                        detected_type: candidate.value_type,
                        method: method_label(candidate.source, candidate.value_type),
                        occurrences: 1,
                        shown_context: sentence_window(text, &value),
                        block_context: block_window(text, &value),
                        value,
                    },
                );
            }
        }
    }
    order
        .into_iter()
        .filter_map(|value| items.remove(&value))
        .collect()
}

/// Apply the decisions: a copy of `document` with every **confirmed** value replaced by a
/// `REDACTED_<FINAL_TYPE>_<NNN>` placeholder (deduped per value); rejected values are left as-is.
pub fn apply(
    document: &Document,
    decisions: &[CensorDecision],
    options: &CensorOptions<'_>,
) -> Document {
    let confirmed: HashMap<&str, &str> = decisions
        .iter()
        .filter_map(|decision| match &decision.verdict {
            Verdict::Confirm { final_type } => Some((decision.value.as_str(), final_type.as_str())),
            Verdict::Reject => None,
        })
        .collect();

    let mut allocator = LabelAllocator::default();
    let blocks = document
        .blocks
        .iter()
        .map(|block| apply_block(block, options, &confirmed, &mut allocator))
        .collect();
    Document {
        source: document.source.clone(),
        blocks,
    }
}

/// Apply confirmed censorings to every text field of a block.
fn apply_block(
    block: &Block,
    options: &CensorOptions<'_>,
    confirmed: &HashMap<&str, &str>,
    allocator: &mut LabelAllocator,
) -> Block {
    match block {
        Block::Heading { level, text } => Block::Heading {
            level: *level,
            text: apply_text(text, options, confirmed, allocator),
        },
        Block::Paragraph { text } => Block::Paragraph {
            text: apply_text(text, options, confirmed, allocator),
        },
        Block::Table { rows } => Block::Table {
            rows: rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|cell| Cell {
                            text: apply_text(&cell.text, options, confirmed, allocator),
                        })
                        .collect()
                })
                .collect(),
        },
    }
}

/// Substitute only the confirmed values in one text field (right-to-left to keep offsets valid).
fn apply_text(
    text: &str,
    options: &CensorOptions<'_>,
    confirmed: &HashMap<&str, &str>,
    allocator: &mut LabelAllocator,
) -> String {
    let reserved = paired_spans(text);
    let claimed = resolve_overlaps(gather_candidates(text, options), &reserved);
    let mut result = text.to_string();
    for span in claimed.iter().rev() {
        let value = &text[span.start..span.end];
        if let Some(&label) = confirmed.get(value) {
            let placeholder = allocator.placeholder_for(value, label);
            result.replace_range(span.start..span.end, &placeholder);
        }
    }
    result
}

/// Allocates `REDACTED_<LABEL>_<NNN>` placeholders, deduping by exact value and counting per
/// label. Labels are the reviewer's final type strings (e.g. `ORG`, `ID`, `other`).
#[derive(Default)]
struct LabelAllocator {
    by_value: HashMap<String, String>,
    counters: HashMap<String, usize>,
}

impl LabelAllocator {
    fn placeholder_for(&mut self, value: &str, label: &str) -> String {
        if let Some(existing) = self.by_value.get(value) {
            return existing.clone();
        }
        let label = label.to_ascii_uppercase();
        let counter = self.counters.entry(label.clone()).or_insert(0);
        *counter += 1;
        let placeholder = format!("REDACTED_{label}_{counter:03}");
        self.by_value.insert(value.to_string(), placeholder.clone());
        placeholder
    }
}

/// Build the schema-3 decision-log records for the human-reviewed decisions (auto-defaulted
/// items are skipped). `items` and `decisions` are parallel (one decision per item).
pub fn decision_records(
    items: &[ReviewItem],
    decisions: &[CensorDecision],
    source: &str,
    timestamp: u64,
) -> Vec<DecisionRecord> {
    items
        .iter()
        .zip(decisions)
        .filter(|(_, decision)| decision.reviewed)
        .map(|(item, decision)| {
            let (verdict, final_type) = match &decision.verdict {
                Verdict::Confirm { final_type } => ("confirm", Some(final_type.clone())),
                Verdict::Reject => ("reject", None),
            };
            DecisionRecord {
                schema: decision_schema(),
                timestamp,
                source: source.to_string(),
                value: item.value.clone(),
                method: item.method.clone(),
                detected_type: item.detected_type.label().to_string(),
                verdict: verdict.to_string(),
                final_type,
                shown_context: item.shown_context.clone(),
                block_context: item.block_context.clone(),
                occurrences: item.occurrences as u32,
            }
        })
        .collect()
}

/// Fold the human-reviewed decisions into the learned store: a `reject` (false positive) marks
/// the value safe to leave in the clear (`allow`); a `confirm` keeps it censored (`deny`). A
/// value seen both ways becomes conflicted and stays censored.
pub fn update_store(store: &mut LearnedStore, items: &[ReviewItem], decisions: &[CensorDecision]) {
    for (item, decision) in items.iter().zip(decisions) {
        if !decision.reviewed {
            continue;
        }
        let allow = matches!(decision.verdict, Verdict::Reject);
        store.record(&item.value, item.detected_type.label(), allow);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn paragraph_doc(text: &str) -> Document {
        Document {
            source: PathBuf::from("test.txt"),
            blocks: vec![Block::Paragraph { text: text.into() }],
        }
    }

    fn censored_text(outcome: &CensorOutcome) -> &str {
        match &outcome.document.blocks[0] {
            Block::Paragraph { text } => text,
            _ => panic!("expected a paragraph"),
        }
    }

    #[test]
    fn email_is_replaced_with_placeholder() {
        let out = censor(
            &paragraph_doc("write to a@b.com please"),
            &CensorOptions::default(),
        );
        assert_eq!(censored_text(&out), "write to REDACTED_EMAIL_001 please");
        assert_eq!(out.mapping.entries[0].value, "a@b.com");
        assert_eq!(out.mapping.entries[0].method, "regex:email");
        assert_eq!(out.mapping.source, "test.txt");
    }

    #[test]
    fn iban_wins_over_overlapping_account() {
        // The IBAN's digits would also match the account pattern; precedence keeps IBAN.
        let out = censor(
            &paragraph_doc("pay GB82WEST12345698765432 now"),
            &CensorOptions::default(),
        );
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].value_type, "IBAN");
        assert_eq!(censored_text(&out), "pay REDACTED_IBAN_001 now");
    }

    #[test]
    fn valid_card_wins_over_account() {
        let out = censor(
            &paragraph_doc("card 4111 1111 1111 1111 saved"),
            &CensorOptions::default(),
        );
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].value_type, "CARD");
    }

    #[test]
    fn duplicate_values_share_one_placeholder() {
        let out = censor(
            &paragraph_doc("a@b.com and again a@b.com"),
            &CensorOptions::default(),
        );
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].occurrences, 2);
        assert_eq!(
            censored_text(&out),
            "REDACTED_EMAIL_001 and again REDACTED_EMAIL_001"
        );
    }

    #[test]
    fn distinct_values_increment_per_type() {
        let out = censor(
            &paragraph_doc("mail a@b.com then c@d.com"),
            &CensorOptions::default(),
        );
        assert_eq!(out.mapping.entries.len(), 2);
        assert_eq!(out.mapping.entries[0].placeholder, "REDACTED_EMAIL_001");
        assert_eq!(out.mapping.entries[1].placeholder, "REDACTED_EMAIL_002");
    }

    #[test]
    fn party_name_is_always_replaced() {
        let list = PartyList::parse("Wonka Corporation").expect("parse");
        let options = CensorOptions {
            parties: Some(&list),
            ..Default::default()
        };
        let out = censor(&paragraph_doc("signed by Wonka Corporation"), &options);
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].value_type, "ORG");
        assert_eq!(out.mapping.entries[0].method, "party-list");
        assert!(censored_text(&out).contains("REDACTED_ORG_001"));
    }

    #[test]
    fn learned_allowed_value_is_not_censored() {
        let allow: std::collections::BTreeSet<String> =
            ["billing@acme.example".to_string()].into_iter().collect();
        let out = censor(
            &paragraph_doc("Reach billing@acme.example or sales@acme.example."),
            &CensorOptions {
                allow: Some(&allow),
                ..Default::default()
            },
        );
        let text = censored_text(&out);
        // The learned-safe value stays in the clear; the other email is still censored.
        assert!(text.contains("billing@acme.example"), "allowed value kept");
        assert!(text.contains("REDACTED_EMAIL_001"), "other value censored");
        assert_eq!(out.mapping.entries.len(), 1);
    }

    #[test]
    fn off_list_proper_nouns_are_guessed_as_entity() {
        // Recall-first (v6): a capitalized name not on the party list is still flagged — as the
        // neutral ENTITY (subtype unknown), to be confirmed or re-typed at review. "Signed"/"by"/
        // "in" are skipped (lone stopword / lowercase).
        let out = censor(
            &paragraph_doc("Signed by Jane Doe in London"),
            &CensorOptions::default(),
        );
        let values: Vec<&str> = out
            .mapping
            .entries
            .iter()
            .map(|e| e.value.as_str())
            .collect();
        assert!(values.contains(&"Jane Doe"), "multi-word name caught");
        assert!(values.contains(&"London"), "single-word location caught");
        assert!(
            out.mapping.entries.iter().all(|e| e.value_type == "ENTITY"),
            "guesses are the neutral ENTITY type"
        );
        assert!(
            out.mapping.entries.iter().all(|e| e.method == "heuristic"),
            "method records the heuristic source"
        );
        let text = censored_text(&out);
        assert!(text.contains("REDACTED_ENTITY_001") && text.contains("REDACTED_ENTITY_002"));
        assert!(!text.contains("Jane Doe"));
    }

    #[test]
    fn party_list_overrides_an_overlapping_entity_guess() {
        // The heuristic also matches "Wonka Corporation", but the authoritative party list wins
        // (precedence), so it is typed ORG, not ENTITY, and appears once.
        let list = PartyList::parse("Wonka Corporation").expect("parse");
        let out = censor(
            &paragraph_doc("countersigned by Wonka Corporation here"),
            &CensorOptions {
                parties: Some(&list),
                ..Default::default()
            },
        );
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].value_type, "ORG");
        assert_eq!(out.mapping.entries[0].method, "party-list");
    }

    #[test]
    fn structured_value_and_entity_guess_coexist() {
        // A guessed name and a structured email are both kept (no overlap); "Contact" is a
        // lone stopword and is skipped.
        let out = censor(
            &paragraph_doc("Contact Jane at a@b.com"),
            &CensorOptions::default(),
        );
        assert_eq!(out.mapping.entries.len(), 2);
        let by_type: std::collections::BTreeSet<&str> = out
            .mapping
            .entries
            .iter()
            .map(|e| e.value_type.as_str())
            .collect();
        assert!(by_type.contains("ENTITY") && by_type.contains("EMAIL"));
    }

    #[test]
    fn entity_guess_respects_bracket_reservation() {
        // A proper noun inside a bracket label is a blank for Claude, not a value — the
        // reservation drops the guess even though the heuristic matches it.
        let out = censor(
            &paragraph_doc("[Buyer Name] shall sign"),
            &CensorOptions::default(),
        );
        assert!(out.mapping.entries.is_empty());
        assert_eq!(censored_text(&out), "[Buyer Name] shall sign");
    }

    #[test]
    fn bracket_interior_is_not_censored() {
        // A bracket's interior is a variable label to send to Claude, never a value to
        // redact — even when it contains something a pattern would otherwise match.
        let out = censor(
            &paragraph_doc("Email [billing@acme.example] today"),
            &CensorOptions::default(),
        );
        assert_eq!(censored_text(&out), "Email [billing@acme.example] today");
        assert!(out.mapping.entries.is_empty());
    }

    #[test]
    fn value_outside_bracket_still_censored_while_label_preserved() {
        // A real value outside a bracket is censored; an identical-looking one inside a
        // bracket label is preserved.
        let out = censor(
            &paragraph_doc("Reach billing@acme.example or [billing@acme.example]."),
            &CensorOptions::default(),
        );
        let text = censored_text(&out);
        assert!(
            text.contains("[billing@acme.example]"),
            "bracket label preserved"
        );
        assert!(
            text.starts_with("Reach REDACTED_EMAIL_001"),
            "outside value censored"
        );
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].value, "billing@acme.example");
    }

    #[test]
    fn malformed_lone_bracket_line_is_still_censored() {
        // A lone '[' (no close) does not reserve its line; real values still get censored.
        let out = censor(
            &paragraph_doc("[ pay billing@acme.example now"),
            &CensorOptions::default(),
        );
        assert!(censored_text(&out).contains("REDACTED_EMAIL_001"));
    }

    #[test]
    fn combined_percent_is_censored_as_one_placeholder() {
        let out = censor(
            &paragraph_doc("overcharged by ten percent (10%) or more"),
            &CensorOptions::default(),
        );
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].value_type, "PERCENT");
        assert_eq!(out.mapping.entries[0].value, "ten percent (10%)");
        assert_eq!(
            censored_text(&out),
            "overcharged by REDACTED_PERCENT_001 or more"
        );
    }

    #[test]
    fn spelled_out_amount_is_censored_as_money() {
        let out = censor(
            &paragraph_doc("pay two thousand dollars on signing"),
            &CensorOptions::default(),
        );
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].value_type, "MONEY");
        assert_eq!(out.mapping.entries[0].value, "two thousand dollars");
    }

    #[test]
    fn no_sensitive_values_leaves_text_untouched() {
        let out = censor(
            &paragraph_doc("just ordinary words"),
            &CensorOptions::default(),
        );
        assert_eq!(censored_text(&out), "just ordinary words");
        assert!(out.mapping.entries.is_empty());
    }

    #[test]
    fn every_character_replaced_at_most_once() {
        // Overlapping candidates must not corrupt the output; the original value is gone.
        let out = censor(
            &paragraph_doc("acct GB82WEST12345698765432 done"),
            &CensorOptions::default(),
        );
        let censored = censored_text(&out);
        assert!(!censored.contains("12345698765432"));
        assert_eq!(censored.matches("REDACTED_").count(), 1);
    }

    // ── Interactive stage (plan → review → apply) ──────────────────────────────

    fn item(value: &str, ty: ValueType) -> ReviewItem {
        ReviewItem {
            value: value.into(),
            detected_type: ty,
            method: "regex:test".into(),
            occurrences: 1,
            shown_context: format!("ctx {value}"),
            block_context: format!("blk {value}"),
        }
    }

    fn confirm_dec(value: &str, ty: ValueType, label: &str) -> CensorDecision {
        CensorDecision {
            value: value.into(),
            detected_type: ty,
            verdict: Verdict::Confirm {
                final_type: label.into(),
            },
            reviewed: true,
        }
    }

    fn reject_dec(value: &str, ty: ValueType) -> CensorDecision {
        CensorDecision {
            value: value.into(),
            detected_type: ty,
            verdict: Verdict::Reject,
            reviewed: true,
        }
    }

    fn para_text(doc: &Document) -> &str {
        match &doc.blocks[0] {
            Block::Paragraph { text } => text,
            _ => panic!("expected a paragraph"),
        }
    }

    #[test]
    fn plan_review_lists_distinct_values_with_occurrences() {
        let doc = paragraph_doc("Email a@b.com then a@b.com again");
        let items = plan_review(&doc, &CensorOptions::default());
        assert_eq!(items.len(), 1, "the value is deduped to one review item");
        assert_eq!(items[0].value, "a@b.com");
        assert_eq!(items[0].detected_type, ValueType::Email);
        assert_eq!(items[0].occurrences, 2);
        assert_eq!(items[0].method, "regex:email");
        assert!(
            items[0].shown_context.contains("a@b.com"),
            "context captured"
        );
    }

    #[test]
    fn plan_review_skips_allowed_values() {
        let allow: std::collections::BTreeSet<String> =
            ["a@b.com".to_string()].into_iter().collect();
        let doc = paragraph_doc("Email a@b.com now");
        let items = plan_review(
            &doc,
            &CensorOptions {
                allow: Some(&allow),
                ..Default::default()
            },
        );
        assert!(items.is_empty(), "allow-listed value is auto-skipped");
    }

    #[test]
    fn apply_censors_only_confirmed_values() {
        let doc = paragraph_doc("Email a@b.com to Jane Doe.");
        let decisions = vec![
            confirm_dec("a@b.com", ValueType::Email, "EMAIL"),
            reject_dec("Jane Doe", ValueType::Entity),
        ];
        let out = apply(&doc, &decisions, &CensorOptions::default());
        let text = para_text(&out);
        assert!(
            text.contains("REDACTED_EMAIL_001"),
            "confirmed value censored"
        );
        assert!(
            text.contains("Jane Doe"),
            "rejected value left in the clear"
        );
        assert!(!text.contains("a@b.com"));
    }

    #[test]
    fn apply_uses_final_type_for_the_placeholder() {
        let doc = paragraph_doc("countersigned by Jane Doe");
        let decisions = vec![confirm_dec("Jane Doe", ValueType::Entity, "PERSON")];
        let out = apply(&doc, &decisions, &CensorOptions::default());
        assert!(
            para_text(&out).contains("REDACTED_PERSON_001"),
            "re-typed ENTITY→PERSON drives the placeholder label"
        );
    }

    #[test]
    fn apply_keeps_unreviewed_defaults_censored() {
        // An item auto-confirmed on quit (reviewed: false) is still censored for safety.
        let doc = paragraph_doc("Email a@b.com now");
        let decisions = vec![CensorDecision {
            value: "a@b.com".into(),
            detected_type: ValueType::Email,
            verdict: Verdict::Confirm {
                final_type: "EMAIL".into(),
            },
            reviewed: false,
        }];
        let out = apply(&doc, &decisions, &CensorOptions::default());
        assert!(para_text(&out).contains("REDACTED_EMAIL_001"));
    }

    #[test]
    fn decision_records_skip_unreviewed_and_map_schema_3_fields() {
        let items = vec![
            item("a@b.com", ValueType::Email),
            item("Jane", ValueType::Entity),
            item("X", ValueType::Person),
        ];
        let decisions = vec![
            confirm_dec("a@b.com", ValueType::Email, "EMAIL"),
            reject_dec("Jane", ValueType::Entity),
            CensorDecision {
                value: "X".into(),
                detected_type: ValueType::Person,
                verdict: Verdict::Confirm {
                    final_type: "PERSON".into(),
                },
                reviewed: false,
            },
        ];
        let records = decision_records(&items, &decisions, "c.txt", 7);
        assert_eq!(records.len(), 2, "the unreviewed item is not logged");
        assert_eq!(records[0].schema, decision_schema());
        assert_eq!(records[0].verdict, "confirm");
        assert_eq!(records[0].final_type.as_deref(), Some("EMAIL"));
        assert_eq!(records[0].detected_type, "EMAIL");
        assert_eq!(records[0].occurrences, 1);
        assert_eq!(records[1].verdict, "reject");
        assert_eq!(records[1].final_type, None, "reject has no final type");
        assert_eq!(records[1].detected_type, "ENTITY");
    }

    #[test]
    fn update_store_allows_rejects_denies_confirms_and_conflicts() {
        let mut store = LearnedStore::default();
        // reject → the value is safe to leave in the clear next run.
        update_store(
            &mut store,
            &[item("Reach", ValueType::Entity)],
            &[reject_dec("Reach", ValueType::Entity)],
        );
        assert!(store.allowed_values().contains("Reach"));
        // confirm → kept censored (never allow-listed).
        update_store(
            &mut store,
            &[item("Acme", ValueType::Org)],
            &[confirm_dec("Acme", ValueType::Org, "ORG")],
        );
        assert!(!store.allowed_values().contains("Acme"));
        // seen both ways across runs → conflicted → stays censored.
        let maybe = [item("Maybe", ValueType::Entity)];
        update_store(
            &mut store,
            &maybe,
            &[reject_dec("Maybe", ValueType::Entity)],
        );
        update_store(
            &mut store,
            &maybe,
            &[confirm_dec("Maybe", ValueType::Entity, "ENTITY")],
        );
        assert!(!store.allowed_values().contains("Maybe"));
    }
}
