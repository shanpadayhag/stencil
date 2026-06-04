//! Censoring: replace sensitive values with `REDACTED_<TYPE>_<NNN>` placeholders.
//!
//! The pipeline gathers candidate spans from [`patterns`] (structured values) and
//! [`names`] (party list + opt-in heuristic), resolves overlaps by a **fixed
//! precedence** with **longest-match-wins** so every character is replaced at most
//! once, dedups by exact value (one value → one stable placeholder), and records a
//! [`Mapping`] for `restore`.

pub mod names;
pub mod patterns;

use std::collections::HashMap;

use crate::detect::paired_spans;
use crate::model::{Block, Cell, Document, MAPPING_VERSION, Mapping, MappingEntry};

pub use names::{IgnoreList, PartyList};

/// The category of a censored value. Determines the placeholder prefix and the
/// detector precedence used to resolve overlaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    /// A person's name (party list or heuristic).
    Person,
    /// An organization's name (party list or heuristic).
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
}

impl ValueType {
    /// The uppercase label used in placeholders (`REDACTED_<LABEL>_<NNN>`) and the
    /// `mapping.json` `type` field.
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
        }
    }
}

/// How a candidate value was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DetectSource {
    /// Matched an explicit party-name list entry (authoritative).
    PartyList,
    /// Matched the opt-in capitalized-sequence heuristic (flagged).
    Heuristic,
    /// Matched a structured-value regex.
    Pattern,
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
    /// Whether to enable the opt-in capitalized-sequence name heuristic.
    pub guess_names: bool,
    /// Phrases the heuristic must never treat as names (defined terms), if any.
    pub ignore: Option<&'a IgnoreList>,
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
    if options.guess_names {
        candidates.extend(names::find_heuristic_names(text, options.ignore));
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

/// Precedence rank (lower wins): party names, then the structured order from the
/// design, then the flagged heuristic names last.
fn precedence(candidate: &Candidate) -> u8 {
    match candidate.source {
        DetectSource::PartyList => 0,
        DetectSource::Heuristic => 10,
        DetectSource::Pattern => match candidate.value_type {
            ValueType::Iban => 1,
            ValueType::Card => 2,
            ValueType::Account => 3,
            ValueType::Phone => 4,
            ValueType::Date => 5,
            ValueType::Money => 6,
            ValueType::Percent => 7,
            ValueType::Email => 8,
            ValueType::Person | ValueType::Org => 9,
        },
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

/// The `mapping.json` `method` label for a detection source.
fn method_label(source: DetectSource, value_type: ValueType) -> String {
    match source {
        DetectSource::PartyList => "party-list".to_string(),
        DetectSource::Heuristic => "heuristic".to_string(),
        DetectSource::Pattern => format!("regex:{}", value_type.label().to_ascii_lowercase()),
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
            guess_names: false,
            ..Default::default()
        };
        let out = censor(&paragraph_doc("signed by Wonka Corporation"), &options);
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].value_type, "ORG");
        assert_eq!(out.mapping.entries[0].method, "party-list");
        assert!(censored_text(&out).contains("REDACTED_ORG_001"));
    }

    #[test]
    fn heuristic_only_runs_when_enabled_and_is_flagged() {
        let text = "Signed by Jane Doe today";

        let off = censor(&paragraph_doc(text), &CensorOptions::default());
        assert!(off.mapping.entries.is_empty());

        let on = censor(
            &paragraph_doc(text),
            &CensorOptions {
                parties: None,
                guess_names: true,
                ..Default::default()
            },
        );
        assert_eq!(on.mapping.entries.len(), 1);
        assert_eq!(on.mapping.entries[0].method, "heuristic");
        assert_eq!(on.mapping.entries[0].value, "Jane Doe");
    }

    #[test]
    fn party_list_beats_overlapping_heuristic() {
        // "Jane Doe" is both a party-list entry and a heuristic match; party-list wins,
        // so the entry's method is authoritative, not heuristic.
        let list = PartyList::parse("Jane Doe").expect("parse");
        let out = censor(
            &paragraph_doc("Signed by Jane Doe"),
            &CensorOptions {
                parties: Some(&list),
                guess_names: true,
                ..Default::default()
            },
        );
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].method, "party-list");
    }

    #[test]
    fn bracket_label_is_not_censored_even_with_guess_names() {
        // "[Client Name]" is a variable label to send to Claude, not a value to redact.
        let out = censor(
            &paragraph_doc("Signed by [Client Name] today"),
            &CensorOptions {
                parties: None,
                guess_names: true,
                ..Default::default()
            },
        );
        assert_eq!(censored_text(&out), "Signed by [Client Name] today");
        assert!(out.mapping.entries.is_empty());
    }

    #[test]
    fn value_outside_bracket_still_censored_while_label_preserved() {
        // Same heuristic name appears both inside a bracket (keep) and outside (censor).
        let out = censor(
            &paragraph_doc("Jane Doe signs as [Client Name]."),
            &CensorOptions {
                parties: None,
                guess_names: true,
                ..Default::default()
            },
        );
        let text = censored_text(&out);
        assert!(text.contains("[Client Name]"), "bracket label preserved");
        assert!(
            text.starts_with("REDACTED_PERSON_001"),
            "outside name censored"
        );
        assert_eq!(out.mapping.entries.len(), 1);
        assert_eq!(out.mapping.entries[0].value, "Jane Doe");
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
}
