//! Name censoring: the required party-name list, always replaced and matched
//! case-insensitively on word boundaries.
//!
//! Each name's type (PERSON vs ORG) is decided by a simple rule: a name containing a
//! known company suffix (Inc, LLC, Ltd, …) is an organization, otherwise a person.

use std::fs;
use std::sync::LazyLock;

use anyhow::{Context, Result};
use regex::Regex;

use super::{Candidate, DetectSource, ValueType};

/// Matches a sequence of capitalized words (`Jane`, `Jane Doe`, `Acme Holdings`). Requires
/// each word to be an uppercase letter followed by ≥1 lowercase letter, so all-caps acronyms
/// (`NASA`), single letters, and codes (`GB82WEST…`) are not mistaken for names.
static NAME_SEQUENCE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)*").expect("valid name regex"));

/// Common function words and contract sentence-starters. A *single* capitalized word that is
/// one of these is skipped (it is almost always capitalized by position, not a proper noun);
/// multi-word sequences are always kept. This trims the worst noise while staying recall-first.
const STOPWORDS: &[&str] = &[
    "the", "a", "an", "this", "that", "these", "those", "it", "we", "you", "they", "he", "she",
    "i", "if", "in", "on", "at", "to", "of", "for", "and", "or", "but", "by", "with", "from", "as",
    "per", "pay", "email", "reach", "contact", "signed", "sign", "deliver", "write", "invoice",
    "note", "see", "please", "dear", "subject", "re", "no", "yes", "all", "any", "each", "such",
];

/// One capitalized word within a matched sequence, used to trim stopword edges.
static NAME_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Z][a-z]+").expect("valid word regex"));

/// Recall-first proper-noun guess: every capitalized word sequence is a candidate `ENTITY`
/// (subtype unknown — the reviewer confirms or re-types it). Leading/trailing lone stopwords are
/// trimmed (so "The Buyer" / "Pay Acme" become "Buyer" / "Acme"), and an all-stopword sequence is
/// dropped. Over-detection is intentional; rejected guesses become negative training data.
pub(crate) fn guess_entities(text: &str) -> Vec<Candidate> {
    NAME_SEQUENCE
        .find_iter(text)
        .filter_map(|m| trim_stopwords(m.as_str(), m.start()))
        .map(|(start, end)| Candidate {
            start,
            end,
            value_type: ValueType::Entity,
            source: DetectSource::Heuristic,
        })
        .collect()
}

/// Trim leading and trailing stopword words from a capitalized sequence, returning the byte
/// range (offset by `base`) of the proper-noun core, or `None` if only stopwords remain.
fn trim_stopwords(span: &str, base: usize) -> Option<(usize, usize)> {
    let words: Vec<(usize, usize)> = NAME_WORD
        .find_iter(span)
        .map(|m| (m.start(), m.end()))
        .collect();
    let mut lo = 0;
    let mut hi = words.len();
    while lo < hi && is_stopword(&span[words[lo].0..words[lo].1]) {
        lo += 1;
    }
    while hi > lo && is_stopword(&span[words[hi - 1].0..words[hi - 1].1]) {
        hi -= 1;
    }
    (lo < hi).then(|| (base + words[lo].0, base + words[hi - 1].1))
}

/// Whether `word` is a common function word / sentence-starter (case-insensitive).
fn is_stopword(word: &str) -> bool {
    STOPWORDS.iter().any(|stop| word.eq_ignore_ascii_case(stop))
}

/// Tokens that mark a name as an organization rather than a person.
const ORG_SUFFIXES: &[&str] = &[
    "inc",
    "incorporated",
    "llc",
    "llp",
    "lp",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "gmbh",
    "plc",
    "ag",
    "sa",
    "nv",
    "bv",
];

/// A parsed list of party names to always censor.
#[derive(Debug)]
pub struct PartyList {
    entries: Vec<PartyEntry>,
}

#[derive(Debug)]
struct PartyEntry {
    matcher: Regex,
    value_type: ValueType,
}

impl PartyList {
    /// Parse a `--parties` specification: either an inline comma-separated list, or
    /// `@path` to read names from a file (one per line and/or comma-separated).
    ///
    /// # Errors
    /// Returns an error if a `@file` cannot be read or a name cannot be compiled.
    pub fn parse(spec: &str) -> Result<Self> {
        let raw = if let Some(path) = spec.strip_prefix('@') {
            fs::read_to_string(path)
                .with_context(|| format!("failed to read parties file `{path}`"))?
        } else {
            spec.to_string()
        };

        let mut entries = Vec::new();
        for name in raw
            .split([',', '\n'])
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            let matcher = Regex::new(&format!(r"(?i)\b{}\b", regex::escape(name)))
                .with_context(|| format!("failed to build matcher for party `{name}`"))?;
            entries.push(PartyEntry {
                matcher,
                value_type: classify_name(name),
            });
        }
        Ok(Self { entries })
    }

    /// Whether the list is empty (no usable names were provided).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find every party-name occurrence in `text`.
    pub(crate) fn find(&self, text: &str) -> Vec<Candidate> {
        let mut found = Vec::new();
        for entry in &self.entries {
            for m in entry.matcher.find_iter(text) {
                found.push(Candidate {
                    start: m.start(),
                    end: m.end(),
                    value_type: entry.value_type,
                    source: DetectSource::PartyList,
                });
            }
        }
        found
    }
}

/// Classify a name as [`ValueType::Org`] if it contains a company suffix, else
/// [`ValueType::Person`].
fn classify_name(name: &str) -> ValueType {
    let is_org = name
        .split_whitespace()
        .map(|word| word.trim_matches(|c: char| !c.is_ascii_alphanumeric()))
        .any(|word| {
            ORG_SUFFIXES
                .iter()
                .any(|suffix| word.eq_ignore_ascii_case(suffix))
        });
    if is_org {
        ValueType::Org
    } else {
        ValueType::Person
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_person_and_org() {
        assert_eq!(classify_name("Jane Doe"), ValueType::Person);
        assert_eq!(classify_name("Acme Corporation"), ValueType::Org);
        assert_eq!(classify_name("Globex LLC"), ValueType::Org);
        assert_eq!(classify_name("John Smith"), ValueType::Person);
    }

    #[test]
    fn party_list_matches_case_insensitively_on_word_boundaries() {
        let list = PartyList::parse("Acme,Jane Doe").expect("parse");
        let text = "ACME hired jane doe; Acme paid.";
        let hits = list.find(text);

        let spans: Vec<&str> = hits.iter().map(|c| &text[c.start..c.end]).collect();
        assert!(spans.contains(&"ACME"));
        assert!(spans.contains(&"jane doe"));
        assert!(spans.contains(&"Acme"));
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn party_list_does_not_match_substring() {
        let list = PartyList::parse("Co").expect("parse");
        // "Co" must not match inside "Company".
        assert!(list.find("the Company agrees").is_empty());
    }

    #[test]
    fn party_entries_get_types() {
        let list = PartyList::parse("Jane Doe, Acme LLC").expect("parse");
        let types: Vec<_> = list.entries.iter().map(|e| e.value_type).collect();
        assert!(types.contains(&ValueType::Person));
        assert!(types.contains(&ValueType::Org));
    }

    #[test]
    fn empty_spec_is_empty_list() {
        assert!(PartyList::parse("   ,  ,").expect("parse").is_empty());
    }

    /// The text spans guessed as entities, in order.
    fn guessed(text: &str) -> Vec<&str> {
        guess_entities(text)
            .iter()
            .map(|c| &text[c.start..c.end])
            .collect()
    }

    #[test]
    fn guesses_multiword_and_singleword_proper_nouns() {
        let hits = guessed("Acme Holdings hired Jane in Paris");
        assert_eq!(hits, vec!["Acme Holdings", "Jane", "Paris"]);
        // All guesses are the neutral ENTITY from the heuristic source.
        for candidate in guess_entities("Acme Holdings hired Jane in Paris") {
            assert_eq!(candidate.value_type, ValueType::Entity);
            assert_eq!(candidate.source, DetectSource::Heuristic);
        }
    }

    #[test]
    fn single_stopwords_are_skipped_but_real_nouns_kept() {
        // "The"/"Pay" are lone stopwords (skipped); "Buyer" and "Acme" are kept.
        let hits = guessed("The Buyer shall Pay Acme");
        assert_eq!(hits, vec!["Buyer", "Acme"]);
    }

    #[test]
    fn acronyms_and_codes_are_not_guessed() {
        // All-caps and alphanumeric codes are not name-like, so they are never flagged here.
        assert!(guessed("NASA and IBM signed").is_empty());
        assert!(guessed("ref GB82WEST12345698765432").is_empty());
    }
}
