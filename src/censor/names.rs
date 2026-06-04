//! Name censoring: the required party-name list, always replaced and matched
//! case-insensitively on word boundaries.
//!
//! Each name's type (PERSON vs ORG) is decided by a simple rule: a name containing a
//! known company suffix (Inc, LLC, Ltd, …) is an organization, otherwise a person.

use std::fs;

use anyhow::{Context, Result};
use regex::Regex;

use super::{Candidate, DetectSource, ValueType};

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
}
