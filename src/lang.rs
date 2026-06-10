//! Per-block language detection — a v7 training feature so the future models can learn across
//! document languages (e.g. bilingual EN/FR contracts).
//!
//! Detection is offline via [`whatlang`]. Because short text detects unreliably, a block that is
//! too short or low-confidence falls back to the document's **dominant** language (the most common
//! confident detection). A `--lang` override forces every block to a given code. The detected
//! `lang` is recorded as a feature; it does **not** gate which detectors fire (v7 runs all of them).

use std::collections::BTreeSet;
use std::collections::HashMap;

use whatlang::{Lang, detect};

/// Minimum confidence for a detection to be trusted on its own (0..1).
const MIN_CONFIDENCE: f64 = 0.5;
/// Texts shorter than this many characters are treated as too short to detect.
const MIN_CHARS: usize = 25;
/// The code used when nothing can be determined (no confident block in the document).
pub const UNKNOWN: &str = "und";

/// A block's detected language: an ISO code and a 0..1 confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockLang {
    /// ISO code (`en`/`fr` for the languages we care about, ISO 639-3 otherwise, or `und`).
    pub lang: String,
    /// Detection confidence in 0..=1 (`1.0` for an override, `0.0` for a fallback).
    pub confidence: f32,
}

impl BlockLang {
    /// A block tagged by a `--lang` override (asserted, so maximally confident).
    fn forced(code: &str) -> Self {
        Self {
            lang: code.to_string(),
            confidence: 1.0,
        }
    }

    /// A block that could not be detected, fell back to the document language `code`.
    fn fallback(code: &str) -> Self {
        Self {
            lang: code.to_string(),
            confidence: 0.0,
        }
    }
}

/// Tag each text with a language.
///
/// With `override_lang = Some(code)` every block is forced to that code. Otherwise each block is
/// detected; a short or low-confidence block falls back to the document's dominant language (or
/// [`UNKNOWN`] when no block could be detected).
pub fn tag_texts(texts: &[&str], override_lang: Option<&str>) -> Vec<BlockLang> {
    if let Some(code) = override_lang {
        return texts.iter().map(|_| BlockLang::forced(code)).collect();
    }
    let detected: Vec<Option<BlockLang>> = texts.iter().map(|text| detect_one(text)).collect();
    let dominant = dominant_lang(&detected);
    detected
        .into_iter()
        .map(|block| block.unwrap_or_else(|| BlockLang::fallback(&dominant)))
        .collect()
}

/// The set of distinct languages across `tags` — the document's languages (a bilingual doc shows
/// more than one).
pub fn document_languages(tags: &[BlockLang]) -> BTreeSet<String> {
    tags.iter().map(|tag| tag.lang.clone()).collect()
}

/// Detect one text's language, if it is long enough and the result is confident.
fn detect_one(text: &str) -> Option<BlockLang> {
    if text.chars().count() < MIN_CHARS {
        return None;
    }
    let info = detect(text)?;
    if !info.is_reliable() || info.confidence() < MIN_CONFIDENCE {
        return None;
    }
    Some(BlockLang {
        lang: lang_code(info.lang()).to_string(),
        confidence: info.confidence() as f32,
    })
}

/// The most common confident language among `detections`, ties broken by the smaller code; falls
/// back to [`UNKNOWN`] when nothing was detected.
fn dominant_lang(detections: &[Option<BlockLang>]) -> String {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for block in detections.iter().flatten() {
        *counts.entry(block.lang.as_str()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)))
        .map_or_else(|| UNKNOWN.to_string(), |(lang, _)| lang.to_string())
}

/// A two-letter code for the languages v7 cares about; ISO 639-3 otherwise.
fn lang_code(lang: Lang) -> &'static str {
    match lang {
        Lang::Eng => "en",
        Lang::Fra => "fr",
        other => other.code(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENGLISH: &str = "This agreement is governed by the laws of the State of New York and \
        the parties agree to binding arbitration of any dispute.";
    const FRENCH: &str = "Le présent contrat est régi par les lois de la province de Québec et \
        les parties conviennent de recourir à l'arbitrage pour tout différend.";

    #[test]
    fn detects_english_and_french() {
        let tags = tag_texts(&[ENGLISH, FRENCH], None);
        assert_eq!(tags[0].lang, "en");
        assert_eq!(tags[1].lang, "fr");
        assert!(tags[0].confidence > 0.0 && tags[1].confidence > 0.0);
    }

    #[test]
    fn bilingual_document_lists_both_languages() {
        let tags = tag_texts(&[ENGLISH, FRENCH], None);
        let langs = document_languages(&tags);
        assert!(langs.contains("en") && langs.contains("fr"));
        assert_eq!(langs.len(), 2);
    }

    #[test]
    fn short_block_falls_back_to_document_dominant() {
        // A long English block sets the dominant language; the short block can't be detected and
        // inherits it (with zero confidence, marking it a fallback).
        let tags = tag_texts(&[ENGLISH, "Oui."], None);
        assert_eq!(tags[0].lang, "en");
        assert_eq!(tags[1].lang, "en", "short block falls back to dominant");
        assert_eq!(tags[1].confidence, 0.0, "fallback is marked low-confidence");
    }

    #[test]
    fn override_forces_every_block() {
        let tags = tag_texts(&[ENGLISH, FRENCH, "x"], Some("fr"));
        assert!(
            tags.iter()
                .all(|tag| tag.lang == "fr" && tag.confidence == 1.0)
        );
    }

    #[test]
    fn unknown_when_nothing_detectable() {
        let tags = tag_texts(&["a", "b"], None);
        assert!(tags.iter().all(|tag| tag.lang == UNKNOWN));
    }
}
