//! Structured-value detectors: regex statics for email, phone, money, date, IBAN,
//! account, and card numbers, plus a Luhn check.
//!
//! Each detector reports candidate spans; it does **not** resolve overlaps. The
//! pipeline (task T7) applies precedence and longest-match-wins across all candidates.
//! The `regex` crate runs in guaranteed linear time, so these patterns cannot trigger
//! catastrophic backtracking.

use std::sync::LazyLock;

use regex::Regex;

use super::ValueType;

/// A candidate value found in some text. `start`/`end` are byte offsets into that text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PatternMatch {
    /// Byte offset of the match start.
    pub start: usize,
    /// Byte offset of the match end (exclusive).
    pub end: usize,
    /// The detected value category.
    pub value_type: ValueType,
}

macro_rules! lazy_regex {
    ($name:ident, $pattern:literal) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new($pattern).expect(concat!("static regex `", stringify!($name), "` is valid"))
        });
    };
}

lazy_regex!(EMAIL, r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}");
lazy_regex!(IBAN, r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{10,30}\b");
lazy_regex!(PHONE, r"\+?[0-9][0-9 ()\-.]{5,}[0-9]");
lazy_regex!(ACCOUNT, r"\b[0-9]{6,17}\b");
// Digit groups optionally separated by single spaces/dashes (13–19 digits validated later).
lazy_regex!(CARD_CANDIDATE, r"\b[0-9](?:[ \-]?[0-9]){12,18}\b");
lazy_regex!(
    MONEY,
    r"(?:[$£€]\s?[0-9][0-9,]*(?:\.[0-9]{1,2})?)|(?:[0-9][0-9,]*(?:\.[0-9]{1,2})?\s?(?i:usd|eur|gbp|dollars?|euros?|pounds?))"
);
lazy_regex!(
    DATE,
    r"(?:\b[0-9]{1,4}[/.\-][0-9]{1,2}[/.\-][0-9]{1,4}\b)|(?:\b[0-9]{1,2}\s+(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+[0-9]{2,4}\b)|(?:\b(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+[0-9]{1,2},?\s+[0-9]{2,4}\b)"
);

/// The "common range" number words: units, teens, tens, and the
/// hundred/thousand/million/billion scales. Rust's regex is leftmost-longest, so
/// `nineteen` wins over `nine` regardless of alternation order.
const NUMBER_WORD: &str = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)";

/// A spelled-out number: one or more [`NUMBER_WORD`]s joined by spaces, hyphens, or
/// `and` (e.g. `two hundred fifty thousand`, `twenty-five`).
fn number_phrase() -> String {
    format!(r"\b{NUMBER_WORD}(?:[ \t\-]+(?:and[ \t]+)?{NUMBER_WORD})*")
}

/// Spelled-out monetary amounts: a number phrase followed by a currency word. Numeric
/// amounts are handled by [`struct@MONEY`]; this catches `two thousand dollars`.
static MONEY_WORDS: LazyLock<Regex> = LazyLock::new(|| {
    let pattern = format!(
        r"(?i){}\s+(?:dollars?|euros?|pounds?|pesos?|cents?|usd|eur|gbp)\b",
        number_phrase()
    );
    Regex::new(&pattern).expect("static regex `MONEY_WORDS` is valid")
});

/// Percentages: numeric (`10%`, `10.5 %`), spelled out (`ten percent`), or the combined
/// `ten percent (10%)` form captured as a single span.
static PERCENT: LazyLock<Regex> = LazyLock::new(|| {
    let pattern = format!(
        r"(?i)(?:{}\s+per ?cent(?:\s*\([^)]*%\s*\))?)|(?:\b[0-9][0-9,]*(?:\.[0-9]+)?\s?%)",
        number_phrase()
    );
    Regex::new(&pattern).expect("static regex `PERCENT` is valid")
});

// ── French detectors (v7) ───────────────────────────────────────────────────
// These run alongside the English detectors on every block (detection is recall-first and not
// gated by the block's detected language). They add French money/percent/date forms; structured
// values (email, IBAN, Luhn cards) are already language-agnostic.

/// French base number words; hyphen/space/`et` joins in [`number_phrase_fr`] cover the compound
/// forms (`dix-sept`, `soixante-dix`, `quatre-vingts`, `deux mille`).
const NUMBER_WORD_FR: &str = r"(?:une?|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingts?|trente|quarante|cinquante|soixante|cents?|mille|millions?|milliards?)";

/// A spelled-out French number: one or more [`NUMBER_WORD_FR`] joined by spaces, hyphens, or `et`.
fn number_phrase_fr() -> String {
    format!(r"\b{NUMBER_WORD_FR}(?:[ \t\-]+(?:et[ \t]+)?{NUMBER_WORD_FR})*")
}

/// A French-formatted number: space/NBSP thousands separators and a comma decimal
/// (`1 000,50`, `10,5`, `42`). `\s` covers the non-breaking space French typography uses.
const FRENCH_NUMBER: &str = r"[0-9]{1,3}(?:\s[0-9]{3})*(?:,[0-9]+)?";

/// French monetary amounts: a French-formatted number or spelled-out phrase with a euro/dollar/
/// franc marker — `1 000,50 €`, `€ 1 000`, `deux mille euros`.
static MONEY_FR: LazyLock<Regex> = LazyLock::new(|| {
    let pattern = format!(
        r"(?i)(?:{FRENCH_NUMBER}\s?(?:€|euros?|dollars?|francs?|centimes?))|(?:€\s?{FRENCH_NUMBER})|(?:{}\s+(?:euros?|dollars?|francs?|centimes?)\b)",
        number_phrase_fr()
    );
    Regex::new(&pattern).expect("static regex `MONEY_FR` is valid")
});

/// French percentages: spelled-out `dix pour cent` (incl. the `dix pour cent (10 %)` form), or
/// numeric with the French space-before-`%` and comma decimal (`10 %`, `10,5 %`).
static PERCENT_FR: LazyLock<Regex> = LazyLock::new(|| {
    let pattern = format!(
        r"(?i)(?:{}\s+pour\s?cent(?:\s*\([^)]*%\s*\))?)|(?:{FRENCH_NUMBER}\s?%)",
        number_phrase_fr()
    );
    Regex::new(&pattern).expect("static regex `PERCENT_FR` is valid")
});

/// French dates: `le 15 janvier 2024`, `1er février 2024`, `15 déc 2024` — accented and accentless
/// month spellings.
static DATE_FR: LazyLock<Regex> = LazyLock::new(|| {
    // The leading `le` is context, not part of the date, so it is not captured.
    Regex::new(
        r"(?i)\b[0-9]{1,2}(?:er)?\s+(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\.?\s+[0-9]{2,4}\b",
    )
    .expect("static regex `DATE_FR` is valid")
});

/// Lower and upper bounds on the digit count of a plausible phone number.
const PHONE_MIN_DIGITS: usize = 7;
const PHONE_MAX_DIGITS: usize = 15;

/// Find every structured-value candidate in `text`.
///
/// Candidates may overlap and are returned in no particular order; the caller resolves
/// conflicts by precedence and length.
///
/// ```
/// use stencil::censor::ValueType;
/// use stencil::censor::patterns::find_candidates;
///
/// let hits = find_candidates("Email me at a@b.com");
/// assert!(hits.iter().any(|m| m.value_type == ValueType::Email));
/// ```
pub fn find_candidates(text: &str) -> Vec<PatternMatch> {
    let mut matches = Vec::new();

    push_all(&mut matches, &EMAIL, text, ValueType::Email);
    push_all(&mut matches, &IBAN, text, ValueType::Iban);
    push_all(&mut matches, &MONEY, text, ValueType::Money);
    push_all(&mut matches, &MONEY_WORDS, text, ValueType::Money);
    push_all(&mut matches, &PERCENT, text, ValueType::Percent);
    push_all(&mut matches, &DATE, text, ValueType::Date);
    push_all(&mut matches, &ACCOUNT, text, ValueType::Account);

    // French detectors, alongside the English ones.
    push_all(&mut matches, &MONEY_FR, text, ValueType::Money);
    push_all(&mut matches, &PERCENT_FR, text, ValueType::Percent);
    push_all(&mut matches, &DATE_FR, text, ValueType::Date);

    push_phones(&mut matches, text);
    push_cards(&mut matches, text);

    matches
}

/// Push every match of `regex` as `value_type`.
fn push_all(matches: &mut Vec<PatternMatch>, regex: &Regex, text: &str, value_type: ValueType) {
    for found in regex.find_iter(text) {
        matches.push(PatternMatch {
            start: found.start(),
            end: found.end(),
            value_type,
        });
    }
}

/// Push phone matches whose digit count is within the plausible range.
fn push_phones(matches: &mut Vec<PatternMatch>, text: &str) {
    for found in PHONE.find_iter(text) {
        let digits = count_digits(found.as_str());
        if (PHONE_MIN_DIGITS..=PHONE_MAX_DIGITS).contains(&digits) {
            matches.push(PatternMatch {
                start: found.start(),
                end: found.end(),
                value_type: ValueType::Phone,
            });
        }
    }
}

/// Push card matches with 13–19 digits that pass the Luhn check.
fn push_cards(matches: &mut Vec<PatternMatch>, text: &str) {
    for found in CARD_CANDIDATE.find_iter(text) {
        let digits = count_digits(found.as_str());
        if (13..=19).contains(&digits) && passes_luhn(found.as_str()) {
            matches.push(PatternMatch {
                start: found.start(),
                end: found.end(),
                value_type: ValueType::Card,
            });
        }
    }
}

/// Count the ASCII digits in `s`.
fn count_digits(s: &str) -> usize {
    s.bytes().filter(u8::is_ascii_digit).count()
}

/// The Luhn checksum, ignoring any non-digit separators. Empty input is not valid.
///
/// ```
/// use stencil::censor::patterns::passes_luhn;
///
/// assert!(passes_luhn("4111 1111 1111 1111")); // valid test Visa
/// assert!(!passes_luhn("4111 1111 1111 1121")); // tampered
/// ```
pub fn passes_luhn(s: &str) -> bool {
    let mut sum = 0u32;
    let mut double = false;
    let mut count = 0usize;

    for byte in s.bytes().rev() {
        if !byte.is_ascii_digit() {
            continue;
        }
        count += 1;
        let mut value = u32::from(byte - b'0');
        if double {
            value *= 2;
            if value > 9 {
                value -= 9;
            }
        }
        sum += value;
        double = !double;
    }

    count > 0 && sum.is_multiple_of(10)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn types_in(text: &str) -> Vec<ValueType> {
        find_candidates(text)
            .into_iter()
            .map(|m| m.value_type)
            .collect()
    }

    fn matched(text: &str, want: ValueType) -> Vec<&str> {
        find_candidates(text)
            .into_iter()
            .filter(|m| m.value_type == want)
            .map(|m| &text[m.start..m.end])
            .collect()
    }

    #[test]
    fn email_matches() {
        assert_eq!(
            matched("contact jane.doe+x@example.co.uk now", ValueType::Email),
            vec!["jane.doe+x@example.co.uk"]
        );
    }

    #[test]
    fn iban_matches() {
        assert_eq!(
            matched("IBAN GB82WEST12345698765432 here", ValueType::Iban),
            vec!["GB82WEST12345698765432"]
        );
    }

    #[test]
    fn money_matches_symbol_and_code() {
        assert!(matched("a deposit of $1,200.50 today", ValueType::Money).contains(&"$1,200.50"));
        assert!(matched("about 2000 USD owed", ValueType::Money).contains(&"2000 USD"));
    }

    #[test]
    fn percent_matches_numeric() {
        assert!(matched("a fee of 10% applies", ValueType::Percent).contains(&"10%"));
        assert!(matched("up to 10.5 % annually", ValueType::Percent).contains(&"10.5 %"));
    }

    #[test]
    fn percent_matches_spelled_out() {
        assert!(
            matched("overcharged by ten percent or more", ValueType::Percent)
                .contains(&"ten percent")
        );
    }

    #[test]
    fn percent_matches_combined_word_and_parenthetical_as_one_span() {
        // The English detector keeps "ten percent (10%)" whole — not split into "ten percent".
        // (The French numeric detector independently also matches the inner "10%"; the pipeline's
        // overlap resolution keeps the longer combined span.)
        let percents = matched(
            "Tenant has been overcharged by ten percent (10%) or more",
            ValueType::Percent,
        );
        assert!(percents.contains(&"ten percent (10%)"));
        assert!(
            !percents.contains(&"ten percent"),
            "the word form is not split off"
        );
    }

    #[test]
    fn longer_number_word_wins_over_prefix() {
        // "nineteen" must not be split into "nine".
        assert_eq!(
            matched("a nineteen percent share", ValueType::Percent),
            vec!["nineteen percent"]
        );
    }

    #[test]
    fn money_matches_spelled_out_amount() {
        assert!(
            matched("pay two thousand dollars now", ValueType::Money)
                .contains(&"two thousand dollars")
        );
        assert!(
            matched("a deposit of five hundred pesos", ValueType::Money)
                .contains(&"five hundred pesos")
        );
    }

    #[test]
    fn number_word_inside_an_ordinary_word_is_not_matched() {
        // "Tenant" contains "ten" but is not followed by percent/currency.
        assert!(matched("Tenant pays the rent", ValueType::Percent).is_empty());
        assert!(matched("Tenant pays the rent", ValueType::Money).is_empty());
    }

    #[test]
    fn date_matches_numeric_and_named() {
        assert!(matched("due 2026-06-04 sharp", ValueType::Date).contains(&"2026-06-04"));
        assert!(matched("on 4 June 2026", ValueType::Date).contains(&"4 June 2026"));
        assert!(matched("by Jan 3, 2027", ValueType::Date).contains(&"Jan 3, 2027"));
    }

    #[test]
    fn phone_matches_with_separators() {
        assert!(
            matched("call +1 (415) 555-0132 please", ValueType::Phone)
                .iter()
                .any(|m| m.contains("415"))
        );
    }

    #[test]
    fn phone_rejects_too_few_digits() {
        // Only 5 digits — below the plausible phone range.
        assert!(matched("code 12-345 only", ValueType::Phone).is_empty());
    }

    #[test]
    fn valid_card_detected_via_luhn() {
        let cards = matched("card 4111 1111 1111 1111 on file", ValueType::Card);
        assert_eq!(cards, vec!["4111 1111 1111 1111"]);
    }

    #[test]
    fn invalid_card_not_detected_as_card() {
        // Luhn-invalid 16-digit run: not a Card (may still be an Account candidate).
        assert!(matched("num 1234 5678 9012 3456 here", ValueType::Card).is_empty());
    }

    #[test]
    fn account_matches_digit_run() {
        assert!(matched("acct 001234567 ref", ValueType::Account).contains(&"001234567"));
    }

    #[test]
    fn luhn_known_vectors() {
        assert!(passes_luhn("79927398713"));
        assert!(!passes_luhn("79927398714"));
        assert!(!passes_luhn(""));
        assert!(!passes_luhn("abc"));
    }

    #[test]
    fn mixed_text_yields_multiple_types() {
        let types = types_in("pay $500 to a@b.com by 2026-01-01");
        assert!(types.contains(&ValueType::Money));
        assert!(types.contains(&ValueType::Email));
        assert!(types.contains(&ValueType::Date));
    }

    #[test]
    fn plain_text_has_no_candidates() {
        assert!(find_candidates("just some ordinary words here").is_empty());
    }

    // ── French detectors ─────────────────────────────────────────────────────

    #[test]
    fn french_money_numeric_and_spelled_out() {
        assert!(matched("un acompte de 1 000,50 €", ValueType::Money).contains(&"1 000,50 €"));
        assert!(matched("payer deux mille euros", ValueType::Money).contains(&"deux mille euros"));
        assert!(matched("environ € 500", ValueType::Money).contains(&"€ 500"));
    }

    #[test]
    fn french_percent_numeric_and_spelled_out() {
        assert!(matched("un taux de 10 %", ValueType::Percent).contains(&"10 %"));
        assert!(matched("jusqu'à 10,5 %", ValueType::Percent).contains(&"10,5 %"));
        assert!(matched("majoré de dix pour cent", ValueType::Percent).contains(&"dix pour cent"));
    }

    #[test]
    fn french_percent_combined_word_and_parenthetical() {
        assert!(
            matched(
                "surfacturé de dix pour cent (10 %) ou plus",
                ValueType::Percent
            )
            .contains(&"dix pour cent (10 %)")
        );
    }

    #[test]
    fn french_dates_with_accented_months() {
        assert!(
            matched("le 15 janvier 2024 au plus tard", ValueType::Date)
                .contains(&"15 janvier 2024")
        );
        assert!(
            matched("signé le 1er février 2024", ValueType::Date).contains(&"1er février 2024")
        );
        assert!(matched("avant le 3 décembre 2024", ValueType::Date).contains(&"3 décembre 2024"));
    }

    #[test]
    fn english_behaviour_unchanged_by_french_detectors() {
        // The French additions must not disturb the English matches.
        assert!(matched("a fee of 10% applies", ValueType::Percent).contains(&"10%"));
        assert!(
            matched("pay two thousand dollars now", ValueType::Money)
                .contains(&"two thousand dollars")
        );
        assert!(matched("due 2026-06-04 sharp", ValueType::Date).contains(&"2026-06-04"));
        assert!(find_candidates("just some ordinary words here").is_empty());
    }
}
