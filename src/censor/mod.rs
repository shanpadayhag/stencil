//! Censoring: replace sensitive values with `REDACTED_<TYPE>_<NNN>` placeholders.
//!
//! The pipeline gathers candidate spans from [`patterns`] (structured values) and
//! [`names`] (the party list), resolves overlaps by a **fixed
//! precedence** with **longest-match-wins** so every character is replaced at most
//! once, and dedups by exact value (one value → one stable placeholder).

pub mod edit;
pub mod names;
pub mod patterns;
pub mod review;

use std::collections::{BTreeSet, HashMap};

use crate::detect::paired_spans;
use crate::learn::{
    DecisionRecord, LearnedStore, Prediction, block_window_at, decision_schema, sentence_window_at,
};
use crate::model::{
    Block, BlockKind, Cell, CensorNeighbors, Document, MAPPING_VERSION, Mapping, MappingEntry,
    Occurrence,
};

pub use names::PartyList;

/// Identifies one text field of a document: the block index, and `(row, col)` for a table cell.
type FieldKey = (usize, Option<(usize, usize)>);

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
///
/// All occurrences of one value are grouped here (deduped by exact string). The default review
/// decides the whole group at once; splitting a group (v7) decides each [`Occurrence`] on its
/// own. The group is never empty — it is built from at least one detected occurrence.
#[derive(Debug, Clone)]
pub struct ReviewItem {
    /// The real value to decide on.
    pub value: String,
    /// The detector's guessed type (possibly the neutral `ENTITY`).
    pub detected_type: ValueType,
    /// How it was detected (`party-list` / `regex:<kind>` / `heuristic`).
    pub method: String,
    /// Every occurrence of the value across the document, in first-seen order.
    pub occurrences: Vec<Occurrence>,
}

impl ReviewItem {
    /// How many times the value occurs across the document.
    pub fn occurrence_count(&self) -> usize {
        self.occurrences.len()
    }

    /// The distinct block kinds the value appears in — drives the mixed-context split hint (v7).
    pub fn block_kinds(&self) -> BTreeSet<BlockKind> {
        self.occurrences.iter().map(|o| o.block_kind).collect()
    }

    /// The first occurrence's sentence window — what a whole-group review shows and records.
    pub fn first_shown_context(&self) -> &str {
        self.occurrences
            .first()
            .map_or("", |o| o.shown_context.as_str())
    }

    /// The first occurrence's paragraph window — the whole-group `block_context`.
    pub fn first_block_context(&self) -> &str {
        self.occurrences
            .first()
            .map_or("", |o| o.block_context.as_str())
    }

    /// The first occurrence's neighbor context — what a whole-group decision records (v10).
    pub fn first_neighbors(&self) -> CensorNeighbors {
        self.occurrences
            .first()
            .map(|o| o.neighbors.clone())
            .unwrap_or_default()
    }
}

/// The reviewer's verdict for a value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Keep censored; the placeholder is typed with this (possibly re-typed) label.
    Confirm { final_type: String },
    /// A false positive — leave the value in the clear.
    Reject,
}

/// What a decision covers: a whole value-group, or a single occurrence carved out by a split.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionScope {
    /// Every occurrence of the value, decided together (the default review).
    Group {
        /// How many occurrences the group covers.
        occurrences: u32,
    },
    /// One occurrence (from a split), with its precise location for offset-based censoring.
    Occurrence(Box<Occurrence>),
}

/// One value's resolved outcome — self-contained, so logging no longer needs the parent
/// [`ReviewItem`] (a v7 add-missed-value decision has no parent item, and a split yields several
/// occurrence-scoped decisions from one item).
///
/// `reviewed` is false for items auto-confirmed when the user quits early: kept censored for
/// safety, but not a human label (excluded from log/store).
#[derive(Debug, Clone)]
pub struct CensorDecision {
    /// The value decided on (the edited string when [`span_edited`](CensorDecision::span_edited)).
    pub value: String,
    /// The detector's guessed type, carried for logging.
    pub detected_type: ValueType,
    /// How the value was detected (`party-list` / `regex:<kind>` / `manual` for added values).
    pub method: String,
    /// The verdict.
    pub verdict: Verdict,
    /// Whether a human explicitly decided this (vs auto-defaulted on quit).
    pub reviewed: bool,
    /// Whether this decides the whole value-group or a single split-out occurrence.
    pub scope: DecisionScope,
    /// The sentence window recorded for this decision (edited when
    /// [`context_edited`](CensorDecision::context_edited)).
    pub shown_context: String,
    /// The paragraph window recorded for this decision.
    pub block_context: String,
    /// The distinct block kind(s) the value sits in: one for an occurrence, the set for a group.
    pub block_kinds: Vec<BlockKind>,
    /// Heading level for a heading occurrence; `None` for a group or a non-heading.
    pub heading_level: Option<u8>,
    /// The distinct block language(s): one for an occurrence, the set for a group.
    pub langs: Vec<String>,
    /// The reviewer adjusted the censored span's boundaries (the value was retargeted).
    pub span_edited: bool,
    /// The reviewer corrected the recorded context window.
    pub context_edited: bool,
    /// The reviewer added this value; the detector did not flag it.
    pub user_added: bool,
    /// The neighbor context around the decided value (v10): for a group, the first occurrence's;
    /// for a split occurrence, that occurrence's.
    pub neighbors: CensorNeighbors,
    /// The model's advisory prediction shown for this value at review time (v11); all-`None` when no
    /// model ran or the value was reviewer-added (no suggestion was shown).
    pub prediction: Prediction,
}

impl CensorDecision {
    /// A whole-group decision built from a reviewed item, using its first occurrence's context and
    /// the distinct kinds/languages across its occurrences. Edit/add flows tweak the result before
    /// recording it.
    pub fn from_item(item: &ReviewItem, verdict: Verdict, reviewed: bool) -> Self {
        Self {
            value: item.value.clone(),
            detected_type: item.detected_type,
            method: item.method.clone(),
            verdict,
            reviewed,
            scope: DecisionScope::Group {
                occurrences: item.occurrence_count() as u32,
            },
            shown_context: item.first_shown_context().to_string(),
            block_context: item.first_block_context().to_string(),
            block_kinds: item.block_kinds().into_iter().collect(),
            heading_level: None,
            langs: distinct_langs(&item.occurrences),
            span_edited: false,
            context_edited: false,
            user_added: false,
            neighbors: item.first_neighbors(),
            prediction: Prediction::default(),
        }
    }

    /// An occurrence-scoped decision carved from a split: it covers exactly `occurrence` (its
    /// location drives the offset-based censoring in [`apply`]) and logs that occurrence's context.
    pub fn from_occurrence(
        value: &str,
        detected_type: ValueType,
        method: &str,
        occurrence: Occurrence,
        verdict: Verdict,
    ) -> Self {
        let block_kinds = vec![occurrence.block_kind];
        let heading_level = occurrence.heading_level;
        let langs = if occurrence.lang.is_empty() {
            Vec::new()
        } else {
            vec![occurrence.lang.clone()]
        };
        let neighbors = occurrence.neighbors.clone();
        Self {
            value: value.to_string(),
            detected_type,
            method: method.to_string(),
            verdict,
            reviewed: true,
            shown_context: occurrence.shown_context.clone(),
            block_context: occurrence.block_context.clone(),
            scope: DecisionScope::Occurrence(Box::new(occurrence)),
            block_kinds,
            heading_level,
            langs,
            span_edited: false,
            context_edited: false,
            user_added: false,
            neighbors,
            prediction: Prediction::default(),
        }
    }

    /// How many occurrences this decision covers (a split occurrence covers one).
    pub fn occurrences(&self) -> u32 {
        match &self.scope {
            DecisionScope::Group { occurrences } => *occurrences,
            DecisionScope::Occurrence(_) => 1,
        }
    }

    /// Whether this decision's value was supplied by the reviewer (edited span or added value)
    /// rather than found by the detector — such values are censored by literal search in
    /// [`apply`], since the detector won't re-find them.
    fn is_user_supplied(&self) -> bool {
        self.span_edited || self.user_added
    }

    /// The wire label for this decision's scope.
    fn scope_label(&self) -> &'static str {
        match self.scope {
            DecisionScope::Group { .. } => "group",
            DecisionScope::Occurrence(_) => "occurrence",
        }
    }
}

/// The distinct, non-empty languages across `occurrences`, sorted for determinism.
fn distinct_langs(occurrences: &[Occurrence]) -> Vec<String> {
    occurrences
        .iter()
        .filter(|occurrence| !occurrence.lang.is_empty())
        .map(|occurrence| occurrence.lang.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// A located text field of a document: one heading/paragraph, or one table cell, tagged with
/// where it sits and its structural kind. Table cells fold into [`BlockKind::TableCell`];
/// every paragraph (list item or not) folds into [`BlockKind::Paragraph`] (v7).
struct FieldRef<'a> {
    block_index: usize,
    cell: Option<(usize, usize)>,
    block_kind: BlockKind,
    heading_level: Option<u8>,
    text: &'a str,
}

/// Every text field of a document, in order (headings, paragraphs, then table cells), each
/// carrying its location and structural kind so detected occurrences can record them.
fn document_fields(document: &Document) -> Vec<FieldRef<'_>> {
    let mut fields = Vec::new();
    for (block_index, block) in document.blocks.iter().enumerate() {
        match block {
            Block::Heading { level, text } => fields.push(FieldRef {
                block_index,
                cell: None,
                block_kind: BlockKind::Heading,
                heading_level: Some(*level),
                text,
            }),
            Block::Paragraph { text } => fields.push(FieldRef {
                block_index,
                cell: None,
                block_kind: BlockKind::Paragraph,
                heading_level: None,
                text,
            }),
            Block::Table { rows } => {
                for (row, cells) in rows.iter().enumerate() {
                    for (col, cell) in cells.iter().enumerate() {
                        fields.push(FieldRef {
                            block_index,
                            cell: Some((row, col)),
                            block_kind: BlockKind::TableCell,
                            heading_level: None,
                            text: &cell.text,
                        });
                    }
                }
            }
        }
    }
    fields
}

/// Resolve the [`CensorNeighbors`] around a field located by `(block_index, cell)` (v10).
///
/// A flow block (`cell == None`) takes the previous/next block's text as `above`/`below` — an
/// adjacent table neighbor is omitted, and the header/label fields stay `None`. A table cell takes
/// the cells one row up/down in its column, the column header (row 0), and the row label (column 0),
/// dropping any header/label that resolves to the cell itself.
fn neighbors_at(
    document: &Document,
    block_index: usize,
    cell: Option<(usize, usize)>,
) -> CensorNeighbors {
    match cell {
        None => CensorNeighbors {
            above: block_index
                .checked_sub(1)
                .and_then(|index| flow_text(document.blocks.get(index))),
            below: flow_text(document.blocks.get(block_index + 1)),
            ..Default::default()
        },
        Some((row, col)) => {
            let Some(Block::Table { rows }) = document.blocks.get(block_index) else {
                return CensorNeighbors::default();
            };
            let cell_text = |r: usize, c: usize| {
                rows.get(r)
                    .and_then(|cells| cells.get(c))
                    .map(|cell| cell.text.clone())
            };
            CensorNeighbors {
                above: row.checked_sub(1).and_then(|r| cell_text(r, col)),
                below: cell_text(row + 1, col),
                // A cell already in row 0 / column 0 is its own header / label → keep it `None`.
                col_header: (row != 0).then(|| cell_text(0, col)).flatten(),
                row_label: (col != 0).then(|| cell_text(row, 0)).flatten(),
            }
        }
    }
}

/// The text of a flow block (heading/paragraph) used as a neighbor; `None` for an absent block or a
/// table (an adjacent table neighbor is omitted in v10).
fn flow_text(block: Option<&Block>) -> Option<String> {
    match block {
        Some(Block::Heading { text, .. } | Block::Paragraph { text }) => Some(text.clone()),
        _ => None,
    }
}

/// Plan the review: detect across the whole document, resolve overlaps per field, drop
/// allow-listed values, and return one [`ReviewItem`] per distinct value in first-seen order.
/// Each item carries every [`Occurrence`] of its value, with that occurrence's location, block
/// kind, and own context windows.
pub fn plan_review(document: &Document, options: &CensorOptions<'_>) -> Vec<ReviewItem> {
    let mut order: Vec<String> = Vec::new();
    let mut items: HashMap<String, ReviewItem> = HashMap::new();
    for field in document_fields(document) {
        let reserved = paired_spans(field.text);
        for candidate in resolve_overlaps(gather_candidates(field.text, options), &reserved) {
            let value = field.text[candidate.start..candidate.end].to_string();
            let occurrence = Occurrence {
                block_index: field.block_index,
                cell: field.cell,
                start: candidate.start,
                end: candidate.end,
                block_kind: field.block_kind,
                heading_level: field.heading_level,
                shown_context: sentence_window_at(field.text, candidate.start, candidate.end),
                block_context: block_window_at(field.text, candidate.start, candidate.end),
                neighbors: neighbors_at(document, field.block_index, field.cell),
                ..Default::default() // lang tagged in a later pass (see `tag_occurrence_languages`)
            };
            if let Some(item) = items.get_mut(&value) {
                item.occurrences.push(occurrence);
            } else {
                order.push(value.clone());
                items.insert(
                    value.clone(),
                    ReviewItem {
                        detected_type: candidate.value_type,
                        method: method_label(candidate.source, candidate.value_type),
                        occurrences: vec![occurrence],
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

/// Tag every occurrence with its block's detected language (a v7 training feature).
///
/// Detection is per text field with a document-dominant fallback for short blocks (see
/// [`crate::lang`]); `override_lang` forces a code on every block. All occurrences in the same
/// field share that field's language.
pub fn tag_occurrence_languages(
    document: &Document,
    items: &mut [ReviewItem],
    override_lang: Option<&str>,
) {
    let fields = document_fields(document);
    let texts: Vec<&str> = fields.iter().map(|field| field.text).collect();
    let tags = crate::lang::tag_texts(&texts, override_lang);
    let by_field: HashMap<FieldKey, &crate::lang::BlockLang> = fields
        .iter()
        .zip(&tags)
        .map(|(field, tag)| ((field.block_index, field.cell), tag))
        .collect();
    for occurrence in items
        .iter_mut()
        .flat_map(|item| item.occurrences.iter_mut())
    {
        if let Some(tag) = by_field.get(&(occurrence.block_index, occurrence.cell)) {
            occurrence.lang = tag.lang.clone();
            occurrence.lang_confidence = tag.confidence;
        }
    }
}

/// Build a [`ReviewItem`] for an exact `value` by locating every literal occurrence across the
/// document's text fields, with each occurrence's location, block kind, and context windows.
///
/// Returns `None` when `value` is empty or does not occur — so it doubles as validation for the
/// reviewer's edited/added values (v7): a value not present cannot be censored.
pub(crate) fn locate_value(
    document: &Document,
    value: &str,
    detected_type: ValueType,
    method: &str,
) -> Option<ReviewItem> {
    if value.is_empty() {
        return None;
    }
    let mut occurrences = Vec::new();
    for field in document_fields(document) {
        for (start, matched) in field.text.match_indices(value) {
            let end = start + matched.len();
            occurrences.push(Occurrence {
                block_index: field.block_index,
                cell: field.cell,
                start,
                end,
                block_kind: field.block_kind,
                heading_level: field.heading_level,
                shown_context: sentence_window_at(field.text, start, end),
                block_context: block_window_at(field.text, start, end),
                neighbors: neighbors_at(document, field.block_index, field.cell),
                ..Default::default()
            });
        }
    }
    if occurrences.is_empty() {
        return None;
    }
    Some(ReviewItem {
        value: value.to_string(),
        detected_type,
        method: method.to_string(),
        occurrences,
    })
}

/// Occurrence-scoped censor spans for one document, keyed by `(block_index, cell)` — the field a
/// span lives in. Each entry is `(start, end, label)`.
type OccurrenceSpans<'a> = HashMap<FieldKey, Vec<(usize, usize, &'a str)>>;

/// Apply the decisions: a copy of `document` with every **confirmed** value replaced by a
/// `REDACTED_<FINAL_TYPE>_<NNN>` placeholder (deduped per value); rejected values are left as-is.
///
/// Three censor mechanisms feed each field: whole-group detector values (censored at every detector
/// span), reviewer-supplied values (censored by literal search), and occurrence-scoped decisions
/// from a split (censored at exactly their recorded offset). A split's *rejected* occurrence
/// contributes no span, so it stays in the clear even though sibling occurrences of the same string
/// are censored.
pub fn apply(
    document: &Document,
    decisions: &[CensorDecision],
    options: &CensorOptions<'_>,
) -> Document {
    // Whole-group, detector-found values: censor every detector span of the value.
    let group_confirmed = confirmed_values(decisions, |decision| {
        matches!(decision.scope, DecisionScope::Group { .. }) && !decision.is_user_supplied()
    });
    // Reviewer-supplied values (edited span / added value): the detector won't re-find them, so
    // they are located by literal search instead.
    let user_supplied = confirmed_values(decisions, CensorDecision::is_user_supplied);
    // Occurrence-scoped confirms: censor exactly their span; a rejected occurrence adds nothing.
    let mut occurrence_spans: OccurrenceSpans<'_> = HashMap::new();
    for decision in decisions {
        if let (DecisionScope::Occurrence(occurrence), Verdict::Confirm { final_type }) =
            (&decision.scope, &decision.verdict)
        {
            occurrence_spans
                .entry((occurrence.block_index, occurrence.cell))
                .or_default()
                .push((occurrence.start, occurrence.end, final_type.as_str()));
        }
    }

    let mut allocator = LabelAllocator::default();
    let blocks = document
        .blocks
        .iter()
        .enumerate()
        .map(|(block_index, block)| {
            apply_block(
                block,
                block_index,
                options,
                &group_confirmed,
                &user_supplied,
                &occurrence_spans,
                &mut allocator,
            )
        })
        .collect();
    Document {
        source: document.source.clone(),
        blocks,
    }
}

/// The `value → final_type` map of confirmed decisions matching `select`.
fn confirmed_values(
    decisions: &[CensorDecision],
    select: impl Fn(&CensorDecision) -> bool,
) -> HashMap<&str, &str> {
    decisions
        .iter()
        .filter(|decision| select(decision))
        .filter_map(|decision| match &decision.verdict {
            Verdict::Confirm { final_type } => Some((decision.value.as_str(), final_type.as_str())),
            Verdict::Reject => None,
        })
        .collect()
}

/// Apply confirmed censorings to every text field of a block.
fn apply_block(
    block: &Block,
    block_index: usize,
    options: &CensorOptions<'_>,
    group_confirmed: &HashMap<&str, &str>,
    user_supplied: &HashMap<&str, &str>,
    occurrence_spans: &OccurrenceSpans<'_>,
    allocator: &mut LabelAllocator,
) -> Block {
    let empty: &[(usize, usize, &str)] = &[];
    let field_spans = |cell: Option<(usize, usize)>| {
        occurrence_spans
            .get(&(block_index, cell))
            .map_or(empty, Vec::as_slice)
    };
    match block {
        Block::Heading { level, text } => Block::Heading {
            level: *level,
            text: apply_text(
                text,
                options,
                group_confirmed,
                user_supplied,
                field_spans(None),
                allocator,
            ),
        },
        Block::Paragraph { text } => Block::Paragraph {
            text: apply_text(
                text,
                options,
                group_confirmed,
                user_supplied,
                field_spans(None),
                allocator,
            ),
        },
        Block::Table { rows } => Block::Table {
            rows: rows
                .iter()
                .enumerate()
                .map(|(row, cells)| {
                    cells
                        .iter()
                        .enumerate()
                        .map(|(col, cell)| Cell {
                            text: apply_text(
                                &cell.text,
                                options,
                                group_confirmed,
                                user_supplied,
                                field_spans(Some((row, col))),
                                allocator,
                            ),
                        })
                        .collect()
                })
                .collect(),
        },
    }
}

/// Substitute the confirmed values in one text field (right-to-left to keep offsets valid).
///
/// Spans come from three sources — detector candidates whose value is a confirmed whole-group
/// value, occurrence-scoped offsets recorded for this field, and literal occurrences of
/// reviewer-supplied values — each carrying its placeholder label. Reserved bracket spans and
/// overlaps are dropped, so every byte is replaced at most once.
fn apply_text(
    text: &str,
    options: &CensorOptions<'_>,
    group_confirmed: &HashMap<&str, &str>,
    user_supplied: &HashMap<&str, &str>,
    occurrence_spans: &[(usize, usize, &str)],
    allocator: &mut LabelAllocator,
) -> String {
    let reserved = paired_spans(text);
    let mut spans: Vec<(usize, usize, &str)> = Vec::new();

    // 1. Detector candidate spans whose value is a confirmed whole-group value.
    for candidate in resolve_overlaps(gather_candidates(text, options), &reserved) {
        if let Some(&label) = group_confirmed.get(&text[candidate.start..candidate.end]) {
            spans.push((candidate.start, candidate.end, label));
        }
    }

    // 2. Occurrence-scoped offsets for this field (already non-reserved, valid byte spans).
    spans.extend_from_slice(occurrence_spans);

    // 3. Literal occurrences of reviewer-supplied values, longest first for determinism,
    //    skipping reserved spans and anything already claimed.
    let mut user_values: Vec<(&str, &str)> = user_supplied.iter().map(|(&v, &l)| (v, l)).collect();
    user_values.sort_unstable_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(b.0)));
    for (value, label) in user_values {
        for (start, end) in literal_spans(text, value) {
            if !intersects(&reserved, start, end) && !overlaps(&spans, start, end) {
                spans.push((start, end, label));
            }
        }
    }

    // Keep a non-overlapping set (greedy by start), then replace right-to-left.
    spans.sort_by_key(|&(start, _, _)| start);
    let mut last_end = 0;
    let mut kept: Vec<(usize, usize, &str)> = Vec::with_capacity(spans.len());
    for span in spans {
        if span.0 >= last_end {
            last_end = span.1;
            kept.push(span);
        }
    }

    let mut result = text.to_string();
    for &(start, end, label) in kept.iter().rev() {
        let placeholder = allocator.placeholder_for(&text[start..end], label);
        result.replace_range(start..end, &placeholder);
    }
    result
}

/// Non-overlapping byte spans of every literal occurrence of `value` in `text`.
fn literal_spans(text: &str, value: &str) -> Vec<(usize, usize)> {
    if value.is_empty() {
        return Vec::new();
    }
    text.match_indices(value)
        .map(|(start, matched)| (start, start + matched.len()))
        .collect()
}

/// Whether `[start, end)` intersects any `[s, e)` span in `spans`.
fn intersects(spans: &[(usize, usize)], start: usize, end: usize) -> bool {
    spans.iter().any(|&(s, e)| start < e && s < end)
}

/// Whether `[start, end)` overlaps any labeled span in `spans`.
fn overlaps(spans: &[(usize, usize, &str)], start: usize, end: usize) -> bool {
    spans.iter().any(|&(s, e, _)| start < e && s < end)
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

/// The v11 censor feature vector for a review `item`, built from the same fields the logged record
/// carries (the label is irrelevant to the encoder, so a placeholder verdict is used). Lets the
/// review wire a prediction before the human decides, with feature parity to what is later logged.
pub fn item_feature_vector(item: &ReviewItem) -> Vec<f64> {
    let decision = CensorDecision::from_item(item, Verdict::Reject, true);
    let records = decision_records(&[decision], "", "", 0);
    // `decision_records` keeps reviewed decisions; the one above is `reviewed: true`, so it is present.
    records
        .first()
        .map(crate::ml::features::censor::censor_features)
        .unwrap_or_default()
}

/// Build the schema-4 decision-log records for the human-reviewed decisions (auto-defaulted items
/// are skipped). Each [`CensorDecision`] is self-contained, so no parallel `items` slice is needed.
/// `doc_id` is the content id keying every record from this document; `source` is the filename.
pub fn decision_records(
    decisions: &[CensorDecision],
    source: &str,
    doc_id: &str,
    timestamp: u64,
) -> Vec<DecisionRecord> {
    decisions
        .iter()
        .filter(|decision| decision.reviewed)
        .map(|decision| {
            let (verdict, final_type) = match &decision.verdict {
                Verdict::Confirm { final_type } => ("confirm", Some(final_type.clone())),
                Verdict::Reject => ("reject", None),
            };
            DecisionRecord {
                schema: decision_schema(),
                timestamp,
                source: source.to_string(),
                doc_id: doc_id.to_string(),
                value: decision.value.clone(),
                method: decision.method.clone(),
                detected_type: decision.detected_type.label().to_string(),
                verdict: verdict.to_string(),
                final_type,
                shown_context: decision.shown_context.clone(),
                block_context: decision.block_context.clone(),
                occurrences: decision.occurrences(),
                scope: decision.scope_label().to_string(),
                block_kinds: decision
                    .block_kinds
                    .iter()
                    .map(|kind| kind.as_str().to_string())
                    .collect(),
                heading_level: decision.heading_level,
                langs: decision.langs.clone(),
                span_edited: decision.span_edited,
                context_edited: decision.context_edited,
                user_added: decision.user_added,
                neighbors: decision.neighbors.clone(),
                prediction: decision.prediction.clone(),
            }
        })
        .collect()
}

/// Fold the human-reviewed decisions into the learned store: a `reject` (false positive) marks
/// the value safe to leave in the clear (`allow`); a `confirm` keeps it censored (`deny`). A
/// value seen both ways becomes conflicted and stays censored.
pub fn update_store(store: &mut LearnedStore, decisions: &[CensorDecision]) {
    for decision in decisions {
        if !decision.reviewed {
            continue;
        }
        let allow = matches!(decision.verdict, Verdict::Reject);
        store.record(&decision.value, decision.detected_type.label(), allow);
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

    fn confirm_dec(value: &str, ty: ValueType, label: &str) -> CensorDecision {
        CensorDecision::from_item(
            &item(value, ty),
            Verdict::Confirm {
                final_type: label.into(),
            },
            true,
        )
    }

    fn reject_dec(value: &str, ty: ValueType) -> CensorDecision {
        CensorDecision::from_item(&item(value, ty), Verdict::Reject, true)
    }

    /// A confirmed decision for a reviewer-supplied (edited/added) value, censored by literal
    /// search rather than the detector.
    fn user_value_dec(value: &str, label: &str) -> CensorDecision {
        let mut decision = CensorDecision::from_item(
            &item(value, ValueType::Entity),
            Verdict::Confirm {
                final_type: label.into(),
            },
            true,
        );
        decision.user_added = true;
        decision
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
        assert_eq!(items[0].occurrence_count(), 2, "both occurrences captured");
        assert_eq!(items[0].method, "regex:email");
        assert!(
            items[0].first_shown_context().contains("a@b.com"),
            "context captured"
        );
    }

    #[test]
    fn plan_review_records_per_occurrence_location_and_kind() {
        // The same value in a heading and a table cell: one item, two located occurrences.
        let doc = Document {
            source: PathBuf::from("c.docx"),
            blocks: vec![
                Block::Heading {
                    level: 2,
                    text: "Email a@b.com".into(),
                },
                Block::Paragraph {
                    text: "filler".into(),
                },
                Block::Table {
                    rows: vec![vec![Cell::new("write to a@b.com please")]],
                },
            ],
        };
        let items = plan_review(&doc, &CensorOptions::default());
        assert_eq!(items.len(), 1);
        let occs = &items[0].occurrences;
        assert_eq!(occs.len(), 2);

        // First occurrence: inside the heading (block 0), kind + level recorded.
        assert_eq!(occs[0].block_index, 0);
        assert_eq!(occs[0].block_kind, BlockKind::Heading);
        assert_eq!(occs[0].heading_level, Some(2));
        assert_eq!(occs[0].cell, None);
        assert_eq!(&"Email a@b.com"[occs[0].start..occs[0].end], "a@b.com");

        // Second occurrence: inside the table cell (block 2), located by (row, col).
        assert_eq!(occs[1].block_index, 2);
        assert_eq!(occs[1].block_kind, BlockKind::TableCell);
        assert_eq!(occs[1].cell, Some((0, 0)));

        // The two occurrences carry their own context windows.
        assert!(occs[0].shown_context.contains("Email"));
        assert!(occs[1].shown_context.contains("please"));
        assert_eq!(
            items[0].block_kinds(),
            [BlockKind::Heading, BlockKind::TableCell]
                .into_iter()
                .collect()
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
        let decisions = vec![CensorDecision::from_item(
            &item("a@b.com", ValueType::Email),
            Verdict::Confirm {
                final_type: "EMAIL".into(),
            },
            false,
        )];
        let out = apply(&doc, &decisions, &CensorOptions::default());
        assert!(para_text(&out).contains("REDACTED_EMAIL_001"));
    }

    #[test]
    fn apply_censors_a_user_supplied_value_by_literal_search() {
        // "Jane Doe" is not a detector candidate (no party list, no name heuristic yet), so it
        // is censored only because the reviewer supplied it — every occurrence, shared placeholder.
        let doc = paragraph_doc("pay Jane Doe and Jane Doe again");
        let decisions = vec![user_value_dec("Jane Doe", "PERSON")];
        let out = apply(&doc, &decisions, &CensorOptions::default());
        let text = para_text(&out);
        assert!(!text.contains("Jane Doe"), "user value censored: {text}");
        assert_eq!(
            text.matches("REDACTED_PERSON_001").count(),
            2,
            "both occurrences share one placeholder: {text}"
        );
    }

    #[test]
    fn decision_scope_distinguishes_group_and_occurrence() {
        // from_item → whole-group scope (the test item() helper has one occurrence).
        let group = confirm_dec("Acme", ValueType::Org, "ORG");
        assert!(matches!(group.scope, DecisionScope::Group { .. }));
        assert_eq!(group.occurrences(), 1);

        // from_occurrence → occurrence scope carrying the precise location + that occurrence's context.
        let occurrence = Occurrence {
            block_index: 4,
            cell: None,
            start: 2,
            end: 6,
            block_kind: BlockKind::Paragraph,
            heading_level: None,
            shown_context: "rate of 3% applies".into(),
            block_context: "the full clause".into(),
            ..Default::default()
        };
        let decision = CensorDecision::from_occurrence(
            "3%",
            ValueType::Percent,
            "regex:percent",
            occurrence,
            Verdict::Reject,
        );
        assert!(matches!(decision.scope, DecisionScope::Occurrence(_)));
        assert_eq!(decision.occurrences(), 1);
        assert_eq!(decision.shown_context, "rate of 3% applies");
        assert!(matches!(decision.verdict, Verdict::Reject));
    }

    fn occ_at(cell: Option<(usize, usize)>, start: usize, end: usize) -> Occurrence {
        Occurrence {
            block_index: 0,
            cell,
            start,
            end,
            block_kind: if cell.is_some() {
                BlockKind::TableCell
            } else {
                BlockKind::Paragraph
            },
            heading_level: None,
            shown_context: String::new(),
            block_context: String::new(),
            ..Default::default()
        }
    }

    fn occ_dec(value: &str, occurrence: Occurrence, verdict: Verdict) -> CensorDecision {
        CensorDecision::from_occurrence(
            value,
            ValueType::Percent,
            "regex:percent",
            occurrence,
            verdict,
        )
    }

    #[test]
    fn apply_offset_path_censors_only_confirmed_split_occurrences() {
        // Three "3%" in one paragraph (offsets 0, 8, 16); confirm the outer two, reject the middle.
        let doc = paragraph_doc("3% then 3% then 3%");
        let confirm = || Verdict::Confirm {
            final_type: "PERCENT".into(),
        };
        let decisions = vec![
            occ_dec("3%", occ_at(None, 0, 2), confirm()),
            occ_dec("3%", occ_at(None, 8, 10), Verdict::Reject),
            occ_dec("3%", occ_at(None, 16, 18), confirm()),
        ];
        let out = apply(&doc, &decisions, &CensorOptions::default());
        // Confirmed occurrences share one placeholder; the rejected middle stays literal — even
        // though the detector would otherwise flag every "3%".
        assert_eq!(
            para_text(&out),
            "REDACTED_PERCENT_001 then 3% then REDACTED_PERCENT_001"
        );
    }

    #[test]
    fn apply_offset_path_targets_the_right_table_cell() {
        let doc = Document {
            source: PathBuf::from("c.docx"),
            blocks: vec![Block::Table {
                rows: vec![vec![Cell::new("Acme"), Cell::new("Acme")]],
            }],
        };
        // Censor only the second cell's "Acme" (row 0, col 1).
        let decisions = vec![CensorDecision::from_occurrence(
            "Acme",
            ValueType::Org,
            "manual",
            occ_at(Some((0, 1)), 0, 4),
            Verdict::Confirm {
                final_type: "ORG".into(),
            },
        )];
        let out = apply(&doc, &decisions, &CensorOptions::default());
        let Block::Table { rows } = &out.blocks[0] else {
            panic!("expected a table");
        };
        assert_eq!(rows[0][0].text, "Acme", "first cell untouched");
        assert!(
            rows[0][1].text.contains("REDACTED_ORG_001"),
            "second cell censored: {}",
            rows[0][1].text
        );
    }

    #[test]
    fn tag_occurrence_languages_tags_per_block_with_fallback_and_override() {
        // A long English block sets the dominant; the short block falls back to it.
        let doc = Document {
            source: PathBuf::from("c.txt"),
            blocks: vec![
                Block::Paragraph {
                    text: "This agreement is governed by the laws of New York and binding \
                           arbitration applies; contact a@b.com for details."
                        .into(),
                },
                Block::Paragraph {
                    text: "See a@b.com.".into(),
                },
            ],
        };

        let mut items = plan_review(&doc, &CensorOptions::default());
        tag_occurrence_languages(&doc, &mut items, None);
        let email = items
            .iter()
            .find(|item| item.value == "a@b.com")
            .expect("email detected in both blocks");
        assert_eq!(email.occurrences.len(), 2);
        assert!(
            email.occurrences.iter().all(|occ| occ.lang == "en"),
            "both blocks resolve to English (the short one via fallback)"
        );

        // Override forces every occurrence's language.
        let mut forced = plan_review(&doc, &CensorOptions::default());
        tag_occurrence_languages(&doc, &mut forced, Some("fr"));
        assert!(
            forced
                .iter()
                .flat_map(|item| &item.occurrences)
                .all(|occ| occ.lang == "fr")
        );
    }

    #[test]
    fn locate_value_finds_every_occurrence_or_none() {
        let doc = Document {
            source: PathBuf::from("c.docx"),
            blocks: vec![
                Block::Heading {
                    level: 1,
                    text: "Acme intro".into(),
                },
                Block::Paragraph {
                    text: "Acme and Acme".into(),
                },
            ],
        };
        let item = locate_value(&doc, "Acme", ValueType::Org, "manual").expect("Acme is present");
        assert_eq!(
            item.occurrence_count(),
            3,
            "1 in the heading + 2 in the paragraph"
        );
        assert_eq!(item.detected_type, ValueType::Org);
        assert_eq!(item.method, "manual");
        assert!(locate_value(&doc, "Nope", ValueType::Org, "manual").is_none());
        assert!(locate_value(&doc, "", ValueType::Org, "manual").is_none());
    }

    #[test]
    fn decision_records_skip_unreviewed_and_map_schema_4_fields() {
        let decisions = vec![
            confirm_dec("a@b.com", ValueType::Email, "EMAIL"),
            reject_dec("Jane", ValueType::Entity),
            CensorDecision::from_item(
                &item("X", ValueType::Person),
                Verdict::Confirm {
                    final_type: "PERSON".into(),
                },
                false,
            ),
        ];
        let records = decision_records(&decisions, "c.txt", "doc0000000000abcd", 7);
        assert_eq!(records.len(), 2, "the unreviewed item is not logged");
        assert_eq!(records[0].schema, decision_schema());
        assert_eq!(records[0].doc_id, "doc0000000000abcd");
        assert_eq!(records[0].scope, "group");
        assert_eq!(records[0].verdict, "confirm");
        assert_eq!(records[0].final_type.as_deref(), Some("EMAIL"));
        assert_eq!(records[0].detected_type, "EMAIL");
        assert_eq!(records[0].occurrences, 1);
        assert_eq!(records[1].verdict, "reject");
        assert_eq!(records[1].final_type, None, "reject has no final type");
        assert_eq!(records[1].detected_type, "ENTITY");
    }

    #[test]
    fn item_feature_vector_has_the_censor_width() {
        // The pre-decision feature vector a prediction is built from matches the encoder width.
        let features = item_feature_vector(&item("jane@acme.com", ValueType::Email));
        assert_eq!(
            features.len(),
            crate::ml::features::censor::CENSOR_FEATURE_LEN
        );
        assert!(features.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn decision_records_carry_the_advisory_prediction() {
        // The prediction stamped on a decision (T67) flows onto the logged row (schema 6).
        let mut decision = confirm_dec("a@b.com", ValueType::Email, "EMAIL");
        decision.prediction = Prediction {
            predicted_verdict: Some("confirm".into()),
            predicted_verdict_score: Some(0.82),
            predicted_reason: Some("EMAIL".into()),
            predicted_reason_score: Some(0.6),
            model_trained_at: Some("stamp42".into()),
        };
        let records = decision_records(&[decision], "c.txt", "docid", 1);
        let prediction = &records[0].prediction;
        assert_eq!(prediction.predicted_verdict.as_deref(), Some("confirm"));
        assert_eq!(prediction.predicted_verdict_score, Some(0.82));
        assert_eq!(prediction.predicted_reason.as_deref(), Some("EMAIL"));
        assert_eq!(prediction.model_trained_at.as_deref(), Some("stamp42"));
    }

    #[test]
    fn occurrence_scoped_record_carries_kind_level_and_lang() {
        // A split occurrence decision logs its single block kind, heading level, and language.
        let mut occurrence = occ_at(None, 0, 2);
        occurrence.block_kind = BlockKind::Heading;
        occurrence.heading_level = Some(2);
        occurrence.lang = "fr".into();
        let decisions = vec![CensorDecision::from_occurrence(
            "3%",
            ValueType::Percent,
            "regex:percent",
            occurrence,
            Verdict::Confirm {
                final_type: "PERCENT".into(),
            },
        )];
        let records = decision_records(&decisions, "c.txt", "docid", 1);
        assert_eq!(records[0].scope, "occurrence");
        assert_eq!(records[0].block_kinds, vec!["heading".to_string()]);
        assert_eq!(records[0].heading_level, Some(2));
        assert_eq!(records[0].langs, vec!["fr".to_string()]);
        assert_eq!(records[0].occurrences, 1);
    }

    #[test]
    fn update_store_allows_rejects_denies_confirms_and_conflicts() {
        let mut store = LearnedStore::default();
        // reject → the value is safe to leave in the clear next run.
        update_store(&mut store, &[reject_dec("Reach", ValueType::Entity)]);
        assert!(store.allowed_values().contains("Reach"));
        // confirm → kept censored (never allow-listed).
        update_store(&mut store, &[confirm_dec("Acme", ValueType::Org, "ORG")]);
        assert!(!store.allowed_values().contains("Acme"));
        // seen both ways across runs → conflicted → stays censored.
        update_store(&mut store, &[reject_dec("Maybe", ValueType::Entity)]);
        update_store(
            &mut store,
            &[confirm_dec("Maybe", ValueType::Entity, "ENTITY")],
        );
        assert!(!store.allowed_values().contains("Maybe"));
    }

    // ── Neighbor context (v10) ──────────────────────────────────────────────────

    #[test]
    fn neighbors_at_flow_blocks_take_prev_and_next() {
        let doc = Document {
            source: PathBuf::from("c.txt"),
            blocks: vec![
                Block::Paragraph {
                    text: "123 Main Street".into(),
                },
                Block::Paragraph {
                    text: "Springfield, IL 62704".into(),
                },
                Block::Paragraph { text: "USA".into() },
            ],
        };
        let middle = neighbors_at(&doc, 1, None);
        assert_eq!(middle.above.as_deref(), Some("123 Main Street"));
        assert_eq!(middle.below.as_deref(), Some("USA"));
        assert_eq!(middle.col_header, None, "flow blocks have no header");
        assert_eq!(middle.row_label, None, "flow blocks have no row label");
        // Document edges: first has no above, last has no below.
        assert_eq!(neighbors_at(&doc, 0, None).above, None);
        assert_eq!(neighbors_at(&doc, 2, None).below, None);
    }

    #[test]
    fn neighbors_at_flow_omits_an_adjacent_table() {
        let doc = Document {
            source: PathBuf::from("c.docx"),
            blocks: vec![
                Block::Table {
                    rows: vec![vec![Cell::new("x")]],
                },
                Block::Paragraph {
                    text: "after".into(),
                },
            ],
        };
        // The paragraph's previous block is a table → omitted, not flattened into text.
        assert_eq!(neighbors_at(&doc, 1, None).above, None);
    }

    #[test]
    fn neighbors_at_table_cell_takes_grid_and_headers() {
        // (Address | Buyer)      row 0  ← headers
        // (123 Main St | Acme)   row 1
        // (Springfield | Wonka)  row 2
        let doc = Document {
            source: PathBuf::from("c.docx"),
            blocks: vec![Block::Table {
                rows: vec![
                    vec![Cell::new("Address"), Cell::new("Buyer")],
                    vec![Cell::new("123 Main St"), Cell::new("Acme")],
                    vec![Cell::new("Springfield"), Cell::new("Wonka")],
                ],
            }],
        };
        // Cell (1,0) = "123 Main St": up/down in its column, header from row 0, self is column 0.
        let mid = neighbors_at(&doc, 0, Some((1, 0)));
        assert_eq!(mid.above.as_deref(), Some("Address"));
        assert_eq!(mid.below.as_deref(), Some("Springfield"));
        assert_eq!(mid.col_header.as_deref(), Some("Address"));
        assert_eq!(mid.row_label, None, "cell is itself in column 0");
        // Cell (2,1) = "Wonka": header + row label present, no row below.
        let corner = neighbors_at(&doc, 0, Some((2, 1)));
        assert_eq!(corner.above.as_deref(), Some("Acme"));
        assert_eq!(corner.below, None);
        assert_eq!(corner.col_header.as_deref(), Some("Buyer"));
        assert_eq!(corner.row_label.as_deref(), Some("Springfield"));
    }

    #[test]
    fn neighbors_at_header_cell_is_its_own_header_and_label() {
        let doc = Document {
            source: PathBuf::from("c.docx"),
            blocks: vec![Block::Table {
                rows: vec![vec![Cell::new("Address"), Cell::new("Buyer")]],
            }],
        };
        // Row 0 / column 0: its own header and label → both None; single-row table → no below.
        let n = neighbors_at(&doc, 0, Some((0, 0)));
        assert_eq!(n.col_header, None);
        assert_eq!(n.row_label, None);
        assert_eq!(n.above, None);
        assert_eq!(n.below, None);
    }

    #[test]
    fn plan_review_records_occurrence_neighbors() {
        let doc = Document {
            source: PathBuf::from("c.txt"),
            blocks: vec![
                Block::Paragraph {
                    text: "123 Main Street".into(),
                },
                Block::Paragraph {
                    text: "Email a@b.com here".into(),
                },
                Block::Paragraph {
                    text: "Trailing line".into(),
                },
            ],
        };
        let items = plan_review(&doc, &CensorOptions::default());
        let email = items
            .iter()
            .find(|item| item.value == "a@b.com")
            .expect("email detected");
        let neighbors = &email.occurrences[0].neighbors;
        assert_eq!(neighbors.above.as_deref(), Some("123 Main Street"));
        assert_eq!(neighbors.below.as_deref(), Some("Trailing line"));
        // A whole-group decision carries the first occurrence's neighbors.
        let decision = CensorDecision::from_item(email, Verdict::Reject, true);
        assert_eq!(decision.neighbors.above.as_deref(), Some("123 Main Street"));
    }
}
