//! Resolve a block's *effective* styling — what it actually looks like — from the document's
//! style and numbering tables, rather than recording only the direct/inline formatting (T47).
//!
//! OOXML formatting is layered: a run's appearance is the document defaults, overlaid by the
//! paragraph style (and its `based_on` ancestors), overlaid by the run's character style, overlaid
//! by the run's own direct formatting — each higher layer winning per property. [`resolve_run`]
//! folds those layers; [`resolve_para`] does the same for alignment / indent / line-spacing; and
//! [`resolve_numbering`] turns a `num_id` + level into the *visible* numbering format (bullet vs
//! `1.` vs `a.`), not the opaque id.
//!
//! Property *values* live in private fields with no getters, so — like [`crate::style::extract`]
//! and the T45 spike — they are read by serializing each `RunProperty` to JSON and pulling the
//! proven keys. The functions here are pure over the parsed tables, so they unit-test without a
//! `.docx` on disk.
//!
//! Deliberately deferred (see the v8 design's open risks): document-default *paragraph* properties
//! (the private `paragraphPropertyDefault` wrapper — line-spacing defaults usually live in the
//! `Normal` style anyway), numbering `level_overrides`, and table-style conditional formatting.

use std::collections::BTreeSet;

use docx_rs::{
    Indent, LevelText, LineSpacing, Numberings, ParagraphProperty, RunProperty, SpecialIndentType,
    Style, Styles,
};
use serde_json::Value;

use crate::model::{EffectiveRun, IndentTwips, NumberingFormat, Spacing};

/// A run's resolved effective styling plus whether any referenced style could not be found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRun {
    /// The effective run styling after folding all layers.
    pub run: EffectiveRun,
    /// `true` when a referenced style id (paragraph/character style or a `based_on` ancestor) was
    /// missing from the table — the resolution is incomplete and the block is "unknown".
    pub unresolved: bool,
}

/// A paragraph's resolved alignment / indent / spacing, plus the same unresolved signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPara {
    /// Effective alignment/justification value (e.g. `center`), if any.
    pub alignment: Option<String>,
    /// Effective indentation in twips.
    pub indent: IndentTwips,
    /// Effective paragraph spacing (including line spacing).
    pub spacing: Spacing,
    /// `true` when the paragraph's style chain referenced a missing style.
    pub unresolved: bool,
}

/// Resolve a run's effective styling by folding, lowest precedence first: document defaults → the
/// paragraph style chain (root-first) → the run's character style chain (root-first) → the run's
/// direct formatting.
///
/// `direct` is the run's own `RunProperty` (its character-style reference, if any, is read from it);
/// `paragraph_style` is the enclosing paragraph's style id.
///
/// ```
/// use docx_rs::{RunFonts, RunProperty, Style, StyleType, Styles};
/// use stencil::style::resolve::resolve_run;
///
/// let styles = Styles::new().add_style(
///     Style::new("Heading2", StyleType::Paragraph)
///         .fonts(RunFonts::new().ascii("Arial"))
///         .size(26)
///         .bold(),
/// );
/// // A heading run with no direct formatting still resolves to the style's look.
/// let resolved = resolve_run(&RunProperty::new(), Some("Heading2"), &styles);
/// assert_eq!(resolved.run.font.as_deref(), Some("Arial"));
/// assert_eq!(resolved.run.size_half_pt, Some(26));
/// assert_eq!(resolved.run.bold, Some(true));
/// assert!(!resolved.unresolved);
/// ```
pub fn resolve_run(
    direct: &RunProperty,
    paragraph_style: Option<&str>,
    styles: &Styles,
) -> ResolvedRun {
    let mut run = doc_default_run(styles);
    let mut unresolved = false;

    let character_style = direct.style.as_ref().map(|style| style.val.as_str());
    for style_id in [paragraph_style, character_style].into_iter().flatten() {
        let (chain, broken) = style_chain(style_id, styles);
        unresolved |= broken;
        for style in chain.iter().rev() {
            run = overlay_run(run, run_layer(&run_property_value(&style.run_property)));
        }
    }
    run = overlay_run(run, run_layer(&run_property_value(direct)));
    ResolvedRun { run, unresolved }
}

/// Resolve a paragraph's alignment / indent / spacing by folding its style chain (root-first) then
/// its direct properties. Document-default *paragraph* properties are not folded in (see the module
/// docs).
pub fn resolve_para(property: &ParagraphProperty, styles: &Styles) -> ResolvedPara {
    let mut layer = ParaLayer::default();
    let mut unresolved = false;

    if let Some(style_id) = property.style.as_ref().map(|style| style.val.as_str()) {
        let (chain, broken) = style_chain(style_id, styles);
        unresolved |= broken;
        for style in chain.iter().rev() {
            layer = overlay_para(layer, read_para(&style.paragraph_property));
        }
    }
    layer = overlay_para(layer, read_para(property));

    ResolvedPara {
        alignment: layer.alignment,
        indent: layer.indent,
        spacing: layer.spacing,
        unresolved,
    }
}

/// Resolve a `num_id` + level to its visible numbering format (the `numFmt` and level-text
/// template). `None` when the numbering, its abstract definition, or the level is missing — the
/// caller records that as `numbering_unresolved`.
///
/// `level_overrides` are not yet applied (see the module docs); the abstract level is used.
pub fn resolve_numbering(
    num_id: usize,
    ilvl: usize,
    numberings: &Numberings,
) -> Option<NumberingFormat> {
    let numbering = numberings.numberings.iter().find(|n| n.id == num_id)?;
    let abstract_num = numberings
        .abstract_nums
        .iter()
        .find(|a| a.id == numbering.abstract_num_id)?;
    let level = abstract_num
        .levels
        .iter()
        .find(|level| level.level == ilvl)?;
    Some(NumberingFormat {
        kind: level.format.val.clone(),
        level_text: level_text(&level.text),
    })
}

/// The chain of styles from `style_id` to its root, nearest-first; the bool is `true` when the id
/// or one of its `based_on` ancestors was missing (a broken reference). A `based_on` cycle is
/// broken by the visited set.
fn style_chain<'a>(style_id: &str, styles: &'a Styles) -> (Vec<&'a Style>, bool) {
    let mut chain = Vec::new();
    let mut seen = BTreeSet::new();
    let mut next = Some(style_id.to_string());
    let mut broken = false;
    while let Some(id) = next.take() {
        if !seen.insert(id.clone()) {
            break; // already visited — a `based_on` cycle; stop.
        }
        match styles.styles.iter().find(|style| style.style_id == id) {
            Some(style) => {
                next = based_on_id(style);
                chain.push(style);
            }
            None => {
                broken = true;
                break;
            }
        }
    }
    (chain, broken)
}

/// The `based_on` parent style id, if any. `BasedOn` serializes to a plain string.
fn based_on_id(style: &Style) -> Option<String> {
    let based_on = style.based_on.as_ref()?;
    serde_json::to_value(based_on)
        .ok()?
        .as_str()
        .map(String::from)
}

/// The document-default run styling (`docDefaults/rPrDefault`), read via serde.
fn doc_default_run(styles: &Styles) -> EffectiveRun {
    let value = serde_json::to_value(&styles.doc_defaults).unwrap_or(Value::Null);
    value
        .pointer("/runPropertyDefault/runProperty")
        .map(run_layer)
        .unwrap_or_default()
}

/// Read one layer's run properties from a serialized `RunProperty` (keys proven by T23 + T45).
fn run_layer(value: &Value) -> EffectiveRun {
    EffectiveRun {
        font: value
            .pointer("/fonts/ascii")
            .and_then(Value::as_str)
            .map(String::from),
        size_half_pt: value.get("sz").and_then(Value::as_u64),
        bold: value.get("bold").and_then(Value::as_bool),
        italic: value.get("italic").and_then(Value::as_bool),
        underline: value
            .get("underline")
            .and_then(Value::as_str)
            .map(String::from),
        color: value.get("color").and_then(Value::as_str).map(String::from),
        strike: value.get("strike").and_then(Value::as_bool),
        caps: value.get("caps").and_then(Value::as_bool),
        char_spacing: value
            .get("characterSpacing")
            .and_then(Value::as_i64)
            .map(|n| n as i32),
    }
}

/// Overlay `top` (higher precedence) onto `base`: a property set in `top` wins; otherwise `base`'s
/// value carries through.
fn overlay_run(base: EffectiveRun, top: EffectiveRun) -> EffectiveRun {
    EffectiveRun {
        font: top.font.or(base.font),
        size_half_pt: top.size_half_pt.or(base.size_half_pt),
        bold: top.bold.or(base.bold),
        italic: top.italic.or(base.italic),
        underline: top.underline.or(base.underline),
        color: top.color.or(base.color),
        strike: top.strike.or(base.strike),
        caps: top.caps.or(base.caps),
        char_spacing: top.char_spacing.or(base.char_spacing),
    }
}

/// Serialize a `RunProperty` to JSON for [`run_layer`]; `Null` (→ all-unset) if serialization fails.
fn run_property_value(property: &RunProperty) -> Value {
    serde_json::to_value(property).unwrap_or(Value::Null)
}

/// The alignment/indent/spacing carried by one paragraph layer (a style or the direct props).
#[derive(Default)]
struct ParaLayer {
    alignment: Option<String>,
    indent: IndentTwips,
    spacing: Spacing,
}

/// Read one paragraph layer's alignment / indent / spacing from a `ParagraphProperty`.
fn read_para(property: &ParagraphProperty) -> ParaLayer {
    ParaLayer {
        alignment: property.alignment.as_ref().map(|just| just.val.clone()),
        indent: property
            .indent
            .as_ref()
            .map(indent_twips)
            .unwrap_or_default(),
        spacing: property
            .line_spacing
            .as_ref()
            .map(spacing_of)
            .unwrap_or_default(),
    }
}

/// Overlay `top` paragraph layer onto `base`, per property.
fn overlay_para(base: ParaLayer, top: ParaLayer) -> ParaLayer {
    ParaLayer {
        alignment: top.alignment.or(base.alignment),
        indent: IndentTwips {
            left: top.indent.left.or(base.indent.left),
            right: top.indent.right.or(base.indent.right),
            hanging: top.indent.hanging.or(base.indent.hanging),
            first_line: top.indent.first_line.or(base.indent.first_line),
        },
        spacing: Spacing {
            before: top.spacing.before.or(base.spacing.before),
            after: top.spacing.after.or(base.spacing.after),
            line: top.spacing.line.or(base.spacing.line),
        },
    }
}

/// Convert docx-rs's [`Indent`] into [`IndentTwips`] (mirrors the extractor).
fn indent_twips(indent: &Indent) -> IndentTwips {
    let (hanging, first_line) = match indent.special_indent {
        Some(SpecialIndentType::Hanging(value)) => (Some(value), None),
        Some(SpecialIndentType::FirstLine(value)) => (None, Some(value)),
        None => (None, None),
    };
    IndentTwips {
        left: indent.start,
        right: indent.end,
        hanging,
        first_line,
    }
}

/// Read paragraph spacing from a [`LineSpacing`] via serde (its fields are private).
fn spacing_of(line_spacing: &LineSpacing) -> Spacing {
    let value = serde_json::to_value(line_spacing).unwrap_or(Value::Null);
    Spacing {
        before: value
            .get("before")
            .and_then(Value::as_u64)
            .map(|n| n as u32),
        after: value.get("after").and_then(Value::as_u64).map(|n| n as u32),
        line: value.get("line").and_then(Value::as_i64).map(|n| n as i32),
    }
}

/// The level-text template (e.g. `%1.`); read via serde since `LevelText`'s field is private.
fn level_text(text: &LevelText) -> String {
    match serde_json::to_value(text) {
        Ok(Value::String(string)) => string,
        Ok(value) => value
            .get("val")
            .and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_default(),
        Err(_) => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use docx_rs::{
        AbstractNumbering, AlignmentType, DocDefaults, Level, LevelJc, LevelText, NumberFormat,
        Numbering, Numberings, ParagraphProperty, RunFonts, RunProperty, Start, Style, StyleType,
        Styles,
    };

    #[test]
    fn doc_defaults_provide_the_floor() {
        // No paragraph style, no direct formatting — the document default size carries through.
        let mut styles = Styles::new();
        styles.doc_defaults = DocDefaults::new().size(22);
        let resolved = resolve_run(&RunProperty::new(), None, &styles);
        assert_eq!(resolved.run.size_half_pt, Some(22));
        assert!(!resolved.unresolved);
    }

    #[test]
    fn paragraph_style_resolves_when_run_is_bare() {
        let styles = Styles::new().add_style(
            Style::new("Heading2", StyleType::Paragraph)
                .fonts(RunFonts::new().ascii("Arial"))
                .size(26)
                .bold(),
        );
        let resolved = resolve_run(&RunProperty::new(), Some("Heading2"), &styles);
        assert_eq!(resolved.run.font.as_deref(), Some("Arial"));
        assert_eq!(resolved.run.size_half_pt, Some(26));
        assert_eq!(resolved.run.bold, Some(true));
    }

    #[test]
    fn based_on_parent_contributes_and_child_overrides() {
        let styles = Styles::new()
            .add_style(
                Style::new("Base", StyleType::Paragraph)
                    .fonts(RunFonts::new().ascii("Arial"))
                    .size(20),
            )
            .add_style(
                Style::new("Child", StyleType::Paragraph)
                    .based_on("Base")
                    .size(26),
            );
        let resolved = resolve_run(&RunProperty::new(), Some("Child"), &styles);
        assert_eq!(
            resolved.run.font.as_deref(),
            Some("Arial"),
            "inherited from Base"
        );
        assert_eq!(
            resolved.run.size_half_pt,
            Some(26),
            "Child overrides the size"
        );
        assert!(!resolved.unresolved);
    }

    #[test]
    fn direct_formatting_wins_over_the_style() {
        let styles =
            Styles::new().add_style(Style::new("Body", StyleType::Paragraph).size(20).bold());
        // Direct size 24 overrides the style's 20; bold still inherited from the style.
        let resolved = resolve_run(&RunProperty::new().size(24), Some("Body"), &styles);
        assert_eq!(resolved.run.size_half_pt, Some(24));
        assert_eq!(resolved.run.bold, Some(true));
    }

    #[test]
    fn missing_style_marks_unresolved() {
        let styles = Styles::new();
        let resolved = resolve_run(&RunProperty::new(), Some("Ghost"), &styles);
        assert!(
            resolved.unresolved,
            "a missing style id is unresolved, not a match"
        );
        assert_eq!(resolved.run, EffectiveRun::default());
    }

    #[test]
    fn based_on_cycle_terminates() {
        // A → B → A. The visited set breaks the loop; both styles still contribute.
        let styles = Styles::new()
            .add_style(Style::new("A", StyleType::Paragraph).based_on("B").bold())
            .add_style(Style::new("B", StyleType::Paragraph).based_on("A").size(20));
        let resolved = resolve_run(&RunProperty::new(), Some("A"), &styles);
        assert_eq!(resolved.run.bold, Some(true));
        assert_eq!(resolved.run.size_half_pt, Some(20));
        assert!(
            !resolved.unresolved,
            "both styles exist; the cycle is not a broken reference"
        );
    }

    #[test]
    fn paragraph_alignment_inherits_from_the_style() {
        let mut quote = Style::new("Quote", StyleType::Paragraph);
        quote.paragraph_property = ParagraphProperty::new().align(AlignmentType::Center);
        let styles = Styles::new().add_style(quote);

        let resolved = resolve_para(&ParagraphProperty::new().style("Quote"), &styles);
        assert_eq!(resolved.alignment.as_deref(), Some("center"));
        assert!(!resolved.unresolved);
    }

    #[test]
    fn numbering_resolves_to_its_visible_format() {
        let numberings = Numberings::new()
            .add_abstract_numbering(AbstractNumbering::new(0).add_level(Level::new(
                0,
                Start::new(1),
                NumberFormat::new("lowerLetter"),
                LevelText::new("%1."),
                LevelJc::new("left"),
            )))
            .add_numbering(Numbering::new(2, 0));

        let format = resolve_numbering(2, 0, &numberings).expect("resolves");
        assert_eq!(format.kind, "lowerLetter");
        assert_eq!(format.level_text, "%1.");

        assert!(
            resolve_numbering(99, 0, &numberings).is_none(),
            "unknown num_id"
        );
        assert!(
            resolve_numbering(2, 5, &numberings).is_none(),
            "level not defined"
        );
    }
}
