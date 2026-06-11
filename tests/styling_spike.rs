//! T23 spike — proves option A and pins the serde keys T28 depends on.
//!
//! docx-rs 0.4.20 keeps run styling (font/size/bold/…) in private fields with no getters,
//! but the structs derive `Serialize`, so `serde_json::to_value(&run.run_property)` exposes
//! them. This test confirms the values are recoverable AND locks the exact JSON key paths,
//! so a future docx-rs bump that renames them fails here loudly (the pin guard) rather than
//! silently producing empty styling features downstream.
//!
//! Verified run_property JSON shape (docx-rs 0.4.20) — flat keys, None fields omitted:
//!   font (ascii)  → `/fonts/ascii`  (string)
//!   size          → `/sz`           (number, **half-points** → pt = sz/2)
//!   bold          → `/bold`         (bool)
//!   italic        → `/italic`       (bool)
//!   underline     → `/underline`    (string, e.g. "single")
//!   color         → `/color`        (string, hex RGB)
//! Complex-script twins `boldCs`/`italicCs`/`szCs` also appear; T28 reads the primary keys.
//!
//! T45 (v8) additions — same `run_property` JSON (`rename_all = "camelCase"`), proven below:
//!   strikethrough     → `/strike`            (bool; `dstrike` is the double-strike twin)
//!   character spacing → `/characterSpacing`  (i32, **twentieths of a point**; negative = condensed)
//!   all-caps          → `/caps`              (bool)
//! Limitations found in docx-rs 0.4.20 (fed back into the v8 design):
//!   - **small-caps is NOT modeled** — no `smallCaps`/`small_caps` field exists on `RunProperty`, so
//!     it cannot be captured without a docx-rs upgrade or custom XML. The v8 catalog drops it.
//!   - `Run` has no `caps()` builder (only `RunProperty::caps()`). This is irrelevant to extraction —
//!     which reads the parsed `RunProperty` regardless of how it was authored — but it's why the caps
//!     fixture sets `run_property` directly instead of chaining a `Run` builder.

use docx_rs::{Docx, Paragraph, ParagraphChild, Run, RunFonts, RunProperty, read_docx};
use std::fs;

/// Build a one-run `.docx` from `make_run`, pack/read it back, and return the first run's
/// `run_property` serialized to a JSON value.
fn first_run_property_json(label: &str, make_run: impl FnOnce(Run) -> Run) -> serde_json::Value {
    let path =
        std::env::temp_dir().join(format!("stencil_spike_{}_{label}.docx", std::process::id()));
    let docx =
        Docx::new().add_paragraph(Paragraph::new().add_run(make_run(Run::new().add_text("Hello"))));
    let file = fs::File::create(&path).expect("create fixture docx");
    docx.build().pack(file).expect("pack fixture docx");
    let bytes = fs::read(&path).expect("read fixture bytes");
    let _ = fs::remove_file(&path);

    let parsed = read_docx(&bytes).expect("parse fixture docx");
    let para = parsed
        .document
        .children
        .iter()
        .find_map(|child| match child {
            docx_rs::DocumentChild::Paragraph(p) => Some(p),
            _ => None,
        })
        .expect("a paragraph");
    let run = para
        .children
        .iter()
        .find_map(|child| match child {
            ParagraphChild::Run(r) => Some(r),
            _ => None,
        })
        .expect("a run");

    serde_json::to_value(&run.run_property).expect("serialize run_property")
}

#[test]
fn styled_run_exposes_font_size_bold_via_serde() {
    let v = first_run_property_json("styled", |run| {
        run.fonts(RunFonts::new().ascii("Courier New"))
            .size(28)
            .bold()
            .italic()
            .underline("single")
            .color("FF0000")
    });

    assert_eq!(
        v.pointer("/fonts/ascii").and_then(|f| f.as_str()),
        Some("Courier New"),
        "font name reachable at /fonts/ascii"
    );
    assert_eq!(
        v.get("sz").and_then(|s| s.as_u64()),
        Some(28),
        "size reachable at /sz (half-points → 14pt)"
    );
    assert_eq!(
        v.get("bold").and_then(|b| b.as_bool()),
        Some(true),
        "bold reachable at /bold"
    );
    assert_eq!(v.get("italic").and_then(|b| b.as_bool()), Some(true));
    assert_eq!(v.get("underline").and_then(|u| u.as_str()), Some("single"));
    assert_eq!(v.get("color").and_then(|c| c.as_str()), Some("FF0000"));
}

#[test]
fn unstyled_run_omits_the_keys() {
    // None-valued properties are skipped in serialization, so absence (not `false`/`0`) is
    // how T28 must detect "not set" — confirm that contract here.
    let v = first_run_property_json("plain", |run| run);
    assert!(v.get("bold").is_none(), "unset bold is omitted, got: {v}");
    assert!(v.get("sz").is_none(), "unset size is omitted, got: {v}");
    assert!(v.get("fonts").is_none(), "unset fonts is omitted, got: {v}");
}

#[test]
fn strike_and_character_spacing_survive_round_trip() {
    // Both have `Run` builders; condensed spacing is a negative value in twentieths of a point
    // (Word's "Condensed by 0.15pt" → -3). The resolver (T47/T48) reads these keys off the parsed
    // RunProperty exactly as it reads bold/size today.
    let v = first_run_property_json("strike_spacing", |run| run.strike().character_spacing(-3));

    assert_eq!(
        v.get("strike").and_then(|b| b.as_bool()),
        Some(true),
        "strikethrough reachable at /strike (bool), got: {v}"
    );
    assert_eq!(
        v.get("characterSpacing").and_then(|n| n.as_i64()),
        Some(-3),
        "character spacing reachable at /characterSpacing (i32, 20ths of a pt; negative = condensed), got: {v}"
    );
}

#[test]
fn caps_reads_back_at_the_caps_key() {
    // `Run` has no caps() builder, so author it on the RunProperty directly. What matters is that a
    // parsed document exposes caps at `/caps` as a bool — which is what the resolver consumes.
    let v = first_run_property_json("caps", |run| {
        let mut run = run;
        run.run_property = RunProperty::new().caps();
        run
    });

    assert_eq!(
        v.get("caps").and_then(|b| b.as_bool()),
        Some(true),
        "all-caps reachable at /caps (bool), got: {v}"
    );
}
