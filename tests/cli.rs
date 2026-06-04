//! End-to-end CLI tests: run the built `stencil` binary over `.txt` and `.docx`
//! inputs and check the produced files and exit codes.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use docx_rs::{Docx, Paragraph, Run};

/// Path to the compiled binary under test (provided by Cargo for integration tests).
const BIN: &str = env!("CARGO_BIN_EXE_stencil");

/// A fresh, isolated work directory for a test. Detect now writes a `snippets/` folder
/// beside its output, so each test needs its own directory to avoid clashing with others
/// running in parallel.
fn work_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("stencil_cli_{}_{label}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create work dir");
    dir
}

fn run(args: &[&str]) -> std::process::Output {
    // Isolate the learned store/log from the developer's real `~/.config/stencil`.
    let cfg = std::env::temp_dir().join(format!("stencil_cli_cfg_{}", std::process::id()));
    Command::new(BIN)
        .args(args)
        .env("XDG_CONFIG_HOME", &cfg)
        .output()
        .expect("failed to run stencil binary")
}

#[test]
fn help_includes_redaction_disclaimer() {
    let output = run(&["detect", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("not a guarantee of complete redaction"),
        "detect --help should carry the review disclaimer; got:\n{stdout}"
    );
}

#[test]
fn txt_detect_writes_markdown() {
    let dir = work_dir("txt_detect");
    let input = dir.join("contract.txt");
    let out = dir.join("contract.stencil.md"); // explicit --out keeps it predictable
    fs::write(&input, "Pay [Buyer Name] the deposit of [Amount].").expect("seed input");

    let output = run(&[
        "detect",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "detect should succeed");

    let md = fs::read_to_string(&out).expect("read output md");
    // The inventory snippet is the whole paragraph each bracket sits in.
    assert!(md.contains("| `Pay [Buyer Name] the deposit of [Amount].` | paired | confident |"));

    // A censored snippet file is written for the paragraph.
    let snippet = dir
        .join("snippets")
        .join("pay-buyer-name-the-deposit-of-amount.md");
    assert!(snippet.exists(), "expected snippet file at {snippet:?}");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn detect_writes_crawlable_snippet_map() {
    let dir = work_dir("crawl_map");
    let input = dir.join("contract.txt");
    let out = dir.join("contract.stencil.md");
    fs::write(&input, "Pay [Buyer Name] the deposit of [Amount].").expect("seed input");

    let output = run(&[
        "detect",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "detect should succeed");

    let md = fs::read_to_string(&out).expect("read output md");
    // A top-level index enumerates the snippet file with an ID and a relative link.
    assert!(md.contains("## Snippet index"), "top index present");
    assert!(md.contains("[snippets/pay-buyer-name-the-deposit-of-amount.md](snippets/pay-buyer-name-the-deposit-of-amount.md)"));
    // The inventory row shares that ID (both brackets are in block 0 → one file, S1).
    assert!(md.contains("| S1 |"), "inventory row carries the shared ID");

    // The snippet file links back up to the main inventory, closing the crawl loop.
    let snippet = dir
        .join("snippets")
        .join("pay-buyer-name-the-deposit-of-amount.md");
    let snippet_md = fs::read_to_string(&snippet).expect("read snippet file");
    assert!(snippet_md.contains("# S1 — snippet, block 0"));
    assert!(snippet_md.contains("[`../contract.stencil.md`](../contract.stencil.md)"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn txt_censor_writes_mapping_and_placeholders() {
    let dir = work_dir("txt_censor");
    let input = dir.join("contract.txt");
    let out = dir.join("contract.stencil.md");
    let map = dir.join("contract.mapping.json");
    fs::write(&input, "Email billing@acme.example about [Invoice].").expect("seed input");

    let output = run(&[
        "detect",
        input.to_str().unwrap(),
        "--censor",
        "--out",
        out.to_str().unwrap(),
        "--map",
        map.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "censor detect should succeed");

    let md = fs::read_to_string(&out).expect("read md");
    assert!(md.contains("REDACTED_EMAIL_001"));
    assert!(
        !md.contains("billing@acme.example"),
        "real value must not leak"
    );

    let mapping = fs::read_to_string(&map).expect("read mapping");
    assert!(mapping.contains("\"billing@acme.example\""));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn refuses_overwrite_without_force() {
    let dir = work_dir("overwrite");
    let input = dir.join("contract.txt");
    let out = dir.join("contract.stencil.md");
    fs::write(&input, "[X]").expect("seed");
    fs::write(&out, "pre-existing").expect("seed out");

    let blocked = run(&[
        "detect",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(!blocked.status.success(), "should refuse without --force");
    let stderr = String::from_utf8_lossy(&blocked.stderr);
    assert!(stderr.contains("refusing to overwrite"));

    let forced = run(&[
        "detect",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--force",
    ]);
    assert!(forced.status.success(), "should overwrite with --force");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn docx_detect_writes_markdown() {
    let dir = work_dir("docx_detect");
    let input = dir.join("contract.docx");
    let out = dir.join("contract.stencil.md");

    let docx = Docx::new()
        .add_paragraph(
            Paragraph::new()
                .style("Heading1")
                .add_run(Run::new().add_text("Scope")),
        )
        .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Deliver [Item] by [Date].")));
    let file = fs::File::create(&input).expect("create docx");
    docx.build().pack(file).expect("pack docx");

    let output = run(&[
        "detect",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "docx detect should succeed");

    let md = fs::read_to_string(&out).expect("read md");
    assert!(md.contains("# Scope"));
    // Both brackets share the paragraph, so the snippet is the whole sentence.
    assert!(md.contains("| `Deliver [Item] by [Date].` | paired | confident |"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn cross_paragraph_span_writes_censored_review_file_and_is_quiet() {
    // A dedicated work dir keeps the generated `cross-paragraph/` subfolder isolated.
    let dir = std::env::temp_dir().join(format!("stencil_xpara_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create work dir");

    let input = dir.join("contract.txt");
    let out = dir.join("contract.stencil.md");
    // `[` opens in the first paragraph, `]` closes in the second (blank line between).
    fs::write(
        &input,
        "[if buyer billing@acme.example defaults\n\nthe deposit is forfeited]",
    )
    .expect("seed input");

    let output = run(&[
        "detect",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "detect should succeed");

    // Main output inventories the span and flags it for review.
    let md = fs::read_to_string(&out).expect("read md");
    assert!(md.contains("paired (cross-paragraph)"));
    assert!(md.contains("⚠ GUESSED"));

    // The cross-paragraph inventory row is censored even without --censor, so the table
    // never dumps a raw value (the section body above it is a separate concern).
    let row = md
        .lines()
        .find(|line| line.contains("paired (cross-paragraph)"))
        .expect("cross-paragraph inventory row");
    assert!(
        row.contains("REDACTED_EMAIL_001"),
        "row preview is censored"
    );
    assert!(
        !row.contains("billing@acme.example"),
        "inventory row must not leak the value"
    );

    // The review file is censored even though the main run had no --censor.
    let review = dir
        .join("snippets")
        .join("cross-paragraph")
        .join("if-buyer-redacted-email-001-defaults.md");
    let review_md = fs::read_to_string(&review).expect("read review file");
    assert!(review_md.contains("REDACTED_EMAIL_001"));
    assert!(
        !review_md.contains("billing@acme.example"),
        "review file must be censored before sharing"
    );

    // Quiet on success: one stdout line, nothing on stderr.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.lines().count(), 1, "exactly one confirmation line");
    assert!(stdout.contains("snippet file"));
    assert!(
        output.stderr.is_empty(),
        "no noisy diagnostics on success; got: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn censor_then_restore_round_trips_via_cli() {
    let dir = work_dir("rt");
    let input = dir.join("contract.txt");
    let out = dir.join("contract.stencil.md");
    let map = dir.join("contract.mapping.json");
    let restored = dir.join("contract.restored.md");
    fs::write(&input, "Invoice billing@acme.example for [Service].").expect("seed");

    let censor = run(&[
        "detect",
        input.to_str().unwrap(),
        "--censor",
        "--out",
        out.to_str().unwrap(),
        "--map",
        map.to_str().unwrap(),
    ]);
    assert!(censor.status.success());

    let restore = run(&[
        "restore",
        out.to_str().unwrap(),
        "--map",
        map.to_str().unwrap(),
        "--out",
        restored.to_str().unwrap(),
    ]);
    assert!(restore.status.success());

    let text = fs::read_to_string(&restored).expect("read restored");
    assert!(
        text.contains("billing@acme.example"),
        "value should be restored"
    );
    assert!(!text.contains("REDACTED_"), "no placeholders should remain");

    let _ = fs::remove_dir_all(&dir);
}
