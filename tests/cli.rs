//! End-to-end CLI tests for the `review` and `style` command surface.
//!
//! These cover the command-line contract — the `review` and `style` commands exist with their
//! flags, the removed `detect`/`restore` commands are gone, `--only`/`--skip` are mutually
//! exclusive, and a non-TTY invocation of either is refused. The interactive pipeline halves that
//! don't need a PTY are exercised end-to-end in `tests/review_pipeline.rs`.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Path to the compiled binary under test (provided by Cargo for integration tests).
const BIN: &str = env!("CARGO_BIN_EXE_stencil");

/// Run the binary with an isolated config dir and **no TTY** on stdin, so the non-TTY gate
/// behaves deterministically regardless of how the test runner is launched.
fn run(args: &[&str]) -> std::process::Output {
    let cfg = std::env::temp_dir().join(format!("stencil_cli_cfg_{}", std::process::id()));
    Command::new(BIN)
        .args(args)
        .env("XDG_CONFIG_HOME", &cfg)
        .stdin(Stdio::null())
        .output()
        .expect("failed to run stencil binary")
}

#[test]
fn help_lists_review_flags_and_carries_disclaimer() {
    let output = run(&["review", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--only") && stdout.contains("--skip"),
        "review --help lists the stage flags; got:\n{stdout}"
    );
    assert!(
        stdout.contains("not a guarantee of complete redaction"),
        "review --help should carry the review disclaimer; got:\n{stdout}"
    );
    assert!(
        !stdout.contains("mapping.json") && !stdout.to_lowercase().contains("restore"),
        "v6 dropped restore/mapping.json; help must not mention them; got:\n{stdout}"
    );
}

#[test]
fn only_and_skip_are_mutually_exclusive() {
    let output = run(&["review", "c.docx", "--only", "censor", "--skip", "snippet"]);
    assert!(
        !output.status.success(),
        "--only and --skip cannot be combined"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--skip") || stderr.contains("cannot be used with"),
        "should report the conflict; got:\n{stderr}"
    );
}

#[test]
fn unknown_stage_is_rejected() {
    let output = run(&["review", "c.docx", "--only", "nonsense"]);
    assert!(
        !output.status.success(),
        "an unknown stage value must be rejected by the parser"
    );
}

#[test]
fn detect_and_restore_are_unknown_commands() {
    assert!(
        !run(&["detect", "c.txt"]).status.success(),
        "the detect command was removed in v6"
    );
    assert!(
        !run(&["restore", "c.md", "--map", "m.json"])
            .status
            .success(),
        "the restore command was removed in v6"
    );
}

#[test]
fn style_help_lists_its_flags() {
    let output = run(&["style", "--help"]);
    assert!(output.status.success(), "style --help should succeed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--pages") && stdout.contains("--lang"),
        "style --help lists --pages and --lang; got:\n{stdout}"
    );
}

#[test]
fn style_without_a_tty_is_refused() {
    // The TTY gate fires before the file is read, so a non-existent path is fine here.
    let output = run(&["style", "contract.docx"]);
    assert!(
        !output.status.success(),
        "style must refuse to run without an interactive terminal"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("interactive terminal"),
        "should explain the TTY requirement; got:\n{stderr}"
    );
}

#[test]
fn review_without_a_tty_is_refused() {
    let input: PathBuf =
        std::env::temp_dir().join(format!("stencil_cli_notty_{}.txt", std::process::id()));
    fs::write(&input, "Pay [Buyer Name].").expect("seed input");

    let output = run(&["review", input.to_str().unwrap()]);
    assert!(
        !output.status.success(),
        "review must refuse to run without an interactive terminal"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("interactive terminal"),
        "should explain the TTY requirement; got:\n{stderr}"
    );

    let _ = fs::remove_file(&input);
}
