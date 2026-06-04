//! End-to-end CLI tests: run the built `stencil` binary over `.txt` and `.docx`
//! inputs and check the produced files and exit codes.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use docx_rs::{Docx, Paragraph, Run};

/// Path to the compiled binary under test (provided by Cargo for integration tests).
const BIN: &str = env!("CARGO_BIN_EXE_stencil");

fn tmp(label: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!("stencil_cli_{}_{label}.{ext}", std::process::id()))
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("failed to run stencil binary")
}

fn cleanup(paths: &[&Path]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
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
    let input = tmp("txt_detect", "txt");
    let out = tmp("txt_detect", "stencil.md"); // explicit --out keeps it predictable
    fs::write(&input, "Pay [Buyer Name] the deposit of [Amount].").expect("seed input");

    let output = run(&[
        "detect",
        input.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "detect should succeed");

    let md = fs::read_to_string(&out).expect("read output md");
    assert!(md.contains("| `[Buyer Name]` | paired | confident |"));
    assert!(md.contains("| `[Amount]` | paired | confident |"));

    cleanup(&[&input, &out]);
}

#[test]
fn txt_censor_writes_mapping_and_placeholders() {
    let input = tmp("txt_censor", "txt");
    let out = tmp("txt_censor", "stencil.md");
    let map = tmp("txt_censor", "mapping.json");
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

    cleanup(&[&input, &out, &map]);
}

#[test]
fn refuses_overwrite_without_force() {
    let input = tmp("overwrite", "txt");
    let out = tmp("overwrite", "stencil.md");
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

    cleanup(&[&input, &out]);
}

#[test]
fn docx_detect_writes_markdown() {
    let input = tmp("docx_detect", "docx");
    let out = tmp("docx_detect", "stencil.md");

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
    assert!(md.contains("| `[Item]` | paired | confident |"));
    assert!(md.contains("| `[Date]` | paired | confident |"));

    cleanup(&[&input, &out]);
}

#[test]
fn censor_then_restore_round_trips_via_cli() {
    let input = tmp("rt", "txt");
    let out = tmp("rt", "stencil.md");
    let map = tmp("rt", "mapping.json");
    let restored = tmp("rt_restored", "md");
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

    cleanup(&[&input, &out, &map, &restored]);
}
