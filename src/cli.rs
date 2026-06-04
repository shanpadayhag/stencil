//! Command-line argument definitions (clap).
//!
//! Kept declarative: each subcommand maps to an `Args` struct that the matching
//! module in [`crate::commands`] consumes.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// Top-level Stencil command.
#[derive(Debug, Parser)]
#[command(
    name = "stencil",
    version,
    about = "Detect bracketed template variables (and optionally censor sensitive values) into a Markdown file for Claude Code.",
    long_about = "Stencil scans a contract template (.docx or .txt) for bracketed fill-in variables and writes a context-rich Markdown file for Claude Code to read. With --censor it first replaces sensitive values (names, money, dates, IDs, emails) with REDACTED_* placeholders and writes a reversible mapping.json.\n\nIMPORTANT: Stencil is a best-effort first-pass filter, NOT a guarantee of complete redaction. Always review the censored output and the censorship summary before pasting anything into Claude."
)]
pub struct Cli {
    /// Which subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// The available subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Detect bracketed template variables and write a Markdown snippet file.
    Detect(DetectArgs),
    /// Restore real values into a file containing `REDACTED_*` placeholders.
    Restore(RestoreArgs),
}

/// Arguments for `stencil detect`.
#[derive(Debug, Args)]
#[command(
    after_help = "Review note: censoring is best-effort and not a guarantee of complete redaction. Check the censorship summary (and any ⚠ GUESSED brackets) before sharing the output."
)]
pub struct DetectArgs {
    /// Input template to scan (`.docx` or `.txt`).
    pub input: PathBuf,

    /// Output Markdown file (default: `<input>.stencil.md`).
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Censor sensitive values before emitting the Markdown.
    #[arg(long)]
    pub censor: bool,

    /// Party names to always censor: inline comma-separated, or `@file` to read from a file.
    #[arg(long)]
    pub parties: Option<String>,

    /// Mapping output, written only with `--censor` (default: `<input>.mapping.json`).
    #[arg(long)]
    pub map: Option<PathBuf>,

    /// Overwrite existing output/mapping files instead of refusing.
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `stencil restore`.
#[derive(Debug, Args)]
pub struct RestoreArgs {
    /// Input file containing `REDACTED_*` placeholders (text or Markdown).
    pub input: PathBuf,

    /// Mapping file produced by a prior `detect --censor` run.
    #[arg(long)]
    pub map: PathBuf,

    /// Restore only these placeholders (exact `REDACTED_*` tokens): inline comma-separated,
    /// or `@file`. Default: every placeholder in the mapping.
    #[arg(long, conflicts_with = "interactive")]
    pub only: Option<String>,

    /// Review each value one at a time: [space] skip, [enter] restore, [q] quit & save.
    #[arg(short, long)]
    pub interactive: bool,

    /// Restored output (default: `<input>.restored.<ext>`).
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Overwrite an existing output file instead of refusing.
    #[arg(long)]
    pub force: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        // clap's own structural assertions: catches conflicting flags, bad arg
        // configs, etc. at test time rather than runtime.
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_detect_with_flags() {
        let cli = Cli::try_parse_from([
            "stencil",
            "detect",
            "contract.txt",
            "--censor",
            "--parties",
            "Acme,Jane Doe",
            "--force",
        ])
        .expect("valid detect invocation should parse");

        match cli.command {
            Command::Detect(args) => {
                assert_eq!(args.input, PathBuf::from("contract.txt"));
                assert!(args.censor);
                assert!(args.force);
                assert_eq!(args.parties.as_deref(), Some("Acme,Jane Doe"));
                assert!(args.out.is_none());
            }
            Command::Restore(_) => panic!("expected the detect subcommand"),
        }
    }

    #[test]
    fn parses_restore_with_required_map() {
        let cli = Cli::try_parse_from([
            "stencil",
            "restore",
            "contract.stencil.md",
            "--map",
            "contract.mapping.json",
        ])
        .expect("valid restore invocation should parse");

        match cli.command {
            Command::Restore(args) => {
                assert_eq!(args.input, PathBuf::from("contract.stencil.md"));
                assert_eq!(args.map, PathBuf::from("contract.mapping.json"));
            }
            Command::Detect(_) => panic!("expected the restore subcommand"),
        }
    }

    #[test]
    fn restore_requires_map() {
        // `--map` is mandatory for restore; omitting it must be a parse error.
        let result = Cli::try_parse_from(["stencil", "restore", "contract.stencil.md"]);
        assert!(
            result.is_err(),
            "restore without --map should fail to parse"
        );
    }
}
