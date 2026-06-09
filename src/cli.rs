//! Command-line argument definitions (clap).
//!
//! Kept declarative: the single `review` subcommand maps to [`ReviewArgs`], which the
//! [`crate::commands::review`] module consumes.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// Top-level Stencil command.
#[derive(Debug, Parser)]
#[command(
    name = "stencil",
    version,
    about = "Interactively review a contract template: censor sensitive values, study document styling, and write a Markdown file for Claude Code.",
    long_about = "Stencil's `review` command walks a contract template (.docx or .txt) through an interactive pipeline:\n\n  1. censor  — over-detect sensitive values; confirm / reject / re-type each one\n  2. styling — walk every block and flag formatting that looks wrong\n  3. snippet — write the context-rich Markdown + per-bracket snippet files for Claude Code\n\nThe review stages need an interactive terminal (TTY). IMPORTANT: censoring is a best-effort first-pass filter, NOT a guarantee of complete redaction — always review the output before sharing."
)]
pub struct Cli {
    /// Which subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// The available subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Review a template: censor, study styling, and write the Markdown snippet file.
    Review(ReviewArgs),
}

/// One stage of the `review` pipeline. Stages always run in pipeline order
/// (`censor` → `styling` → `snippet`); `--only`/`--skip` just choose which run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Stage {
    /// Detect and interactively review sensitive values.
    Censor,
    /// Walk every block and review its styling.
    Styling,
    /// Write the Markdown inventory and per-bracket snippet files.
    Snippet,
}

impl Stage {
    /// Every stage, in pipeline order.
    pub const ALL: [Stage; 3] = [Stage::Censor, Stage::Styling, Stage::Snippet];

    /// The lowercase stage name as used on the command line.
    pub fn label(self) -> &'static str {
        match self {
            Stage::Censor => "censor",
            Stage::Styling => "styling",
            Stage::Snippet => "snippet",
        }
    }
}

/// Arguments for `stencil review`.
#[derive(Debug, Args)]
#[command(
    after_help = "Review note: censoring is best-effort and not a guarantee of complete redaction. Check the censorship summary (and any ⚠ GUESSED brackets) before sharing the output."
)]
pub struct ReviewArgs {
    /// Input template to review (`.docx` or `.txt`).
    pub input: PathBuf,

    /// Output Markdown file (default: `<input>.stencil.md`).
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Party names to always censor: inline comma-separated, or `@file` to read from a file.
    #[arg(long)]
    pub parties: Option<String>,

    /// Run only these stages (comma-separated or repeated). Mutually exclusive with `--skip`.
    #[arg(long, value_enum, value_delimiter = ',', conflicts_with = "skip")]
    pub only: Vec<Stage>,

    /// Skip these stages (comma-separated or repeated). Mutually exclusive with `--only`.
    #[arg(long, value_enum, value_delimiter = ',')]
    pub skip: Vec<Stage>,

    /// Root directory for the learning stores (default: `$XDG_CONFIG_HOME/stencil` or
    /// `~/.config/stencil`). Per-model subdirs `censor/` and `styling/` live under it.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,

    /// Override the censor store location (else `<data_dir>/censor/`; env `STENCIL_CENSOR_DIR`).
    #[arg(long)]
    pub censor_dir: Option<PathBuf>,

    /// Override the styling store location (else `<data_dir>/styling/`; env `STENCIL_STYLING_DIR`).
    #[arg(long)]
    pub styling_dir: Option<PathBuf>,

    /// Overwrite existing output/snippet files instead of refusing.
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
    fn parses_review_with_flags() {
        let cli = Cli::try_parse_from([
            "stencil",
            "review",
            "contract.docx",
            "--parties",
            "Acme,Jane Doe",
            "--force",
        ])
        .expect("valid review invocation should parse");

        let Command::Review(args) = cli.command;
        assert_eq!(args.input, PathBuf::from("contract.docx"));
        assert!(args.force);
        assert_eq!(args.parties.as_deref(), Some("Acme,Jane Doe"));
        assert!(args.out.is_none());
        assert!(args.only.is_empty());
        assert!(args.skip.is_empty());
    }

    #[test]
    fn parses_only_as_comma_list() {
        let cli = Cli::try_parse_from(["stencil", "review", "c.docx", "--only", "censor,snippet"])
            .expect("comma-separated stages should parse");
        let Command::Review(args) = cli.command;
        assert_eq!(args.only, vec![Stage::Censor, Stage::Snippet]);
    }

    #[test]
    fn rejects_unknown_stage() {
        let result = Cli::try_parse_from(["stencil", "review", "c.docx", "--skip", "bogus"]);
        assert!(result.is_err(), "an unknown stage name must fail to parse");
    }

    #[test]
    fn only_and_skip_are_mutually_exclusive() {
        let result = Cli::try_parse_from([
            "stencil", "review", "c.docx", "--only", "censor", "--skip", "snippet",
        ]);
        assert!(result.is_err(), "--only and --skip together must error");
    }

    #[test]
    fn detect_and_restore_are_gone() {
        assert!(Cli::try_parse_from(["stencil", "detect", "c.txt"]).is_err());
        assert!(Cli::try_parse_from(["stencil", "restore", "c.md", "--map", "m.json"]).is_err());
    }
}
