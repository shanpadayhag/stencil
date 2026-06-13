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
    about = "Interactively review a contract template: censor sensitive values and write a Markdown file for Claude Code, or study document styling.",
    long_about = "Stencil has two interactive commands over a contract template, plus two that work with the models it learns:\n\n  review <doc>  — over-detect sensitive values (confirm / reject / re-type / edit / split each), then write the context-rich Markdown + per-bracket snippet files for Claude Code\n  style  <doc>  — walk every block and flag formatting that looks wrong (fix it in Word, then run `review`)\n  train         — rebuild the suggestive styling/censor models from your logged reviews (full batch)\n  accuracy      — show each model's recent (prequential) accuracy\n\nTypical flow: run `style` first, fix the flagged blocks, then `review`; run `train` once you have enough reviews and the models add an advisory green/red suggestion line (never changing the output). review/style need an interactive terminal (TTY). IMPORTANT: censoring is a best-effort first-pass filter, NOT a guarantee of complete redaction — always review the output before sharing."
)]
pub struct Cli {
    /// Which subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// The available subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Review a template: censor sensitive values and write the Markdown snippet file.
    Review(ReviewArgs),
    /// Review document styling, surfacing formatting that looks wrong (fix it in Word, then
    /// `review`). Records the per-block fine/weird labels for the future styling model.
    Style(StyleArgs),
    /// Rebuild the suggestive models from the logged reviews (full batch). No flags trains both;
    /// `--styling`/`--censor` scope to one. Advisory only — never changes detection or output.
    Train(TrainArgs),
    /// Show each model's recent (prequential) accuracy over the last 100 logged predictions.
    Accuracy(AccuracyArgs),
}

/// One stage of the `review` pipeline. Stages run in pipeline order (`censor` → `snippet`);
/// `--only`/`--skip` just choose which run. (Styling is its own `stencil style` command in v7.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Stage {
    /// Detect and interactively review sensitive values.
    Censor,
    /// Write the Markdown inventory and per-bracket snippet files.
    Snippet,
}

impl Stage {
    /// Every stage, in pipeline order.
    pub const ALL: [Stage; 2] = [Stage::Censor, Stage::Snippet];

    /// The lowercase stage name as used on the command line.
    pub fn label(self) -> &'static str {
        match self {
            Stage::Censor => "censor",
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

    /// Language for the per-block training feature: `auto` (detect, default) or a forced code
    /// such as `en` or `fr`.
    #[arg(long, default_value = "auto")]
    pub lang: String,

    /// Scope the censor review to these pages (e.g. `2-3` or `1,3,5-7`); other pages are still
    /// censored, just not reviewed. Requires explicit `.docx` page breaks.
    #[arg(long)]
    pub pages: Option<String>,

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

    /// Overwrite existing output/snippet files instead of refusing.
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `stencil style` — the standalone styling review (v7).
#[derive(Debug, Args)]
pub struct StyleArgs {
    /// Input template to review (`.docx`; styling is `.docx`-only).
    pub input: PathBuf,

    /// Language for the per-block training feature: `auto` (detect, default) or a forced code.
    #[arg(long, default_value = "auto")]
    pub lang: String,

    /// Scope the review to these pages (e.g. `2-3` or `1,3,5-7`). Requires explicit page breaks.
    #[arg(long)]
    pub pages: Option<String>,

    /// Root directory for the learning stores (default: `$XDG_CONFIG_HOME/stencil` or
    /// `~/.config/stencil`).
    #[arg(long)]
    pub data_dir: Option<PathBuf>,

    /// Override the styling store location (else `<data_dir>/styling/`; env `STENCIL_STYLING_DIR`).
    #[arg(long)]
    pub styling_dir: Option<PathBuf>,
}

/// Arguments for `stencil train` — rebuild the suggestive models from the logs (v11).
#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Train only the styling model (default: train both).
    #[arg(long)]
    pub styling: bool,

    /// Train only the censor model (default: train both).
    #[arg(long)]
    pub censor: bool,

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
}

/// Arguments for `stencil accuracy` — the prequential accuracy meters (v11).
#[derive(Debug, Args)]
pub struct AccuracyArgs {
    /// Root directory for the learning stores (default: `$XDG_CONFIG_HOME/stencil` or
    /// `~/.config/stencil`).
    #[arg(long)]
    pub data_dir: Option<PathBuf>,

    /// Override the censor store location (else `<data_dir>/censor/`; env `STENCIL_CENSOR_DIR`).
    #[arg(long)]
    pub censor_dir: Option<PathBuf>,

    /// Override the styling store location (else `<data_dir>/styling/`; env `STENCIL_STYLING_DIR`).
    #[arg(long)]
    pub styling_dir: Option<PathBuf>,
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

        let Command::Review(args) = cli.command else {
            panic!("expected the review subcommand");
        };
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
        let Command::Review(args) = cli.command else {
            panic!("expected the review subcommand");
        };
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
    fn parses_train_defaults_to_both_models() {
        let cli = Cli::try_parse_from(["stencil", "train"]).expect("bare train should parse");
        let Command::Train(args) = cli.command else {
            panic!("expected the train subcommand");
        };
        // No flags ⇒ neither is set; the command treats that as "train both".
        assert!(!args.styling);
        assert!(!args.censor);
    }

    #[test]
    fn parses_train_scoped_to_styling() {
        let cli = Cli::try_parse_from(["stencil", "train", "--styling", "--data-dir", "/tmp/d"])
            .expect("scoped train should parse");
        let Command::Train(args) = cli.command else {
            panic!("expected the train subcommand");
        };
        assert!(args.styling);
        assert!(!args.censor);
        assert_eq!(args.data_dir, Some(PathBuf::from("/tmp/d")));
    }

    #[test]
    fn detect_and_restore_are_gone() {
        assert!(Cli::try_parse_from(["stencil", "detect", "c.txt"]).is_err());
        assert!(Cli::try_parse_from(["stencil", "restore", "c.md", "--map", "m.json"]).is_err());
    }
}
