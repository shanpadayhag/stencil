//! Stencil — a local-only CLI that turns a contract template into a Markdown file
//! Claude Code can read.
//!
//! The library is the testable core; the binary ([`main`](../main/index.html)) is a
//! thin shell that parses arguments and calls [`run`]. The processing pipeline lives
//! in focused modules:
//!
//! - [`extract`] — read `.docx`/`.txt` into the [`model`] block tree
//! - [`censor`] — replace sensitive values with `REDACTED_*` placeholders
//! - [`detect`] — find bracketed variables and tally bracket balance
//! - [`section`] — slice the document into heading-delimited sections
//! - [`render`] — emit the per-section Markdown
//! - [`review`] — write per-candidate censored files for cross-paragraph spans
//! - [`style`] — read each block's formatting for the styling-review stage
//! - [`learn`] — persist review decisions so censoring improves over time
//! - [`commands`] — orchestrate the above for the `review` pipeline
#![forbid(unsafe_code)]

pub mod censor;
pub mod cli;
pub mod commands;
pub mod detect;
pub mod extract;
pub mod learn;
pub mod model;
pub mod render;
pub mod review;
pub mod section;
pub mod style;

use anyhow::Result;

use crate::cli::{Cli, Command};

/// Dispatch a parsed [`Cli`] to the matching subcommand.
///
/// This is the single entry point the binary calls.
///
/// ```no_run
/// use clap::Parser;
/// use stencil::cli::Cli;
///
/// let cli = Cli::parse();
/// stencil::run(cli)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Review(args) => commands::review::run(args),
    }
}
