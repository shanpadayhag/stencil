//! Thin binary shell: load a local `.env`, parse arguments, and hand off to the library.

use clap::Parser;

use stencil::cli::Cli;

fn main() -> anyhow::Result<()> {
    // Load `.env` from the working directory (or a parent) so settings like
    // STENCIL_DATA_DIR can live in a file. Missing file is fine; already-exported
    // variables win (dotenvy never overrides them), keeping precedence
    // `--data-dir` flag > real env > `.env` > default.
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();
    stencil::run(cli)
}
