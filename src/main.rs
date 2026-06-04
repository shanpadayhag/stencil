//! Thin binary shell: parse arguments and hand off to the library.

use clap::Parser;

use stencil::cli::Cli;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    stencil::run(cli)
}
