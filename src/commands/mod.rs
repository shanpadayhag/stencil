//! Subcommand orchestration: each module wires the pipeline stages together for
//! one CLI subcommand.

pub mod detect;
pub mod restore;

use std::path::Path;

use anyhow::{Result, bail};

/// Error if `path` already exists and `force` was not given. Shared by the commands
/// that write output files.
pub(crate) fn ensure_writable(path: &Path, force: bool) -> Result<()> {
    if path.exists() && !force {
        bail!(
            "refusing to overwrite existing file `{}` (pass --force to overwrite)",
            path.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn refuses_existing_without_force() {
        let path: PathBuf =
            std::env::temp_dir().join(format!("stencil_ew_{}.tmp", std::process::id()));
        fs::write(&path, "x").expect("seed");

        assert!(ensure_writable(&path, false).is_err());
        assert!(ensure_writable(&path, true).is_ok());

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn allows_missing_file() {
        let path = std::env::temp_dir().join("stencil_ew_missing_zzz.tmp");
        let _ = fs::remove_file(&path);
        assert!(ensure_writable(&path, false).is_ok());
    }
}
