//! Subcommand orchestration: each module wires the pipeline stages together for the CLI.
//!
//! `review` runs censor → snippet; `style` is the standalone styling review (v7).

pub mod review;
pub mod style;

use std::path::Path;

use anyhow::{Result, bail};

/// Whether `path` has a `.docx` extension (case-insensitive).
pub(crate) fn is_docx(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("docx"))
}

/// The forced language code from a `--lang` value, or `None` for auto-detect.
pub(crate) fn lang_override(lang: &str) -> Option<&str> {
    (lang != "auto").then_some(lang)
}

/// Error if `path` already exists and `force` was not given. Shared by the stages that write
/// output files so a run never silently clobbers a prior one.
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

    #[test]
    fn is_docx_is_case_insensitive_and_extension_only() {
        assert!(is_docx(Path::new("dir/Contract.docx")));
        assert!(is_docx(Path::new("dir/Contract.DOCX")));
        assert!(!is_docx(Path::new("dir/contract.txt")));
        assert!(!is_docx(Path::new("docx")));
    }

    #[test]
    fn lang_override_maps_auto_to_none() {
        assert_eq!(lang_override("auto"), None);
        assert_eq!(lang_override("fr"), Some("fr"));
    }
}
