//! Document extraction: read an input file into the [`crate::model`] block tree.
//!
//! [`from_path`] dispatches to a format-specific reader by file extension. Each
//! format lives in its own submodule.

pub mod docx;
pub mod txt;

use std::path::Path;

use anyhow::{Result, bail};

use crate::model::Document;

/// Read an input file into the block model, choosing the reader by file extension.
///
/// Supported: `.txt` (now) and `.docx` (task T10). The match is case-insensitive.
///
/// # Errors
/// Returns an error for an unsupported or missing extension, or if the chosen reader
/// fails.
pub fn from_path(path: &Path) -> Result<Document> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);

    match extension.as_deref() {
        Some("txt") => txt::from_path(path),
        Some("docx") => docx::from_path(path),
        Some(other) => {
            bail!("unsupported input extension `.{other}` (expected .txt or .docx)")
        }
        None => bail!(
            "input `{}` has no file extension (expected .txt or .docx)",
            path.display()
        ),
    }
}

/// The 1-based page number of each block (parallel to [`from_path`]'s blocks), or `None` for a
/// format without pages (`.txt`). Only explicit `.docx` page breaks delimit pages — see
/// [`docx::page_numbers`].
///
/// # Errors
/// Returns an error for an unsupported/missing extension, or if the reader fails.
pub fn page_numbers(path: &Path) -> Result<Option<Vec<u32>>> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);

    match extension.as_deref() {
        Some("txt") => Ok(None),
        Some("docx") => Ok(Some(docx::page_numbers(path)?)),
        Some(other) => bail!("unsupported input extension `.{other}` (expected .txt or .docx)"),
        None => bail!(
            "input `{}` has no file extension (expected .txt or .docx)",
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn unsupported_extension_errors() {
        let err = from_path(&PathBuf::from("notes.pdf")).unwrap_err();
        assert!(err.to_string().contains("unsupported input extension"));
    }

    #[test]
    fn missing_extension_errors() {
        let err = from_path(&PathBuf::from("README")).unwrap_err();
        assert!(err.to_string().contains("no file extension"));
    }

    #[test]
    fn docx_routes_to_docx_reader() {
        // Routing check: a missing .docx reaches the docx reader and fails to read it.
        let err = from_path(&PathBuf::from("definitely_missing_contract.docx")).unwrap_err();
        assert!(err.to_string().contains("failed to read"));
    }
}
