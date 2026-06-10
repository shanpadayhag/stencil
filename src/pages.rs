//! Parse and apply a `--pages` range selection (v7): scope the interactive review to part of a
//! document while the rest is auto-censored. Pages are delimited by explicit `.docx` page breaks
//! (see [`crate::extract::page_numbers`]).

use anyhow::{Result, anyhow, bail};

/// A selection of 1-based page numbers, parsed from a `--pages` spec like `2-3`, `5`, or
/// `1,3,5-7` (comma-separated pages and inclusive ranges).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageSelection {
    /// Inclusive `(start, end)` ranges, in input order.
    ranges: Vec<(u32, u32)>,
}

impl PageSelection {
    /// Parse a `--pages` spec.
    ///
    /// # Errors
    /// Returns an error on a non-numeric page, a `0` page (pages are 1-based), a reversed range,
    /// or an empty selection.
    ///
    /// ```
    /// use stencil::pages::PageSelection;
    ///
    /// let sel = PageSelection::parse("2-3,5").expect("valid");
    /// assert!(sel.contains(2) && sel.contains(3) && sel.contains(5));
    /// assert!(!sel.contains(4));
    /// assert_eq!(sel.max_page(), 5);
    /// ```
    pub fn parse(spec: &str) -> Result<Self> {
        let mut ranges = Vec::new();
        for part in spec
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
        {
            let (start, end) = match part.split_once('-') {
                Some((start, end)) => (parse_page(start)?, parse_page(end)?),
                None => {
                    let page = parse_page(part)?;
                    (page, page)
                }
            };
            if start > end {
                bail!("invalid --pages range `{part}`: start page {start} is after end page {end}");
            }
            ranges.push((start, end));
        }
        if ranges.is_empty() {
            bail!("--pages selection is empty");
        }
        Ok(Self { ranges })
    }

    /// Whether `page` is in the selection.
    pub fn contains(&self, page: u32) -> bool {
        self.ranges
            .iter()
            .any(|&(start, end)| (start..=end).contains(&page))
    }

    /// The highest page the selection references — used to reject ranges beyond the document.
    pub fn max_page(&self) -> u32 {
        self.ranges.iter().map(|&(_, end)| end).max().unwrap_or(1)
    }
}

/// Parse one 1-based page number.
fn parse_page(text: &str) -> Result<u32> {
    let page: u32 = text
        .trim()
        .parse()
        .map_err(|_| anyhow!("invalid page number `{text}` in --pages"))?;
    if page == 0 {
        bail!("page numbers are 1-based; got `0` in --pages");
    }
    Ok(page)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_range_and_list() {
        let sel = PageSelection::parse("2-3").expect("range");
        assert!(!sel.contains(1) && sel.contains(2) && sel.contains(3) && !sel.contains(4));

        let list = PageSelection::parse("1, 3, 5-7").expect("list");
        assert!(list.contains(1) && !list.contains(2) && list.contains(3));
        assert!(list.contains(5) && list.contains(6) && list.contains(7) && !list.contains(8));
        assert_eq!(list.max_page(), 7);
    }

    #[test]
    fn single_page_is_a_unit_range() {
        let sel = PageSelection::parse("4").expect("single");
        assert!(sel.contains(4) && !sel.contains(3));
        assert_eq!(sel.max_page(), 4);
    }

    #[test]
    fn rejects_bad_specs() {
        assert!(PageSelection::parse("3-1").is_err(), "reversed range");
        assert!(PageSelection::parse("0").is_err(), "zero page");
        assert!(PageSelection::parse("x").is_err(), "non-numeric");
        assert!(PageSelection::parse("").is_err(), "empty");
        assert!(PageSelection::parse("  ,  ").is_err(), "all-empty parts");
    }
}
