# Stencil

A fully-local Rust CLI that walks a contract template through interactive review and writes a
context-rich Markdown file Claude Code can read.

Stencil takes a template (`.docx` or `.txt`) and gives you two interactive commands:

- **`review`** — over-detect sensitive values, decide each one, and write a censored, navigable
  `<stem>.stencil.md` plus per-bracket snippet files (censor → snippet).
- **`style`** *(`.docx` only)* — walk every block's formatting and flag what looks wrong, so you can
  fix it in Word before reviewing.

Every decision you make also trains Stencil's local, deterministic memory and writes append-only
training logs for two future models — no ML runs yet; this is data collection only.

> ⚠️ **Review before pasting.** Stencil is a best-effort first-pass filter, **not a guarantee of
> complete redaction**. Always review the output (and any ⚠ GUESSED brackets) before pasting
> anything into Claude.

## Install

Requires the Rust toolchain (`rustup`). Build the release binary:

```sh
cargo build --release
# binary at target/release/stencil
```

## Usage

Both commands are interactive — they read single keypresses — so they must be run directly in a
terminal (TTY). The intended flow is **`style` first, fix the flagged blocks in Word, then
`review`**:

```sh
stencil style contract.docx                                   # flag odd formatting; fix it in Word
stencil review contract.docx --parties "Acme Corporation, Jane Doe"
# → contract.stencil.md   (context shows REDACTED_* placeholders)
# → snippets/             (one censored file per bracket span)
```

### `review` — censor → snippet

1. **censor** — over-detects sensitive values (names, money, percentages, dates, emails, phones,
   IDs, …) in **English and French**, and walks you through each one:

   | key | action |
   |-----|--------|
   | `c` | confirm (keep censored) |
   | `t` | re-type (confirm, pick the correct category) |
   | `x` | reject (false positive — left in the clear) |
   | `e` | edit the censored span (extend/trim the value) |
   | `n` | add a value the detector missed |
   | `w` | correct the recorded context window |
   | `s` | split a value into its occurrences and decide each separately |
   | `b` / `q` | back · quit & save |

   Only confirmed values are censored. Splitting is for context-dependent values — e.g. `3%` as a
   contractual rate in one place and a throwaway statistic in another.

2. **snippet** — writes the `.stencil.md` section inventory plus a censored snippet file per bracket
   span, cross-linked for navigation. Always whole-document, even under `--pages`.

Run a subset with `--only` / `--skip` (mutually exclusive, comma-separated `censor`/`snippet`):

```sh
stencil review contract.docx --only censor     # just the censor pass
stencil review contract.docx --skip snippet    # censor only
```

### `style` — standalone styling review

Walks every block (paragraphs, headings, list items, table cells) showing its formatting against the
document's norms; you mark each **[space]** fine or **[w]** weird (with a category and an optional
note). It **never edits the document** — fix the flagged blocks yourself in Word, then run `review`.
The per-block labels are recorded as training data for a future "should this fix be applied?" model.

### Flags

Shared by both commands:

- `--lang <auto|code>` — language for the per-block training feature: `auto` (detect, default) or a
  forced code like `en` / `fr`.
- `--pages <range>` — scope the **review** to part of the document (e.g. `2-3` or `1,3,5-7`); other
  pages are still censored, just not reviewed. Requires explicit `.docx` page breaks.
- `--data-dir <dir>` — root for the learning stores (default `$XDG_CONFIG_HOME/stencil` or
  `~/.config/stencil`; env `STENCIL_DATA_DIR`). Per-model subdirs `censor/` and `styling/` live
  under it.

`review` only:

- `--parties <list|@file>` — names to always censor (inline comma-separated, or `@path` to a file).
- `--only <stages>` / `--skip <stages>` — choose stages (`censor`, `snippet`).
- `--out <file>` — override the Markdown output path (default `<input>.stencil.md`).
- `--force` — overwrite existing output/snippet files instead of refusing.
- `--censor-dir <dir>` — override the censor store location (env `STENCIL_CENSOR_DIR`).

`style` only:

- `--styling-dir <dir>` — override the styling store location (env `STENCIL_STYLING_DIR`).

## How it works

```
                ┌──────── stencil style ────────┐   (fix flagged blocks in Word, save)
contract.docx → extract → per-block review ──────┼→ styling.jsonl + profiles/<doc_id>.json
                          (fine / weird)          │
                                                  ▼
                                         corrected contract.docx
                ┌──────── stencil review ─────────────────────────────────────────┐
corrected.docx → extract → censor (confirm/edit/split/add) → snippet ───────────────┼→ <stem>.stencil.md
                              │                                                     │   + snippets/
                       decisions.jsonl + learned.json
```

- **Input is read-only.** `.docx` is never written back by either command.
- **One-way pass.** No `restore` step and no `mapping.json`; the censored text shown and stored
  never carries real values. `--pages` keeps out-of-scope pages censored too, so a confirmed value
  never leaks into a snippet.
- **Stable document id.** Records and the style profile are keyed by a content hash (`doc_id`), not
  the filename, so reusing one filename across folders never collides.
- **Deterministic learning.** Your censor decisions feed a per-user store: a value you reject
  becomes an auto-skip on later runs; a value seen both ways stays censored. The append-only
  `decisions.jsonl` and `styling.jsonl` logs — tagged with block kind, language, and edit
  provenance — are the labelled training sets for future models.
- **Detection is lenient.** Lone (unpaired) brackets are still detected and flagged *guessed* for
  review, with a `[`/`]` balance diagnostic.

## Scope

Stencil detects variables and censors values. It does **not** name variables, write the questions to
ask, choose input placeholders, classify block conditions, fill the document, or apply styling fixes
— those are your job (styling) or Claude's, downstream.

## License

MIT
