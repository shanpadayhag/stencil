# Stencil

A fully-local Rust CLI that turns a contract template into a Markdown file Claude Code can read.

Stencil scans a template (`.docx` or `.txt`) for **bracketed fill-in variables** (e.g. `[Buyer Name]`)
and writes a context-rich Markdown file — one entry per section, each with a bracket inventory — for
Claude Code to read. Optionally it first **censors** sensitive values (names, money, dates, IDs, emails)
into `REDACTED_*` placeholders and writes a reversible `mapping.json`.

> ⚠️ **Review before pasting.** Stencil is a best-effort first-pass filter, **not a guarantee of
> complete redaction**. Always review the censored output and the printed censorship summary (and any
> ⚠ GUESSED brackets) before pasting anything into Claude.

## Install

Requires the Rust toolchain (`rustup`). Build the release binary:

```sh
cargo build --release
# binary at target/release/stencil
```

## Usage

### Detect bracketed variables

```sh
stencil detect contract.txt
# → contract.stencil.md  (+ bracket-balance report on stderr)
```

### Detect + censor sensitive values

```sh
stencil detect contract.docx --censor \
    --parties "Acme Corporation, Jane Doe" \
    --guess-names
# → contract.stencil.md   (context shows REDACTED_* placeholders)
# → contract.mapping.json (reversible mapping)
# censorship summary printed to stderr
```

- `--parties <list|@file>` — names to always censor (inline comma-separated, or `@path` to a file).
- `--guess-names` — opt-in capitalized-sequence name heuristic. Noisy by design; every guess is
  **flagged** in the summary for you to verify.
- `--out <file>` / `--map <file>` — override the default output paths.
- `--force` — overwrite existing output/mapping files.

### Restore real values

`restore` reverses censorship on any text/Markdown file containing `REDACTED_*` tokens (it does **not**
fill bracket variables, and never writes `.docx`):

```sh
stencil restore contract.stencil.md --map contract.mapping.json
# → contract.stencil.restored.md
```

## How it works

```
template.docx/.txt → extract → [--censor] → detect brackets → section → render → <stem>.stencil.md
                                    │                                              (+ mapping.json)
                                    └→ REDACTED_* placeholders + censorship summary
```

- **Input is read-only.** `.docx` is never written back.
- **Detection is lenient.** Lone (unpaired) brackets are still detected, with a *guessed* span flagged
  for review, and a `[`/`]` balance diagnostic is printed.
- **Censoring is reversible** via `mapping.json` and dedups by exact value (one value → one placeholder).

## Scope

Stencil detects variables and censors values. It does **not** name variables, write the questions to
ask, choose input placeholders, classify block conditions, or fill the document — those are Claude's
job, downstream.

## License

MIT
