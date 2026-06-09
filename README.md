# Stencil

A fully-local Rust CLI that walks a contract template through an interactive **review** pass and
writes a context-rich Markdown file Claude Code can read.

Stencil takes a template (`.docx` or `.txt`) and runs three stages in order — **censor**,
**styling**, **snippet** — to produce a censored, navigable `<stem>.stencil.md` plus per-bracket
snippet files. Every decision you make also trains Stencil's local, deterministic memory so the
next run needs less input.

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

`review` is the only command. It is interactive — it reads single keypresses — so it must be run
directly in a terminal (TTY).

```sh
stencil review contract.docx --parties "Acme Corporation, Jane Doe"
# → contract.stencil.md   (context shows REDACTED_* placeholders)
# → snippets/             (one censored file per bracket span)
```

### The three stages

1. **censor** — over-detects sensitive values (names, money, percentages, dates, emails, phones,
   IDs, …) and walks you through each one: **[c]** confirm · **[t]** re-type · **[x]** reject
   (false positive) · **[b]** back · **[q]** quit & save. Only confirmed values are censored.
2. **styling** *(`.docx` only)* — walks every block and shows its formatting against the
   document's norms; you mark each **[space]** fine or **[w]** weird (with a category and an
   optional note). Purely a labelling pass — it changes no output.
3. **snippet** — writes the `.stencil.md` section inventory plus a censored snippet file per
   bracket span, cross-linked for navigation.

Run a subset with `--only` or `--skip` (mutually exclusive, comma-separated):

```sh
stencil review contract.docx --only censor        # just the censor pass
stencil review contract.docx --skip styling       # censor + snippet
```

### Flags

- `--parties <list|@file>` — names to always censor (inline comma-separated, or `@path` to a file).
- `--only <stages>` / `--skip <stages>` — choose stages (`censor`, `styling`, `snippet`).
- `--out <file>` — override the Markdown output path (default `<input>.stencil.md`).
- `--force` — overwrite existing output/snippet files instead of refusing.
- `--data-dir <dir>` — root for the learning stores (default `$XDG_CONFIG_HOME/stencil` or
  `~/.config/stencil`); also settable via `STENCIL_DATA_DIR`. Per-model subdirs `censor/` and
  `styling/` live under it.
- `--censor-dir <dir>` / `--styling-dir <dir>` — override a single model's store location (env:
  `STENCIL_CENSOR_DIR` / `STENCIL_STYLING_DIR`).

## How it works

```
template.docx/.txt → extract → censor (confirm/reject) → styling (fine/weird) → snippet
                                   │              │                                  │
                                   │              │                          <stem>.stencil.md
                                   │              │                          + snippets/
                              decisions.jsonl   styling.jsonl + profiles/
                              + learned.json    (per-doc style sidecar)
```

- **Input is read-only.** `.docx` is never written back.
- **One-way pass.** v6 has no `restore` step and writes no `mapping.json`; the censored text shown
  and stored never carries real values.
- **Deterministic learning.** Your decisions feed a per-user store: a value you reject becomes an
  auto-skip on later runs; a value seen both ways stays censored. The append-only `decisions.jsonl`
  and `styling.jsonl` logs are the labelled training sets for future models — no ML runs yet.
- **Detection is lenient.** Lone (unpaired) brackets are still detected and flagged *guessed* for
  review, with a `[`/`]` balance diagnostic.

## Scope

Stencil detects variables and censors values. It does **not** name variables, write the questions
to ask, choose input placeholders, classify block conditions, or fill the document — those are
Claude's job, downstream.

## License

MIT
