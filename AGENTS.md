# Instructions for coding agents

This repository is a small, carefully edited site (https://homogenization.microprediction.org) plus the numerical
certificates behind it. Every PR is reviewed and merged by hand, so make each one easy to review: short, correct,
and in the house style below.

## Build and check

```
python3 tools/assemble.py && node docs/header-check.js   # builds docs/*.html from tools/pages/*.html
python3 tools/check_pages.py tools/pages/<page>.html ...  # run on every page you changed; must print OK
```

Edit `tools/pages/*.html`, never `docs/*.html` by hand; commit the rebuilt `docs/` with the source. Thirteen pricing
pages (the `MOVED` list in `tools/assemble.py`) publish as redirects to regimelib.org, so changes to them are not
visible on this site.

## Pages: short summaries, detail in a full note

- A page is a summary of at most about 250 source lines, and never more than 450 (`check_pages.py` enforces 450).
  State the setup, the handful of results a reader should remember (each once, with its key formula), a few headline
  numbers, and what is not claimed.
- Put proofs, extra theorems, long tables and side results in `tools/pages/<page>-full.html`, linked from the summary
  and linking back. Nine pages already follow this pattern (for example `cumulants.html` and `cumulants-full.html`).
  When you extend a result, add the detail to the full note and change at most a sentence or two of the summary.
- Paragraphs have at most three sentences. Run `python3 tools/split_paragraphs.py tools/pages/<page>.html`.
- Plain, direct prose. No lists of every theorem in running text, no stacked qualifiers.
- Maths: every LaTeX command needs its backslash (`\qquad`, `\mathrm`, `\operatorname`, `\infty`, `\{ \}`).
  `check_pages.py` catches the common ones.

## Papers list (`tools/pages/papers.html`)

Each entry's summary is one to three sentences, about 20 to 60 words. Do not append a new sentence to an entry every
time a result is added; rewrite the entry so it stays that length.

## Certificates (`papers/<topic>/verify_*.py`)

- Every number quoted on a page must be printed by a certificate, and the certificate must end by printing `PASS`
  (or `FAIL`) on its last line. Do not rely on exit status alone.
- Tolerances must hold on other machines and library versions, not just yours. Never assert below 1e-14 in absolute
  terms. Scale tolerances to the size of the quantity: for example, a variance computed as E[N^2] - E[N]^2 has
  rounding error proportional to E[N^2]. If an assertion compares against a truncated asymptotic series, the
  tolerance must exceed the next omitted term.
- Do not format arbitrary-precision numbers (`mpmath.mpf`) with f-string format specs; convert with `float()` first.
- Do not commit `results.json` files that changed only in the last digits.

## Bibliography

Add references through `tools/pages/_bib/<section>.json` and `python3 tools/merge_bib.py <section>`, which writes
`tools/biblio.json` and `tools/pages/bibliography.html` together. Every cited `./bibliography.html#Key` must exist on
the bibliography page, and each key must appear once. Give a DOI where one exists, and cite only works you have
checked exist with the stated title, authors, year and venue.

## Pull requests and issues

- Branch from current `main` and keep each PR to one result: ideally under 300 added lines outside `docs/`.
  Split larger work into several PRs.
- Before opening a PR: rebuild, run `check_pages.py` on the changed pages, and run the changed certificates.
- Do not open a PR that repeats a fix already on `main`; check `git log origin/main` first.
- Open issues only about text or code that is on `main`. A defect in your own unmerged branch should be fixed on
  that branch, not filed as an issue.
- After a PR is merged, start new work from the updated `main`, not from the old branch.
