# Contributing to Multimodal-Multisensor

Thanks for your interest. This is a small, reproducible research repo: notebooks
and a shared `mms` analysis package built on top of committed sensor summaries
from a 10-participant longitudinal study. Contributions that improve clarity,
reproducibility, or the statistical rigour of the analysis are especially
welcome. Because the data are special-category health data, please also read
[DATA_ETHICS.md](DATA_ETHICS.md) before working with anything under `data/`.

## Development setup

The repo targets Python >= 3.11 (CI runs 3.11 and 3.12). Create a virtualenv and
install the project editable, which also brings in the `mms` package and the
test toolchain:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # deps + mms + pytest / nbmake / ipykernel
```

For a byte-for-byte reproducible environment, install from the pinned, hashed
lockfile instead (generated with `uv pip compile`):

```bash
pip install -r requirements.lock
pip install -e . --no-deps       # add the mms package on top, without re-resolving
```

## Running the tests

The suite validates the committed data and checks every notebook is valid and
was saved without error outputs. No notebook execution is required:

```bash
pytest tests/ -q
```

Optional top-to-bottom notebook execution (slow, and what the manual
`execute-notebooks` CI job runs) is opt-in:

```bash
pytest --nbmake --nbmake-kernel=python3 --nbmake-timeout=900 case-study group individual
```

If you change the analysis code or committed data, also confirm the group
summaries still regenerate and reconcile:

```bash
python pipeline/build_group_summaries.py
```

## Linting

There is no separate linter or formatter configured — keep style consistent with
the surrounding code. CI does run a `pip-audit` dependency scan; it is advisory
today, so a flagged CVE won't block a PR, but do check the output.

## Code style

- Clean code, few comments: let names and structure carry the meaning.
- Put shared loaders and metrics in the `mms` package (`mms.io`, `mms.hrv`,
  `mms.stats`, `mms.fixation`) and import them from notebooks — don't
  re-implement analysis inline. New shared logic needs a test in `tests/`.
- Re-run a notebook top to bottom before committing it, so it lands with clean
  outputs and no saved errors (the smoke test enforces this).
- Never commit raw or re-identifiable data. Keep to the de-identified summaries
  under `data/`; see [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for scope.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) with a terse
description:

```
feat:     a new analysis, metric, or notebook
fix:      a correctness fix in mms/, the pipeline, or a notebook
docs:     README, provenance, or other docs
test:     tests only
refactor: restructuring with no behaviour change
build:    dependencies, packaging, lockfile
chore:    tooling and housekeeping
```

For example: `fix: correct ICC(1) confidence interval` or
`docs: clarify data licence`. Keep one logical change per commit.

## Filing an issue

Open an issue describing what you observed and how to reproduce it. For a numeric
discrepancy, include the exact command and the values you got versus expected —
the reliability numbers are reproducible from `mms.stats.icc1`, so a mismatch is
worth a report.

## Opening a pull request

1. Branch off `main`.
2. Make your change and keep commits focused.
3. Run `pytest tests/ -q` locally (and `python pipeline/build_group_summaries.py`
   if you touched the analysis or data).
4. Open a pull request against `main`. CI runs the data and notebook validation
   on Python 3.11 and 3.12 and must be green before merge.

Code (notebooks, `mms/`, scripts) is released under the [MIT license](LICENSE);
data and figures under the terms in [DATA_LICENSE.md](DATA_LICENSE.md).
