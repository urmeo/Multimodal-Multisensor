"""Lightweight data smoke test: every committed CSV must load and be non-trivial.

Catches corrupted, truncated, or empty data files before they silently break a
notebook. CSVs are discovered automatically, so newly committed data is covered
without touching this test.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV_FILES = sorted(
    p for p in ROOT.rglob("*.csv") if ".ipynb_checkpoints" not in p.parts
)


def test_csv_files_present():
    assert CSV_FILES, "No CSV files found in the repository."


@pytest.mark.parametrize(
    "path", CSV_FILES, ids=[str(p.relative_to(ROOT)) for p in CSV_FILES]
)
def test_csv_loads_and_is_non_trivial(path):
    rel = path.relative_to(ROOT)
    assert path.stat().st_size > 0, f"{rel} is a zero-byte file"
    df = pd.read_csv(path)
    assert df.shape[0] >= 1, f"{rel} has no data rows"
    assert df.shape[1] >= 1, f"{rel} has no columns"
