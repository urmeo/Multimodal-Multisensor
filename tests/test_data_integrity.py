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


# --- Group-summary integrity ------------------------------------------------
# Byte-identical values in a *continuous* measure are a copy-paste red flag. Two
# such cases are known and documented (see DATA_PROVENANCE.md) — their source
# data is not in the repo, so they are allow-listed rather than guessed. Any NEW,
# undocumented duplicate (within a participant OR across participants) fails.
GROUP_SUMMARIES = sorted((ROOT / "data" / "group_results").glob("*.csv"))

# (file, participant, sorted session-pair) — same participant, two equal sessions
KNOWN_WITHIN_PARTICIPANT_DUPES = {
    ("HRV_SDNN.csv", "P01", ("Session 02", "Session 03")),
}
# (file, session, value) — two participants sharing one session value
KNOWN_CROSS_PARTICIPANT_DUPES = {
    ("Psychometric_Test_Duration_STD.csv", "Session 01", 4.409281089248826),
}


def _summary(path):
    df = pd.read_csv(path)
    session_cols = [c for c in df.columns if c.lower().startswith("session")]
    if "Participant" not in df.columns or len(session_cols) < 2:
        pytest.skip("not a participant-by-session summary")
    return df, session_cols


@pytest.mark.parametrize("path", GROUP_SUMMARIES, ids=[p.name for p in GROUP_SUMMARIES])
def test_no_undocumented_within_participant_duplicates(path):
    df, session_cols = _summary(path)
    offenders = []
    for _, row in df.iterrows():
        for i, ci in enumerate(session_cols):
            for cj in session_cols[i + 1:]:
                if float(row[ci]) == float(row[cj]):
                    pair = tuple(sorted((ci, cj)))
                    if (path.name, str(row["Participant"]), pair) not in KNOWN_WITHIN_PARTICIPANT_DUPES:
                        offenders.append((row["Participant"], ci, cj, float(row[ci])))
    assert not offenders, (
        f"Undocumented within-participant session duplicates in {path.name}: {offenders}. "
        "If real, document in DATA_PROVENANCE.md and add to KNOWN_WITHIN_PARTICIPANT_DUPES."
    )


@pytest.mark.parametrize("path", GROUP_SUMMARIES, ids=[p.name for p in GROUP_SUMMARIES])
def test_no_undocumented_cross_participant_duplicates(path):
    df, session_cols = _summary(path)
    offenders = []
    for col in session_cols:
        for val, count in df[col].value_counts().items():
            if count >= 2 and (path.name, col, float(val)) not in KNOWN_CROSS_PARTICIPANT_DUPES:
                shared = df.loc[df[col] == val, "Participant"].tolist()
                offenders.append((col, float(val), shared))
    assert not offenders, (
        f"Undocumented cross-participant duplicates in {path.name}: {offenders}. "
        "If real, document in DATA_PROVENANCE.md and add to KNOWN_CROSS_PARTICIPANT_DUPES."
    )
