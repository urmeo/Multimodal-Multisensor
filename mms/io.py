"""Data loaders — one tested path for reading the study's CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import paths


_SOURCES = {"case-study": paths.CASE_STUDY, "individual": paths.INDIVIDUAL}


def _base(source: str) -> Path:
    try:
        return _SOURCES[source]
    except KeyError:
        raise ValueError(
            f"unknown source {source!r}; expected one of {sorted(_SOURCES)}"
        ) from None


def load_ibi(session: int, *, source: str = "case-study") -> pd.DataFrame:
    """Inter-beat-interval stream for a session (1-3)."""
    return pd.read_csv(_base(source) / "processed" / f"ibi_{session:02d}.csv")


def load_hr(session: int, *, high_confidence_only: bool = True,
            source: str = "case-study") -> pd.DataFrame:
    """Heart-rate stream for a session; drops low-confidence rows by default."""
    df = pd.read_csv(_base(source) / "processed" / f"hr_{session:02d}.csv")
    if high_confidence_only and "confidence" in df.columns:
        df = df[df["confidence"] == 1.0].copy()
    return df


def load_fixation(session: int, *, source: str = "case-study") -> pd.DataFrame:
    """Processed eye-tracking / fixation ('sed_fix') stream for a session."""
    return pd.read_csv(_base(source) / "processed" / f"sed_fix_{session:02d}.csv")


def load_psychometric(session: int, *, source: str = "case-study") -> pd.DataFrame:
    """Psychometric test results for a session."""
    return pd.read_csv(_base(source) / "psychometric"
                       / f"Psychometric_Test_Results_{session:02d}.csv")


def load_group_summary(metric: str) -> pd.DataFrame:
    """Group summary CSV (row per participant, column per session).

    ``metric`` ∈ {``HRV_SDNN``, ``Pupil_Dilation_STD``,
    ``Psychometric_Test_Duration_STD``}.
    """
    return pd.read_csv(paths.GROUP_RESULTS / f"{metric}.csv")


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse a stream datetime column to tz-naive timestamps."""
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)
