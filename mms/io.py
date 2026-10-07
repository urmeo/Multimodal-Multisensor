"""Immutable dataset reads, explicit schemas and clock awareness."""

from __future__ import annotations

from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd
from pytz.exceptions import InvalidTimeError  # type: ignore[import-untyped]

from . import paths
from .hrv import _real_series

KNOWN_WITHIN_DUPLICATES = {
    ("HRV_SDNN", "P01", ("Session 02", "Session 03"), 65.39381048265373),
}
KNOWN_CROSS_DUPLICATES = {
    ("Psychometric_Test_Duration_STD", "Session 01", ("P07", "P08"), 4.409281089248826),
}


def _base(source: str, data_root: str | Path | None = None) -> Path:
    if source not in paths.SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {paths.SOURCES}")
    return paths.resolve_data_root(data_root) / source


def _suffix(session: int | None) -> str:
    if session is None:
        return ""
    if (
        isinstance(session, bool)
        or not isinstance(session, (int, np.integer))
        or session not in (1, 2, 3)
    ):
        raise ValueError("session must be None (baseline), 1, 2 or 3")
    return f"_{session:02d}"


def _read(path: Path, *, sep: str = ",") -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file is absent: {path}. Wheels contain code only; "
            "supply data_root=PATH to the separately obtained dataset."
        )
    return pd.read_csv(path, sep=sep)


def load_raw(
    sensor: str,
    session: int | None = None,
    *,
    source: str = "case-study",
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Read a semicolon export; this does not identify its participant/hardware."""
    if sensor not in ("hr", "ibi", "sed"):
        raise ValueError("raw sensor must be hr, ibi or sed")
    return _read(
        _base(source, data_root) / "raw" / f"{sensor}{_suffix(session)}.txt", sep=";"
    )


def load_ibi(
    session: int | None,
    *,
    source: str = "case-study",
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Interval samples, retaining channel IDs, repeated readings and row order."""
    return _read(_base(source, data_root) / "processed" / f"ibi{_suffix(session)}.csv")


def load_hr(
    session: int | None,
    *,
    high_confidence_only: bool = True,
    source: str = "case-study",
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """HR samples; confidence==1 is the default operational filter."""
    frame = _read(_base(source, data_root) / "processed" / f"hr{_suffix(session)}.csv")
    if high_confidence_only:
        if "confidence" not in frame:
            raise ValueError("HR stream lacks required confidence column")
        frame = frame.loc[frame["confidence"] == 1.0].copy()
    return frame


def load_fixation(
    session: int | None,
    *,
    source: str = "case-study",
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Historical eye samples/low-movement labels; not an event table."""
    return _read(
        _base(source, data_root) / "processed" / f"sed_fix{_suffix(session)}.csv"
    )


def parse_datetime(
    series: pd.Series, *, naive_timezone: str | None = None, allow_missing: bool = False
) -> pd.Series:
    """Complete parsing, retaining awareness; unknown naive clocks stay naive.

    Explicit naive_timezone is a caller's documented acquisition-clock mapping.
    Mixed naive/aware values are rejected. Aware offsets normalize to UTC only
    when they differ, preserving instants; an existing single zone is retained.
    """
    parsed = []
    for value in series:
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            if not allow_missing:
                raise ValueError("timestamp column contains missing values")
            parsed.append(pd.NaT)
            continue
        if isinstance(value, (Real, bool)):
            raise ValueError("numeric timestamps require an explicit unit contract")
        try:
            stamp = pd.Timestamp(value)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("timestamp column contains an invalid value") from error
        if pd.isna(stamp):
            raise ValueError("timestamp column contains an invalid value")
        parsed.append(stamp)
    present = [stamp for stamp in parsed if not pd.isna(stamp)]
    awareness = {stamp.tzinfo is not None for stamp in present}
    if len(awareness) > 1:
        raise ValueError("timestamp column mixes naive and timezone-aware clocks")
    if awareness == {True} and len({str(stamp.tzinfo) for stamp in present}) > 1:
        out = pd.Series(
            pd.to_datetime(parsed, utc=True), index=series.index, name=series.name
        )
    else:
        out = pd.Series(pd.DatetimeIndex(parsed), index=series.index, name=series.name)
    if out.dt.tz is None and naive_timezone is not None:
        try:
            out = out.dt.tz_localize(
                naive_timezone, ambiguous="raise", nonexistent="raise"
            )
        except (ValueError, TypeError, KeyError, InvalidTimeError) as error:
            raise ValueError("cannot apply the declared naive timezone") from error
    return out


def validate_psychometric(frame: pd.DataFrame) -> None:
    required = {
        "Type",
        "Test",
        "Question",
        "Answer",
        "Time(s)",
        "Question Start Time",
        "Question Answer Time",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            "noncanonical psychometric schema; legacy _00 has shifted headers "
            "and Score holds duration, not a verified scale score"
        )
    duration = _real_series(frame["Time(s)"], "Time(s)")
    if not np.isfinite(duration.to_numpy(float)).all() or (duration < 0).any():
        raise ValueError(
            "psychometric Time(s) must contain finite nonnegative durations"
        )
    start = parse_datetime(frame["Question Start Time"])
    end = parse_datetime(frame["Question Answer Time"])
    if start.dt.tz is None or end.dt.tz is None:
        raise ValueError(
            "canonical question boundaries require explicit timezone-aware clocks"
        )
    if (end < start).any() or not np.allclose(
        (end - start).dt.total_seconds(), duration, rtol=0, atol=0.0021
    ):
        raise ValueError("question boundaries disagree with response durations")


def load_psychometric(
    session: int, *, source: str = "case-study", data_root: str | Path | None = None
) -> pd.DataFrame:
    """Canonical item responses; stored answers are not verified keyed scores."""
    if session == 0:
        raise ValueError(
            "legacy psychometric _00 is malformed: shifted time headers; "
            "Score represents duration. Read explicitly as source evidence."
        )
    frame = _read(
        _base(source, data_root)
        / "psychometric"
        / f"Psychometric_Test_Results{_suffix(session)}.csv"
    )
    validate_psychometric(frame)
    return frame


def summary_duplicates(frame: pd.DataFrame, metric: str) -> tuple[set, set]:
    within, cross = set(), set()
    for _, row in frame.iterrows():
        for i, first in enumerate(paths.SESSION_COLUMNS):
            for second in paths.SESSION_COLUMNS[i + 1 :]:
                if row[first] == row[second]:
                    within.add(
                        (
                            metric,
                            str(row["Participant"]),
                            (first, second),
                            float(row[first]),
                        )
                    )
    for session in paths.SESSION_COLUMNS:
        for value, count in frame[session].value_counts().items():
            if count > 1:
                identities = tuple(
                    sorted(frame.loc[frame[session] == value, "Participant"])
                )
                cross.add((metric, session, identities, float(value)))
    return within, cross


def validate_group_summary(
    frame: pd.DataFrame, metric: str, *, require_known_duplicates: bool = False
) -> None:
    if metric not in paths.GROUP_METRICS:
        raise ValueError(f"unknown group metric {metric!r}")
    if list(frame.columns) != ["Participant", *paths.SESSION_COLUMNS]:
        raise ValueError(
            "group summary must have Participant and exact Session01–03 columns"
        )
    if (
        len(frame) != 10
        or frame["Participant"].duplicated().any()
        or set(frame["Participant"]) != set(paths.PARTICIPANTS)
    ):
        raise ValueError("group summary requires unique P01–P10 rows")
    values = frame[list(paths.SESSION_COLUMNS)].apply(
        lambda column: _real_series(column, column.name)
    )
    if not np.isfinite(values.to_numpy(float)).all() or (values < 0).any().any():
        raise ValueError("group summary values must be finite nonnegative numbers")
    numeric = frame.copy()
    numeric[list(paths.SESSION_COLUMNS)] = values
    within, cross = summary_duplicates(numeric, metric)
    expected_within = {item for item in KNOWN_WITHIN_DUPLICATES if item[0] == metric}
    expected_cross = {item for item in KNOWN_CROSS_DUPLICATES if item[0] == metric}
    if not within.issubset(expected_within) or not cross.issubset(expected_cross):
        raise ValueError("undocumented group duplicate identity/value/multiplicity")
    if require_known_duplicates and (
        within != expected_within or cross != expected_cross
    ):
        raise ValueError("released known duplicate identity/value/multiplicity changed")


def load_group_summary(
    metric: str, *, data_root: str | Path | None = None
) -> pd.DataFrame:
    frame = (
        _read(paths.resolve_data_root(data_root) / "group_results" / f"{metric}.csv")
        if metric in paths.GROUP_METRICS
        else None
    )
    if frame is None:
        raise ValueError(
            f"unknown group metric {metric!r}; expected {paths.GROUP_METRICS}"
        )
    validate_group_summary(frame, metric)
    return frame
