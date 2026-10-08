"""Descriptive variability of interval samples, preserving their recorded adjacency."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

NN_MIN_MS = 300.0
NN_MAX_MS = 2000.0


def _real_series(values, name: str) -> pd.Series:
    s = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
        raise ValueError(f"{name} must contain real numbers")
    if any(isinstance(v, (bool, np.bool_, complex, np.complexfloating)) for v in s):
        raise ValueError(
            f"{name} must contain real numbers, not boolean or complex values"
        )
    try:
        return pd.to_numeric(s, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc


def _finite_real(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'real'}"
        )
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'real'}"
        )
    return result


def _bounds(lo, hi) -> tuple[float, float]:
    lo, hi = _finite_real(lo, "lo"), _finite_real(hi, "hi")
    if lo > hi:
        raise ValueError("lo must not exceed hi")
    return lo, hi


def _intervals(ibi, lo, hi) -> pd.Series:
    lo, hi = _bounds(lo, hi)
    s = _real_series(ibi, "ibi")
    return s.where(np.isfinite(s) & (s > 0) & (s >= lo) & (s <= hi)).astype(float)


def _elapsed(values, name: str) -> np.ndarray:
    s = _real_series(values, name)
    if s.isna().any() or not np.isfinite(s).all() or (s < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative times")
    if not s.is_monotonic_increasing:
        raise ValueError(f"{name} must be chronological")
    if s.empty:
        return np.empty(0, dtype=float)
    origin = s.iloc[0].item() if isinstance(s.iloc[0], np.generic) else s.iloc[0]
    elapsed = np.asarray(
        [(v.item() if isinstance(v, np.generic) else v) - origin for v in s],
        dtype=float,
    )
    if not np.isfinite(elapsed).all():
        raise ValueError(f"{name} span is too large")
    return elapsed


def _gap(max_gap_s) -> float | None:
    return (
        None
        if max_gap_s is None
        else _finite_real(max_gap_s, "max_gap_s", positive=True)
    )


def _time_differences(values) -> np.ndarray:
    raw = list(values)
    raw = [value.item() if isinstance(value, np.generic) else value for value in raw]
    return np.asarray([b - a for a, b in zip(raw[:-1], raw[1:])], dtype=float)


def _groups(df: pd.DataFrame, channel_col: str | None):
    if channel_col is None or channel_col not in df.columns:
        return [(None, np.arange(len(df)))], None
    if df[channel_col].isna().any():
        raise ValueError(f"{channel_col} must not contain missing channel identities")
    return list(
        df.groupby(channel_col, sort=False, observed=True).indices.items()
    ), channel_col


def clean_nn(ibi, lo: float = NN_MIN_MS, hi: float = NN_MAX_MS) -> pd.Series:
    """Return in-range interval samples; the name is retained for compatibility."""
    return _intervals(ibi, lo, hi).dropna()


def sdnn(ibi, lo: float = NN_MIN_MS, hi: float = NN_MAX_MS) -> float:
    """Sample SD in milliseconds after interval filtering."""
    samples = clean_nn(ibi, lo, hi)
    scale = float(samples.abs().max()) if len(samples) else 0.0
    return (
        float((samples / scale).std(ddof=1) * scale)
        if len(samples) > 1
        else float("nan")
    )


def rmssd(ibi, lo: float = NN_MIN_MS, hi: float = NN_MAX_MS) -> float:
    """RMS difference of valid adjacent interval samples, without joining gaps."""
    diffs = _intervals(ibi, lo, hi).diff().dropna().to_numpy(float)
    if not len(diffs):
        return float("nan")
    scale = np.max(np.abs(diffs))
    return float(scale * np.sqrt(np.mean((diffs / scale) ** 2))) if scale else 0.0


def hrv_rolling(
    df: pd.DataFrame,
    ibi_col: str = "ibi",
    window_beats: int = 30,
    lo: float = NN_MIN_MS,
    hi: float = NN_MAX_MS,
    *,
    channel_col: str | None = "iSensor",
    time_col: str = "reltime",
    max_gap_s: float | None = None,
) -> pd.DataFrame:
    """Trailing N-sample variability by channel; RMSSD uses N-1 internal pairs.

    The compatibility argument ``window_beats`` counts samples. Warm-up needs
    max(2, N//3) valid samples for SD and one fewer valid pairs for RMSSD.
    """
    if (
        isinstance(window_beats, (bool, np.bool_))
        or not isinstance(window_beats, Integral)
        or window_beats < 2
        or window_beats > np.iinfo(np.intp).max
    ):
        raise ValueError("window_beats must be an integer >= 2 within platform limits")
    gap = _gap(max_gap_s)
    samples = _intervals(df[ibi_col], lo, hi)
    times = _elapsed(df[time_col], time_col) if time_col in df else None
    if gap is not None and times is None:
        raise ValueError(f"{time_col} is required with max_gap_s")
    groups, _ = _groups(df, channel_col)
    out = df.copy()
    sd, rms = np.full(len(df), np.nan), np.full(len(df), np.nan)
    min_samples = max(2, int(window_beats) // 3)
    for _, positions in groups:
        part = samples.iloc[positions].reset_index(drop=True)
        scale = float(part.abs().max()) if part.notna().any() else 1.0
        scaled = part / scale
        differences = scaled.diff()
        if gap is not None:
            differences = differences.where(
                pd.Series(
                    np.r_[False, _time_differences(df[time_col].iloc[positions]) <= gap]
                )
            )
        sd[positions] = (
            scaled.rolling(int(window_beats), min_periods=min_samples).std(ddof=1)
            * scale
        )
        rms[positions] = (differences**2).rolling(
            int(window_beats) - 1, min_periods=min_samples - 1
        ).mean() ** 0.5 * scale
    out["sdnn"], out["rmssd"] = sd, rms
    return out


def hrv_over_time(
    df: pd.DataFrame,
    ibi_col: str = "ibi",
    time_col: str = "reltime",
    window_s: float = 30.0,
    step_s: float | None = None,
    lo: float = NN_MIN_MS,
    hi: float = NN_MAX_MS,
    *,
    channel_col: str | None = "iSensor",
    max_gap_s: float | None = None,
    max_windows: int = 100_000,
) -> pd.DataFrame:
    """Elapsed-time windows by channel; larger steps can leave gaps.

    A window reaching the recording end includes the last sample.
    ``n_beats`` is a compatibility alias for ``n_samples``. Samples are not
    verified NN beats. Invalid intervals retain their positions between pairs.
    """
    window = _finite_real(window_s, "window_s", positive=True)
    step = window if step_s is None else _finite_real(step_s, "step_s", positive=True)
    gap = _gap(max_gap_s)
    if (
        isinstance(max_windows, (bool, np.bool_))
        or not isinstance(max_windows, Integral)
        or not 1 <= max_windows <= 1_000_000
    ):
        raise ValueError("max_windows must be an integer between 1 and 1000000")
    samples = _intervals(df[ibi_col], lo, hi)
    t = _elapsed(df[time_col], time_col)
    groups, channel = _groups(df, channel_col)
    columns = ["window_start_s", "n_samples", "n_pairs", "n_beats", "sdnn", "rmssd"]
    if channel is not None:
        columns.insert(0, channel)
    if not len(t):
        return pd.DataFrame(columns=columns)
    span = float(t[-1])
    count_ratio = span / step
    if not math.isfinite(count_ratio) or count_ratio > max_windows:
        raise ValueError("requested windows exceed max_windows")
    count = max(1, math.ceil(count_ratio))
    if count * len(groups) > max_windows:
        raise ValueError("requested channel windows exceed max_windows")
    last = (count - 1) * step
    if not math.isfinite(last + window) or last + window <= last:
        raise ValueError("window end must be finite and advance in floating-point time")
    rows = []
    previous = -1.0
    for index in range(count):
        start = index * step
        if start <= previous:
            raise ValueError("step_s does not advance in floating-point time")
        previous = start
        top = start + window
        in_window = (t >= start) & ((t <= top) if top >= span else (t < top))
        for identity, positions in groups:
            selected = positions[in_window[positions]]
            part = samples.iloc[selected].reset_index(drop=True)
            differences = part.diff()
            if gap is not None:
                differences = differences.where(
                    pd.Series(
                        np.r_[
                            False, _time_differences(df[time_col].iloc[selected]) <= gap
                        ]
                    )
                )
            valid_differences = differences.dropna().to_numpy(float)
            scale = (
                np.max(np.abs(valid_differences)) if len(valid_differences) else np.nan
            )
            rms = (
                (
                    float(scale * np.sqrt(np.mean((valid_differences / scale) ** 2)))
                    if scale
                    else 0.0
                )
                if len(valid_differences)
                else float("nan")
            )
            row = {
                "window_start_s": start,
                "n_samples": int(part.notna().sum()),
                "n_pairs": len(valid_differences),
                "n_beats": int(part.notna().sum()),
                "sdnn": sdnn(part, lo, hi),
                "rmssd": rms,
            }
            if channel is not None:
                row[channel] = identity
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)
