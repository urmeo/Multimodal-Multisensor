"""Heart-rate-variability metrics: windowed SDNN/RMSSD with NN artifact filtering."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Plausible NN-interval bounds in ms (2000 ≈ 30 bpm, 300 ≈ 200 bpm).
NN_MIN_MS = 300.0
NN_MAX_MS = 2000.0


def clean_nn(ibi, lo: float = NN_MIN_MS, hi: float = NN_MAX_MS) -> pd.Series:
    """Normal-to-normal intervals: numeric, in-range, zeros/artifacts dropped."""
    s = pd.to_numeric(pd.Series(ibi), errors="coerce").dropna()
    return s[(s >= lo) & (s <= hi)]


def sdnn(ibi, lo: float = NN_MIN_MS, hi: float = NN_MAX_MS) -> float:
    """SD of NN intervals (ms); NaN if fewer than 2 valid beats."""
    nn = clean_nn(ibi, lo, hi)
    return float(nn.std(ddof=1)) if len(nn) > 1 else float("nan")


def rmssd(ibi, lo: float = NN_MIN_MS, hi: float = NN_MAX_MS) -> float:
    """RMS of successive NN differences (ms); NaN if fewer than 2 valid beats."""
    nn = clean_nn(ibi, lo, hi)
    diff = nn.diff().dropna()
    return float(np.sqrt((diff ** 2).mean())) if len(diff) else float("nan")


def hrv_rolling(
    df: pd.DataFrame,
    ibi_col: str = "ibi",
    window_beats: int = 30,
    lo: float = NN_MIN_MS,
    hi: float = NN_MAX_MS,
) -> pd.DataFrame:
    """Add per-row rolling ``sdnn``/``rmssd`` over a trailing ``window_beats`` window.

    Keeps the input rows (drop-in for the old broadcast-scalar output) so the
    ``reltime, datetime, sdnn, rmssd`` schema is preserved but the values vary.
    """
    d = df.copy()
    ibi = pd.to_numeric(d[ibi_col], errors="coerce")
    nn = ibi.where((ibi >= lo) & (ibi <= hi))
    min_p = max(2, window_beats // 3)
    d["sdnn"] = nn.rolling(window_beats, min_periods=min_p).std(ddof=1)
    # A filtered-out beat voids the two successive differences that span it, so
    # rmssd may rest on slightly fewer beats than sdnn near an artifact — intended.
    d["rmssd"] = (nn.diff() ** 2).rolling(window_beats, min_periods=min_p).mean() ** 0.5
    return d


def hrv_over_time(
    df: pd.DataFrame,
    ibi_col: str = "ibi",
    time_col: str = "reltime",
    window_s: float = 30.0,
    step_s: float | None = None,
    lo: float = NN_MIN_MS,
    hi: float = NN_MAX_MS,
) -> pd.DataFrame:
    """Time-resolved SDNN/RMSSD: one row per ``window_s``-second window.

    Returns ``window_start_s, n_beats, sdnn, rmssd``. ``step_s`` defaults to
    ``window_s`` (non-overlapping windows).
    """
    step_s = step_s or window_s
    d = df[[time_col, ibi_col]].copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col])
    if d.empty:
        return pd.DataFrame(columns=["window_start_s", "n_beats", "sdnn", "rmssd"])

    t = d[time_col].to_numpy(dtype=float)
    start, end = float(t.min()), float(t.max())
    rows = []
    w = start
    while w < end or not rows:
        top = w + window_s
        # Include the boundary only on the final window, so the last beat at
        # t == end is counted exactly once and never lost.
        in_win = (t >= w) & (t <= top if top >= end else t < top)
        seg = d[in_win][ibi_col]
        rows.append({
            "window_start_s": round(w - start, 3),
            "n_beats": int(len(clean_nn(seg, lo, hi))),
            "sdnn": sdnn(seg, lo, hi),
            "rmssd": rmssd(seg, lo, hi),
        })
        w += step_s
    return pd.DataFrame(rows)
