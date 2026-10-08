"""Quality-filtered pupil samples and descriptive, explicitly bounded event runs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hrv import _elapsed, _finite_real, _gap, _real_series

_EVENT_COLUMNS = [
    "event_id",
    "start_s",
    "end_s",
    "duration_s",
    "n_samples",
    "start_pos",
    "end_pos",
    "left_censored",
    "right_censored",
    "end_reason",
]


def _quality_min(value) -> float:
    value = _finite_real(value, "quality_min")
    if not 0 <= value <= 1:
        raise ValueError("quality_min must be between 0 and 1")
    return value


def pupil_samples(
    sed: pd.DataFrame, quality_min: float = 0.5, *, missing_quality: str = "reject"
) -> pd.Series:
    """Keep the original index and replace invalid pupil samples with NaN."""
    threshold = _quality_min(quality_min)
    if missing_quality not in {"reject", "allow"}:
        raise ValueError("missing_quality must be 'reject' or explicit 'allow'")
    if "pupil" not in sed:
        return pd.Series(np.nan, index=sed.index, name="pupil")
    pupil = _real_series(sed["pupil"], "pupil")
    valid = np.isfinite(pupil) & (pupil > 0)
    if "pupilQ" not in sed:
        if missing_quality == "reject":
            raise ValueError(
                "pupilQ is required; use missing_quality='allow' explicitly if justified"
            )
    else:
        quality = _real_series(sed["pupilQ"], "pupilQ")
        valid &= np.isfinite(quality) & (quality >= threshold) & (quality <= 1)
    return pupil.where(valid).astype(float)


def pupil_std(
    sed: pd.DataFrame, quality_min: float = 0.5, *, missing_quality: str = "reject"
) -> float:
    """Sample SD of finite positive pupil values passing the declared quality rule."""
    values = pupil_samples(sed, quality_min, missing_quality=missing_quality).dropna()
    if len(values) < 2:
        return float("nan")
    scale = float(values.abs().max())
    return float((values / scale).std(ddof=1) * scale)


def _mask(values, name: str, length: int) -> tuple[np.ndarray, np.ndarray]:
    s = pd.Series(values).reset_index(drop=True)
    if len(s) != length:
        raise ValueError(f"{name} must have the same length as time")
    if any(not isinstance(value, (bool, np.bool_)) for value in s.dropna()):
        raise ValueError(f"{name} must contain boolean values")
    known = s.notna().to_numpy(bool)
    return np.asarray(
        [bool(value) if present else False for value, present in zip(s, known)],
        dtype=bool,
    ), known


def contiguous_events(
    time, mask, *, valid=None, max_gap_s: float | None = None
) -> pd.DataFrame:
    """One row per observed run; duration is last minus first qualifying sample.

    Times are elapsed seconds. Invalid rows and explicit excessive gaps break
    runs. Boundary, missing-quality and gap endings are marked as censored.
    """
    time_series = _real_series(time, "time").reset_index(drop=True)
    _elapsed(time_series, "time")
    t = time_series.to_numpy()
    flags, mask_valid = _mask(mask, "mask", len(t))
    good, valid_known = (
        _mask(valid, "valid", len(t))
        if valid is not None
        else (np.ones(len(t), dtype=bool), np.ones(len(t), dtype=bool))
    )
    good &= valid_known & mask_valid
    gap = _gap(max_gap_s)
    breaks = np.zeros(len(t), dtype=bool)
    if gap is not None and len(t) > 1:
        breaks[1:] = [
            float(
                (b.item() if isinstance(b, np.generic) else b)
                - (a.item() if isinstance(a, np.generic) else a)
            )
            > gap
            for a, b in zip(t[:-1], t[1:])
        ]
    rows: list[dict[str, object]] = []
    start = None
    left_censored = False
    for position in range(len(t) + 1):
        active = position < len(t) and good[position] and flags[position]
        interrupted = position < len(t) and breaks[position]
        if start is not None and (not active or interrupted):
            last = position - 1
            end_reason = (
                "recording_end"
                if position == len(t)
                else "gap"
                if interrupted
                else "invalid"
                if not good[position]
                else "transition"
            )
            a = t[start].item() if isinstance(t[start], np.generic) else t[start]
            b = t[last].item() if isinstance(t[last], np.generic) else t[last]
            rows.append(
                {
                    "event_id": len(rows) + 1,
                    "start_s": a,
                    "end_s": b,
                    "duration_s": float(b - a),
                    "n_samples": last - start + 1,
                    "start_pos": start,
                    "end_pos": last,
                    "left_censored": left_censored,
                    "right_censored": end_reason != "transition",
                    "end_reason": end_reason,
                }
            )
            start = None
        if active and start is None:
            start = position
            left_censored = position == 0 or interrupted or not good[position - 1]
    result = pd.DataFrame(rows, columns=_EVENT_COLUMNS).astype(
        {
            "event_id": "int64",
            "duration_s": "float64",
            "n_samples": "int64",
            "start_pos": "int64",
            "end_pos": "int64",
            "left_censored": "bool",
            "right_censored": "bool",
        }
    )
    if result.empty:
        result = result.astype({"start_s": "float64", "end_s": "float64"})
    return result


def fixation_events(
    sed: pd.DataFrame,
    *,
    mask=None,
    time_col: str = "reltime",
    quality_min: float = 0.5,
    displacement_threshold: float = 0.01,
    max_gap_s: float | None = None,
) -> pd.DataFrame:
    """Describe supplied fixation runs or a historical Euclidean displacement rule.

    The rule is unvalidated and is distance per sample, not angular velocity.
    ``gazeQ`` and finite source coordinates gate samples without dropping gaps.
    """
    threshold = _quality_min(quality_min)
    distance = _finite_real(
        displacement_threshold, "displacement_threshold", positive=True
    )
    if "gazeQ" not in sed:
        raise ValueError("gazeQ is required for fixation events")
    quality = _real_series(sed["gazeQ"], "gazeQ")
    valid = np.isfinite(quality) & (quality >= threshold) & (quality <= 1)
    coordinates = next(
        (
            names
            for names in [
                ("gazeDir.x", "gazeDir.y", "gazeDir.z"),
                ("gaze_x", "gaze_y", "gaze_z"),
            ]
            if all(name in sed for name in names)
        ),
        None,
    )
    vectors = None
    if coordinates is not None:
        vectors = np.column_stack(
            [_real_series(sed[name], name).to_numpy(float) for name in coordinates]
        )
        valid &= np.isfinite(vectors).all(axis=1) & (
            np.max(np.abs(vectors), axis=1) > 0
        )
    if mask is None:
        if "fixation" in sed:
            mask = sed["fixation"]
        else:
            if vectors is None:
                raise ValueError(
                    "a fixation mask or complete gaze coordinates is required"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                delta = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
            mask = np.zeros(len(sed), dtype=bool)
            mask[1:] = np.isfinite(delta) & (delta < distance)
            preceding_valid = np.zeros(len(sed), dtype=bool)
            preceding_valid[1:] = valid.to_numpy(bool)[:-1]
            valid &= preceding_valid
    return contiguous_events(sed[time_col], mask, valid=valid, max_gap_s=max_gap_s)


def blink_events(
    time,
    closed,
    *,
    valid,
    min_duration_s: float,
    max_duration_s: float,
    max_gap_s: float,
) -> pd.DataFrame:
    """Return bounded open-closed-open candidates under a caller-declared rule.

    Tracking loss, boundary closures and excessive gaps are excluded. Duration
    ends at the observed valid reopening; no frame-rate assumption is made.
    """
    minimum = _finite_real(min_duration_s, "min_duration_s")
    maximum = _finite_real(max_duration_s, "max_duration_s", positive=True)
    if minimum < 0 or minimum > maximum:
        raise ValueError(
            "duration bounds must satisfy 0 <= min_duration_s <= max_duration_s"
        )
    gap = _gap(max_gap_s)
    if gap is None:
        raise ValueError("max_gap_s is required for blink candidates")
    times = _real_series(time, "time").reset_index(drop=True)
    events = contiguous_events(times, closed, valid=valid, max_gap_s=gap)
    selected = events[~events["left_censored"] & ~events["right_censored"]].copy()
    for index, row in selected.iterrows():
        reopening = times.iloc[int(row["end_pos"]) + 1]
        onset = times.iloc[int(row["start_pos"])]
        reopening = reopening.item() if isinstance(reopening, np.generic) else reopening
        onset = onset.item() if isinstance(onset, np.generic) else onset
        selected.loc[index, "end_s"] = reopening
        selected.loc[index, "duration_s"] = float(reopening - onset)
    return selected[selected["duration_s"].between(minimum, maximum)].reset_index(
        drop=True
    )
