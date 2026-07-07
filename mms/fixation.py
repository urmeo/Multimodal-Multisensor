"""Pupil-variability metric from the processed 'sed_fix' eye-tracking streams."""
from __future__ import annotations

import pandas as pd


def pupil_std(sed: pd.DataFrame, quality_min: float = 0.5) -> float:
    """Standard deviation of pupil diameter over good-quality samples."""
    if "pupil" not in sed.columns:
        return float("nan")
    good = sed[sed["pupilQ"] >= quality_min] if "pupilQ" in sed.columns else sed
    p = pd.to_numeric(good["pupil"], errors="coerce").dropna()
    return float(p.std(ddof=1)) if len(p) > 1 else float("nan")
