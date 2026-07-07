"""Shared analysis code for the Multimodal-Multisensor study.

    import mms
    ibi = mms.io.load_ibi(1)
    series = mms.hrv.hrv_over_time(ibi)   # time-resolved SDNN/RMSSD
    res = mms.stats.icc1(sdnn_matrix)     # {'icc': ..., 'ci95': (lo, hi), ...}

Not installed? The package sits at the repo root, so a notebook can reach it with
``sys.path.insert(0, str(pathlib.Path.cwd().parent)); import mms``.
"""
from __future__ import annotations

from . import fixation, hrv, io, paths, stats

__all__ = ["paths", "io", "hrv", "stats", "fixation"]
__version__ = "1.0.0"
