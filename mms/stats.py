"""Finite complete-case ICC and explicitly scoped multiple-comparison statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .hrv import _real_series

_NAN_ICC = {
    "icc": float("nan"),
    "ci95": (float("nan"), float("nan")),
    "F": float("nan"),
}


def icc1(data) -> dict:
    """ICC(1,1), participants by ratings, with a 95% CI and finite complete rows."""
    raw = np.asarray(data, dtype=object)
    if raw.ndim != 2 or raw.shape[1] < 2:
        return {**_NAN_ICC, "n": 0, "k": 0}
    x = _real_series(raw.ravel(), "data").to_numpy(float).reshape(raw.shape)
    x = x[np.isfinite(x).all(axis=1)]
    n, k = x.shape
    if n < 2:
        return {**_NAN_ICC, "n": int(n), "k": int(k)}
    # Scale before squared deviations to avoid overflow without changing ICC.
    scale = float(np.max(np.abs(x)))
    if not scale:
        return {**_NAN_ICC, "n": int(n), "k": int(k)}
    x = x / scale
    deviations = x - x[:, :1]
    mean_deviation = deviations.mean(axis=1, keepdims=True)
    mean_rows = x[:, :1] + mean_deviation
    centered_rows = mean_rows[:, 0] - mean_rows[:, 0].mean()
    ms_between = k * (centered_rows**2).sum() / (n - 1)
    ms_within = ((deviations - mean_deviation) ** 2).sum() / (n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    if not denom:
        return {**_NAN_ICC, "n": int(n), "k": int(k)}
    icc = (ms_between - ms_within) / denom
    if ms_within == 0:
        return {
            "icc": float(icc),
            "ci95": (float("nan"), float("nan")),
            "n": int(n),
            "k": int(k),
            "F": float("inf"),
        }
    f = ms_between / ms_within
    f_lo = f / stats.f.ppf(0.975, n - 1, n * (k - 1))
    f_hi = f * stats.f.ppf(0.975, n * (k - 1), n - 1)
    lower = (f_lo - 1) / (f_lo + (k - 1))
    upper = (f_hi - 1) / (f_hi + (k - 1)) if np.isfinite(f_hi) else 1.0
    return {
        "icc": float(icc),
        "ci95": (float(lower), float(upper)),
        "n": int(n),
        "k": int(k),
        "F": float(f),
    }


def _pvalues(pvals) -> np.ndarray:
    raw = np.asarray(pvals, dtype=object)
    if raw.ndim != 1:
        raise ValueError("p-values must be a one-dimensional vector")
    p = _real_series(raw, "p-values").to_numpy(float)
    if np.isinf(p).any() or ((p[~np.isnan(p)] < 0) | (p[~np.isnan(p)] > 1)).any():
        raise ValueError("p-values must be in [0, 1], with NaN for unrun tests")
    return p


def _adjust(pvals, method: str) -> np.ndarray:
    p = _pvalues(pvals)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    values = p[valid]
    count = len(values)
    if count:
        order = np.argsort(values, kind="stable")
        if method == "holm":
            adjusted = np.maximum.accumulate(values[order] * np.arange(count, 0, -1))
        else:
            adjusted = values[order] * count / np.arange(1, count + 1)
            adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        original_order = np.empty(count)
        original_order[order] = np.clip(adjusted, 0, 1)
        out[valid] = original_order
    return out


def holm(pvals) -> np.ndarray:
    """Holm family-wise adjusted p-values in input order; unrun NaNs excluded."""
    return _adjust(pvals, "holm")


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH FDR adjusted p-values in input order; unrun NaNs excluded."""
    return _adjust(pvals, "bh")


def corr_matrix_fdr(df: pd.DataFrame, method: str = "pearson") -> dict:
    """Pairwise finite Pearson or Spearman correlations with one BH test family.

    Constant or fewer-than-three complete pairs are undefined, including their
    p-values. Diagonal correlations are descriptive and never counted as tests.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    if not df.columns.is_unique:
        raise ValueError("correlation columns must have unique names")
    cols = list(df.columns)
    values = [_real_series(df[col], str(col)).to_numpy(float) for col in cols]
    size = len(cols)
    r, p_raw, p_fdr = (np.full((size, size), np.nan) for _ in range(3))
    counts = np.zeros((size, size), dtype=int)
    pairs = []
    correlate = stats.pearsonr if method == "pearson" else stats.spearmanr
    for i, a in enumerate(values):
        finite = a[np.isfinite(a)]
        counts[i, i] = len(finite)
        if len(finite) >= 3 and np.any(finite != finite[0]):
            r[i, i] = 1.0
        for j in range(i + 1, size):
            b = values[j]
            mask = np.isfinite(a) & np.isfinite(b)
            counts[i, j] = counts[j, i] = int(mask.sum())
            aa, bb = a[mask], b[mask]
            if len(aa) >= 3 and np.any(aa != aa[0]) and np.any(bb != bb[0]):
                if method == "pearson":
                    aa = aa / float(np.max(np.abs(aa)))
                    bb = bb / float(np.max(np.abs(bb)))
                    aa = aa - aa.mean()
                    bb = bb - bb.mean()
                rr, pp = correlate(aa, bb)
                r[i, j] = r[j, i] = float(rr)
                p_raw[i, j] = p_raw[j, i] = float(pp)
            pairs.append((i, j, p_raw[i, j]))
    adjusted = benjamini_hochberg([p for _, _, p in pairs])
    for (i, j, _), value in zip(pairs, adjusted):
        p_fdr[i, j] = p_fdr[j, i] = value

    def frame(matrix):
        return pd.DataFrame(matrix, index=cols, columns=cols)

    return {
        "r": frame(r),
        "p_raw": frame(p_raw),
        "p_fdr": frame(p_fdr),
        "n_pairs": frame(counts),
        "n_tests": int(np.isfinite(adjusted).sum()),
        "method": method,
        "adjustment": "Benjamini-Hochberg",
        "family": "all valid off-diagonal column pairs",
    }
