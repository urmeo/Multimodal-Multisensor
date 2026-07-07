"""Statistics for a small within-subject design: ICC with CIs, FDR-corrected correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

_NAN_ICC = {"icc": float("nan"), "ci95": (float("nan"), float("nan")),
            "F": float("nan")}


def icc1(data) -> dict:
    """ICC(1,1) one-way random-effects reliability with a 95% CI.

    ``data`` has shape (n_subjects, k_ratings) — here participants × sessions.
    Returns ``icc``, ``ci95``, ``n``, ``k``, ``F`` (Shrout & Fleiss 1979); the
    values are NaN for degenerate input (fewer than 2 subjects or ratings).
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        return {**_NAN_ICC, "n": 0, "k": 0}
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 2:
        return {**_NAN_ICC, "n": int(n), "k": int(k)}

    grand = x.mean()
    ms_between = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_within = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    denom = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom else float("nan")

    # CI is undefined without within-subject variance (perfect agreement).
    if ms_within == 0:
        return {"icc": float(icc), "ci95": (float("nan"), float("nan")),
                "n": int(n), "k": int(k), "F": float("inf")}

    f = ms_between / ms_within
    f_lo = f / stats.f.ppf(0.975, n - 1, n * (k - 1))
    f_hi = f * stats.f.ppf(0.975, n * (k - 1), n - 1)
    lower = (f_lo - 1) / (f_lo + (k - 1))
    upper = (f_hi - 1) / (f_hi + (k - 1))
    return {"icc": float(icc), "ci95": (float(lower), float(upper)),
            "n": int(n), "k": int(k), "F": float(f)}


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values, in the input order.

    NaN inputs — tests that never validly ran (e.g. a constant column) — are
    excluded from the comparison count ``m`` and returned as NaN rather than
    collapsing the rest of the vector.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = pv.size
    if m:
        order = np.argsort(pv)
        adj = pv[order] * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        tmp = np.empty(m)
        tmp[order] = np.clip(adj, 0, 1)
        out[valid] = tmp
    return out


def corr_matrix_fdr(df: pd.DataFrame) -> dict:
    """Pearson correlation matrix with raw and FDR-adjusted p-values.

    Returns ``{'r', 'p_raw', 'p_fdr', 'n_tests'}``. Judge significance across the
    whole matrix on ``p_fdr``, not ``p_raw``.
    """
    cols = list(df.columns)
    n = len(cols)
    r = np.eye(n)
    p_raw = np.ones((n, n))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = df[cols[i]], df[cols[j]]
            mask = a.notna() & b.notna()
            rr, pp = (stats.pearsonr(a[mask], b[mask]) if mask.sum() >= 3
                      else (np.nan, 1.0))
            r[i, j] = r[j, i] = rr
            p_raw[i, j] = p_raw[j, i] = pp
            pairs.append((i, j, pp))

    p_fdr = np.ones((n, n))
    for (i, j, _), pa in zip(pairs, benjamini_hochberg([p for *_, p in pairs])):
        p_fdr[i, j] = p_fdr[j, i] = pa

    frame = lambda m: pd.DataFrame(m, index=cols, columns=cols)
    return {"r": frame(r), "p_raw": frame(p_raw), "p_fdr": frame(p_fdr),
            "n_tests": len(pairs)}
