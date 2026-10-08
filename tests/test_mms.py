"""Numerical regressions and checks against the preserved study summaries."""

import math

import numpy as np
import pandas as pd
import pytest

import mms


def test_icc1_reproduces_readme_hrv_value():
    g = mms.io.load_group_summary("HRV_SDNN")
    mat = g[["Session 01", "Session 02", "Session 03"]].to_numpy(float)
    res = mms.stats.icc1(mat)
    assert res["icc"] == pytest.approx(0.22, abs=0.02)
    lo, hi = res["ci95"]
    assert lo < 0 < hi


def test_icc1_perfect_agreement_is_one():
    data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    res = mms.stats.icc1(data)
    assert res["icc"] == pytest.approx(1.0, abs=1e-9)
    assert math.isnan(res["ci95"][0]) and math.isnan(res["ci95"][1])
    assert math.isinf(res["F"])


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0], [2.0]]),
        np.array([[1.0, 2.0]]),
    ],
)
def test_icc1_degenerate_input_returns_nan(bad):
    res = mms.stats.icc1(bad)
    assert math.isnan(res["icc"])


def test_sdnn_filters_artifacts_and_zeros():
    beats = pd.Series([800, 810, 0, 790, 5000, 805])
    expected = pd.Series([800, 810, 790, 805]).std(ddof=1)
    assert mms.hrv.sdnn(beats) == pytest.approx(expected)


def test_rmssd_preserves_invalid_interval_gaps():
    beats = pd.Series([800, 810, 0, 5000, 790, 805])
    expected = float(np.sqrt((10**2 + 15**2) / 2))
    assert mms.hrv.rmssd(beats) == pytest.approx(expected)


def test_hrv_over_time_is_not_a_broadcast_scalar():
    rng = np.random.default_rng(0)
    n = len(np.arange(0, 300, 0.8))
    df = pd.DataFrame(
        {"reltime": np.arange(0, 300, 0.8), "ibi": 800 + rng.normal(0, 40, n)}
    )
    ts = mms.hrv.hrv_over_time(df, window_s=30)
    assert len(ts) > 1
    assert ts["sdnn"].nunique() > 1


def test_hrv_over_time_conserves_beats():
    df = pd.DataFrame({"reltime": [0, 10, 20, 30, 59.9, 60.0], "ibi": [800] * 6})
    ts = mms.hrv.hrv_over_time(df, window_s=30)
    assert ts["n_beats"].sum() == 6


def test_benjamini_hochberg_bounds_and_dominates_raw():
    p = [0.9, 0.001, 0.5, 0.01, 0.04]
    adj = mms.stats.benjamini_hochberg(p)
    assert np.all((adj >= 0) & (adj <= 1))
    assert np.all(adj >= np.array(p) - 1e-12)
    assert np.array_equal(np.argsort(adj), np.argsort(p))


def test_benjamini_hochberg_is_nan_safe():
    adj = mms.stats.benjamini_hochberg([0.001, 0.5, np.nan, 0.02])
    assert np.isnan(adj[2])
    assert np.all(np.isfinite(adj[[0, 1, 3]]))


def test_corr_matrix_fdr_shrinks_significance():
    g = pd.concat(
        [
            mms.io.load_group_summary("HRV_SDNN")
            .set_index("Participant")
            .add_suffix("_HRV"),
            mms.io.load_group_summary("Pupil_Dilation_STD")
            .set_index("Participant")
            .add_suffix("_Pupil"),
        ],
        axis=1,
    )
    res = mms.stats.corr_matrix_fdr(g)
    n = res["r"].shape[0]
    iu = np.triu_indices(n, k=1)
    sig_raw = int((res["p_raw"].values[iu] < 0.05).sum())
    sig_fdr = int((res["p_fdr"].values[iu] < 0.05).sum())
    assert sig_raw > 0
    assert sig_fdr <= sig_raw


def test_pupil_std_quality_gates_and_matches_std():
    sed = pd.DataFrame(
        {"pupil": [3.0, 4.0, 100.0, 5.0], "pupilQ": [1.0, 1.0, 0.1, 1.0]}
    )
    expected = pd.Series([3.0, 4.0, 5.0]).std(ddof=1)
    assert mms.fixation.pupil_std(sed, quality_min=0.5) == pytest.approx(expected)


def test_parse_datetime_preserves_explicit_zone():
    out = mms.io.parse_datetime(pd.Series(["2024-05-28T15:37:00.695Z"]))
    assert str(out.dt.tz) == "UTC"
    assert out.iloc[0].year == 2024


def test_load_hr_individual_source_and_raw_confidence():
    hi = mms.io.load_hr(1, source="individual")
    lo = mms.io.load_hr(1, source="individual", high_confidence_only=False)
    assert len(lo) >= len(hi)


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        mms.io.load_ibi(1, source="casestudy")


@pytest.mark.parametrize(
    "window_s, step_s",
    [
        (0, None),
        (-1, None),
        (float("nan"), None),
        (float("inf"), None),
        (30, 0),
        (30, -1),
        (30, float("nan")),
        (30, float("inf")),
    ],
)
def test_hrv_over_time_rejects_invalid_windows(window_s, step_s):
    df = pd.DataFrame({"reltime": [0, 1], "ibi": [800, 810]})
    with pytest.raises(ValueError, match="finite and positive"):
        mms.hrv.hrv_over_time(df, window_s=window_s, step_s=step_s)


def test_hrv_rolling_differences_stay_inside_beat_window():
    df = pd.DataFrame({"ibi": [700, 800, 800, 800]})
    out = mms.hrv.hrv_rolling(df, window_beats=3)
    assert out["rmssd"].iloc[2] == pytest.approx(np.sqrt(5000))
    assert out["rmssd"].iloc[3] == 0


def test_hrv_over_time_rejects_nonfinite_timestamps():
    df = pd.DataFrame(
        {
            "reltime": [0, np.inf, -np.inf, np.nan, 1],
            "ibi": [800, 1900, 1900, 1900, 810],
        }
    )
    with pytest.raises(ValueError, match="finite nonnegative"):
        mms.hrv.hrv_over_time(df)
