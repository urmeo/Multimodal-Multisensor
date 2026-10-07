"""Synthetic numeric edge cases, without participant-data writers."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mms import hrv
from mms import stats as mstats


@pytest.mark.parametrize(
    "values", [[800, 0, 1000], [800, np.nan, 1000], [800, np.inf, 1000], []]
)
def test_rmssd_never_joins_invalid_intervals(values):
    assert math.isnan(hrv.rmssd(values))


def test_rmssd_uses_only_real_adjacent_pairs_after_gap():
    assert hrv.rmssd([800, 810, 0, 1000, 1020]) == pytest.approx(np.sqrt(250))


@pytest.mark.parametrize("bad", [True, 1.5, 1, 0, -1, np.nan, np.inf, "3", 2**100])
def test_rolling_window_must_be_integer_at_least_two(bad):
    with pytest.raises(ValueError, match="window_beats"):
        hrv.hrv_rolling(pd.DataFrame({"ibi": [800, 810]}), window_beats=bad)


@pytest.mark.parametrize(
    "lo,hi", [(np.nan, 2000), (300, np.inf), (900, 800), (True, 2000), (300, "2000")]
)
@pytest.mark.parametrize("function", [hrv.clean_nn, hrv.sdnn, hrv.rmssd])
def test_interval_bounds_rejected(lo, hi, function):
    with pytest.raises(ValueError):
        function([800, 810], lo, hi)


@pytest.mark.parametrize(
    "values",
    [[True, False], [800, 1j], pd.Series([800, 2 + 3j], dtype=object), [800, "bad"]],
)
@pytest.mark.parametrize("function", [hrv.clean_nn, hrv.sdnn, hrv.rmssd])
def test_interval_input_must_be_real(values, function):
    with pytest.raises(ValueError, match="real"):
        function(values)


def test_two_sample_rolling_and_gap_keep_index_and_rows():
    frame = pd.DataFrame({"ibi": [800, 810, 0, 1000, 1020]}, index=[4, 4, 2, 8, 9])
    result = hrv.hrv_rolling(frame, window_beats=2)
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert result.rmssd.tolist()[1] == pytest.approx(10)
    assert result.rmssd.isna().tolist() == [True, False, True, True, False]
    assert result.rmssd.iloc[4] == pytest.approx(20)


def test_rolling_separates_sample_channels():
    frame = pd.DataFrame(
        {
            "reltime": [0, 0, 1, 1],
            "iSensor": [1, 2, 1, 2],
            "ibi": [800, 1500, 810, 1520],
        }
    )
    result = hrv.hrv_rolling(frame, window_beats=2)
    assert result.rmssd.iloc[2:].tolist() == pytest.approx([10, 20])
    windows = hrv.hrv_over_time(frame)
    assert windows.iSensor.tolist() == [1, 2]
    assert windows.n_samples.tolist() == [2, 2]
    assert windows.n_pairs.tolist() == [1, 1]
    assert windows.rmssd.tolist() == pytest.approx([10, 20])


def test_elapsed_gap_excludes_pairs():
    frame = pd.DataFrame({"reltime": [0, 1, 9, 10], "ibi": [800, 810, 1000, 1020]})
    windows = hrv.hrv_over_time(frame, max_gap_s=2)
    assert windows.n_pairs.tolist() == [2]
    assert windows.rmssd.iloc[0] == pytest.approx(np.sqrt(250))
    rolling = hrv.hrv_rolling(frame, window_beats=2, max_gap_s=2)
    assert math.isnan(rolling.rmssd.iloc[2])


@pytest.mark.parametrize(
    "times", [[1, 0], [0, np.inf], [0, np.nan], [-1, 0], [False, True], [0, 1j]]
)
@pytest.mark.parametrize("function", [hrv.hrv_rolling, hrv.hrv_over_time])
def test_elapsed_times_must_be_real_finite_and_ordered(times, function):
    with pytest.raises(ValueError):
        function(pd.DataFrame({"reltime": times, "ibi": [800, 810]}))


def test_integer_order_and_resolution_preserved_before_float_conversion():
    times = np.array([2**60, 2**60 + 1, 2**60 + 2], dtype=np.uint64)
    result = hrv.hrv_over_time(
        pd.DataFrame({"reltime": times, "ibi": [800, 810, 820]}), window_s=1
    )
    assert result.window_start_s.tolist() == [0, 1]
    assert result.n_samples.tolist() == [1, 2]
    times[-1] = 2**60
    with pytest.raises(ValueError, match="chronological"):
        hrv.hrv_over_time(pd.DataFrame({"reltime": times, "ibi": [800, 810, 820]}))


def test_large_absolute_float_time_uses_elapsed_origin_without_looping():
    frame = pd.DataFrame({"reltime": [1e16, 1e16 + 2], "ibi": [800, 810]})
    result = hrv.hrv_over_time(frame, window_s=0.1, step_s=0.1)
    assert len(result) == 20
    assert result.n_samples.sum() == 2


def test_gap_checks_preserve_small_integer_steps_after_large_span():
    times = np.array([0, 2**60, 2**60 + 10], dtype=np.uint64)
    frame = pd.DataFrame({"reltime": times, "ibi": [800, 810, 820]})
    assert hrv.hrv_over_time(frame, window_s=2**61, max_gap_s=1).n_pairs.tolist() == [0]
    assert hrv.hrv_rolling(frame, window_beats=2, max_gap_s=1).rmssd.isna().all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_s": 1e-320},
        {"max_windows": 1},
        {"max_windows": True},
        {"max_windows": 0},
        {"max_windows": 1000001},
        {"window_s": True},
        {"step_s": "1"},
    ],
)
def test_window_limits_reject_unbounded_or_invalid_requests(kwargs):
    frame = pd.DataFrame({"reltime": [0, 100], "ibi": [800, 810]})
    with pytest.raises(ValueError):
        hrv.hrv_over_time(frame, **kwargs)


def test_window_top_overflow_rejected_without_warnings():
    frame = pd.DataFrame({"reltime": [0, 1.7e308], "ibi": [800, 810]})
    with pytest.raises(ValueError, match="finite and advance"):
        hrv.hrv_over_time(frame, window_s=1.5e308, step_s=1e308)


def test_empty_single_and_exact_final_boundary_windows():
    empty = hrv.hrv_over_time(pd.DataFrame({"reltime": [], "ibi": []}))
    assert empty.empty and {"n_samples", "n_pairs", "rmssd"} <= set(empty)
    single = hrv.hrv_over_time(pd.DataFrame({"reltime": [0], "ibi": [800]}))
    assert single.n_samples.tolist() == [1]
    assert math.isnan(single.rmssd.iloc[0])
    final = hrv.hrv_over_time(pd.DataFrame({"reltime": [0, 30, 60], "ibi": [800] * 3}))
    assert final.n_samples.tolist() == [1, 2]


def test_large_interval_arithmetic_remains_finite():
    values = [1e307, 5e307, 1e308]
    frame = pd.DataFrame({"reltime": [0, 1, 2], "ibi": values})
    assert math.isfinite(hrv.sdnn(values, lo=0, hi=1e308))
    assert math.isfinite(hrv.rmssd(values, lo=0, hi=1e308))
    assert np.isfinite(
        hrv.hrv_rolling(frame, window_beats=3, lo=0, hi=1e308).rmssd.iloc[-1]
    )


def test_holm_and_bh_exact_unsorted_vectors_and_unrun_tests():
    assert mstats.holm([0.04, 0.041, 0.042]) == pytest.approx([0.12] * 3)
    assert mstats.benjamini_hochberg([0.04, 0.001, 0.03]) == pytest.approx(
        [0.04, 0.003, 0.04]
    )
    result = mstats.holm([0.041, np.nan, 0.04, 0.042])
    assert result[[0, 2, 3]] == pytest.approx([0.12] * 3)
    assert math.isnan(result[1])
    assert mstats.holm([]).size == 0


@pytest.mark.parametrize(
    "values",
    [
        [-0.1],
        [1.1],
        [np.inf],
        [-np.inf],
        [True],
        [True, 0.1],
        [1j],
        [[0.1, 0.2]],
        [0.1, "bad"],
    ],
)
@pytest.mark.parametrize("adjust", [mstats.holm, mstats.benjamini_hochberg])
def test_invalid_pvalues_fail_clearly(values, adjust):
    with pytest.raises(ValueError):
        adjust(values)


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlation_joint_finite_masks_constants_and_family(method):
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3, np.inf, 5],
            "b": [1, 4, 2, 8, np.nan],
            "constant": [2] * 5,
            "sparse": [1, 2, np.nan, np.nan, np.nan],
        }
    )
    result = mstats.corr_matrix_fdr(frame, method=method)
    correlate = stats.pearsonr if method == "pearson" else stats.spearmanr
    expected_r, expected_p = correlate([1, 2, 3], [1, 4, 2])
    assert result["r"].loc["a", "b"] == pytest.approx(expected_r)
    assert result["p_raw"].loc["a", "b"] == pytest.approx(expected_p)
    assert result["n_pairs"].loc["a", "b"] == 3
    assert result["n_tests"] == 1
    assert result["method"] == method
    assert result["adjustment"] == "Benjamini-Hochberg"
    assert result["p_raw"].loc["constant"].isna().all()
    assert result["p_fdr"].loc["sparse"].isna().all()
    assert math.isnan(result["r"].loc["constant", "constant"])


def test_spearman_preserves_tiny_distinct_ranks():
    frame = pd.DataFrame({"a": [1e-300, 2e-300, 1e308], "b": [1, 2, 3]})
    assert mstats.corr_matrix_fdr(frame, method="spearman")["r"].iloc[
        0, 1
    ] == pytest.approx(1)


@pytest.mark.parametrize(
    "frame,kwargs",
    [
        (pd.DataFrame({"x": [1, 2, 3]}), {"method": "kendall"}),
        (pd.DataFrame([[1, 2]], columns=["x", "x"]), {}),
        (pd.DataFrame({"x": [1j, 2, 3], "y": [1, 2, 3]}), {}),
    ],
)
def test_correlation_contract_rejected(frame, kwargs):
    with pytest.raises(ValueError):
        mstats.corr_matrix_fdr(frame, **kwargs)


def test_icc_complete_finite_rows_match_explicit_subset():
    valid = np.array([[1, 2], [2, 4], [4, 3]], dtype=float)
    actual = mstats.icc1(np.vstack([valid, [np.inf, 1], [2, np.nan]]))
    expected = mstats.icc1(valid)
    assert actual == expected
    assert actual["n"] == 3


def test_icc_all_constant_undefined_and_large_finite_scale_invariant():
    result = mstats.icc1(np.ones((4, 3)))
    assert math.isnan(result["icc"]) and math.isnan(result["F"])
    values = np.array([[1, 2], [3, 2], [4, 5]], dtype=float)
    assert mstats.icc1(values * 1e307)["icc"] == pytest.approx(
        mstats.icc1(values)["icc"]
    )


def test_identical_ratings_stay_perfect_without_rounding_variance():
    values = np.repeat(np.array([[1.0], [2.0], [3.0]]), 5, axis=1)
    result = mstats.icc1(values)
    assert result["icc"] == 1.0
    assert math.isinf(result["F"])
    assert math.isnan(result["ci95"][0])


@pytest.mark.parametrize(
    "values",
    [
        [[1j, 2], [2, 3]],
        [[True, False], [False, True]],
        [[True, 2], [3, 4]],
        [["bad", 2], [2, 3]],
    ],
)
def test_icc_rejects_nonreal_values(values):
    with pytest.raises(ValueError, match="real"):
        mstats.icc1(values)
