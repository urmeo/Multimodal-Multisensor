"""Observed event boundaries, tracking gaps and finite pupil-quality rules."""

import math

import numpy as np
import pandas as pd
import pytest

from mms.fixation import (
    blink_events,
    contiguous_events,
    fixation_events,
    pupil_samples,
    pupil_std,
)


def test_runs_are_one_row_each_with_observed_duration():
    result = contiguous_events(
        [0, 1, 2, 3, 4, 5], [False, True, True, False, True, False]
    )
    assert result.n_samples.tolist() == [2, 1]
    assert result.duration_s.tolist() == [1, 0]
    assert result.start_pos.tolist() == [1, 4]
    assert not result.left_censored.any() and not result.right_censored.any()


def test_recording_boundaries_and_all_run_are_censored():
    result = contiguous_events([0, 1, 2], [True] * 3)
    assert result.n_samples.tolist() == [3]
    assert result.duration_s.tolist() == [2]
    assert result.left_censored.tolist() == [True]
    assert result.right_censored.tolist() == [True]
    assert result.end_reason.tolist() == ["recording_end"]


def test_invalid_quality_and_gaps_split_runs():
    result = contiguous_events(
        [0, 1, 2, 3, 8, 9],
        [True] * 6,
        valid=[True, True, False, True, True, True],
        max_gap_s=2,
    )
    assert result.n_samples.tolist() == [2, 1, 2]
    assert result.end_reason.tolist() == ["invalid", "gap", "recording_end"]
    assert result.left_censored.all() and result.right_censored.all()


def test_missing_mask_cannot_join_runs():
    result = contiguous_events(
        [0, 1, 2], pd.Series([True, pd.NA, True], dtype="boolean")
    )
    assert result.n_samples.tolist() == [1, 1]
    assert result.end_reason.tolist() == ["invalid", "recording_end"]


def test_uint64_event_times_keep_exact_durations_and_gap_boundaries():
    times = np.array([2**60, 2**60 + 1, 2**60 + 4], dtype=np.uint64)
    result = contiguous_events(times, [True] * 3, max_gap_s=2)
    assert result.n_samples.tolist() == [2, 1]
    assert result.duration_s.tolist() == [1, 0]


@pytest.mark.parametrize(
    "time,mask,kwargs",
    [
        ([1, 0], [True, True], {}),
        ([0, np.inf], [True, True], {}),
        ([0, 1], [1, 0], {}),
        ([0, 1], [True], {}),
        ([0, 1], [True, True], {"valid": [True]}),
        ([0, 1], [True, True], {"max_gap_s": 0}),
    ],
)
def test_invalid_event_contract_rejected(time, mask, kwargs):
    with pytest.raises(ValueError):
        contiguous_events(time, mask, **kwargs)


def test_empty_events_have_stable_columns():
    result = contiguous_events([], [])
    assert result.empty
    assert {"duration_s", "left_censored", "right_censored"} <= set(result)
    blink = blink_events(
        [], [], valid=[], min_duration_s=0.1, max_duration_s=0.5, max_gap_s=0.1
    )
    assert blink.empty and blink.columns.equals(result.columns)
    assert result.left_censored.dtype == bool
    assert result.duration_s.dtype == float
    raw = pd.DataFrame(
        columns=["reltime", "gazeQ", "gazeDir.x", "gazeDir.y", "gazeDir.z"]
    )
    fixation = fixation_events(raw)
    assert fixation.empty and fixation.columns.equals(result.columns)


def test_blink_requires_valid_opening_closure_reopening_and_elapsed_bounds():
    times = [0, 0.05, 0.13, 0.2, 0.3, 0.35, 0.5, 0.55]
    closed = [False, True, True, False, False, True, True, False]
    result = blink_events(
        times,
        closed,
        valid=[True] * 8,
        min_duration_s=0.1,
        max_duration_s=0.3,
        max_gap_s=0.1,
    )
    assert result.start_s.tolist() == [0.05]
    assert result.end_s.tolist() == [0.2]
    assert result.duration_s.tolist() == pytest.approx([0.15])
    assert result.n_samples.tolist() == [2]


@pytest.mark.parametrize(
    "closed,valid",
    [
        ([True, True, False], [True] * 3),
        ([False, True, True], [True] * 3),
        ([False, True, False], [True, True, False]),
        ([False, True, False], [False, True, True]),
    ],
)
def test_tracking_loss_or_boundary_closure_is_not_blink(closed, valid):
    assert blink_events(
        [0, 0.05, 0.1],
        closed,
        valid=valid,
        min_duration_s=0.01,
        max_duration_s=0.5,
        max_gap_s=0.1,
    ).empty


def test_blink_duration_bounds_rejected():
    with pytest.raises(ValueError, match="bounds"):
        blink_events(
            [0],
            [False],
            valid=[True],
            min_duration_s=0.5,
            max_duration_s=0.1,
            max_gap_s=0.1,
        )


def test_pupil_samples_keep_index_and_exclude_nonfinite_nonpositive_quality():
    frame = pd.DataFrame(
        {
            "pupil": [3, 4, 0, np.inf, 5, 6, 7],
            "pupilQ": [1, 0.8, 1, 1, np.inf, 1.1, -0.1],
        },
        index=list("abcdefg"),
    )
    result = pupil_samples(frame)
    assert result.index.equals(frame.index)
    assert result.dropna().tolist() == [3, 4]
    assert pupil_std(frame) == pytest.approx(np.sqrt(0.5))


def test_missing_quality_requires_explicit_policy():
    frame = pd.DataFrame({"pupil": [3, 4]})
    with pytest.raises(ValueError, match="pupilQ"):
        pupil_samples(frame)
    assert pupil_samples(frame, missing_quality="allow").tolist() == [3, 4]
    assert math.isnan(pupil_std(pd.DataFrame({"pupil": [3], "pupilQ": [1]})))


@pytest.mark.parametrize("column", ["pupil", "pupilQ"])
@pytest.mark.parametrize("bad", [True, 2 + 3j, "bad"])
def test_pupil_rejects_nonreal_values_without_casting(column, bad):
    frame = pd.DataFrame({"pupil": [3, 4], "pupilQ": [1, 1]})
    frame[column] = pd.Series([1, bad], dtype=object)
    with pytest.raises(ValueError, match="real"):
        pupil_samples(frame)


@pytest.mark.parametrize("quality", [True, np.nan, np.inf, -1, 1.1])
def test_quality_threshold_rejected(quality):
    with pytest.raises(ValueError, match="quality_min"):
        pupil_std(
            pd.DataFrame({"pupil": [3, 4], "pupilQ": [1, 1]}), quality_min=quality
        )


def test_fixation_source_mask_quality_and_coordinates_gate_events():
    frame = pd.DataFrame(
        {
            "reltime": [0, 0.1, 0.2, 0.3, 0.4],
            "fixation": [False, True, True, True, False],
            "gazeQ": [1, 1, 0.1, 1, 1],
        }
    )
    result = fixation_events(frame, max_gap_s=0.2)
    assert result.n_samples.tolist() == [1, 1]
    assert result.right_censored.tolist() == [True, False]
    assert result.left_censored.tolist() == [False, True]


def test_raw_euclidean_fixation_rule_never_compacts_invalid_gaze():
    frame = pd.DataFrame(
        {
            "reltime": [0, 0.1, 0.2, 0.3, 0.4],
            "gazeDir.x": [1] * 5,
            "gazeDir.y": [0, 0.001, np.nan, 0.002, 0.003],
            "gazeDir.z": [0] * 5,
            "gazeQ": [1] * 5,
        }
    )
    result = fixation_events(frame, max_gap_s=0.2)
    assert result.n_samples.tolist() == [1, 1]
    assert result.start_pos.tolist() == [1, 4]


def test_fixation_requires_quality_and_mask_or_coordinates():
    with pytest.raises(ValueError, match="gazeQ"):
        fixation_events(pd.DataFrame({"reltime": [0], "fixation": [True]}))
    with pytest.raises(ValueError, match="mask or complete"):
        fixation_events(pd.DataFrame({"reltime": [0], "gazeQ": [1]}))
