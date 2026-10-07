"""Synthetic regressions for notebook alignment, summaries and export boundaries."""

import numpy as np
import pandas as pd
import pytest

from mms import exploration as ex


def questions(
    starts=("2024-01-01T00:00:00Z", "2024-01-01T00:00:02Z"), durations=(1, 1)
):
    start = pd.Series(starts)
    end = [
        (pd.Timestamp(s) + pd.Timedelta(d, unit="s")).isoformat()
        for s, d in zip(starts, durations)
    ]
    return pd.DataFrame(
        {
            "Type": "HADS",
            "Test": [f"{i + 1}. item" for i in range(len(start))],
            "Question": "synthetic",
            "Answer": 1,
            "Time(s)": durations,
            "Question Start Time": start,
            "Question Answer Time": end,
        }
    )


def complete_questions():
    rows = []
    for kind, count in ex.EXPECTED_ITEMS.items():
        for i in range(count):
            stamp = pd.Timestamp("2024-01-01T00:00:00Z") + pd.Timedelta(
                len(rows) * 2, unit="s"
            )
            rows.append(
                {
                    "Type": kind,
                    "Test": f"{i + 1}. item",
                    "Question": "synthetic",
                    "Answer": i % 4,
                    "Time(s)": 1.0,
                    "Question Start Time": stamp.isoformat(),
                    "Question Answer Time": (
                        stamp + pd.Timedelta(1, unit="s")
                    ).isoformat(),
                }
            )
    return pd.DataFrame(rows)


def eye_frame():
    return pd.DataFrame(
        {
            "reltime": [0, 0.05, 0.10, 0.15],
            "gazeDir.x": [1, 1.1, 1.101, 1.2],
            "gazeDir.y": 0.0,
            "gazeDir.z": 0.0,
            "gazeQ": 1.0,
            "pupil": [3, 4, 5, 6],
            "pupilQ": [1, 0.5, 0, 1],
            "leftEyeOpen": [10, 0, 0, 10],
            "rightEyeOpen": [10, 0, 0, 10],
            "leftEyeOpenQ": 1.0,
            "rightEyeOpenQ": 1.0,
        }
    )


def test_question_containment_does_not_assign_future_or_past_rows():
    times = pd.Series(
        [
            "2023-12-31T23:59:59Z",
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:01Z",
            "2024-01-01T00:00:02Z",
            "2024-01-01T00:00:03Z",
        ],
        index=[8, 8, 2, 9, 1],
    )
    result = ex.assign_questions(times, questions())
    assert result.index.tolist() == [8, 8, 2, 9, 1]
    assert result.fillna(-1).tolist() == [-1, 0, -1, 1, -1]


def test_touching_question_boundaries_select_only_new_interval():
    result = ex.assign_questions(
        pd.Series(["2024-01-01T00:00:01Z"]),
        questions(starts=("2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z")),
    )
    assert result.tolist() == [1]


def test_unknown_naive_sensor_zone_is_rejected_then_explicit_mapping_works():
    times = pd.Series(["2024-01-01 01:00:00"])
    with pytest.raises(ValueError, match="timezone metadata"):
        ex.assign_questions(times, questions())
    assert ex.assign_questions(
        times, questions(), sensor_timezone="Europe/Paris"
    ).tolist() == [0]


def test_aware_offset_conversion_preserves_instant():
    assert ex.assign_questions(
        pd.Series(["2024-01-01T01:00:02+01:00"]), questions()
    ).tolist() == [1]


def test_overlapping_questions_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        ex.assign_questions(
            pd.Series(["2024-01-01T00:00:01Z"]), questions(durations=(3, 1))
        )


def test_zero_length_interval_has_no_sample():
    assert (
        ex.assign_questions(
            pd.Series(["2024-01-01T00:00:00Z"]), questions(durations=(0, 1))
        )
        .isna()
        .all()
    )


@pytest.mark.parametrize(
    "mutation",
    ["drop", "duplicate", "wrong_id", "unknown_type", "nan", "complex", "boolean"],
)
def test_raw_codes_require_complete_real_response_identities(mutation):
    frame = complete_questions()
    if mutation == "drop":
        frame = frame.iloc[:-1]
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif mutation == "wrong_id":
        frame.loc[0, "Test"] = "99. item"
    elif mutation == "unknown_type":
        frame.loc[0, "Type"] = "unknown"
    else:
        frame["Answer"] = frame.Answer.astype(object)
        frame.loc[0, "Answer"] = {"nan": np.nan, "complex": 1 + 2j, "boolean": True}[
            mutation
        ]
    with pytest.raises(ValueError):
        ex.raw_answer_summary(frame)


def test_raw_codes_do_not_create_keyed_totals_or_clinical_labels():
    summary = ex.raw_answer_summary(complete_questions())
    assert summary.n_items.sum() == 88
    assert list(summary) == [
        "Type",
        "n_items",
        "mean_raw_code",
        "min_raw_code",
        "max_raw_code",
    ]
    assert "HADS" in set(summary.Type)


def test_duration_summary_uses_seconds_and_sample_sd():
    frame = questions(durations=(1, 3))
    row = ex.duration_summary(frame).iloc[0]
    assert row.total_response_s == 4
    assert row.mean_response_s == 2
    assert row.response_sd_s == pytest.approx(np.sqrt(2))


def test_interval_summary_separates_channels_and_keeps_invalid_positions():
    frame = pd.DataFrame(
        {
            "reltime": [0, 0.1, 0.2, 0.3, 0.4],
            "iSensor": [5, 6, 5, 6, 5],
            "ibi": [800, 1500, 0, 1600, 1000],
        }
    )
    result = ex.interval_summary({"Baseline": frame}).set_index("Channel")
    assert result.loc[5, "n_samples"] == 2
    assert result.loc[5, "n_pairs"] == 0
    assert np.isnan(result.loc[5, "Adjacent-sample RMS difference (ms)"])
    assert result.loc[6, "Adjacent-sample RMS difference (ms)"] == 100


def test_hr_summary_filters_invalid_values_and_confidence():
    frame = pd.DataFrame(
        {
            "iSensor": [5] * 5,
            "heart_rate": [80, np.inf, -1, 100, 90],
            "confidence": [1, 1, 1, 0, 1],
        }
    )
    row = ex.hr_summary({"Baseline": frame}).iloc[0]
    assert row.n_samples == 2
    assert row["Mean HR (bpm)"] == 85


def test_closure_candidates_use_elapsed_duration_and_valid_observation():
    events, seconds = ex.closure_candidates(eye_frame())
    assert len(events) == 1
    assert events.duration_s.iloc[0] == pytest.approx(0.1)
    assert seconds == pytest.approx(0.15)


@pytest.mark.parametrize("column", ["leftEyeOpen", "rightEyeOpenQ", "reltime"])
def test_closure_numeric_contract_rejects_object_complex(column):
    frame = eye_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[1, column] = 1 + 2j
    with pytest.raises(ValueError):
        ex.closure_candidates(frame)


def test_closure_gap_and_invalid_quality_are_not_false_blinks():
    frame = eye_frame()
    frame.loc[2, "leftEyeOpenQ"] = 0
    assert ex.closure_candidates(frame)[0].empty
    frame = eye_frame()
    frame.loc[2:, "reltime"] += 1
    assert ex.closure_candidates(frame)[0].empty


def test_derived_eye_keeps_source_rows_and_separate_event_duration():
    frame = eye_frame()
    samples, events = ex.derive_eye(frame)
    pd.testing.assert_frame_equal(samples[frame.columns], frame)
    assert "duration_s" in events
    assert "duration_s" not in samples
    assert len(samples) == len(frame)
    assert samples.legacy_low_movement.sum() == 1


def test_pupil_quality_and_missing_values_are_consistent():
    row = ex.eye_summary({"Baseline": eye_frame()}).iloc[0]
    assert row.n_valid_pupil == 3
    assert row["Mean pupil (source units)"] == pytest.approx(13 / 3)


def test_baseline_change_uses_matching_channel_and_original_units():
    table = pd.DataFrame(
        {
            "Recording": ["Baseline", "Baseline", "Session 01", "Session 01"],
            "Channel": [5, 6, 5, 6],
            "Mean": [80, 100, 90, 95],
        }
    )
    result = ex.baseline_changes(table, ["Mean"])
    assert result["Change in Mean"].tolist() == [0, 0, 10, -5]


def test_export_requires_opt_in_and_protects_source_root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("MMS_DATA_ROOT", str(data))
    monkeypatch.delenv("MMS_OUTPUT_ROOT", raising=False)
    assert ex.export_frame(pd.DataFrame({"value": [1]}), "derived/test.csv") is None
    assert not list(data.iterdir())
    monkeypatch.setenv("MMS_OUTPUT_ROOT", str(data))
    with pytest.raises(ValueError, match="separate"):
        ex.export_frame(pd.DataFrame({"value": [1]}), "test.csv")
    monkeypatch.setenv("MMS_OUTPUT_ROOT", str(tmp_path / "scratch"))
    destination = ex.export_frame(pd.DataFrame({"value": [1]}), "derived/test.csv")
    assert pd.read_csv(destination).value.tolist() == [1]
    assert not list(data.iterdir())


def test_alignment_skips_without_acquisition_metadata(monkeypatch):
    monkeypatch.delenv("MMS_SENSOR_TIMEZONE", raising=False)
    assert ex.aligned_questions().empty
    assert ex.synchronized_features(1).empty


@pytest.mark.parametrize(
    "frame,k",
    [
        (pd.DataFrame(columns=["a", "b"]), 2),
        (pd.DataFrame({"a": [1, 2], "b": [2, 3]}), 2),
        (pd.DataFrame({"a": [1] * 5, "b": [2] * 5}), 2),
        (pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}), True),
    ],
)
def test_cluster_small_constant_and_invalid_count_guards(frame, k):
    with pytest.raises(ValueError):
        ex.cluster_features(frame, n_clusters=k)


def test_cluster_projection_preserves_complete_row_identity():
    frame = pd.DataFrame(
        {"a": [0, 1, 0, 10, 11, 10, np.nan], "b": [0, 0, 1, 10, 10, 11, 5]},
        index=list("abcdefg"),
    )
    projection, score = ex.cluster_features(frame, n_clusters=2)
    assert projection.index.tolist() == list("abcdef")
    assert set(projection.Cluster) == {0, 1}
    assert 0 < score <= 1


def test_paired_group_zero_change_and_missing_complete_participant():
    frame = pd.DataFrame(
        {
            "Session 01": [1, 2, 3, np.nan],
            "Session 02": [1, 2, 3, 4],
            "Session 03": [1, 2, 3, 5],
        }
    )
    omnibus, pairs = ex.paired_group_tests({"metric": frame})
    assert omnibus.n_participants.iloc[0] == 3
    assert pairs.n_participants.tolist() == [3, 3, 3]
    assert pairs.p_holm.tolist() == [1, 1, 1]
    assert pairs["Paired d_z"].isna().all()
    assert (pairs["CI low"] == 0).all()


def test_interval_spectrum_uses_recorded_time_per_channel():
    frame = pd.DataFrame(
        {
            "reltime": np.arange(30) * 0.5,
            "iSensor": 5,
            "ibi": 800 + 20 * np.sin(np.arange(30)),
        }
    )
    result = ex.interval_spectra({"Baseline": frame})
    assert list(result) == ["Baseline, channel 5"]
    assert list(next(iter(result.values()))) == [
        "Frequency (Hz)",
        "Normalized sample power",
    ]
    assert np.isfinite(next(iter(result.values())).to_numpy()).all()


def test_question_feature_integration_retains_invalid_adjacency_and_empty_coverage(
    monkeypatch,
):
    frame = complete_questions()
    times = [f"2024-01-01T00:00:00.{value:03d}" for value in (100, 200, 300, 400, 500)]
    intervals = pd.DataFrame(
        {
            "datetime": times,
            "reltime": np.arange(5) * 0.1,
            "iSensor": [5, 6, 5, 6, 5],
            "ibi": [800, 1500, 0, 1600, 1000],
        }
    )
    heart = intervals[["datetime", "reltime", "iSensor"]].assign(
        heart_rate=[80, 120, np.inf, 130, 90], confidence=[1, 0, 1, 1, 0]
    )
    eye = eye_frame().assign(
        datetime=[f"2024-01-01T00:00:00.{value:03d}" for value in (100, 150, 200, 250)]
    )
    monkeypatch.setattr(ex.io, "load_psychometric", lambda *a, **k: frame)
    monkeypatch.setattr(ex.io, "load_ibi", lambda *a, **k: intervals)
    monkeypatch.setattr(ex.io, "load_hr", lambda *a, **k: heart)
    monkeypatch.setattr(ex.io, "load_raw", lambda *a, **k: eye)
    result = ex.question_features(1, sensor_timezone="UTC")
    first = result.loc[result.Type.eq("HADS") & result.Item.eq(1)].set_index("Channel")
    assert len(result) == 176
    assert first.loc[5, "n_interval_pairs"] == 0
    assert np.isnan(first.loc[5, "interval_rmsdiff_ms"])
    assert first.loc[6, "interval_rmsdiff_ms"] == 100
    assert first.loc[5, "n_hr"] == 1
    assert first.loc[5, "mean_hr_bpm"] == 80
    assert first.loc[6, "n_hr"] == 1
    assert first.loc[6, "mean_hr_bpm"] == 130
    assert first.loc[5, "n_closure_candidates"] == 1
    assert first.loc[5, "n_pupil"] == 3
    assert first.loc[5, "raw_answer_code"] == 0
    unobserved = result.loc[result.Type.eq("HADS") & result.Item.eq(2)]
    assert (unobserved.n_interval_samples == 0).all()
    assert unobserved.mean_hr_bpm.isna().all()
    assert unobserved.closure_candidates_per_min.isna().all()


def test_synchronized_features_gate_finite_hr_and_confidence(monkeypatch):
    times = [f"2024-01-01T00:00:00.{value:03d}" for value in (100, 200, 300, 400)]
    ibi = pd.DataFrame(
        {
            "datetime": times,
            "reltime": [0, 0.1, 0.2, 0.3],
            "iSensor": 5,
            "ibi": [800, 900, 1000, 1100],
        }
    )
    heart = ibi[["datetime", "reltime", "iSensor"]].assign(
        heart_rate=[80, np.inf, 90, 95], confidence=[1, 1, 0, 1]
    )
    eye = eye_frame().assign(datetime=times)
    monkeypatch.setenv("MMS_SENSOR_TIMEZONE", "UTC")
    monkeypatch.setenv("MMS_CLOCKS_SYNCHRONIZED", "1")
    monkeypatch.setattr(ex.io, "load_ibi", lambda *a, **k: ibi)
    monkeypatch.setattr(ex.io, "load_hr", lambda *a, **k: heart)
    monkeypatch.setattr(ex.io, "load_raw", lambda *a, **k: eye)
    result = ex.synchronized_features(1)
    assert result.heart_rate.tolist() == [80, 95]
    assert result.ibi.tolist() == [800, 1100]
    assert np.isfinite(result[["heart_rate", "ibi", "pupil"]]).all().all()


@pytest.mark.parametrize("column", ["heart_rate", "confidence"])
def test_hr_contract_rejects_complex_measurements(column):
    frame = pd.DataFrame({"heart_rate": [80], "confidence": [1]})
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = 1 + 2j
    with pytest.raises(ValueError, match="real measurements"):
        ex.hr_summary({"Baseline": frame})


def test_paired_group_rejects_complex_and_empty_family():
    with pytest.raises(ValueError, match="at least one"):
        ex.paired_group_tests({})
    frame = pd.DataFrame({"s1": [1 + 2j, 3 + 0j], "s2": [1, 2], "s3": [2, 3]})
    with pytest.raises(ValueError, match="real measurements"):
        ex.paired_group_tests({"metric": frame})


def test_numeric_string_durations_are_aggregated_as_seconds():
    frame = questions(durations=(1, 3))
    frame["Time(s)"] = ["1", "3"]
    row = ex.duration_summary(frame).iloc[0]
    assert row.total_response_s == 4
    assert row.mean_response_s == 2


def test_repeated_question_correlations_do_not_treat_rows_as_replications():
    features = pd.DataFrame(
        {
            "Type": "HADS",
            "Channel": 5,
            "Session": [1, 1, 2, 2, 3, 3],
            "raw_answer_code": [1, 2, 1, 2, 1, 2],
            "response_s": [2, 4, 2, 4, 2, 4],
            "mean_hr_bpm": [80, 90, 80, 90, 80, 90],
            "interval_rmsdiff_ms": [10, 20, 10, 20, 10, 20],
            "mean_pupil_source_units": [3, np.nan, 3, np.nan, 3, np.nan],
        }
    )
    result = ex.question_correlations(features)[("HADS", 5)]
    assert result["r"].loc["raw_answer_code", "response_s"] == pytest.approx(1)
    assert result["n_pairs"].loc["raw_answer_code", "response_s"] == 6
    assert result["n_pairs"].loc["raw_answer_code", "mean_pupil_source_units"] == 3
    assert result["p_raw"].isna().all().all()
    assert result["p_fdr"].isna().all().all()
    assert result["n_tests"] == 0
    assert result["adjustment"] == "none"
    assert "inferential tests omitted" in result["family"]


def test_correlation_plot_labels_actual_adjustment(monkeypatch):
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    result = ex.descriptive_correlations(pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}))
    figure = ex.plot_correlations(result)
    assert "p values omitted" in figure.axes[0].get_title()
    assert "BH" not in figure.axes[0].get_title()
    group = ex.stats.corr_matrix_fdr(
        pd.DataFrame({"a": [1, 2, 3, 4], "b": [3, 2, 4, 1]}), method="spearman"
    )
    figure = ex.plot_correlations(group)
    assert "Benjamini-Hochberg-adjusted" in figure.axes[0].get_title()
