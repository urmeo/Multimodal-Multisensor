"""Synthetic immutable-input, clock, schema and command regressions."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mms import io, paths
from pipeline import build_group_summaries as build
from scripts import deidentify_timestamps as shift

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "source"
    for name in paths.SOURCE_CSV_RELATIVE:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "datetime": ["2001/02/03 04:05:00.000", "2001/02/03 04:05:01.000"],
                "value": [1, 2],
            }
        ).to_csv(destination, index=False)
    for name in paths.SOURCE_RAW_RELATIVE:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("reltime;ibi;iSensor\n0;800;3\n1;810;3\n")
    expected = {metric: [] for metric in paths.GROUP_METRICS}
    for session in (1, 2, 3):
        ibi = pd.DataFrame(
            {
                "iSensor": [3, 3, 5, 5],
                "ibi": [800, 800 + 10 * session, 900, 900 + 20 * session],
            }
        )
        eye = pd.DataFrame(
            {
                "iSensor": [0] * 4,
                "pupilQ": [0.8] * 4,
                "pupil": [2, 2 + 0.1 * session, 2 + 0.4 * session, 2 + 0.7 * session],
            }
        )
        durations = np.array([1, 2 * session, 3 * session, 4 * session], dtype=float)
        starts = pd.date_range("2001-02-03T04:05:00Z", periods=4, freq="1min")
        psych = pd.DataFrame(
            {
                "Type": ["synthetic"] * 4,
                "Test": ["synthetic"] * 4,
                "Question": ["fixture"] * 4,
                "Answer": [1] * 4,
                "Time(s)": durations,
                "Question Start Time": starts.astype(str),
                "Question Answer Time": (
                    starts + pd.to_timedelta(durations, unit="s")
                ).astype(str),
            }
        )
        for name, frame in (
            (f"processed/ibi_{session:02d}.csv", ibi),
            (f"processed/sed_fix_{session:02d}.csv", eye),
            (f"psychometric/Psychometric_Test_Results_{session:02d}.csv", psych),
        ):
            frame.to_csv(root / "case-study" / name, index=False)
        expected["HRV_SDNN"].append(ibi["ibi"].std(ddof=1))
        expected["Pupil_Dilation_STD"].append(eye["pupil"].std(ddof=1))
        expected["Psychometric_Test_Duration_STD"].append(psych["Time(s)"].std(ddof=1))
    for metric, values in expected.items():
        frame = pd.DataFrame(
            {
                "Participant": paths.PARTICIPANTS,
                "Session 01": np.arange(100.0, 110),
                "Session 02": np.arange(200.0, 210),
                "Session 03": np.arange(300.0, 310),
            }
        )
        frame.loc[1, list(paths.SESSION_COLUMNS)] = values
        if metric == "Psychometric_Test_Duration_STD":
            frame.loc[1, "Session 01"] += 2e-5
        frame.to_csv(root / "group_results" / f"{metric}.csv", index=False)
    return root


def command(module, *args):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )


def test_explicit_dataset_root_and_code_only_missing_data(
    dataset, monkeypatch, tmp_path
):
    assert len(io.load_ibi(1, data_root=dataset)) == 4
    assert io.load_raw("ibi", data_root=dataset)["ibi"].tolist() == [800, 810]
    monkeypatch.setattr(paths, "DATA", tmp_path / "wheel-without-data")
    with pytest.raises(FileNotFoundError, match="Wheels contain code only.*data_root"):
        io.load_ibi(1)


@pytest.mark.parametrize("session", [0, 4, -1, True, 1.2, "01"])
def test_invalid_stream_session_rejected(session, dataset):
    with pytest.raises(ValueError, match="session"):
        io.load_ibi(session, data_root=dataset)


def test_malformed_legacy_psychometric_rejected(dataset):
    with pytest.raises(ValueError, match="legacy.*malformed"):
        io.load_psychometric(0, source="individual", data_root=dataset)
    malformed = pd.DataFrame({"Score": [3.0], "Time(s)": ["2001-02-03T04:05:00Z"]})
    with pytest.raises(ValueError, match="noncanonical.*Score holds duration"):
        io.validate_psychometric(malformed)


def test_datetime_preserves_awareness_and_requires_declared_naive_zone():
    aware = io.parse_datetime(pd.Series(["2001-02-03T04:05:00+02:00"], dtype="string"))
    assert aware.dt.tz is not None and aware.iloc[0].utcoffset().total_seconds() == 7200
    naive = io.parse_datetime(pd.Series(["2001/02/03 04:05:00.000"]))
    assert naive.dt.tz is None
    mapped = io.parse_datetime(
        pd.Series(["2001/02/03 04:05:00.000"]), naive_timezone="UTC"
    )
    assert str(mapped.dt.tz) == "UTC"


@pytest.mark.parametrize(
    "values",
    [
        ["2001-01-01", "bad"],
        ["2001-01-01", None],
        ["2001-01-01", "2001-01-01T00:00:00Z"],
        [1000, 2000],
    ],
)
def test_datetime_rejects_incomplete_ambiguous_or_mixed_columns(values):
    with pytest.raises(ValueError):
        io.parse_datetime(pd.Series(values))


def test_explicit_missing_datetime_keeps_original_position():
    parsed = io.parse_datetime(
        pd.Series([None, "2001-01-01T00:00:00Z"], index=[4, 9]), allow_missing=True
    )
    assert parsed.index.tolist() == [4, 9] and pd.isna(parsed.iloc[0])
    assert parsed.dt.tz is not None


@pytest.mark.parametrize(
    "kind", ["third_identity", "changed_value", "wrong_identity", "within_value"]
)
def test_duplicate_allowlist_exact_values_identities_and_multiplicity(kind):
    metric = "HRV_SDNN" if kind == "within_value" else "Psychometric_Test_Duration_STD"
    frame = io.load_group_summary(metric).copy()
    if kind == "third_identity":
        frame.loc[frame.Participant == "P09", "Session 01"] = 4.409281089248826
    elif kind == "changed_value":
        frame.loc[frame.Participant.isin(["P07", "P08"]), "Session 01"] = 9.123
    elif kind == "wrong_identity":
        frame.loc[frame.Participant == "P08", "Session 01"] = 8.123
        frame.loc[frame.Participant == "P09", "Session 01"] = 4.409281089248826
    else:
        frame.loc[frame.Participant == "P01", ["Session 02", "Session 03"]] = 9.123
    with pytest.raises(ValueError, match="duplicate"):
        io.validate_group_summary(frame, metric)


@pytest.mark.parametrize(
    "kind", ["missing", "duplicated", "wrong_columns", "nonfinite"]
)
def test_group_summary_canonical_schema_required(kind, dataset):
    frame = io.load_group_summary("HRV_SDNN", data_root=dataset)
    if kind == "missing":
        frame = frame.iloc[:-1]
    elif kind == "duplicated":
        frame.loc[1, "Participant"] = "P01"
    elif kind == "wrong_columns":
        frame = frame.rename(columns={"Session 01": "Session 1"})
    else:
        frame.loc[0, "Session 01"] = np.inf
    with pytest.raises(ValueError):
        io.validate_group_summary(frame, "HRV_SDNN")


def test_complex_summary_and_interval_values_are_rejected(dataset):
    frame = io.load_group_summary("HRV_SDNN", data_root=dataset)
    frame["Session 01"] = frame["Session 01"].astype(complex) + 1j
    with pytest.raises(ValueError, match="real"):
        io.validate_group_summary(frame, "HRV_SDNN")
    with pytest.raises(ValueError, match="real"):
        build._numeric(pd.DataFrame({"ibi": [800 + 1j, 810 + 2j]}), "ibi")


def test_summary_check_metric_tolerances_and_no_writes(dataset, tmp_path):
    before = snapshot(dataset)
    result = command(
        "pipeline.build_group_summaries", "--check", "--data-root", dataset
    )
    assert result.returncode == 0, result.stderr
    assert "within tolerance" in result.stdout and snapshot(dataset) == before
    assert not (tmp_path / "review").exists()
    target = dataset / "group_results/Psychometric_Test_Duration_STD.csv"
    frame = pd.read_csv(target)
    frame.loc[1, "Session 01"] += 4e-5
    frame.to_csv(target, index=False)
    before = snapshot(dataset)
    result = command(
        "pipeline.build_group_summaries", "--check", "--data-root", dataset
    )
    assert result.returncode == 1 and "DRIFT" in result.stdout
    assert snapshot(dataset) == before


def test_summary_explicit_output_and_both_duplicate_flags(dataset, tmp_path):
    for metric, identities, columns, value in (
        ("HRV_SDNN", ["P01"], ["Session 02", "Session 03"], 65.39381048265373),
        (
            "Psychometric_Test_Duration_STD",
            ["P07", "P08"],
            ["Session 01"],
            4.409281089248826,
        ),
    ):
        target = dataset / "group_results" / f"{metric}.csv"
        frame = pd.read_csv(target)
        frame.loc[frame.Participant.isin(identities), columns] = value
        frame.to_csv(target, index=False)
    before = snapshot(dataset)
    output = tmp_path / "review"
    assert build.main(["--data-root", str(dataset), "--output-root", str(output)]) == 0
    assert (output / "MANIFEST.json").is_file()
    manifest = json.loads((output / "MANIFEST.json").read_text())
    assert len(manifest["known_equal_value_issues"]) == 2
    assert manifest["participant_identity_independently_established"] is False
    assert len(list(output.glob("*.csv"))) == 4
    assert snapshot(dataset) == before


def test_summary_full_prevalidation_prevents_partial_output(dataset, tmp_path):
    target = dataset / "case-study/psychometric/Psychometric_Test_Results_03.csv"
    frame = pd.read_csv(target).drop(columns="Question Answer Time")
    frame.to_csv(target, index=False)
    before = snapshot(dataset)
    output = tmp_path / "review"
    assert build.main(["--data-root", str(dataset), "--output-root", str(output)]) == 1
    assert not output.exists() and snapshot(dataset) == before


@pytest.mark.parametrize("kind", ["ibi", "sed_fix"])
def test_summary_fractional_channels_fail_before_output(
    kind, dataset, tmp_path, capsys
):
    target = dataset / "case-study/processed" / f"{kind}_03.csv"
    frame = pd.read_csv(target)
    frame["iSensor"] = [1.1, 1.1, 1.9, 1.9]
    frame.to_csv(target, index=False)
    before = snapshot(dataset)
    output = tmp_path / "review"
    assert build.main(["--data-root", str(dataset), "--output-root", str(output)]) == 1
    assert "integer channel identifiers" in capsys.readouterr().err
    assert not output.exists() and snapshot(dataset) == before
    streams = {
        3: (
            io.load_ibi(3, data_root=dataset),
            io.load_fixation(3, data_root=dataset),
            pd.DataFrame(),
        )
    }
    with pytest.raises(ValueError, match="integer channel identifiers"):
        build.filtered_sample_metrics(streams)


@pytest.mark.parametrize(
    "module", ["pipeline.build_group_summaries", "scripts.deidentify_timestamps"]
)
def test_help_and_parser_errors_never_write_sources(module, dataset):
    before = snapshot(dataset)
    assert command(module, "--help").returncode == 0
    assert command(module, "--typo").returncode == 2
    assert snapshot(dataset) == before


def test_output_roots_reject_source_parent_symlink_and_hardlink(dataset, tmp_path):
    for destination in (dataset, dataset / "generated", dataset.parent):
        with pytest.raises(ValueError, match="separate"):
            paths.resolve_output_root(destination, data_root=dataset)
    link = tmp_path / "link"
    link.symlink_to(dataset, target_is_directory=True)
    with pytest.raises(ValueError, match="separate"):
        paths.resolve_output_root(link, data_root=dataset)
    output = tmp_path / "review"
    output.mkdir()
    alias = output / "HRV_SDNN_P02_consistency.csv"
    os.link(dataset / "group_results/HRV_SDNN.csv", alias)
    before = snapshot(dataset)
    assert build.main(["--data-root", str(dataset), "--output-root", str(output)]) == 1
    assert not (output / "MANIFEST.json").exists() and snapshot(dataset) == before
    with pytest.raises(ValueError):
        paths.output_path(
            output, "../source/group_results/HRV_SDNN.csv", data_root=dataset
        )


def test_calendar_shift_stringdtype_renamed_columns_and_aware_elapsed_time():
    frame = pd.DataFrame(
        {
            "renamed clock": pd.Series(
                ["2001-02-03T04:05:00+02:00", "2001-02-03T04:05:01+02:00"],
                dtype="string",
            ),
            "value": [2, 3],
        }
    )
    before = frame.copy(deep=True)
    shifted, columns = shift.shift_frame(frame)
    assert columns == ["renamed clock"]
    parsed = io.parse_datetime(shifted["renamed clock"])
    assert parsed.iloc[0].year == 2000 and parsed.dt.tz is not None
    assert (parsed.iloc[1] - parsed.iloc[0]).total_seconds() == 1
    pd.testing.assert_frame_equal(frame, before)


def test_calendar_detection_checks_later_rows_and_refuses_parse_loss():
    frame = pd.DataFrame(
        {
            "renamed clock": pd.Series(
                ["bad"] * 60 + ["2001-02-03T04:05:00Z"], dtype="string"
            )
        }
    )
    with pytest.raises(ValueError, match="invalid"):
        shift.shift_frame(frame)


def test_shift_validates_full_inventory_before_write_without_offsets_in_logs(
    dataset, tmp_path, capsys
):
    legacy = dataset / "individual/psychometric/Psychometric_Test_Results_00.csv"
    pd.DataFrame(
        {
            "Time(s)": ["2001-02-03T04:05:00Z"],
            "Question Start Time": ["2001-02-03T04:05:03Z"],
            "Answer Time": [None],
            "Score": [3.0],
        }
    ).to_csv(legacy, index=False)
    before = snapshot(dataset)
    output = tmp_path / "shifted"
    assert shift.main(["--data-root", str(dataset), "--output-root", str(output)]) == 0
    assert snapshot(dataset) == before
    text = capsys.readouterr().out
    assert "2001" not in text and "offset" not in text and "04:05" not in text
    assert "Raw TXT" in text and "awareness-specific" in text
    assert (output / legacy.relative_to(dataset)).is_file()
    assert not list(output.rglob("*.txt"))
    shifted = pd.read_csv(output / legacy.relative_to(dataset))
    assert pd.isna(shifted["Answer Time"].iloc[0])
    assert (
        io.parse_datetime(shifted["Question Start Time"])
        - io.parse_datetime(shifted["Time(s)"])
    ).dt.total_seconds().iloc[0] == 3


def test_shift_parse_failure_and_missing_inventory_never_create_output(
    dataset, tmp_path
):
    path = dataset / "individual/processed/sed_03.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "datetime"] = "bad"
    frame.to_csv(path, index=False)
    before = snapshot(dataset)
    output = tmp_path / "shifted"
    assert shift.main(["--data-root", str(dataset), "--output-root", str(output)]) == 1
    assert not output.exists() and snapshot(dataset) == before
    path.unlink()
    assert shift.main(["--data-root", str(dataset), "--output-root", str(output)]) == 1
    assert not output.exists()


def test_shift_in_place_flag_rejected_and_dry_run_immutable(dataset):
    before = snapshot(dataset)
    result = command("scripts.deidentify_timestamps", "--data-root", dataset)
    assert result.returncode == 0, result.stderr
    assert snapshot(dataset) == before
    result = command("scripts.deidentify_timestamps", "--apply", "--data-root", dataset)
    assert result.returncode == 2 and snapshot(dataset) == before


def test_generated_csv_is_excluded_from_inventory(dataset):
    generated = dataset / "group_results/reconstructed"
    generated.mkdir()
    (generated / "bad.csv").write_text("corrupt")
    assert len(shift.collect(dataset)) == 51
    assert generated / "bad.csv" not in paths.source_files(dataset)


def test_shift_source_hardlink_and_directory_destination_fail_before_any_write(
    dataset, tmp_path
):
    output = tmp_path / "review"
    relative = Path("individual/processed/sed_03.csv")
    destination = output / relative
    destination.parent.mkdir(parents=True)
    os.link(dataset / relative, destination)
    before = snapshot(dataset)
    output_before = snapshot(output)
    assert shift.main(["--data-root", str(dataset), "--output-root", str(output)]) == 1
    assert snapshot(output) == output_before and snapshot(dataset) == before
    destination.unlink()
    destination.mkdir()
    with pytest.raises(ValueError, match="must be a file"):
        paths.output_path(output, relative, data_root=dataset)


def test_filtered_metrics_keep_channels_separate_and_share_pupil_quality():
    streams = {
        1: (
            pd.DataFrame({"iSensor": [3, 5, 3, 5], "ibi": [800, 1000, 810, 1020]}),
            pd.DataFrame(
                {
                    "iSensor": [0] * 4,
                    "pupil": [2, 3, 99, 99],
                    "pupilQ": [0.5, 0.5, 0.4, 1.1],
                }
            ),
            pd.DataFrame(),
        )
    }
    records = build.filtered_sample_metrics(streams)
    assert len(records) == 3
    intervals = [
        record for record in records if record["metric"] == "interval_sample_sd_ms"
    ]
    assert [record["channel"] for record in intervals] == [3, 5]
    assert [record["value"] for record in intervals] == pytest.approx(
        [10 / 2**0.5, 20 / 2**0.5]
    )
    pupil = next(
        record
        for record in records
        if record["metric"] == "pupil_sample_sd_reported_mm"
    )
    assert pupil["n_samples"] == 2 and pupil["value"] == pytest.approx(1 / 2**0.5)


def test_output_hardlink_to_other_dataset_artifact_is_rejected(dataset, tmp_path):
    artifact = dataset / "group_results/historical.png"
    artifact.write_bytes(b"historical image bytes")
    output = tmp_path / "review"
    output.mkdir()
    os.link(artifact, output / "result.csv")
    with pytest.raises(ValueError, match="aliases"):
        paths.output_path(output, "result.csv", data_root=dataset)
    assert artifact.read_bytes() == b"historical image bytes"


@pytest.mark.parametrize(
    "module", ["pipeline.build_group_summaries", "scripts.deidentify_timestamps"]
)
@pytest.mark.parametrize("kind", ["input", "output"])
def test_cyclic_roots_fail_helpfully_without_traceback_or_writes(
    module, kind, dataset, tmp_path
):
    loop = tmp_path / "loop"
    loop.symlink_to("loop")
    before = snapshot(dataset)
    arguments = (
        ["--data-root", loop]
        if kind == "input"
        else ["--data-root", dataset, "--output-root", loop]
    )
    result = command(module, *arguments)
    assert result.returncode == 1
    assert "cyclic symlink" in result.stderr and "Traceback" not in result.stderr
    assert snapshot(dataset) == before


@pytest.mark.parametrize(
    "module", ["pipeline.build_group_summaries", "scripts.deidentify_timestamps"]
)
def test_cli_flags_cannot_be_abbreviated(module, dataset):
    before = snapshot(dataset)
    assert command(module, "--data-r", dataset).returncode == 2
    assert command(module, "--output-r", dataset).returncode == 2
    assert snapshot(dataset) == before


def test_custom_dataset_outputs_still_protect_canonical_dataset(
    dataset, tmp_path, monkeypatch
):
    project = tmp_path / "project"
    canonical = project / "data"
    canonical.mkdir(parents=True)
    source = canonical / "observation.csv"
    source.write_text("value\n1\n")
    monkeypatch.setattr(paths, "DATA", canonical)
    monkeypatch.setattr(paths, "ROOT", project)
    for destination in (canonical, canonical / "generated", project):
        with pytest.raises(ValueError, match="separate"):
            paths.resolve_output_root(destination, data_root=dataset)
    output = project / "outputs"
    output.mkdir()
    os.link(source, output / "result.csv")
    before = source.read_bytes()
    with pytest.raises(ValueError, match="aliases"):
        paths.output_path(output, "result.csv", data_root=dataset)
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "relative",
    [
        "docs/reports/snapshot.pdf",
        "images/photo.jpg",
        "outputs/silhouette_score.png",
        "LICENSE",
    ],
)
def test_outputs_protect_declared_preserved_artifacts(
    relative, dataset, tmp_path, monkeypatch
):
    project = tmp_path / "project"
    canonical = project / "data"
    canonical.mkdir(parents=True)
    original = project / relative
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(b"preserved evidence")
    monkeypatch.setattr(paths, "DATA", canonical)
    monkeypatch.setattr(paths, "ROOT", project)
    output = project / "outputs"
    output.mkdir(exist_ok=True)
    os.link(original, output / "reliability.png")
    with pytest.raises(ValueError, match="aliases"):
        paths.output_path(output, "reliability.png", data_root=dataset)
    assert original.read_bytes() == b"preserved evidence"


def test_current_outputs_can_be_regenerated_without_protecting_unrelated_files(
    dataset, tmp_path, monkeypatch
):
    project = tmp_path / "project"
    canonical = project / "data"
    canonical.mkdir(parents=True)
    monkeypatch.setattr(paths, "DATA", canonical)
    monkeypatch.setattr(paths, "ROOT", project)
    output = project / "outputs"
    output.mkdir()
    for name in ("reliability.png", "correlation.png", "figures.csv", "figures.toml"):
        target = output / name
        target.write_bytes(b"current generated artifact")
        assert paths.output_path(output, name, data_root=dataset) == target


def test_canonical_question_clock_awareness_cannot_be_silently_removed(dataset):
    frame = io.load_psychometric(1, data_root=dataset)
    for name in ("Question Start Time", "Question Answer Time"):
        frame[name] = io.parse_datetime(frame[name]).dt.tz_convert(None).astype(str)
    with pytest.raises(ValueError, match="timezone-aware"):
        io.validate_psychometric(frame)


@pytest.mark.parametrize("kind", ["bool", "complex"])
def test_canonical_duration_rejects_nonreal_values_instead_of_casting(dataset, kind):
    frame = io.load_psychometric(1, data_root=dataset)
    starts = io.parse_datetime(frame["Question Start Time"])
    frame["Question Answer Time"] = (starts + pd.Timedelta(1, unit="s")).astype(str)
    frame["Time(s)"] = (
        pd.Series([True] * len(frame))
        if kind == "bool"
        else pd.Series([1 + 2j] * len(frame), dtype=object)
    )
    with pytest.raises(ValueError, match="real"):
        io.validate_psychometric(frame)


def test_group_and_pipeline_reject_boolean_numbers(dataset):
    frame = io.load_group_summary("HRV_SDNN", data_root=dataset)
    frame["Session 01"] = frame["Session 01"].astype(object)
    frame.loc[0, "Session 01"] = True
    with pytest.raises(ValueError, match="real"):
        io.validate_group_summary(frame, "HRV_SDNN")
    with pytest.raises(ValueError, match="real"):
        build._numeric(pd.DataFrame({"ibi": [True, False]}), "ibi")


def test_case_insensitive_directory_alias_cannot_write_into_source(dataset):
    alias = dataset.with_name(dataset.name.upper())
    if not alias.exists() or not os.path.samefile(dataset, alias):
        pytest.skip("filesystem is case-sensitive")
    for destination in (alias, alias / "generated"):
        with pytest.raises(ValueError, match="separate"):
            paths.resolve_output_root(destination, data_root=dataset)


@pytest.mark.parametrize("timestamp", ["2024-10-27T02:30:00", "2024-03-31T02:30:00"])
def test_declared_dst_zone_rejects_ambiguous_or_nonexistent_times_without_calendar_logs(
    timestamp, dataset, tmp_path
):
    with pytest.raises(ValueError, match="declared naive timezone"):
        io.parse_datetime(pd.Series([timestamp]), naive_timezone="Europe/Paris")
    source = dataset / "case-study/processed/hr.csv"
    pd.DataFrame({"datetime": [timestamp], "value": [1]}).to_csv(source, index=False)
    before = snapshot(dataset)
    output = tmp_path / "review"
    result = command(
        "scripts.deidentify_timestamps",
        "--data-root",
        dataset,
        "--naive-timezone",
        "Europe/Paris",
        "--output-root",
        output,
    )
    assert result.returncode == 1 and "declared naive timezone" in result.stderr
    assert "Traceback" not in result.stderr and timestamp not in result.stderr
    assert not output.exists() and snapshot(dataset) == before
