#!/usr/bin/env python3
"""Read-only reconciliation, or explicit output outside the source dataset.

The legacy pooled raw-STD recipe is numerically consistent with group row P02.
That does not independently establish participant identity or NN-beat provenance.
Filtered interval samples are reported separately per channel, not as beat HRV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mms  # noqa: E402

SESSIONS = (1, 2, 3)
CONSISTENT_GROUP_ROW = "P02"
# Absolute tolerance, rtol=0. Duration includes the known rounding discrepancy.
METRIC_TOLERANCES = {
    "HRV_SDNN": 1e-9,
    "Pupil_Dilation_STD": 1e-9,
    "Psychometric_Test_Duration_STD": 3e-5,
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"required column absent: {column}")
    values = mms.hrv._real_series(frame[column], column)
    if len(values) < 2 or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError(f"{column} requires at least two finite source samples")
    return values


def _inputs(data_root: str | Path | None) -> tuple[dict, dict]:
    streams, committed = {}, {}
    for session in SESSIONS:
        ibi = mms.io.load_ibi(session, data_root=data_root)
        eye = mms.io.load_fixation(session, data_root=data_root)
        psych = mms.io.load_psychometric(session, data_root=data_root)
        for frame, columns in (
            (ibi, ("ibi", "iSensor")),
            (eye, ("pupil", "pupilQ", "iSensor")),
            (psych, ("Time(s)",)),
        ):
            for column in columns:
                _numeric(frame, column)
        streams[session] = (ibi, eye, psych)
    for metric in mms.paths.GROUP_METRICS:
        committed[metric] = mms.io.load_group_summary(metric, data_root=data_root)
    return streams, committed


def original_recipe(
    data_root: str | Path | None = None, *, streams: dict | None = None
) -> dict:
    """Historical pooled sample standard deviations, with no artifact filtering."""
    if streams is None:
        streams, _ = _inputs(data_root)
    values: dict[str, list[float]] = {metric: [] for metric in mms.paths.GROUP_METRICS}
    for session in SESSIONS:
        ibi, eye, psych = streams[session]
        for metric, frame, column in (
            ("HRV_SDNN", ibi, "ibi"),
            ("Pupil_Dilation_STD", eye, "pupil"),
            ("Psychometric_Test_Duration_STD", psych, "Time(s)"),
        ):
            values[metric].append(float(_numeric(frame, column).std(ddof=1)))
    return values


def filtered_sample_metrics(streams: dict) -> list[dict]:
    """Descriptive variability per channel; no beat identity is inferred."""
    records = []
    for session, (ibi, eye, _) in streams.items():
        for channel, group in ibi.groupby("iSensor", sort=True):
            kept = mms.hrv.clean_nn(group["ibi"])
            if len(kept) >= 2:
                records.append(
                    {
                        "session": session,
                        "channel": int(channel),
                        "metric": "interval_sample_sd_ms",
                        "n_samples": len(kept),
                        "value": mms.hrv.sdnn(group["ibi"]),
                    }
                )
        for channel, group in eye.groupby("iSensor", sort=True):
            kept = mms.fixation.pupil_samples(group, quality_min=0.5).dropna()
            if len(kept) >= 2:
                records.append(
                    {
                        "session": session,
                        "channel": int(channel),
                        "metric": "pupil_sample_sd_reported_mm",
                        "n_samples": len(kept),
                        "value": mms.fixation.pupil_std(group, quality_min=0.5),
                    }
                )
    return records


def reconcile(original: dict, summaries: dict) -> dict:
    reconciliation = {}
    for metric, tolerance in METRIC_TOLERANCES.items():
        frame = summaries[metric]
        mms.io.validate_group_summary(frame, metric)
        row = (
            frame.loc[
                frame["Participant"] == CONSISTENT_GROUP_ROW,
                list(mms.paths.SESSION_COLUMNS),
            ]
            .to_numpy(float)
            .ravel()
        )
        values = np.asarray(original[metric], dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("reconciliation requires three finite session values")
        maximum = float(np.max(np.abs(values - row)))
        reconciliation[metric] = {
            "committed_P02": row.tolist(),
            "legacy_pooled_sample_std": values.tolist(),
            "max_abs_error": maximum,
            "absolute_tolerance": tolerance,
            "relative_tolerance": 0,
            "within_tolerance": maximum <= tolerance,
        }
    return reconciliation


def _source_hashes(data_root: Path) -> dict:
    names = [
        f"case-study/processed/{kind}_{s:02d}.csv"
        for s in SESSIONS
        for kind in ("ibi", "sed_fix")
    ]
    names += [
        f"case-study/psychometric/Psychometric_Test_Results_{s:02d}.csv"
        for s in SESSIONS
    ]
    names += [f"group_results/{metric}.csv" for metric in mms.paths.GROUP_METRICS]
    return {
        name: hashlib.sha256((data_root / name).read_bytes()).hexdigest()
        for name in names
    }


def run(
    data_root: str | Path | None = None, output_root: str | Path | None = None
) -> bool:
    source = mms.paths.resolve_data_root(data_root)
    streams, committed = _inputs(source)
    original = original_recipe(source, streams=streams)
    reconciliation = reconcile(original, committed)
    matched = all(record["within_tolerance"] for record in reconciliation.values())
    for metric, record in reconciliation.items():
        print(
            f"{metric}: {'within tolerance' if record['within_tolerance'] else 'DRIFT'} "
            f"(absolute tolerance {record['absolute_tolerance']:g})"
        )
    if output_root is None:
        print("Read-only check; no files written.")
        return matched
    if not matched:
        raise ValueError("unexpected reconciliation drift; no output written")
    records = filtered_sample_metrics(streams)
    duplicate_flags: list[dict] = []
    for metric, frame in committed.items():
        within, cross = mms.io.summary_duplicates(frame, metric)
        duplicate_flags.extend(
            {
                "metric": item[0],
                "participant": item[1],
                "sessions": list(item[2]),
                "value": item[3],
                "status": "suspected equal-value issue; source unknown",
            }
            for item in sorted(within)
        )
        duplicate_flags.extend(
            {
                "metric": item[0],
                "session": item[1],
                "participants": list(item[2]),
                "value": item[3],
                "status": "suspected equal-value issue; source unknown",
            }
            for item in sorted(cross)
        )
    manifest = {
        "consistency_match_to_group_row": CONSISTENT_GROUP_ROW,
        "participant_identity_independently_established": False,
        "raw_inventory": {
            "case-study": "12 released streams",
            "individual": "12 distinct released streams; participant mapping unknown",
        },
        "other_group_rows": "No released source-to-participant mapping establishes reconstruction.",
        "reconciliation": reconciliation,
        "known_equal_value_issues": duplicate_flags,
        "source_sha256": _source_hashes(source),
        "settings": {
            "legacy_std_ddof": 1,
            "interval_sample_range_ms": [300, 2000],
            "pupil_quality": "finite pupil>0 and 0.5<=pupilQ<=1",
            "channel_separation": True,
        },
        "limits": "Sample variability is descriptive, not verified NN-beat HRV or anxiety detection.",
        "source_modified": False,
    }
    frames = {
        f"{metric}_P02_consistency.csv": pd.DataFrame(
            [[CONSISTENT_GROUP_ROW, *values]],
            columns=["Participant", *mms.paths.SESSION_COLUMNS],
        )
        for metric, values in original.items()
    }
    frames["filtered_channel_sample_variability.csv"] = pd.DataFrame(records)
    # All source/schema/metric/path validation finishes before mkdir or a write.
    destinations = {
        name: mms.paths.output_path(output_root, name, data_root=source)
        for name in (*frames, "MANIFEST.json")
    }
    serialized = json.dumps(manifest, indent=2, allow_nan=False)
    for name, frame in frames.items():
        destination = destinations[name]
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
    destinations["MANIFEST.json"].write_text(serialized + "\n")
    print(
        f"Wrote {len(destinations)} explicit review outputs; source inputs unchanged."
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="folder containing case-study/individual/group_results",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="read-only reconciliation (default)"
    )
    mode.add_argument(
        "--output-root",
        type=Path,
        help="explicit review output directory outside source data",
    )
    args = parser.parse_args(argv)
    try:
        return 0 if run(args.data_root, args.output_root) else 1
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"Reconciliation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
