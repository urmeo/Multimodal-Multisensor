#!/usr/bin/env python3
"""Regenerate group summaries from committed per-participant data.

The case-study participant maps to group row **P02**; the original recipe
reproduces its HRV and pupil rows exactly (~1e-13) and its response-duration row
to ~2e-5 (see the MANIFEST reconciliation), and an artifact-filtered recipe is
also emitted. The other 9 participants' raw streams were not released, so their
rows are not regenerable and are never fabricated. Writes to
``data/group_results/reconstructed/`` only — committed data is untouched.
Run: ``python pipeline/build_group_summaries.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Make the repo-root `mms` package importable when run as a plain script
# (`python pipeline/build_group_summaries.py`) without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mms  # noqa: E402

OUT = mms.paths.GROUP_RESULTS / "reconstructed"
SESSIONS = (1, 2, 3)
CASE_STUDY_PARTICIPANT = "P02"  # verified: case-study IBI/pupil/duration == committed P02 row


def original_recipe() -> dict[str, list[float]]:
    """The recipe that reproduces the committed summaries (no artifact filtering)."""
    sdnn, pupil, dur = [], [], []
    for s in SESSIONS:
        ibi = mms.io.load_ibi(s)
        sdnn.append(float(pd.to_numeric(ibi["ibi"], errors="coerce").std(ddof=1)))
        sed = mms.io.load_fixation(s)
        pupil.append(float(pd.to_numeric(sed["pupil"], errors="coerce").std(ddof=1)))
        psy = mms.io.load_psychometric(s)
        dur.append(float(pd.to_numeric(psy["Time(s)"], errors="coerce").std(ddof=1)))
    return {"HRV_SDNN": sdnn, "Pupil_Dilation_STD": pupil,
            "Psychometric_Test_Duration_STD": dur}


def improved_recipe() -> dict[str, list[float]]:
    """Physiologically defensible recipe: NN-filtered HRV, quality-gated pupil."""
    sdnn, pupil, dur = [], [], []
    for s in SESSIONS:
        sdnn.append(mms.hrv.sdnn(mms.io.load_ibi(s)["ibi"]))
        pupil.append(mms.fixation.pupil_std(mms.io.load_fixation(s)))
        psy = mms.io.load_psychometric(s)
        dur.append(float(pd.to_numeric(psy["Time(s)"], errors="coerce").std(ddof=1)))
    return {"HRV_SDNN": sdnn, "Pupil_Dilation_STD": pupil,
            "Psychometric_Test_Duration_STD": dur}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    orig = original_recipe()
    improved = improved_recipe()
    cols = ["Session 01", "Session 02", "Session 03"]

    reconciliation = {}
    for metric, values in orig.items():
        committed = mms.io.load_group_summary(metric)
        committed_row = committed.loc[
            committed["Participant"] == CASE_STUDY_PARTICIPANT, cols
        ].to_numpy(float).ravel()
        assert committed_row.size == len(values), (
            f"expected {len(values)} committed values for {CASE_STUDY_PARTICIPANT} "
            f"in {metric}, got {committed_row.size} (row missing or duplicated)"
        )
        max_abs_err = max(abs(a - b) for a, b in zip(values, committed_row))
        reconciliation[metric] = {
            "committed_P02": [round(x, 6) for x in committed_row],
            "regenerated_original_recipe": [round(x, 6) for x in values],
            "regenerated_improved_recipe": [round(x, 6) for x in improved[metric]],
            "max_abs_error_vs_committed": max_abs_err,
            "exact_match": bool(max_abs_err < 1e-6),
        }

        # write reconstructed rows (original + improved) for the one regenerable participant
        pd.DataFrame(
            [["P02 (original recipe)"] + values,
             ["P02 (improved recipe)"] + improved[metric]],
            columns=["Participant"] + cols,
        ).to_csv(OUT / f"{metric}_P02_reconstructed.csv", index=False)

    # flag the known integrity issue in the committed data (do not "fix" it — we
    # have no source of truth for it)
    hrv = mms.io.load_group_summary("HRV_SDNN")
    dup_flags = []
    for _, row in hrv.iterrows():
        if abs(row["Session 02"] - row["Session 03"]) < 1e-9:
            dup_flags.append({"participant": row["Participant"],
                              "issue": "Session 02 == Session 03 (suspected copy artifact)",
                              "value": float(row["Session 02"])})

    manifest = {
        "regenerable_from_committed_data": [CASE_STUDY_PARTICIPANT],
        "summary_only_not_regenerable": [f"P{i:02d}" for i in range(1, 11)
                                         if f"P{i:02d}" != CASE_STUDY_PARTICIPANT],
        "reason_others_absent": "Raw per-participant streams for P01 and P03-P10 "
                                "were not released (privacy; see DATA_ETHICS.md).",
        "reconciliation": reconciliation,
        "known_integrity_issues": dup_flags,
        "note": "Committed summaries were NOT modified by this script.",
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"Case-study participant maps to {CASE_STUDY_PARTICIPANT}.")
    for metric, rec in reconciliation.items():
        status = "EXACT MATCH" if rec["exact_match"] else f"max err {rec['max_abs_error_vs_committed']:.2e}"
        print(f"  {metric:32s} regenerated vs committed P02: {status}")
    if dup_flags:
        print("\nKnown integrity issues flagged (not modified):")
        for d in dup_flags:
            print(f"  {d['participant']}: {d['issue']} (value {d['value']})")
    print(f"\nWrote reconstructed rows + MANIFEST.json to {OUT.relative_to(mms.paths.ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
