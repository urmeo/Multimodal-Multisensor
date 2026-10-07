#!/usr/bin/env python3
"""Build or verify aggregate figures from unchanged released group tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mms import io, paths, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ["Session 01", "Session 02", "Session 03"]
LABELS = {
    "HRV_SDNN": "Interval-sample SD",
    "Pupil_Dilation_STD": "Pupil SD",
    "Psychometric_Test_Duration_STD": "Response-duration SD",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(data_root: Path | None = None) -> tuple[list[dict], pd.DataFrame, dict]:
    root = paths.resolve_data_root(data_root)
    rows, features, hashes = [], [], {}
    for key, label in LABELS.items():
        frame = io.load_group_summary(key, data_root=root).set_index("Participant")[
            SESSIONS
        ]
        result = stats.icc1(frame.to_numpy(float))
        rows.append(
            {
                "measure": label,
                "icc": result["icc"],
                "lower": result["ci95"][0],
                "upper": result["ci95"][1],
                "n": result["n"],
                "k": result["k"],
            }
        )
        features.append(
            frame.rename(
                columns={name: f"{label} S{i + 1}" for i, name in enumerate(SESSIONS)}
            )
        )
        hashes[key] = digest(root / "group_results" / f"{key}.csv")
    correlation = stats.corr_matrix_fdr(pd.concat(features, axis=1), method="spearman")[
        "r"
    ]
    return rows, correlation, hashes


def records(rows: list[dict], corr: pd.DataFrame) -> list[dict]:
    result = [
        {"metric": "icc", "row": row["measure"], "column": field, "value": row[field]}
        for row in rows
        for field in ("icc", "lower", "upper", "n", "k")
    ]
    result += [
        {"metric": "spearman", "row": a, "column": b, "value": corr.loc[a, b]}
        for a in corr.index
        for b in corr.columns
    ]
    return result


def verify(output: Path, rows: list[dict], corr: pd.DataFrame, hashes: dict) -> None:
    metadata = tomllib.loads((output / "figures.toml").read_text())
    if metadata["source"] != hashes or metadata["method"] != {
        "icc": "ICC(1,1), finite complete rows, 95% F interval",
        "correlation": "Spearman, descriptive, no significance stars",
    }:
        raise ValueError("figure sources or settings differ; regenerate review outputs")
    expected = records(rows, corr)
    with (output / "figures.csv").open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    if len(actual) != len(expected):
        raise ValueError("figure metric inventory differs")
    for saved, current in zip(actual, expected, strict=True):
        if any(saved[key] != current[key] for key in ("metric", "row", "column")):
            raise ValueError("figure metric labels differ")
        if not np.isclose(
            float(saved["value"]),
            float(current["value"]),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError("figure values differ")
    if set(metadata["figure"]) != {"reliability.png", "correlation.png"}:
        raise ValueError("figure hash inventory must contain both current PNGs")
    for name, expected_hash in metadata["figure"].items():
        if (
            name not in {"reliability.png", "correlation.png"}
            or digest(output / name) != expected_hash
        ):
            raise ValueError("figure bytes differ from their recorded hashes")


def build(output: Path, rows: list[dict], corr: pd.DataFrame, hashes: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(12, 5.4), layout="constrained")
    bounds = [
        row[field]
        for row in rows
        for field in ("icc", "lower", "upper")
        if np.isfinite(row[field])
    ]
    minimum = min([-0.25, *bounds]) - 0.04
    for i, row in enumerate(rows):
        point, lower, upper = (row[field] for field in ("icc", "lower", "upper"))
        ax.errorbar(
            point,
            i,
            xerr=[[point - lower], [upper - point]],
            fmt="o",
            color="#2263a6",
            capsize=6,
            markersize=9,
            linewidth=2,
        )
        ax.text(
            1.02,
            i,
            f"{point:.2f}  [{lower:.2f}, {upper:.2f}]",
            va="center",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )
    ax.set(
        yticks=range(3),
        yticklabels=[row["measure"] for row in rows],
        xlim=(minimum, 1),
        ylim=(2.6, -0.6),
        xlabel="ICC(1,1) and 95% interval",
        title="Session consistency: released group summaries\n10 participants · 3 sessions · no diagnostic validation",
    )
    ax.axvline(0, color="#9aa6b2", linestyle=":")
    ax.grid(axis="x", alpha=0.15)
    fig.savefig(output / "reliability.png", dpi=150)
    plt.close(fig)

    short = [
        label.replace("Interval-sample SD", "Interval")
        .replace("Response-duration SD", "Duration")
        .replace("Pupil SD", "Pupil")
        for label in corr.columns
    ]
    fig, ax = plt.subplots(figsize=(12, 7.5), layout="constrained")
    plot = ax.imshow(corr.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r")
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(value) > 0.7 else "#172b42",
            )
    ax.set(
        xticks=range(len(short)),
        yticks=range(len(short)),
        xticklabels=short,
        yticklabels=short,
        title="Spearman correlation: 10 released participant rows",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    fig.colorbar(plot, ax=ax, shrink=0.8, label="Spearman r; descriptive")
    fig.savefig(output / "correlation.png", dpi=150)
    plt.close(fig)
    with (output / "figures.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("metric", "row", "column", "value"), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records(rows, corr))
    metadata = [
        "[method]",
        'icc = "ICC(1,1), finite complete rows, 95% F interval"',
        'correlation = "Spearman, descriptive, no significance stars"',
        "",
        "[source]",
    ]
    metadata += [f'{key} = "{value}"' for key, value in hashes.items()]
    metadata += ["", "[figure]"]
    metadata += [
        f'"{name}" = "{digest(output / name)}"'
        for name in ("reliability.png", "correlation.png")
    ]
    (output / "figures.toml").write_text("\n".join(metadata) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--data-root", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check", action="store_true", help="read-only verification, default"
    )
    group.add_argument(
        "--output-root", type=Path, help="explicit separate output directory"
    )
    args = parser.parse_args(argv)
    try:
        rows, corr, hashes = calculate(args.data_root)
        if args.output_root is not None:
            root = paths.resolve_data_root(args.data_root)
            destinations = [
                paths.output_path(args.output_root, name, data_root=root)
                for name in (
                    "reliability.png",
                    "correlation.png",
                    "figures.csv",
                    "figures.toml",
                )
            ]
            output = paths.resolve_output_root(args.output_root, data_root=root)
            if any(destination.parent != output for destination in destinations):
                raise ValueError(
                    "aggregate figure paths must remain in one flat output directory"
                )
            build(output, rows, corr, hashes)
        else:
            verify(ROOT / "outputs", rows, corr, hashes)
        print(
            "Verified 3 ICC estimates and 81 descriptive correlations; inputs unchanged."
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
