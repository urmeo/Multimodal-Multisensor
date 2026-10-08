"""Checkout defaults and explicit dataset/output roots."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CASE_STUDY = DATA / "case-study"
INDIVIDUAL = DATA / "individual"
GROUP_RESULTS = DATA / "group_results"

SOURCES = ("case-study", "individual")
GROUP_METRICS = ("HRV_SDNN", "Pupil_Dilation_STD", "Psychometric_Test_Duration_STD")
SESSION_COLUMNS = ("Session 01", "Session 02", "Session 03")
PARTICIPANTS = tuple(f"P{i:02d}" for i in range(1, 11))
LEGACY_OUTPUT_NAMES = (
    "legacy-spearman-heatmap.png",
    "Group Duration.png",
    "Group Eye.png",
    "Group SD.png",
    "Standard Deviation of HRV (SDNN).png",
    "Standard Deviation of Pupil Dilation.png",
    "avg_answer_duration.png",
    "correlation_heatmap_with_values_final.png",
    "fixation_duration_by_session.png",
    "pca_kmeans_clusters.png",
    "silhouette_score.png",
)

SOURCE_CSV_RELATIVE = tuple(
    sorted(
        [
            f"case-study/processed/{metric}{suffix}.csv"
            for metric in ("hr", "hrv", "ibi", "sed", "sed_fix")
            for suffix in ("", "_01", "_02", "_03")
        ]
        + [
            f"case-study/psychometric/Psychometric_Test_Results_{s:02d}{suffix}.csv"
            for s in (1, 2, 3)
            for suffix in ("", "_modified")
        ]
        + [
            f"individual/processed/{metric}{suffix}.csv"
            for metric in ("hr", "ibi", "sed")
            for suffix in ("", "_01", "_02", "_03")
        ]
        + [f"individual/processed/sed_fix_{s:02d}.csv" for s in (1, 2, 3)]
        + [f"individual/processed/{name}.csv" for name in ("QQ", "QQ2", "QQHRV")]
        + [
            f"individual/psychometric/Psychometric_Test_Results_{s:02d}.csv"
            for s in (0, 1, 2, 3)
        ]
        + [f"group_results/{metric}.csv" for metric in GROUP_METRICS]
    )
)
SOURCE_RAW_RELATIVE = tuple(
    f"{source}/raw/{metric}{suffix}.txt"
    for source in SOURCES
    for metric in ("hr", "ibi", "sed")
    for suffix in ("", "_01", "_02", "_03")
)


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    try:
        return path.resolve()
    except RuntimeError as error:
        raise ValueError("cannot resolve path: cyclic symlink") from error


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Folder containing case-study, individual and group_results.

    Wheels contain code only. A caller must supply the separately obtained data.
    """
    root = _resolve(DATA if data_root is None else Path(data_root))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory is absent: {root}. Wheels contain code only; "
            "supply data_root=PATH containing case-study/individual/group_results."
        )
    return root


def source_files(data_root: str | Path | None = None) -> tuple[Path, ...]:
    root = resolve_data_root(data_root)
    return tuple(root / name for name in SOURCE_CSV_RELATIVE + SOURCE_RAW_RELATIVE)


def _dataset_roots(data_root: str | Path | None) -> tuple[Path, ...]:
    selected = resolve_data_root(data_root)
    canonical = _resolve(DATA)
    return (
        (selected, canonical)
        if canonical.is_dir() and canonical != selected
        else (selected,)
    )


def _preserved_artifacts() -> tuple[Path, ...]:
    files = [
        ROOT / "LICENSE",
        *(ROOT / "outputs" / name for name in LEGACY_OUTPUT_NAMES),
    ]
    for directory in (ROOT / "docs/reports", ROOT / "images"):
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    return tuple(path for path in files if path.is_file())


def _contains(ancestor: Path, child: Path) -> bool:
    if ancestor == child or ancestor in child.parents:
        return True
    if not ancestor.exists():
        return False
    return any(
        path.exists() and os.path.samefile(ancestor, path)
        for path in (child, *child.parents)
    )


def resolve_output_root(
    output_root: str | Path, *, data_root: str | Path | None = None
) -> Path:
    """Validate a separate output directory without creating it."""
    sources = _dataset_roots(data_root)
    output = _resolve(Path(output_root))
    if any(
        _contains(source, output) or _contains(output, source) for source in sources
    ):
        raise ValueError(
            "output_root must be separate from the source dataset directory"
        )
    if output.exists() and not output.is_dir():
        raise ValueError("output_root must be a directory")
    return output


def output_path(
    output_root: str | Path,
    relative: str | Path,
    *,
    data_root: str | Path | None = None,
) -> Path:
    """Reject traversal, symlink and hard-link aliases before writing."""
    root = resolve_output_root(output_root, data_root=data_root)
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("output paths must be relative and contained in output_root")
    destination = _resolve(root / relative)
    if destination != root and root not in destination.parents:
        raise ValueError("output path escapes output_root")
    if destination.exists() and not destination.is_file():
        raise ValueError("output destination must be a file")
    sources = (
        tuple(
            path
            for source in _dataset_roots(data_root)
            for path in source.rglob("*")
            if path.is_file()
        )
        + _preserved_artifacts()
        if destination.exists()
        else ()
    )
    for source in sources:
        if destination == _resolve(source) or (
            destination.exists()
            and source.exists()
            and os.path.samefile(destination, source)
        ):
            raise ValueError("output path aliases a source observation")
    return destination
