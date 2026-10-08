#!/usr/bin/env python3
"""Calendar-shift selected CSV copies with complete validation before writing.

This covers the exact released CSV inventory, including QQ, modified and legacy
exports. Raw TXT, notebooks, reports, screenshots and photographs are unchanged.
Each file/clock-awareness domain gets its own origin; no cross-stream alignment
or anonymity is established. No source file is ever overwritten.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mms import io, paths  # noqa: E402

ANCHOR = pd.Timestamp("2000-01-01T00:00:00")
TIME_COLUMNS = {
    "datetime",
    "Question Start Time",
    "Question Answer Time",
    "Answer Time",
    "Start Time",
    "End Time",
}
CALENDAR = re.compile(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]|$)")


def _looks_like_datetime(series: pd.Series) -> bool:
    """Inspect all string values, including pandas StringDtype, without parsing loss."""
    if not (pd.api.types.is_string_dtype(series.dtype) or series.dtype == object):
        return False
    return any(CALENDAR.match(str(value)) for value in series.dropna())


def time_columns_of(frame: pd.DataFrame) -> dict[str, str]:
    columns = {}
    for name in frame:
        if name in TIME_COLUMNS or _looks_like_datetime(frame[name]):
            values = frame[name].dropna().astype(str)
            columns[name] = (
                "slash"
                if not values.empty and values.str.match(r"^\d{4}/").all()
                else "iso"
            )
    return columns


def collect(data_root: str | Path | None = None) -> tuple[Path, ...]:
    """Exact 51 source CSVs; generated files are outside this policy."""
    root = paths.resolve_data_root(data_root)
    files = tuple(root / name for name in paths.SOURCE_CSV_RELATIVE)
    missing = [str(path.relative_to(root)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source CSV inventory is incomplete: {missing}")
    return files


def shift_frame(
    frame: pd.DataFrame, *, naive_timezone: str | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Shift each compatible clock domain within one file, retaining aware times.

    Missing source cells remain missing; any malformed nonempty timestamp fails.
    An explicit naive timezone is a caller's declared mapping, never a guess.
    """
    columns = time_columns_of(frame)
    parsed = {
        name: io.parse_datetime(
            frame[name], naive_timezone=naive_timezone, allow_missing=True
        )
        for name in columns
    }
    domains: dict[bool, list[str]] = {}
    for name, stamps in parsed.items():
        domains.setdefault(stamps.dt.tz is not None, []).append(name)
    out = frame.copy(deep=True)
    for aware, names in domains.items():
        minima = [parsed[name].min() for name in names if parsed[name].notna().any()]
        if not minima:
            continue
        origin = min(minima)
        anchor = ANCHOR.tz_localize(origin.tzinfo) if aware else ANCHOR
        offset = origin - anchor
        for name in names:
            stamps = parsed[name] - offset
            fmt = columns[name]
            if fmt == "slash" and stamps.dt.tz is None:
                out[name] = stamps.dt.strftime("%Y/%m/%d %H:%M:%S.%f").where(
                    stamps.notna()
                )
            else:
                out[name] = stamps.map(
                    lambda stamp: stamp.isoformat() if not pd.isna(stamp) else None
                )
    return out, list(columns)


def run(
    data_root: str | Path | None = None,
    *,
    output_root: str | Path | None = None,
    naive_timezone: str | None = None,
) -> int:
    root = paths.resolve_data_root(data_root)
    prepared = []
    for source in collect(root):
        shifted, columns = shift_frame(
            pd.read_csv(source), naive_timezone=naive_timezone
        )
        if columns:
            prepared.append((source.relative_to(root), shifted, columns))
    destinations = (
        [
            paths.output_path(output_root, relative, data_root=root)
            for relative, _, _ in prepared
        ]
        if output_root is not None
        else []
    )
    print(f"Validated 51 source CSVs; {len(prepared)} files contain calendar columns.")
    for relative, _, columns in prepared:
        print(f"{relative}: {len(columns)} calendar columns")
    print(
        "Scope: calendar-bearing CSV copies only. Raw TXT, non-time CSVs, notebooks, "
        "reports, screenshots and photos are not copied or shifted."
    )
    print(
        "Origins are file-local and awareness-specific; no cross-stream synchronization "
        "or anonymity is established. Sensitive responses/measurements remain."
    )
    if output_root is None:
        print("Read-only validation; no files written.")
        return 0
    for (_, shifted, _), destination in zip(prepared, destinations, strict=True):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shifted.to_csv(destination, index=False)
    print(f"Wrote {len(destinations)} explicit CSV copies; source inputs unchanged.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="folder containing case-study/individual/group_results",
    )
    parser.add_argument(
        "--naive-timezone", help="explicit documented timezone for naive source clocks"
    )
    parser.add_argument(
        "--apply", action="store_true", help="write copies only with --output-root"
    )
    parser.add_argument(
        "--output-root", "--out", type=Path, help="separate explicit output directory"
    )
    args = parser.parse_args(argv)
    if args.apply and args.output_root is None:
        parser.error("--apply requires --output-root; in-place writes are unsupported")
    try:
        return run(
            args.data_root,
            output_root=args.output_root,
            naive_timezone=args.naive_timezone,
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        print(f"Calendar shift failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
