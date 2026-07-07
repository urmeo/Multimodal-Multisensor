#!/usr/bin/env python3
"""Shift absolute timestamps to relative session time (privacy hardening).

Absolute appointment date + start time is a re-identification vector for an n=10
cohort. This anchors each session at ``2000-01-01T00:00:00``, removing the
wall-clock date/time while preserving relative timings.

- **Dry-run by default**; ``--apply`` writes in place, ``--apply --out DIR`` writes
  the shifted time-bearing files under DIR (not a full copy of ``data/`` — non-time
  files are not copied).
- **Alignment-preserving**: files sharing a (folder, session) shift by one offset,
  so cross-stream joins still line up.
- **Format-preserving**: slash or ISO-8601 output as in the source.

Scope: per-session streams + psychometric originals. Aggregates (``QQ*``) and
``*_modified`` files are left alone. Timestamps are treated as UTC wall-clock.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = pd.Timestamp("2000-01-01T00:00:00")

# column -> ("slash" | "iso") output format
TIME_COLUMNS = {
    "datetime": "slash",
    "Question Start Time": "iso",
    "Question Answer Time": "iso",
    "Answer Time": "iso",  # older psychometric export header
}
SLASH_FMT = "%Y/%m/%d %H:%M:%S.%f"


def session_key(path: Path) -> str:
    """Group files that must share a time origin: (folder, session suffix)."""
    folder = path.relative_to(ROOT).parts[1]  # case-study | individual
    m = re.search(r"_(\d{2})(?:_modified)?\.csv$", path.name)
    sess = m.group(1) if m else "baseline"
    return f"{folder}/{sess}"


def _parse(series: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(series, format=SLASH_FMT if fmt == "slash" else None,
                          utc=(fmt == "iso"), errors="coerce")


def _emit(ts: pd.Series, fmt: str) -> pd.Series:
    """Format shifted timestamps; NaT rows become blank, never 'nan'/'NaT'/'.0'."""
    valid = ts.notna()
    if fmt == "slash":
        s = ts.dt.strftime(SLASH_FMT).str.slice(0, -2)  # trim to .ffff (4 dp)
    else:
        ms = (ts.dt.microsecond.fillna(0) // 1000).astype("int64").map("{:03d}".format)
        s = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.") + ms + "Z"
    return s.where(valid)


def _looks_like_datetime(s: pd.Series) -> bool:
    """True if a string column parses mostly as timestamps (a possible PII leak)."""
    if s.dtype != object:
        return False
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return False
    # bare numbers (e.g. a "Time(s)" duration) parse as epochs — not timestamps
    if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
        return False
    with warnings.catch_warnings():  # heuristic probe; ignore format hints
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    return parsed.notna().mean() > 0.8


def time_columns_of(df: pd.DataFrame) -> dict[str, str]:
    """Map every timestamp column to its format: known columns + auto-detected ones.

    Auto-detection (treated as ISO) catches renamed/mislabeled absolute-timestamp
    columns — e.g. a psychometric file whose 'Time(s)' column holds timestamps —
    so they cannot silently escape de-identification.
    """
    cols = {c: TIME_COLUMNS[c] for c in df.columns if c in TIME_COLUMNS}
    for c in df.columns:
        if c not in cols and _looks_like_datetime(df[c]):
            cols[c] = "iso"
    return cols


def collect() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in sorted(ROOT.glob("data/*/**/*.csv")):
        if "_modified" in path.name or path.name.startswith("QQ"):
            continue
        tcols = time_columns_of(pd.read_csv(path, nrows=200))
        extra = [c for c in tcols if c not in TIME_COLUMNS]
        if extra:
            warnings.warn(
                f"{path.relative_to(ROOT)}: auto-detected timestamp column(s) {extra} "
                "not in TIME_COLUMNS — shifting them as ISO-8601"
            )
        if tcols:
            groups.setdefault(session_key(path), []).append(path)
    return groups


def group_offset(paths: list[Path]) -> pd.Timedelta:
    mins = []
    for p in paths:
        df = pd.read_csv(p)
        for col, fmt in time_columns_of(df).items():
            t = _parse(df[col], fmt)
            if t.notna().any():
                mins.append(t.min().tz_localize(None) if t.dt.tz else t.min())
    return min(mins) - ANCHOR if mins else pd.Timedelta(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--out", type=Path, help="write copies under this dir instead of in place")
    args = ap.parse_args()

    groups = collect()
    total = sum(len(v) for v in groups.values())
    print(f"{'APPLY' if args.apply else 'DRY-RUN'}: {total} files across "
          f"{len(groups)} (folder, session) groups\n")

    for key, paths in groups.items():
        offset = group_offset(paths)
        print(f"[{key}] shift -{offset}  ({len(paths)} files)")
        if not args.apply:
            continue
        for p in paths:
            df = pd.read_csv(p)
            for col, fmt in time_columns_of(df).items():
                shifted = _parse(df[col], fmt)
                shifted = (shifted.dt.tz_localize(None) if fmt == "iso" else shifted) - offset
                df[col] = _emit(shifted, fmt)
            dest = (args.out / p.relative_to(ROOT)) if args.out else p
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dest, index=False)

    if not args.apply:
        print("\nNo files written. Re-run with --apply to shift timestamps in place,")
        print("or --apply --out data_deid/ to write de-identified copies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
