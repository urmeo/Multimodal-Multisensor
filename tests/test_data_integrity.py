"""Intentional released CSV inventory and exact known duplicate contracts."""

from pathlib import Path

import pandas as pd
import pytest

from mms import io, paths

ROOT = Path(__file__).resolve().parents[1]
CSV_FILES = tuple(ROOT / "data" / name for name in paths.SOURCE_CSV_RELATIVE)


def test_exact_source_inventory():
    assert len(CSV_FILES) == 51
    assert len(set(CSV_FILES)) == 51
    assert all(path.is_file() for path in CSV_FILES)
    for source in paths.SOURCES:
        for folder in ("processed", "psychometric"):
            found = {
                path.relative_to(ROOT / "data").as_posix()
                for path in (ROOT / "data" / source / folder).glob("*.csv")
            }
            expected = {
                name
                for name in paths.SOURCE_CSV_RELATIVE
                if name.startswith(f"{source}/{folder}/")
            }
            assert found == expected
    assert {path.name for path in (ROOT / "data/group_results").glob("*.csv")} == {
        f"{metric}.csv" for metric in paths.GROUP_METRICS
    }


@pytest.mark.parametrize(
    "path", CSV_FILES, ids=[str(path.relative_to(ROOT)) for path in CSV_FILES]
)
def test_csv_loads_and_is_non_trivial(path):
    assert path.stat().st_size > 0
    frame = pd.read_csv(path)
    assert len(frame) > 0 and len(frame.columns) > 0


@pytest.mark.parametrize("metric", paths.GROUP_METRICS)
def test_group_summary_schema_and_exact_duplicate_cases(metric):
    frame = pd.read_csv(ROOT / "data" / "group_results" / f"{metric}.csv")
    io.validate_group_summary(frame, metric, require_known_duplicates=True)


def test_exact_raw_inventory_and_distinct_sources():
    assert len(paths.SOURCE_RAW_RELATIVE) == 24
    assert all((ROOT / "data" / name).is_file() for name in paths.SOURCE_RAW_RELATIVE)
    for source in paths.SOURCES:
        assert {
            path.relative_to(ROOT / "data").as_posix()
            for path in (ROOT / "data" / source / "raw").glob("*.txt")
        } == {
            name for name in paths.SOURCE_RAW_RELATIVE if name.startswith(f"{source}/")
        }
    for metric in ("hr", "ibi", "sed"):
        for suffix in ("", "_01", "_02", "_03"):
            name = f"raw/{metric}{suffix}.txt"
            assert (ROOT / "data/case-study" / name).read_bytes() != (
                ROOT / "data/individual" / name
            ).read_bytes()
