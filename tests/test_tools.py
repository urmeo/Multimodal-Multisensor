"""CLI help, current figure drift and prevalidated output regressions."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_figures

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script", ["build_figures", "verify_notebooks", "normalize_kernelspec"]
)
def test_cli_help_and_unknown_argument(script):
    command = [sys.executable, str(ROOT / "scripts" / f"{script}.py")]
    help_result = subprocess.run(command + ["--help"], capture_output=True, text=True)
    invalid = subprocess.run(
        command + ["--unknown-option"], capture_output=True, text=True
    )
    assert help_result.returncode == 0 and "usage:" in help_result.stdout
    assert invalid.returncode == 2 and "unrecognized arguments" in invalid.stderr


@pytest.mark.parametrize(
    "corruption", ["empty hashes", "truncated hashes", "changed metric"]
)
def test_figure_check_rejects_incomplete_or_stale_artifacts(tmp_path, corruption):
    for name in ("figures.csv", "figures.toml", "reliability.png", "correlation.png"):
        shutil.copy2(ROOT / "outputs" / name, tmp_path / name)
    path = tmp_path / (
        "figures.csv" if corruption == "changed metric" else "figures.toml"
    )
    text = path.read_text()
    if corruption == "empty hashes":
        text = text[: text.index("[figure]")] + "[figure]\n"
    elif corruption == "truncated hashes":
        text = "\n".join(
            line
            for line in text.splitlines()
            if not line.startswith('"correlation.png"')
        )
    else:
        lines = text.splitlines()
        fields = lines[1].split(",")
        fields[-1] = "0.99"
        lines[1] = ",".join(fields)
        text = "\n".join(lines)
    path.write_text(text)
    rows, corr, hashes = build_figures.calculate()
    with pytest.raises(ValueError):
        build_figures.verify(tmp_path, rows, corr, hashes)


@pytest.mark.parametrize("alias", ["source hardlink", "contained nested symlink"])
def test_all_figure_destinations_validated_before_first_write(tmp_path, alias):
    if alias == "source hardlink":
        source = ROOT / "data" / "group_results" / "HRV_SDNN.csv"
        before = source.read_bytes()
        os.link(source, tmp_path / "figures.toml")
    else:
        (tmp_path / "nested").mkdir()
        (tmp_path / "reliability.png").symlink_to(
            tmp_path / "nested" / "reliability.png"
        )
    assert build_figures.main(["--output-root", str(tmp_path)]) == 1
    assert not (tmp_path / "correlation.png").exists()
    assert not (tmp_path / "figures.csv").exists()
    if alias == "source hardlink":
        assert source.read_bytes() == before
    else:
        assert not (tmp_path / "nested" / "reliability.png").exists()
