#!/usr/bin/env python3
"""Execute 33 notebooks on copied inputs and verify source bytes stay unchanged."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("case-study", "group", "individual")


def hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args(argv)
    before = hashes(ROOT / "data")
    with tempfile.TemporaryDirectory(prefix="mms-notebooks-") as name:
        scratch = Path(name)
        shutil.copytree(ROOT / "data", scratch / "data")
        for folder in FOLDERS:
            destination = scratch / folder
            destination.mkdir()
            for source in (ROOT / folder).glob("*.ipynb"):
                shutil.copy2(source, destination / source.name)
        notebooks = sorted(
            path for folder in FOLDERS for path in (scratch / folder).glob("*.ipynb")
        )
        if len(notebooks) != 33:
            raise ValueError("expected all 33 source notebooks")
        environment = os.environ.copy()
        environment.update(
            MMS_DATA_ROOT=str(scratch / "data"),
            MMS_OUTPUT_ROOT=str(scratch / "outputs"),
            IPYTHONDIR=str(scratch / "ipython"),
            JUPYTER_DATA_DIR=str(scratch / "jupyter-data"),
            JUPYTER_CONFIG_DIR=str(scratch / "jupyter-config"),
            JUPYTER_RUNTIME_DIR=str(scratch / "jupyter-runtime"),
            MPLCONFIGDIR=str(scratch / "matplotlib"),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        environment.pop("MMS_SENSOR_TIMEZONE", None)
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--nbmake",
            "--nbmake-kernel=python3",
            "--nbmake-timeout=900",
            "-q",
            *map(str, notebooks),
        ]
        result = subprocess.run(command, cwd=scratch, env=environment, check=False)
        if hashes(ROOT / "data") != before or hashes(scratch / "data") != before:
            raise ValueError("notebook execution changed source inputs")
        if result.returncode:
            return result.returncode
        print("All 33 notebooks executed; original and copied inputs unchanged.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
