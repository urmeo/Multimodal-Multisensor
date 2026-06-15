"""Lightweight notebook smoke test (no execution required).

Every committed notebook must be valid nbformat and must not contain saved
error outputs — i.e. it was last run cleanly, not committed in a broken state.
Optional top-to-bottom execution lives in the manual `execute-notebooks` CI job.
"""
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(
    p for p in ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts
)


def test_notebooks_present():
    assert NOTEBOOKS, "No notebooks found in the repository."


@pytest.mark.parametrize(
    "path", NOTEBOOKS, ids=[str(p.relative_to(ROOT)) for p in NOTEBOOKS]
)
def test_notebook_is_valid(path):
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize(
    "path", NOTEBOOKS, ids=[str(p.relative_to(ROOT)) for p in NOTEBOOKS]
)
def test_notebook_has_no_error_outputs(path):
    nb = nbformat.read(path, as_version=4)
    bad = [
        (i, out.get("ename"))
        for i, cell in enumerate(nb.cells)
        if cell.get("cell_type") == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not bad, f"{path.name} was committed with error outputs: {bad}"
