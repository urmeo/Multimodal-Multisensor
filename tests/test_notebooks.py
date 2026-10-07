"""Validate the exact 33 research notebooks without scanning environments."""

import ast
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(
    path
    for folder in ("case-study", "individual", "group")
    for path in (ROOT / folder).glob("*.ipynb")
)


def test_notebooks_present():
    assert len(NOTEBOOKS) == 33
    assert {
        folder: sum(path.parent.name == folder for path in NOTEBOOKS)
        for folder in ("case-study", "individual", "group")
    } == {"case-study": 10, "individual": 14, "group": 9}


@pytest.mark.parametrize(
    "path", NOTEBOOKS, ids=[str(p.relative_to(ROOT)) for p in NOTEBOOKS]
)
def test_notebook_contract(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert notebook.cells
    first_code = next(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "from mms import exploration as ex" in first_code
    for cell in notebook.cells:
        if cell.cell_type == "code":
            tree = ast.parse(cell.source)
            assert cell.execution_count is None
            assert not cell.outputs, (
                "Repaired source must not publish fresh participant-derived outputs"
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in {"to_csv", "savefig"}, (
                        "Notebook writes must use validated scratch helpers"
                    )
            assert "utc=True" not in cell.source
            assert "tz_localize(None)" not in cell.source
    assert "MMS_OUTPUT_ROOT" in notebook.cells[0].source
