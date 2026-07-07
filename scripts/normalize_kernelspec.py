#!/usr/bin/env python3
"""Pin every notebook's kernelspec to the portable ``python3``.

Private kernel names (``pdf_processing``, ``conda-base-py``) break a clean
``git clone`` with ``NoSuchKernel``. Idempotent. Run:
``python scripts/normalize_kernelspec.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = {"name": "python3", "display_name": "Python 3", "language": "python"}


def main() -> int:
    changed = []
    notebooks = [
        p for p in ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts
    ]
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        ks = nb.metadata.get("kernelspec", {})
        if ks.get("name") != "python3" or ks.get("display_name") != "Python 3":
            nb.metadata["kernelspec"] = dict(PORTABLE)
            li = nb.metadata.setdefault("language_info", {})
            li["name"] = "python"
            nbformat.write(nb, path)
            changed.append(path.relative_to(ROOT))

    print(f"Scanned {len(notebooks)} notebooks; normalized {len(changed)}.")
    for c in changed:
        print(f"  fixed: {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
