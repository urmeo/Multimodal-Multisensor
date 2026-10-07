#!/usr/bin/env python3
"""Validate portable kernels; update only with --write."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = {"name": "python3", "display_name": "Python 3", "language": "python"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    pending = []
    for folder in ("case-study", "individual", "group"):
        for path in sorted((ROOT / folder).glob("*.ipynb")):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            if notebook.metadata.get("kernelspec") != PORTABLE:
                notebook.metadata["kernelspec"] = dict(PORTABLE)
                notebook.metadata.setdefault("language_info", {})["name"] = "python"
                pending.append((path, notebook))
    if args.write:
        for path, notebook in pending:
            nbformat.write(notebook, path)
    print(
        f"Portable kernels: {len(pending)} changes {'written' if args.write else 'needed'}."
    )
    return 1 if pending and not args.write else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, nbformat.ValidationError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
