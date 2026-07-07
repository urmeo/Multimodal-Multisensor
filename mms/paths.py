"""Repository paths, resolved from this file so any working directory works."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CASE_STUDY = DATA / "case-study"
INDIVIDUAL = DATA / "individual"
GROUP_RESULTS = DATA / "group_results"
