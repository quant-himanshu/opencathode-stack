#!/usr/bin/env python3
"""
scripts/check_datasets.py — Dataset presence and health check.

Prints file counts, total size, detected formats, and READY/MISSING
for each dataset expected by the OpenCATHODE validation harness.

Usage:
    python scripts/check_datasets.py

Exit codes:
    0  all datasets READY
    1  one or more datasets MISSING or INCOMPLETE
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _collect_files(directory: Path, patterns: List[str]) -> List[Path]:
    """Glob for files matching any of the given patterns under directory."""
    found: List[Path] = []
    for pat in patterns:
        found.extend(directory.rglob(pat))
    return sorted(set(found))


def _total_bytes(files: List[Path]) -> int:
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def _format_extensions(files: List[Path]) -> str:
    exts: Dict[str, int] = {}
    for f in files:
        ext = f.suffix.lower() or "(no ext)"
        exts[ext] = exts.get(ext, 0) + 1
    return ", ".join(f"{ext}×{n}" for ext, n in sorted(exts.items()))


class DatasetSpec(NamedTuple):
    name: str
    directory: Path
    required_patterns: List[str]       # at least one file must match each
    any_patterns: List[str]            # used for size/format inventory
    min_required_files: int            # minimum file count to call READY
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Dataset specifications
# ─────────────────────────────────────────────────────────────────────────────

DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        name="VED",
        directory=_ROOT / "data" / "ved",
        required_patterns=[
            # Static metadata (XLSX from git repo — two files for ICE+HEV and PHEV+EV)
            "VED_Static_Data_PHEV&EV.xlsx",
            # Dynamic data (7z archives OR extracted CSVs)
            "VED_DynamicData_Part*.7z",
        ],
        any_patterns=["*.csv", "*.xlsx", "*.7z", "*.xz", "*.zip"],
        min_required_files=2,
        notes=(
            "Static: VED_Static_Data_PHEV&EV.xlsx (in Data/ subdir). "
            "Dynamic: VED_DynamicData_Part{1,2}.7z — extract with: "
            "  7z x 'Data/VED_DynamicData_Part1.7z' -o data/ved/ "
            "  7z x 'Data/VED_DynamicData_Part2.7z' -o data/ved/ "
            "Extracted files are weekly CSVs named VED_mmddyy_week.csv."
        ),
    ),
    DatasetSpec(
        name="BMW_i3",
        directory=_ROOT / "data" / "bmw_i3",
        required_patterns=["*.csv"],
        any_patterns=["*.csv", "*.xlsx", "*.zip"],
        min_required_files=1,
        notes=(
            "Download from: https://ieee-dataport.org/open-access/real-driving-cycles-bmw-i3 "
            "(Lüth et al. 2020, doi:10.21227/4p3e-e843). "
            "Extract CSVs to data/bmw_i3/."
        ),
    ),
    DatasetSpec(
        name="Renault_Zoe",
        directory=_ROOT / "data" / "renault",
        required_patterns=["*.csv"],
        any_patterns=["*.csv", "*.xlsx", "*.zip"],
        min_required_files=1,
        notes=(
            "Download from: https://ieee-dataport.org/documents/renault-zoe-kangoo "
            "or search IEEE DataPort for 'Renault Zoe CAN'. "
            "Extract CSVs to data/renault/."
        ),
    ),
    DatasetSpec(
        name="Deng20_BAIC",
        directory=_ROOT / "data" / "deng20",
        required_patterns=["*.csv"],
        any_patterns=["*.csv", "*.xlsx", "*.zip", "*.mat"],
        min_required_files=5,
        notes=(
            "Mendeley Data: https://data.mendeley.com/datasets/nsc7xybrr6/2 "
            "(doi:10.17632/nsc7xybrr6.2). "
            "Mirror: IEEE DataPort doi:10.21227/j60n-4t77. "
            "Extract to data/deng20/ — expected layout: "
            "  vehicle_01/session_YYYY-MM-DD.csv  (or flat vehicle_01_session*.csv). "
            "Paper: Deng et al. (2022) Applied Energy 322:119513."
        ),
    ),
    DatasetSpec(
        name="EV300_NatureComms",
        directory=_ROOT / "data" / "ev300",
        required_patterns=["*.csv"],
        any_patterns=["*.csv", "*.xlsx", "*.zip", "*.mat", "*.h5", "*.hdf5"],
        min_required_files=10,
        notes=(
            "300-EV fleet dataset (Liu et al. 2025, Nature Comms doi:10.1038/s41467-025-56485-7). "
            "Raw data server: http://ivstskl.changan.com.cn/?p=2697 (Chang'an Automobile; "
            "requires institutional/registration access — direct download not available publicly). "
            "Code + processed source data (~1.3 MB): "
            "  https://github.com/HoraceLiu1010/Multi-modal-SOH-estimation-framework "
            "This dataset enables CellMode.PER_CELL (individual cell voltages available). "
            "Extract data files to data/ev300/."
        ),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset check
# ─────────────────────────────────────────────────────────────────────────────

def _check_required_pattern(directory: Path, pattern: str) -> bool:
    """True if at least one file in directory matches the glob pattern."""
    return len(list(directory.rglob(pattern))) > 0


def check_dataset(spec: DatasetSpec) -> bool:
    """Print status for one dataset. Returns True if READY."""
    print(f"\n{'─'*60}")
    print(f"  {spec.name}")
    print(f"{'─'*60}")

    if not spec.directory.exists():
        print(f"  STATUS : MISSING  (directory does not exist: {spec.directory})")
        print(f"  Action : {spec.notes}")
        return False

    all_files = _collect_files(spec.directory, spec.any_patterns)
    n_files   = len(all_files)
    total_sz  = _total_bytes(all_files)
    fmts      = _format_extensions(all_files) if all_files else "(none)"

    print(f"  Dir    : {spec.directory}")
    print(f"  Files  : {n_files}  ({_human_size(total_sz)})")
    print(f"  Formats: {fmts}")

    # Check each required pattern
    missing_patterns = []
    for pat in spec.required_patterns:
        if not _check_required_pattern(spec.directory, pat):
            missing_patterns.append(pat)

    if n_files < spec.min_required_files or missing_patterns:
        print(f"  STATUS : INCOMPLETE")
        if missing_patterns:
            print(f"  Missing: {missing_patterns}")
        if n_files < spec.min_required_files:
            print(f"  Need   : at least {spec.min_required_files} files, found {n_files}")
        print(f"  Action : {spec.notes}")
        return False

    # Peek at first CSV column names
    csvs = [f for f in all_files if f.suffix.lower() == ".csv"]
    if csvs:
        try:
            import pandas as pd
            sample = None
            for enc, sep in [("utf-8", ","), ("latin-1", ";"), ("latin-1", ",")]:
                try:
                    sample = pd.read_csv(csvs[0], nrows=0, encoding=enc, sep=sep)
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            if sample is not None:
                cols = list(sample.columns)
                preview = cols[:8]
                print(f"  Cols   : {preview}{'…' if len(cols) > 8 else ''}")
        except Exception as e:
            print(f"  Cols   : (could not peek: {e})")

    # Peek at first 7z/xlsx if no CSV yet
    if not csvs:
        sevenz = [f for f in all_files if f.suffix.lower() in (".7z", ".xz", ".zip")]
        xlsx   = [f for f in all_files if f.suffix.lower() in (".xlsx", ".xls")]
        if sevenz:
            print(f"  Note   : Dynamic data is compressed ({sevenz[0].name}). "
                  "Run extraction step before validate_generic.py.")
        if xlsx:
            try:
                import pandas as pd
                xl = pd.ExcelFile(xlsx[0])
                print(f"  XLSX   : {xlsx[0].name}  sheets={xl.sheet_names}")
            except Exception:
                pass

    print(f"  STATUS : READY")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("  OPENCATHODE — DATASET PRESENCE CHECK")
    print("=" * 60)

    results: Dict[str, bool] = {}
    for spec in DATASETS:
        results[spec.name] = check_dataset(spec)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for name, ok in results.items():
        status = "READY    " if ok else "MISSING/INCOMPLETE"
        print(f"  {name:25s}  {status}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  All datasets READY.  Run: python data/validate_generic.py --all")
    else:
        print("  One or more datasets not ready.  See Action lines above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
