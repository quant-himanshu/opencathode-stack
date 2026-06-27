"""
deng_loader.py — Load and clean Deng BAIC EU500 fleet CSVs.

ASSUMPTIONS:
  - record_time is integer YYYYMMDDHHMMSS
  - charge_current (A) < 0 means charging, > 0 means discharging (verified from V01)
  - min_cell_voltage == 0 indicates sensor dropout (filter out)
  - max current magnitude capped at 250 A (V06/V17 have -400 A outliers — hardware fault)
  - Vehicles are labeled V01..V20 corresponding to filenames #1.csv..#20.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "deng20"

_REQUIRED = [
    "record_time",
    "soc",
    "pack_voltage (V)",
    "charge_current (A)",
    "max_cell_voltage (V)",
    "min_cell_voltage (V)",
    "max_temperature (℃)",
    "min_temperature (℃)",
    "available_capacity (Ah)",
]

_I_MAX_ABS   = 250.0   # A  — clip threshold for current outliers (V06/V17 fault)
_Q_NOMINAL   = 136.2   # Ah — nameplate capacity from data (max observed available_capacity)
_Q_MIN_VALID = 10.0    # Ah — below this, likely sensor error (filter as missing)
_V_CELL_MIN  = 2.5     # V  — sensor dropout guard

_CHEMISTRY_THRESHOLDS = {
    # NMC: sloped 3.6–4.2 V; LFP: flat plateau 3.2–3.3 V
    "LFP_max": 3.45,   # if mean cell voltage < this → LFP
    "NMC_min": 3.55,   # if mean cell voltage > this → NMC
}


def detect_chemistry(df: pd.DataFrame) -> str:
    """
    Infer cell chemistry from cell voltage range.

    LFP: flat plateau 3.2–3.35 V (mean ~3.28 V)
    NMC: sloped 3.6–4.2 V (mean >3.55 V)

    Returns 'NMC' or 'LFP'.
    """
    valid = df[df["max_cell_voltage (V)"] > _V_CELL_MIN]["max_cell_voltage (V)"]
    if len(valid) == 0:
        return "UNKNOWN"
    v_mean = float(valid.mean())
    if v_mean < _CHEMISTRY_THRESHOLDS["LFP_max"]:
        return "LFP"
    if v_mean > _CHEMISTRY_THRESHOLDS["NMC_min"]:
        return "NMC"
    return "UNKNOWN"


def load_vehicle(
    vehicle_idx: int,
    data_dir: Optional[Path] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load and clean one Deng vehicle CSV.

    Args:
        vehicle_idx: 1-based (1 = #1.csv = V01).
        data_dir   : Override default data path.
        verbose    : Print cleaning summary.

    Returns:
        Cleaned DataFrame sorted by timestamp, with added columns:
          ts          — datetime64
          dt_s        — seconds since previous row (0 at session start)
          I_A         — clipped current (negative=charging, positive=discharging)
          T_mean_C    — mean of max/min temperature
          Q_Ah        — available_capacity, NaN where sensor invalid
          chemistry   — string, same for every row
    """
    ddir = data_dir or DATA_DIR
    csv_path = ddir / f"#{vehicle_idx}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    n_raw = len(df)

    # ── parse timestamp ───────────────────────────────────────────────
    df["ts"] = pd.to_datetime(
        df["record_time"].astype(str).str.strip(),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    df = df.dropna(subset=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    # ── time delta ────────────────────────────────────────────────────
    df["dt_s"] = df["ts"].diff().dt.total_seconds().fillna(0).clip(lower=0)

    # ── sensor dropout filter: zero cell voltage rows ─────────────────
    v_dropout = df["min_cell_voltage (V)"] <= _V_CELL_MIN
    df = df[~v_dropout].reset_index(drop=True)
    n_after_v = len(df)

    # ── current clipping (outlier hardware faults) ────────────────────
    df["I_A"] = df["charge_current (A)"].clip(-_I_MAX_ABS, _I_MAX_ABS)

    # ── capacity validation (sensor errors show Q≈0 during normal use) ─
    df["Q_Ah"] = df["available_capacity (Ah)"].where(
        df["available_capacity (Ah)"] >= _Q_MIN_VALID, other=np.nan
    )

    # ── temperature proxy ─────────────────────────────────────────────
    df["T_mean_C"] = (df["max_temperature (℃)"] + df["min_temperature (℃)"]) / 2.0

    # ── chemistry detection ───────────────────────────────────────────
    chem = detect_chemistry(df)
    df["chemistry"] = chem

    if verbose:
        n_clipped = int(np.sum(np.abs(df["charge_current (A)"]) > _I_MAX_ABS))
        n_q_nan   = int(df["Q_Ah"].isna().sum())
        print(
            f"  V{vehicle_idx:02d}: {n_raw} raw → {n_after_v} after dropout filter "
            f"| I clipped: {n_clipped} | Q NaN: {n_q_nan} | chemistry: {chem}"
        )

    return df[
        ["ts", "dt_s", "soc", "pack_voltage (V)", "I_A",
         "max_cell_voltage (V)", "min_cell_voltage (V)",
         "max_temperature (℃)", "min_temperature (℃)", "T_mean_C",
         "Q_Ah", "chemistry"]
    ]


def load_all(
    data_dir: Optional[Path] = None,
    n_vehicles: int = 20,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load all vehicles. Returns dict keyed by 'V01'..'V20'.
    Prints chemistry detection result for each vehicle.
    """
    ddir = data_dir or DATA_DIR
    out: Dict[str, pd.DataFrame] = {}
    for idx in range(1, n_vehicles + 1):
        key = f"V{idx:02d}"
        try:
            out[key] = load_vehicle(idx, data_dir=ddir, verbose=verbose)
        except FileNotFoundError:
            if verbose:
                print(f"  {key}: file not found, skipping")
    return out


if __name__ == "__main__":
    print("=== Deng Loader — Quick Check ===")
    vehicles = load_all(verbose=True)
    print(f"\nLoaded {len(vehicles)} vehicles")

    # Chemistry summary
    chems = {k: v["chemistry"].iloc[0] for k, v in vehicles.items()}
    nmc_count = sum(1 for c in chems.values() if c == "NMC")
    lfp_count = sum(1 for c in chems.values() if c == "LFP")
    print(f"Chemistry: NMC={nmc_count}  LFP={lfp_count}  OTHER={len(vehicles)-nmc_count-lfp_count}")

    v01 = vehicles.get("V01")
    if v01 is not None:
        print(f"\nV01: {len(v01)} rows  |  date range: {v01['ts'].min()} → {v01['ts'].max()}")
        print(f"     Q_Ah: min={v01['Q_Ah'].min():.1f}  max={v01['Q_Ah'].max():.1f}  "
              f"NaN={v01['Q_Ah'].isna().sum()}")
        print(f"     I_A:  min={v01['I_A'].min():.1f}  max={v01['I_A'].max():.1f}")
        print(f"     T:    min={v01['T_mean_C'].min():.1f}  max={v01['T_mean_C'].max():.1f} °C")
