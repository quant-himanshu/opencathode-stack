"""
cycle_segmentor.py — Segment Deng fleet data into charge/discharge cycles.

ASSUMPTIONS:
  - A new session begins when time gap > SESSION_GAP_S (30 min)
  - We use charging sessions (I < 0) as degradation cycles because:
      (a) charging data is far more consistent in Deng dataset (all vehicles I <= 0)
      (b) C-rate, DoD, and temperature are well-defined per session
  - SOC provided by BMS is used directly as state indicator (not re-computed from current)
  - A valid cycle requires at least MIN_CYCLE_ROWS rows AND SOC span >= MIN_DOD_PCT
  - Cycle date is set to the START timestamp of the session
  - Q_Ah for the cycle is median of valid available_capacity readings in that session
    (median is more robust to transient sensor errors than mean or first/last)

RETURNED COLUMNS per cycle row:
  vehicle     — string key (V01 etc.)
  cycle_date  — timestamp of session start
  cycle_idx   — 0-based ordinal within vehicle
  n_rows      — number of data rows in session
  dt_total_s  — wall time of session [s]
  soc_start   — SOC at session start [0–100]
  soc_end     — SOC at session end [0–100]
  DoD_pct     — |soc_end - soc_start| [%]
  I_mean_A    — mean current (negative = charging)
  I_rms_A     — RMS current
  C_rate      — |I_mean| / Q_nominal (fraction of C)
  T_mean_C    — mean temperature across session
  T_max_C     — peak temperature
  Q_Ah        — median available_capacity in session (NaN if no valid readings)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from degradation.deng_loader import _Q_NOMINAL

SESSION_GAP_S  = 1800.0   # 30 min — gap that separates two distinct drive/charge sessions
MIN_CYCLE_ROWS = 5        # discard very short sessions (< 40 s of data)
MIN_DOD_PCT    = 2.0      # discard near-zero SOC change (parked with charger fluctuating)


def _summarise_session(
    sess: pd.DataFrame,
    vehicle: str,
    cycle_idx: int,
) -> Optional[Dict]:
    """Return a summary dict for one session, or None if it fails quality checks."""
    if len(sess) < MIN_CYCLE_ROWS:
        return None

    soc_start = float(sess["soc"].iloc[0])
    soc_end   = float(sess["soc"].iloc[-1])
    dod       = abs(soc_end - soc_start)
    if dod < MIN_DOD_PCT:
        return None

    dt_total = float(sess["dt_s"].sum())
    I_vals   = sess["I_A"].values
    T_vals   = sess["T_mean_C"].values

    # Q_Ah: median of valid readings in session
    q_valid  = sess["Q_Ah"].dropna()
    q_median = float(q_valid.median()) if len(q_valid) > 0 else float("nan")

    return {
        "vehicle"    : vehicle,
        "cycle_date" : sess["ts"].iloc[0],
        "cycle_idx"  : cycle_idx,
        "n_rows"     : len(sess),
        "dt_total_s" : dt_total,
        "soc_start"  : soc_start,
        "soc_end"    : soc_end,
        "DoD_pct"    : dod,
        "I_mean_A"   : float(np.mean(I_vals)),
        "I_rms_A"    : float(np.sqrt(np.mean(I_vals ** 2))),
        "C_rate"     : abs(float(np.mean(I_vals))) / _Q_NOMINAL,
        "T_mean_C"   : float(np.mean(T_vals)),
        "T_max_C"    : float(np.max(T_vals)),
        "Q_Ah"       : q_median,
    }


def segment_vehicle(
    df: pd.DataFrame,
    vehicle: str,
    session_gap_s: float = SESSION_GAP_S,
) -> pd.DataFrame:
    """
    Split one vehicle's dataframe into sessions and return a cycles DataFrame.

    Each row of the output is one session summary (see module docstring).
    """
    # Mark session boundaries: wherever dt_s exceeds the gap threshold
    df = df.copy()
    df["session_id"] = (df["dt_s"] > session_gap_s).cumsum()

    records: List[Dict] = []
    cycle_idx = 0
    for _, sess in df.groupby("session_id", sort=True):
        summary = _summarise_session(sess, vehicle, cycle_idx)
        if summary is not None:
            records.append(summary)
            cycle_idx += 1

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def segment_all(
    vehicles: Dict[str, pd.DataFrame],
    session_gap_s: float = SESSION_GAP_S,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Segment all vehicles. Returns a single concatenated cycles DataFrame
    sorted by (vehicle, cycle_date).
    """
    parts: List[pd.DataFrame] = []
    for veh, df in sorted(vehicles.items()):
        cyc = segment_vehicle(df, veh, session_gap_s=session_gap_s)
        if len(cyc) > 0:
            parts.append(cyc)
            if verbose:
                q_valid = cyc["Q_Ah"].dropna()
                print(
                    f"  {veh}: {len(cyc):4d} cycles | "
                    f"DoD mean={cyc['DoD_pct'].mean():.1f}% | "
                    f"Q_Ah: {q_valid.min():.1f}–{q_valid.max():.1f} "
                    f"(NaN={cyc['Q_Ah'].isna().sum()})"
                )

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True).sort_values(
        ["vehicle", "cycle_date"]
    ).reset_index(drop=True)


if __name__ == "__main__":
    from degradation.deng_loader import load_all

    print("=== Cycle Segmentor — Quick Check ===")
    vehicles = load_all(verbose=False)
    cycles = segment_all(vehicles, verbose=True)
    print(f"\nTotal cycles: {len(cycles)}")
    print(f"Vehicles: {cycles['vehicle'].nunique()}")
    print(f"DoD: mean={cycles['DoD_pct'].mean():.1f}%  median={cycles['DoD_pct'].median():.1f}%")
    print(f"C-rate: mean={cycles['C_rate'].mean():.3f}  max={cycles['C_rate'].max():.3f}")
    print(f"Q_Ah non-null: {cycles['Q_Ah'].notna().sum()} / {len(cycles)}")
    print(f"\nFirst 5 cycles (V01):")
    print(cycles[cycles["vehicle"] == "V01"][
        ["cycle_date", "DoD_pct", "C_rate", "T_mean_C", "Q_Ah"]
    ].head().to_string(index=False))
