"""
fatigue.py — Rainflow cycle counting + Palmgren-Miner damage accumulation.

APPROACH:
  1. SOC timeseries within a session → rainflow cycle counting (ASTM E1049)
     via the `rainflow` package (established, ASTM-compliant)
  2. Per extracted half-cycle: compute stress amplitude using stress_model
  3. S-N Basquin power law: N_f(Δσ) = A × Δσ^(-m)
     where A, m are calibrated per-vehicle from training set
  4. Palmgren-Miner linear damage: D = Σ (n_i / N_f(Δσ_i))
  5. Cumulative damage across all sessions → D_total(cycle_number)

ASSUMPTIONS:
  - Linear damage accumulation (Miner's rule) — ignores sequence effects
  - Each rainflow half-cycle counted as n=0.5 (full cycle = n=1)
  - Basquin S-N defaults: A=1e6, m=2.5  (NMC literature range: m=1.5–3.5)
    These are nominal; calibration in soh_predictor.py adjusts them.
  - Minimum stress amplitude for cycle inclusion: 0.005 (filters rest noise)
  - Rainflow is applied to SOC timeseries within each session because
    SOC represents the mechanical strain state of electrode particles
"""

from __future__ import annotations

from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import rainflow  # ASTM E1049 compliant

from degradation.stress_model import compute_stress

# ── S-N defaults ──────────────────────────────────────────────────────────────
SN_A_DEFAULT = 1.0e6   # Basquin coefficient (cycles to failure at unit stress)
SN_M_DEFAULT = 2.5     # Basquin exponent (higher m = stress-sensitive material)
MIN_STRESS   = 0.005   # minimum stress amplitude to count as a fatigue cycle


def n_cycles_to_failure(stress_amp: float | np.ndarray, A: float, m: float) -> np.ndarray:
    """
    Basquin power law: N_f = A × Δσ^(-m).

    Args:
        stress_amp : Dimensionless stress amplitude (from stress_model.compute_stress)
        A, m       : Fitted S-N parameters
    Returns:
        N_f        : Cycles to failure (clamped ≥ 1 to avoid division by zero)
    """
    s = np.maximum(np.asarray(stress_amp, dtype=float), 1e-10)
    return np.maximum(A * s ** (-m), 1.0)


def rainflow_damage(
    soc_series: np.ndarray,
    T_series: np.ndarray,
    c_rate: float,
    A: float = SN_A_DEFAULT,
    m: float = SN_M_DEFAULT,
) -> float:
    """
    Compute Palmgren-Miner damage for one session's SOC timeseries.

    Args:
        soc_series : SOC values [0–100] for this session
        T_series   : Temperature [°C] aligned with soc_series
        c_rate     : Session-level C-rate (used to scale stress)
        A, m       : S-N parameters
    Returns:
        damage     : Miner's accumulated damage (0 = no damage, 1 = failure)
    """
    soc = np.asarray(soc_series, dtype=float)
    if len(soc) < 3:
        return 0.0

    T_mean = float(np.mean(T_series))

    damage = 0.0
    try:
        for rng, mean, count, i_start, i_end in rainflow.extract_cycles(soc):
            # stress_model expects DoD in % and positive c_rate
            stress_amp = float(compute_stress(
                dod_pct  = abs(rng),   # rng is already the SOC range [%]
                c_rate   = abs(c_rate),
                T_mean_C = T_mean,
            ))
            if stress_amp < MIN_STRESS:
                continue
            nf = float(n_cycles_to_failure(stress_amp, A, m))
            damage += count / nf  # count is 0.5 or 1.0 per ASTM convention
    except Exception:
        # rainflow can raise on edge cases (constant series etc.)
        pass

    return damage


def accumulate_damage(
    cycles: pd.DataFrame,
    raw_vehicles: Dict[str, pd.DataFrame],
    A: float = SN_A_DEFAULT,
    m: float = SN_M_DEFAULT,
    session_gap_s: float = 1800.0,
) -> pd.DataFrame:
    """
    Compute per-cycle cumulative damage for a cycles DataFrame.

    For each session we apply rainflow to the raw SOC timeseries
    (which is already available in raw_vehicles), then accumulate D.

    Args:
        cycles       : Output of segment_all() (one row per cycle)
        raw_vehicles : Dict of raw DataFrames from deng_loader.load_all()
        A, m         : S-N parameters
        session_gap_s: Must match the gap used in segment_vehicle()

    Returns:
        cycles DataFrame with added columns:
          d_cycle     — Miner's damage increment for this cycle
          D_cumul     — cumulative damage up to and including this cycle
    """
    all_parts: List[pd.DataFrame] = []

    for veh, grp in cycles.groupby("vehicle", sort=True):
        raw = raw_vehicles.get(veh)
        if raw is None:
            grp = grp.copy()
            grp["d_cycle"] = np.nan
            grp["D_cumul"] = np.nan
            all_parts.append(grp)
            continue

        raw = raw.copy()
        raw["session_id"] = (raw["dt_s"] > session_gap_s).cumsum()

        # Map session_id to sorted cycle_idx
        session_ids = raw.groupby("session_id").first().reset_index()
        # We need to match each cycle row to a raw session by its order
        # (cycle_idx is the 0-based ordinal within the vehicle after filtering)
        # Build a map: cycle_idx → matching raw session df
        valid_sessions: List[pd.DataFrame] = []
        for _, sess in raw.groupby("session_id", sort=True):
            if len(sess) >= 5:
                soc_start = float(sess["soc"].iloc[0])
                soc_end   = float(sess["soc"].iloc[-1])
                if abs(soc_end - soc_start) >= 2.0:
                    valid_sessions.append(sess)

        grp = grp.copy().reset_index(drop=True)
        d_cycle = np.zeros(len(grp))

        for i, (_, row) in enumerate(grp.iterrows()):
            cidx = int(row["cycle_idx"])
            if cidx < len(valid_sessions):
                sess = valid_sessions[cidx]
                soc_arr = sess["soc"].values
                T_arr   = sess["T_mean_C"].values
                cr      = abs(float(row["C_rate"]))
                d_cycle[i] = rainflow_damage(soc_arr, T_arr, cr, A=A, m=m)

        grp["d_cycle"] = d_cycle
        grp["D_cumul"] = np.cumsum(d_cycle)
        all_parts.append(grp)

    if not all_parts:
        return cycles.copy()

    return pd.concat(all_parts, ignore_index=True)


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, ".")
    from degradation.deng_loader import load_all
    from degradation.cycle_segmentor import segment_all

    print("=== Fatigue Module — Quick Check ===")
    print("Loading V01 only...")
    vehicles = load_all(verbose=False)
    v01_only = {"V01": vehicles["V01"]}
    cycles = segment_all(v01_only, verbose=False)

    t0 = time.time()
    cycles_d = accumulate_damage(cycles, v01_only)
    dt = time.time() - t0

    v01c = cycles_d[cycles_d["vehicle"] == "V01"]
    print(f"V01: {len(v01c)} cycles in {dt:.1f}s")
    print(f"d_cycle: mean={v01c['d_cycle'].mean():.2e}  max={v01c['d_cycle'].max():.2e}")
    print(f"D_cumul: final={v01c['D_cumul'].iloc[-1]:.4f}")
    print(f"\nFirst 5 cycles:")
    print(v01c[["cycle_date","DoD_pct","C_rate","T_mean_C","d_cycle","D_cumul"]].head().to_string(index=False))
