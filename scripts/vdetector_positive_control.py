"""
Stage 1 Positive Control for VDetector.

PURPOSE: Confirm that V-detector can rank a known-degraded cell as top-1
BEFORE any contact with real Quartz data. This separates algorithm validation
from data signal: the prior Null Type A was a data problem (47× cycling-vs-
inter-cell variance domination), not an algorithm problem. This script proves
the algorithm works on a ground-truth case.

SYNTHETIC PACK CONSTRUCTION:
  - 12 distinct real cells drawn WITHOUT replacement from the 35 Quartz cells.
  - Cells 0–10 (healthy): form the V-detector self-set.
  - Cell 11 (degraded): trajectory NOT seen during training; δ injected post-hoc.
  - Trajectories = real per-cell (V_norm, T_norm) sequences from all 8 WLTP cycles.
  - Stride = 220 (same as prior NSA self-set construction, ~2,880 timesteps/cell).

INJECTION MODES (pre-registered):
  Mode A — constant offset:  V_norm(t) -= δ  at every timestep.
  Mode B — load-dependent:   V_norm(t) -= δ × (1 – V_norm(t)).
                              Peak offset = δ at deep discharge (V_norm≈0);
                              zero offset at full charge (V_norm≈1).
  Post-injection clip to [0, 1] applied; clipped count reported.

δ VALUES (pre-registered):
  3x: 3 × 0.004 = 0.012  (3× inter-cell spread)
  1x: 1 × 0.004 = 0.004  (at noise floor)
  0.5x: 0.5 × 0.004 = 0.002  (below noise floor)
  Inter-cell spread 0.004 from stage_1 v_norm_span_across_cells=0.0084 / 2.

PRE-REGISTERED SUCCESS CRITERIA (locked before running):
  SUCCESS: Cell #12 ranks top-1 in PRIMARY score at δ=3× for BOTH Mode A and
           Mode B AND PRIMARY(#12) > max(PRIMARY(healthy)) + std(PRIMARY(healthy)).
  SENSITIVITY: smallest δ at which top-1 still holds (reported after δ=3× check).
  NULL (algorithm inadequate): cell #12 fails top-1 at δ=3× for EITHER mode.
           If this occurs, do NOT proceed to Stage 2 (Quartz). Report as algorithm
           failure, not a data problem.

STAGE 2 NOTE (pre-registered, not run here):
  After Stage 1 passes: run V-detector on the real 35-cell Quartz pack.
  Expected result: NULL (Null Type A, same reason as prior NSA — cycling
  amplitude 0.188 >> inter-cell spread 0.004 by 47×). A null in Stage 2
  confirms the prior finding with a now-validated algorithm.
"""

from __future__ import annotations

import os
import sys
import glob
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from diagnosis.weakest_cell import VDetector, FEATURE_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "quartz_wltp")
STRIDE = 220          # same as prior NSA self-set construction
RNG_SEED = 42         # fixed for reproducibility; cell selection documented below

INTER_CELL_SPREAD = 0.004   # V_norm std across healthy cells (stage_1 span / 2)
DELTA_3X = 3 * INTER_CELL_SPREAD   # 0.012
DELTA_1X = 1 * INTER_CELL_SPREAD   # 0.004
DELTA_05X = 0.5 * INTER_CELL_SPREAD  # 0.002

# All 35 Quartz cells (P3S11 excluded — known faulty temperature sensor 476°C mean)
ALL_CELLS = [
    f"P{p}S{s}"
    for p in range(1, 4)
    for s in range(1, 13)
    if not (p == 3 and s == 11)
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_quartz_trajectories(data_dir: str, stride: int = STRIDE) -> Dict[str, np.ndarray]:
    """
    Load per-cell (V_norm, T_norm) trajectories from all WLTP parquet files.

    Returns dict mapping cell_id (e.g. 'P1S3') to (N_timesteps, 2) array.
    Temperature = mean of Top and Bottom sensors per cell.
    P3S11 excluded (faulty sensor; 651°C observed in raw data).
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*WLTP*.parquet")))
    if not files:
        raise FileNotFoundError(f"No WLTP parquet files found in {data_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.iloc[::stride].reset_index(drop=True)

    trajectories: Dict[str, np.ndarray] = {}
    for cell in ALL_CELLS:
        v_col   = f"Voltage_Cell_{cell} [V]"
        t_top   = f"Temperature_Cell_Top_{cell} [degC]"
        t_bot   = f"Temperature_Cell_Bottom_{cell} [degC]"

        v   = df[v_col].to_numpy(dtype=np.float64)
        t_c = (df[t_top].to_numpy(dtype=np.float64)
               + df[t_bot].to_numpy(dtype=np.float64)) / 2.0

        v_norm = np.clip((v - 3.0) / 1.5, 0.0, 1.0)
        t_norm = (t_c - 25.0) / 50.0   # (°C – 25) / 50 ≡ (K – 298.15) / 50

        trajectories[cell] = np.column_stack([v_norm, t_norm])  # (N, 2)

    return trajectories


# ---------------------------------------------------------------------------
# Injection modes
# ---------------------------------------------------------------------------

def inject_mode_a(traj: np.ndarray, delta: float) -> Tuple[np.ndarray, int]:
    """Constant offset: V_norm -= δ at every timestep."""
    v_deg = traj[:, 0] - delta
    n_clipped = int((v_deg < 0.0).sum())
    v_deg = np.clip(v_deg, 0.0, 1.0)
    return np.column_stack([v_deg, traj[:, 1]]), n_clipped


def inject_mode_b(traj: np.ndarray, delta: float) -> Tuple[np.ndarray, int]:
    """
    Load-dependent sag: V_norm(t) -= δ × (1 – V_norm(t)).
    Peak sag = δ at deep discharge (V_norm≈0); zero at full charge (V_norm≈1).
    δ here is the peak offset.
    """
    v_deg = traj[:, 0] - delta * (1.0 - traj[:, 0])
    n_clipped = int((v_deg < 0.0).sum())
    v_deg = np.clip(v_deg, 0.0, 1.0)
    return np.column_stack([v_deg, traj[:, 1]]), n_clipped


# ---------------------------------------------------------------------------
# Positive control runner
# ---------------------------------------------------------------------------

def run_positive_control(delta: float, show_sensitivity: bool = False) -> dict:
    """
    Run Stage 1 positive control at a single δ value.
    Reports per-cell PRIMARY and SECONDARY scores for both injection modes.
    Checks pre-registered success criteria.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1 POSITIVE CONTROL  δ = {delta:.4f}  ({delta/INTER_CELL_SPREAD:.1f}× inter-cell spread)")
    print("=" * 60)

    # --- Load trajectories ---
    trajs = load_quartz_trajectories(DATA_DIR)
    n_timesteps = next(iter(trajs.values())).shape[0]

    # --- Select 12 distinct cells WITHOUT replacement ---
    rng = np.random.default_rng(RNG_SEED)
    chosen_idx = rng.choice(len(ALL_CELLS), 12, replace=False)
    chosen_cells = [ALL_CELLS[i] for i in chosen_idx]
    healthy_cells = chosen_cells[:11]
    target_cell   = chosen_cells[11]

    print(f"\nCell selection (seed={RNG_SEED}, without replacement):")
    print(f"  Healthy (training self-set, cells 0–10): {healthy_cells}")
    print(f"  Target cell #12 (degraded, NOT in training): {target_cell}")

    # --- Dataset stats ---
    print(f"\nDataset stats:")
    print(f"  WLTP files loaded:  8 cycles")
    print(f"  Stride:             {STRIDE}")
    print(f"  Timesteps per cell: {n_timesteps}")
    print(f"  Self-set size:      {11 * n_timesteps}  (11 healthy × {n_timesteps})")

    self_arr = np.vstack([trajs[c] for c in healthy_cells])

    # Sanity: inter-cell V_norm std on healthy cells
    healthy_cell_means = np.array([trajs[c][:, 0].mean() for c in healthy_cells])
    print(f"  Healthy-cell V_norm means:  min={healthy_cell_means.min():.5f}  "
          f"max={healthy_cell_means.max():.5f}  "
          f"std={healthy_cell_means.std():.5f}")

    # --- Train V-detector on 11 healthy cells ---
    print(f"\nTraining V-detector (n_candidate={VDetector().n_candidate})...")
    vd = VDetector(rng_seed=RNG_SEED)
    vd.observe_normal_array(self_arr)
    stats = vd.train()

    print(f"  n_survivors:      {stats['n_survivors']}")
    print(f"  n_filtered_r_min: {stats['n_filtered_r_min']}  "
          f"(r_i < r_min={vd.r_min})")
    print(f"  p95_self_cloud:   {stats['p95_self_cloud']:.5f}")
    print(f"  centroid:         V_norm={stats['centroid'][0]:.5f}  "
          f"T_norm={stats['centroid'][1]:.5f}")
    print(f"  r distribution:   min={stats['r_min_actual']:.5f}  "
          f"p5={stats['r_p5']:.5f}  median={stats['r_median']:.5f}  "
          f"p95={stats['r_p95']:.5f}  max={stats['r_max_actual']:.5f}")
    print(f"  [PRE-REG CHECK] saturation_flag (any detector >50% activation): "
          f"{'*** FLAGGED ***' if stats['saturation_flag'] else 'CLEAR'}")

    # --- Score both injection modes ---
    results = {}
    for mode_name, inject_fn in [("Mode_A", inject_mode_a), ("Mode_B", inject_mode_b)]:
        print(f"\n--- {mode_name} scoring ---")

        tgt_traj, n_clipped = inject_fn(trajs[target_cell], delta)
        if n_clipped > 0:
            print(f"  Note: {n_clipped}/{n_timesteps} injected V_norm values "
                  f"clipped to 0 after injection")

        # Score all 12 cells
        scores_primary   = {}
        scores_secondary = {}

        for i, cell in enumerate(healthy_cells):
            fracs = vd.activation_fraction_array(trajs[cell])
            scores_primary[cell]   = float(fracs.mean())
            scores_secondary[cell] = float(fracs.max())

        # Target cell (degraded)
        tgt_fracs = vd.activation_fraction_array(tgt_traj)
        scores_primary[target_cell]   = float(tgt_fracs.mean())
        scores_secondary[target_cell] = float(tgt_fracs.max())

        # Sort by PRIMARY score descending
        sorted_cells = sorted(scores_primary, key=lambda c: scores_primary[c], reverse=True)

        print(f"\n  Per-cell PRIMARY scores (mean activation fraction), ranked:")
        for rank, cell in enumerate(sorted_cells, 1):
            marker = " ← INJECTED" if cell == target_cell else ""
            print(f"    {rank:2d}. {cell:6s}  PRIMARY={scores_primary[cell]:.6f}  "
                  f"SECONDARY={scores_secondary[cell]:.6f}{marker}")

        # Pre-registered success check
        healthy_primaries = np.array([scores_primary[c] for c in healthy_cells])
        tgt_primary  = scores_primary[target_cell]
        top1_cell    = sorted_cells[0]
        top1_pass    = (top1_cell == target_cell)
        gap_threshold = healthy_primaries.max() + healthy_primaries.std()
        gap_pass     = tgt_primary > gap_threshold

        print(f"\n  [PRE-REG SUCCESS CHECK] {mode_name}:")
        print(f"    Top-1 is target cell: {'PASS' if top1_pass else 'FAIL'} "
              f"(top-1={top1_cell}, target={target_cell})")
        print(f"    Gap criterion: PRIMARY(target)={tgt_primary:.6f} "
              f"> max(healthy)+std(healthy)={gap_threshold:.6f}: "
              f"{'PASS' if gap_pass else 'FAIL'}")

        overall = top1_pass and gap_pass
        print(f"    OVERALL {mode_name}: {'SUCCESS' if overall else 'FAIL'}")

        results[mode_name] = {
            "top1_pass": top1_pass,
            "gap_pass": gap_pass,
            "overall": overall,
            "target_primary": tgt_primary,
            "healthy_max_primary": float(healthy_primaries.max()),
            "healthy_std_primary": float(healthy_primaries.std()),
            "gap_threshold": float(gap_threshold),
            "scores_primary": scores_primary,
            "scores_secondary": scores_secondary,
            "sorted_ranking": sorted_cells,
            "n_clipped": n_clipped,
        }

    # --- Combined verdict ---
    both_pass = results["Mode_A"]["overall"] and results["Mode_B"]["overall"]
    print(f"\n{'='*60}")
    print(f"STAGE 1 VERDICT at δ={delta:.4f} ({delta/INTER_CELL_SPREAD:.1f}× inter-cell spread):")
    print(f"  Mode A: {'SUCCESS' if results['Mode_A']['overall'] else 'FAIL'}")
    print(f"  Mode B: {'SUCCESS' if results['Mode_B']['overall'] else 'FAIL'}")
    print(f"  COMBINED: {'ALGORITHM VALIDATED — proceed to sensitivity sweep' if both_pass else 'ALGORITHM NULL — do NOT proceed'}")
    if not both_pass and not results["Mode_A"]["overall"]:
        print("  NOTE: Mode A fail → algorithm cannot detect even a rigid translation.")
    if not both_pass and results["Mode_A"]["overall"] and not results["Mode_B"]["overall"]:
        print("  NOTE: Mode A pass, Mode B fail → detects translation only, not shape change.")
    print("=" * 60)

    return {
        "delta": delta,
        "delta_x_spread": delta / INTER_CELL_SPREAD,
        "healthy_cells": healthy_cells,
        "target_cell": target_cell,
        "n_timesteps": n_timesteps,
        "train_stats": stats,
        "Mode_A": results["Mode_A"],
        "Mode_B": results["Mode_B"],
        "combined_pass": both_pass,
    }


# ---------------------------------------------------------------------------
# Entry point — Stage 1: show δ=3× results, stop before sensitivity sweep
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("VDETECTOR POSITIVE CONTROL — Stage 1")
    print("Pre-registered criteria locked in module docstring.")
    print("Running δ=3× only. Sensitivity sweep NOT yet triggered.")
    print("=" * 60)

    result_3x = run_positive_control(delta=DELTA_3X, show_sensitivity=False)

    # Write results to JSON for record
    out_path = os.path.join(os.path.dirname(__file__), "..",
                            "data", "vdetector_positive_control.json")
    with open(out_path, "w") as f:
        # Convert np types for JSON serialisation
        def to_serialisable(obj):
            if isinstance(obj, bool):          # before int — bool is subclass of int
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_serialisable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serialisable(x) for x in obj]
            return obj

        json.dump(to_serialisable(result_3x), f, indent=2)

    print(f"\nResults written to data/vdetector_positive_control.json")
    print("Show this output to the user before running sensitivity sweep.")
