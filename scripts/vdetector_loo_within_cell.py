"""
Stage 1 (redesigned) — Within-cell delta positive control for VDetector.

MOTIVATION
The original positive control (scripts/vdetector_positive_control.py) used
cross-cell rank as its success metric. Diagnostic 2 showed that P1S7 ranks
top-1 at δ=0 due to natural cell-to-cell variation, confounding the result.
The cross-cell rank metric cannot distinguish "injection detected" from
"this cell was already a natural outlier".

This script uses within-cell delta instead:
    delta_i = score_3x_i − score_0_i
Each cell is its own control. Natural baseline elevation is subtracted out.
No cell is hand-picked; all 35 Quartz cells are tested identically via LOO.

DIAGNOSTIC 2 RESULT (recorded, not discarded)
P1S7 at δ=0 ranked top-1 among 12 cells, score=0.000073 vs gap threshold
0.000018. This is a RESULT: in this healthy Quartz pack, natural cell-to-cell
variation exceeds the δ=3× injected signal (0.000084) for at least some cells.
This is the Null Type A message stated a third way — healthy pack inter-cell
variance is large relative to any individual weak-cell signal. The prior finding
is reinforced, not overturned.

PRE-REGISTERED CRITERIA (locked before running)
Self-set for cell i: trajectories of the other 34 cells
  (34 × 2005 = 27,070 points, stride=220, all Stage 1 hyperparameters).
Two seeds: seed_primary=42 (score_0, score_3x_A, score_3x_B),
           seed_noise=43   (score_0 only, for repeat-noise estimation).
Repeat noise per cell i: |score_0_seed42_i − score_0_seed43_i|.
noise_ceiling = p95 of the 35 repeat-noise values.
delta_A_i = score_3x_Mode_A_i − score_0_i  (seed=42).
delta_B_i = score_3x_Mode_B_i − score_0_i  (seed=42).

SUCCESS (Mode A): median(delta_A) > noise_ceiling
                  AND count(delta_A > 0) >= 24  (>= two-thirds of 35 cells).
SUCCESS (Mode B): same thresholds applied to delta_B.
NULL:             either condition fails.
PARTIAL:          Mode A passes, Mode B fails → "detects rigid translation only".

δ VALUES: 3× = 0.012 only (sensitivity sweep deferred to after this check passes).
"""

from __future__ import annotations

import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from diagnosis.weakest_cell import VDetector
from scripts.vdetector_positive_control import (
    load_quartz_trajectories, inject_mode_a, inject_mode_b,
    ALL_CELLS, DATA_DIR, STRIDE, RNG_SEED,
    DELTA_3X, INTER_CELL_SPREAD,
)

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------
SEED_PRIMARY = RNG_SEED        # 42 — main scores
SEED_NOISE   = RNG_SEED + 1   # 43 — repeat-noise estimation
SUCCESS_MAJORITY = 24          # >= 24/35 cells must show positive delta
SUCCESS_LABEL = ">=24/35 (two-thirds)"

# ---------------------------------------------------------------------------
# LOO runner
# ---------------------------------------------------------------------------

def loo_scores_for_cell(
    cell_idx: int,
    all_trajs: dict,
    seed: int,
    compute_injected: bool = True,
) -> dict:
    """
    Train V-detector on 34 cells (all except cell_idx), then score cell_idx
    at δ=0, δ=3× Mode A, and δ=3× Mode B.

    Returns dict with score_0, score_3x_A, score_3x_B (only score_0 if
    compute_injected=False).
    """
    cell = ALL_CELLS[cell_idx]
    leave_out = set([cell])
    training_cells = [c for c in ALL_CELLS if c not in leave_out]

    self_arr = np.vstack([all_trajs[c] for c in training_cells])  # (34×2005, 2)

    vd = VDetector(rng_seed=seed)
    vd.observe_normal_array(self_arr)
    vd.train()

    traj = all_trajs[cell]

    # δ=0
    fracs_0 = vd.activation_fraction_array(traj)
    score_0 = float(fracs_0.mean())

    if not compute_injected:
        return {"cell": cell, "score_0": score_0, "seed": seed}

    # δ=3× Mode A
    traj_a, _ = inject_mode_a(traj, DELTA_3X)
    fracs_a = vd.activation_fraction_array(traj_a)
    score_3x_a = float(fracs_a.mean())

    # δ=3× Mode B
    traj_b, _ = inject_mode_b(traj, DELTA_3X)
    fracs_b = vd.activation_fraction_array(traj_b)
    score_3x_b = float(fracs_b.mean())

    return {
        "cell": cell,
        "score_0": score_0,
        "score_3x_A": score_3x_a,
        "score_3x_B": score_3x_b,
        "delta_A": score_3x_a - score_0,
        "delta_B": score_3x_b - score_0,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_loo_within_cell() -> dict:
    print("=" * 65)
    print("VDETECTOR LOO WITHIN-CELL DELTA — 35-cell sweep")
    print("Pre-registered criteria locked in module docstring.")
    print("=" * 65)

    trajs = load_quartz_trajectories(DATA_DIR, stride=STRIDE)
    n_cells = len(ALL_CELLS)
    n_timesteps = next(iter(trajs.values())).shape[0]

    print(f"\nSetup:")
    print(f"  Cells: {n_cells}  (P3S11 excluded)")
    print(f"  Timesteps per cell: {n_timesteps}")
    print(f"  Self-set per LOO iteration: {(n_cells-1) * n_timesteps}")
    print(f"  δ=3×: {DELTA_3X:.4f}  ({DELTA_3X/INTER_CELL_SPREAD:.1f}× inter-cell spread)")
    print(f"  Seeds: primary={SEED_PRIMARY}, noise={SEED_NOISE}")
    print(f"  SUCCESS threshold: median(delta) > noise_ceiling")
    print(f"                     AND count(delta>0) >= {SUCCESS_MAJORITY}/{n_cells}")

    # --- Primary run (seed=42): score_0, score_3x_A, score_3x_B ---
    print(f"\nRunning LOO primary pass (seed={SEED_PRIMARY})...")
    t0 = time.time()
    primary = []
    for i in range(n_cells):
        r = loo_scores_for_cell(i, trajs, seed=SEED_PRIMARY, compute_injected=True)
        primary.append(r)
        if (i + 1) % 5 == 0 or i == n_cells - 1:
            elapsed = time.time() - t0
            print(f"  [{i+1:2d}/{n_cells}] {r['cell']:6s}  "
                  f"score_0={r['score_0']:.2e}  "
                  f"delta_A={r['delta_A']:+.2e}  "
                  f"delta_B={r['delta_B']:+.2e}  "
                  f"({elapsed:.0f}s)")

    # --- Noise run (seed=43): score_0 only ---
    print(f"\nRunning LOO noise pass (seed={SEED_NOISE}, δ=0 only)...")
    t1 = time.time()
    noise_pass = []
    for i in range(n_cells):
        r = loo_scores_for_cell(i, trajs, seed=SEED_NOISE, compute_injected=False)
        noise_pass.append(r)
        if (i + 1) % 5 == 0 or i == n_cells - 1:
            elapsed = time.time() - t1
            print(f"  [{i+1:2d}/{n_cells}] {r['cell']:6s}  "
                  f"score_0_retest={r['score_0']:.2e}  ({elapsed:.0f}s)")

    # --- Compute deltas and repeat noise ---
    delta_A   = np.array([r["delta_A"] for r in primary])
    delta_B   = np.array([r["delta_B"] for r in primary])
    score_0   = np.array([r["score_0"] for r in primary])
    score_0_r = np.array([r["score_0"] for r in noise_pass])
    repeat_noise = np.abs(score_0 - score_0_r)
    noise_ceiling = float(np.percentile(repeat_noise, 95))

    # --- Results table ---
    print(f"\n{'='*65}")
    print(f"PER-CELL RESULTS  (sorted by delta_A descending)")
    print(f"{'='*65}")
    print(f"{'Cell':6s}  {'score_0':>10s}  {'score_3xA':>10s}  {'delta_A':>10s}  "
          f"{'delta_B':>10s}  {'rpt_noise':>10s}")
    print("-" * 65)
    order = np.argsort(delta_A)[::-1]
    for i in order:
        r = primary[i]
        rn = repeat_noise[i]
        flag_a = ">" if delta_A[i] > noise_ceiling else " "
        print(f"{r['cell']:6s}  {r['score_0']:10.2e}  {r['score_3x_A']:10.2e}  "
              f"{delta_A[i]:+10.2e}{flag_a} {delta_B[i]:+10.2e}  {rn:10.2e}")
    print(f"  > = delta_A exceeds noise_ceiling ({noise_ceiling:.2e})")

    # --- Distribution summary ---
    print(f"\n{'='*65}")
    print(f"DISTRIBUTION SUMMARY")
    print(f"{'='*65}")
    print(f"\nRepeat noise (|score_0_seed42 − score_0_seed43|):")
    for p in [0, 5, 25, 50, 75, 95, 100]:
        print(f"  p{p:3d}: {np.percentile(repeat_noise, p):.3e}")
    print(f"  noise_ceiling (p95): {noise_ceiling:.3e}")

    print(f"\ndelta_A (score_3x_Mode_A − score_0) across 35 cells:")
    for p in [0, 5, 25, 50, 75, 95, 100]:
        print(f"  p{p:3d}: {np.percentile(delta_A, p):+.3e}")
    n_pos_a = int((delta_A > 0).sum())
    n_exceed_a = int((delta_A > noise_ceiling).sum())
    print(f"  count(delta_A > 0):             {n_pos_a}/{n_cells}")
    print(f"  count(delta_A > noise_ceiling): {n_exceed_a}/{n_cells}")

    print(f"\ndelta_B (score_3x_Mode_B − score_0) across 35 cells:")
    for p in [0, 5, 25, 50, 75, 95, 100]:
        print(f"  p{p:3d}: {np.percentile(delta_B, p):+.3e}")
    n_pos_b = int((delta_B > 0).sum())
    n_exceed_b = int((delta_B > noise_ceiling).sum())
    print(f"  count(delta_B > 0):             {n_pos_b}/{n_cells}")
    print(f"  count(delta_B > noise_ceiling): {n_exceed_b}/{n_cells}")

    # --- Pre-registered verdict ---
    med_a = float(np.median(delta_A))
    med_b = float(np.median(delta_B))
    pass_med_a  = med_a > noise_ceiling
    pass_maj_a  = n_pos_a >= SUCCESS_MAJORITY
    pass_med_b  = med_b > noise_ceiling
    pass_maj_b  = n_pos_b >= SUCCESS_MAJORITY
    success_a = pass_med_a and pass_maj_a
    success_b = pass_med_b and pass_maj_b

    print(f"\n{'='*65}")
    print(f"PRE-REGISTERED VERDICT  (δ=3×={DELTA_3X:.4f})")
    print(f"{'='*65}")
    print(f"\nMode A:")
    print(f"  median(delta_A) = {med_a:.3e}  > noise_ceiling {noise_ceiling:.3e}? "
          f"{'PASS' if pass_med_a else 'FAIL'}")
    print(f"  count(delta_A>0) = {n_pos_a}/{n_cells} >= {SUCCESS_MAJORITY}? "
          f"{'PASS' if pass_maj_a else 'FAIL'}")
    print(f"  Mode A overall: {'SUCCESS' if success_a else 'FAIL'}")

    print(f"\nMode B:")
    print(f"  median(delta_B) = {med_b:.3e}  > noise_ceiling {noise_ceiling:.3e}? "
          f"{'PASS' if pass_med_b else 'FAIL'}")
    print(f"  count(delta_B>0) = {n_pos_b}/{n_cells} >= {SUCCESS_MAJORITY}? "
          f"{'PASS' if pass_maj_b else 'FAIL'}")
    print(f"  Mode B overall: {'SUCCESS' if success_b else 'FAIL'}")

    if success_a and success_b:
        verdict = "ALGORITHM VALIDATED (both modes)"
    elif success_a and not success_b:
        verdict = "PARTIAL — detects rigid translation (Mode A), not load-dependent shape change (Mode B)"
    elif not success_a:
        verdict = "NULL — algorithm cannot detect δ=3× signal above noise floor"
    print(f"\nCOMBINED VERDICT: {verdict}")
    print("=" * 65)

    result = {
        "pre_registration": {
            "success_criterion": (
                "median(delta) > noise_ceiling (p95 repeat noise) "
                f"AND count(delta>0) >= {SUCCESS_MAJORITY}/{n_cells}"
            ),
            "noise_definition": "|score_0_seed42 - score_0_seed43| per cell",
            "seeds": {"primary": SEED_PRIMARY, "noise": SEED_NOISE},
            "delta_3x": DELTA_3X,
        },
        "noise_ceiling_p95": noise_ceiling,
        "repeat_noise_stats": {p: float(np.percentile(repeat_noise, p))
                               for p in [0, 5, 25, 50, 75, 95, 100]},
        "mode_A": {
            "median_delta": med_a,
            "n_positive": n_pos_a,
            "n_exceed_ceiling": n_exceed_a,
            "pass_median": pass_med_a,
            "pass_majority": pass_maj_a,
            "success": success_a,
        },
        "mode_B": {
            "median_delta": med_b,
            "n_positive": n_pos_b,
            "n_exceed_ceiling": n_exceed_b,
            "pass_median": pass_med_b,
            "pass_majority": pass_maj_b,
            "success": success_b,
        },
        "combined_verdict": verdict,
        "per_cell": [
            {
                "cell": primary[i]["cell"],
                "score_0": float(score_0[i]),
                "score_3x_A": float(primary[i]["score_3x_A"]),
                "score_3x_B": float(primary[i]["score_3x_B"]),
                "delta_A": float(delta_A[i]),
                "delta_B": float(delta_B[i]),
                "repeat_noise": float(repeat_noise[i]),
            }
            for i in range(n_cells)
        ],
    }

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "vdetector_loo_within_cell.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to data/vdetector_loo_within_cell.json")

    return result


if __name__ == "__main__":
    run_loo_within_cell()
