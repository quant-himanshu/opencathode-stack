#!/usr/bin/env python3
"""
degradation/lambda_pooled_estimate.py  —  Pooled λ estimate with per-vehicle spread reporting

PURPOSE
-------
Estimate a fleet-representative SEI calendar-aging rate λ by pooling per-vehicle OLS
estimates from Tier 1+2 vehicles (V01-V09), then evaluate how well this pooled λ
predicts Tier 3 (V10-V20) test-window trajectories versus the frozen production scalar.

METHOD
------
This is a simple pooling heuristic, not formal hierarchical Bayesian shrinkage —
no informative prior or posterior uncertainty is computed.

Per-vehicle λ_v values are loaded directly from cell_to_field_temporal_report.json
(train-window OLS fits, 50% of each vehicle's timeline). No new OLS fit is performed.
This choice was made so the pre-registered pooled mean of 0.024895 is exactly correct —
re-fitting on full vehicle records would produce different λ_v estimates and invalidate
the pre-registration. The train-window fits are already-computed, already-committed
per-vehicle estimates; reusing them is simpler and methodologically transparent.

No flooring of negative λ_v is applied. Flooring would asymmetrically bias the pooled
mean upward and would be inconsistent with how the existing fleet-median baseline
(0.018476) was computed in cell_to_field_temporal.py (raw np.median, no floor).
Raw values are used throughout for methodological consistency.

DATA SPLIT (Tier structure, unchanged from all prior modules)
-------------------------------------------------------------
  Tier 1 = V01-V04  (λ source for M2 frozen scalar; included in Tier 1+2 pool here)
  Tier 2 = V05-V09  (pool training — no new fits)
  Tier 3 = V10-V20  (pure test — never seen during pool computation)

PRE-REGISTRATION (stated before code runs, computed from existing JSON)
----------------------------------------------------------------------
  V01-V09 per-vehicle λ_v (train-window, raw, no floor):
    V01=+0.057639  V02=+0.004216  V03=+0.020574  V04=+0.027598  V05=-0.019982
    V06=+0.036280  V07=-0.049065  V08=+0.083915  V09=+0.062882
    n_negative = 2 (V05, V07)

  Expected pooled mean (arithmetic, raw)  = 0.024895
  Expected pooled median (raw)            = 0.027598

  Qualitative expectation: λ_pool (mean=0.024895) is slightly LOWER than the frozen
  scalar λ_sei=0.02639332 (Δλ≈0.0015) and substantially HIGHER than the all-20 fleet
  median 0.018476 (because Tier 2 over-samples high-fade vehicles: V08=0.084, V09=0.063).
  Neither converges toward the full-fleet median.

  Expected Tier 3 RMSE outcome: null result. Δλ≈0.0015 implies ΔSOH difference
  ≈0.0015·√t ≈0.001-0.002 SOH units over 1 year — well within the per-vehicle noise
  floor of 0.02-0.04 observed in cell_to_field_temporal.py. If B_sei wins or ties,
  that is reported as-is. No post-hoc adjustment.

EVALUATION BASELINES (on Tier 3 test windows, fresh computation from data)
---------------------------------------------------------------------------
  B_sei  : λ = 0.02639332  (production frozen scalar, V01-V04 aggregate M2 fit)
  B_fleet: λ = 0.018476    (all-20 fleet median from temporal report, no floor)
  B_pool : λ = λ_pool      (V01-V09 pooled mean, this module's output)

  Prediction: ΔSOH_pred(t) = λ · √t  (no intercept, same as all prior modules)
  Metric: trajectory RMSE on test-window cycles (t > t_cut, same split as temporal script)

OUTPUT
------
  data/lambda_pooled_report.json

CITATION
--------
  Calendar aging kinetics (√t form): Pinson & Bazant 2013 J. Electrochem. Soc. 160:A243.
  Additive calendar+cycling model structure: Schmalstieg et al. 2014 J. Power Sources 257:325.
  Source data: Deng et al. 2023 Applied Energy 339:120954.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT         = Path(__file__).resolve().parent.parent
TEMPORAL_RPT = ROOT / "data" / "cell_to_field_temporal_report.json"
OUT_JSON     = ROOT / "data" / "lambda_pooled_report.json"

# Tier split — identical to all prior modules
TIER1_2 = ["V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V09"]
TIER3   = ["V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20"]

# Reference constants (frozen from prior modules — never re-tuned here)
LAMBDA_SEI_FROZEN  = 0.02639332   # M2 V01-V04 aggregate OLS (cell_to_field_report.json)
LAMBDA_FLEET_MED   = 0.018476     # all-20 median, no floor (cell_to_field_temporal_report.json)
TRAIN_FRAC         = 0.50         # 50/50 temporal split, same as temporal script
Q_NOMINAL          = 136.2        # Ah — consistent with all prior modules
SOH_WIN            = 50           # rolling-window size for SOH smoothing


# ---------------------------------------------------------------------------
# Data helpers (minimal re-implementation of temporal script mechanics)
# ---------------------------------------------------------------------------

_CYCLES_CACHE: Optional[pd.DataFrame] = None

def _get_fleet_cycles() -> pd.DataFrame:
    """Load full fleet cycles once, cache for reuse. Mirrors temporal script exactly."""
    global _CYCLES_CACHE
    if _CYCLES_CACHE is not None:
        return _CYCLES_CACHE
    from degradation.deng_loader    import load_all
    from degradation.cycle_segmentor import segment_all
    from degradation.soh_predictor  import add_t_years
    print("  (Loading full Deng fleet — first call only)")
    raw = load_all(verbose=False)
    cycles = segment_all(raw, verbose=False)
    cycles = add_t_years(cycles)
    _CYCLES_CACHE = cycles
    return cycles



def _observed_dsoh(fleet_cycles: pd.DataFrame, veh: str) -> Optional[np.ndarray]:
    """ΔSOH[k] = SOH[0] - SOH[k]. Delegates to soh_predictor.observed_delta_soh."""
    from degradation.soh_predictor import observed_delta_soh
    return observed_delta_soh(fleet_cycles, veh)


def _rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def _eval_lambda_on_vehicle(
    veh: str,
    lambda_pool: float,
    fleet_cycles: pd.DataFrame,
) -> Optional[Dict]:
    """
    Evaluate B_sei, B_fleet, B_pool on the test window (t > t_cut) of one vehicle.
    Returns None if vehicle has insufficient data.
    """
    vc = fleet_cycles[fleet_cycles["vehicle"] == veh].copy().reset_index(drop=True)
    if len(vc) < 2 * SOH_WIN:
        return None

    dsoh = _observed_dsoh(fleet_cycles, veh)
    if dsoh is None:
        return None

    t = vc["t_years"].values.astype(float)
    t_max  = float(t[-1])
    t_cut  = TRAIN_FRAC * t_max
    te_mask = t > t_cut

    if te_mask.sum() < 10:
        return None

    t_te    = t[te_mask]
    dsoh_te = dsoh[te_mask]
    sqrt_te = np.sqrt(np.maximum(t_te, 0.0))

    pred_sei   = LAMBDA_SEI_FROZEN * sqrt_te
    pred_fleet = LAMBDA_FLEET_MED  * sqrt_te
    pred_pool  = lambda_pool       * sqrt_te

    return {
        "t_cut_years":      round(float(t_cut), 3),
        "t_max_years":      round(float(t_max), 3),
        "n_test_cycles":    int(te_mask.sum()),
        "trajectory_rmse_B_sei":   round(_rmse(pred_sei,   dsoh_te), 5),
        "trajectory_rmse_B_fleet": round(_rmse(pred_fleet, dsoh_te), 5),
        "trajectory_rmse_B_pool":  round(_rmse(pred_pool,  dsoh_te), 5),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 68)
    print("lambda_pooled_estimate.py — Pooled λ with per-vehicle spread")
    print("=" * 68)

    # ------------------------------------------------------------------
    # Step 1 — Load Tier 1+2 λ_v from temporal report (no new OLS fit)
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading Tier 1+2 λ_v from cell_to_field_temporal_report.json")
    temporal = json.loads(TEMPORAL_RPT.read_text())
    pv       = temporal["per_vehicle"]

    tier1_2_lambdas: Dict[str, float] = {}
    for veh in TIER1_2:
        lam = float(pv[veh]["lambda_v"])
        tier1_2_lambdas[veh] = lam
        flag = "  [NEGATIVE]" if lam < 0 else ""
        print(f"  {veh}: λ_v = {lam:+.6f}{flag}")

    raw_vals = list(tier1_2_lambdas.values())
    lam_pool_mean   = float(np.mean(raw_vals))
    lam_pool_median = float(np.median(raw_vals))
    lam_pool_sd     = float(np.std(raw_vals, ddof=1))
    n_negative      = sum(1 for v in raw_vals if v < 0)

    print(f"\n  Pooled mean   (primary):  {lam_pool_mean:.6f}")
    print(f"  Pooled median (secondary):{lam_pool_median:.6f}")
    print(f"  SD:                       {lam_pool_sd:.6f}")
    print(f"  n_negative:               {n_negative} of {len(raw_vals)}")

    # Verify pre-registration
    PRE_MEAN   = 0.024895
    PRE_MEDIAN = 0.027598
    assert abs(lam_pool_mean   - PRE_MEAN)   < 1e-5, \
        f"Pre-registered mean mismatch: got {lam_pool_mean:.6f}, expected {PRE_MEAN}"
    assert abs(lam_pool_median - PRE_MEDIAN) < 1e-5, \
        f"Pre-registered median mismatch: got {lam_pool_median:.6f}, expected {PRE_MEDIAN}"
    print("\n  [OK] Pre-registered values confirmed exactly.")

    # ------------------------------------------------------------------
    # Step 2 — Freeze λ_pool before touching Tier 3
    # ------------------------------------------------------------------
    LAMBDA_POOL = lam_pool_mean   # primary: arithmetic mean, raw, no floor
    print(f"\n[Step 2] λ_pool FROZEN = {LAMBDA_POOL:.6f}  (Tier 3 data loading begins now)")

    # ------------------------------------------------------------------
    # Step 3 — Evaluate on Tier 3 (V10-V20)
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Evaluating on Tier 3 (V10-V20)")
    print(f"  Baselines:  B_sei={LAMBDA_SEI_FROZEN}  B_fleet={LAMBDA_FLEET_MED}  B_pool={LAMBDA_POOL:.6f}")
    fleet_cycles = _get_fleet_cycles()

    tier3_results: Dict[str, Dict] = {}
    rmse_sei_list:   List[float] = []
    rmse_fleet_list: List[float] = []
    rmse_pool_list:  List[float] = []

    for veh in TIER3:
        res = _eval_lambda_on_vehicle(veh, LAMBDA_POOL, fleet_cycles)
        if res is None:
            print(f"  {veh}: SKIPPED (load error or insufficient data)")
            tier3_results[veh] = {"skipped": True}
            continue
        tier3_results[veh] = res
        rmse_sei_list.append(res["trajectory_rmse_B_sei"])
        rmse_fleet_list.append(res["trajectory_rmse_B_fleet"])
        rmse_pool_list.append(res["trajectory_rmse_B_pool"])

        delta_pool_vs_sei = res["trajectory_rmse_B_pool"] - res["trajectory_rmse_B_sei"]
        sign = "+" if delta_pool_vs_sei >= 0 else ""
        print(
            f"  {veh}: RMSE  sei={res['trajectory_rmse_B_sei']:.4f}"
            f"  fleet={res['trajectory_rmse_B_fleet']:.4f}"
            f"  pool={res['trajectory_rmse_B_pool']:.4f}"
            f"  (pool−sei: {sign}{delta_pool_vs_sei:.4f})"
        )

    # Fleet-mean RMSE across evaluable Tier 3 vehicles
    n_eval = len(rmse_sei_list)
    fleet_rmse_sei   = float(np.mean(rmse_sei_list))   if rmse_sei_list   else float("nan")
    fleet_rmse_fleet = float(np.mean(rmse_fleet_list)) if rmse_fleet_list else float("nan")
    fleet_rmse_pool  = float(np.mean(rmse_pool_list))  if rmse_pool_list  else float("nan")

    print(f"\n  Fleet-mean RMSE (n={n_eval} vehicles):")
    print(f"    B_sei   (λ=0.02639332): {fleet_rmse_sei:.5f}")
    print(f"    B_fleet (λ=0.018476):   {fleet_rmse_fleet:.5f}")
    print(f"    B_pool  (λ={LAMBDA_POOL:.6f}): {fleet_rmse_pool:.5f}")

    # ------------------------------------------------------------------
    # Step 4 — Verdict (pre-committed decision rule)
    # ------------------------------------------------------------------
    if n_eval == 0:
        verdict = "INSUFFICIENT_DATA"
    else:
        delta = fleet_rmse_pool - fleet_rmse_sei
        if abs(delta) < 0.001:
            verdict = "NULL_RESULT: B_pool and B_sei are within 0.001 RMSE — indistinguishable given fleet noise floor (pre-registered expectation confirmed)."
        elif delta < 0:
            verdict = f"B_pool WINS: fleet-mean RMSE lower by {abs(delta):.5f} vs B_sei. Marginal — interpret cautiously given noise floor."
        else:
            verdict = f"B_sei WINS: B_pool fleet-mean RMSE higher by {delta:.5f}. Pooling V01-V09 does not improve on frozen scalar."

    print(f"\n  VERDICT: {verdict}")

    # ------------------------------------------------------------------
    # Step 5 — Write JSON
    # ------------------------------------------------------------------
    report = {
        "meta": {
            "script":  "degradation/lambda_pooled_estimate.py",
            "method":  "pooled_lambda_estimate_with_per_vehicle_spread",
            "method_note": (
                "Simple pooling heuristic. Not formal hierarchical Bayesian shrinkage — "
                "no informative prior or posterior uncertainty is computed. "
                "Per-vehicle λ_v loaded from cell_to_field_temporal_report.json "
                "(train-window OLS, no new fit). No flooring of negatives: flooring "
                "would bias the mean upward and be inconsistent with how the fleet-median "
                "baseline (0.018476) was computed (raw np.median, no floor)."
            ),
            "tier_split": {
                "tier1_2": TIER1_2,
                "tier3":   TIER3,
            },
            "source_lambda_v": "cell_to_field_temporal_report.json — train-window OLS fits (50% of each vehicle timeline)",
        },
        "pre_registration": {
            "expected_pool_mean":   PRE_MEAN,
            "expected_pool_median": PRE_MEDIAN,
            "qualitative_expectation": (
                "λ_pool (mean=0.024895) is slightly lower than frozen scalar 0.02639332 "
                "(Δλ≈0.0015) and substantially higher than all-20 fleet median 0.018476 "
                "because Tier 2 over-samples high-fade vehicles (V08=0.084, V09=0.063). "
                "Expected Tier 3 RMSE outcome: null result (Δλ≈0.0015 → ΔSOH≈0.001-0.002, "
                "within fleet noise floor of 0.02-0.04)."
            ),
        },
        "tier1_2_fit": {
            "per_vehicle_lambda_v": {v: round(lam, 6) for v, lam in tier1_2_lambdas.items()},
            "pooled_mean_raw":   round(lam_pool_mean,   6),
            "pooled_median_raw": round(lam_pool_median, 6),
            "pooled_sd":         round(lam_pool_sd,     6),
            "n_vehicles":        len(raw_vals),
            "n_negative":        n_negative,
            "lambda_used_for_tier3": round(LAMBDA_POOL, 6),
            "lambda_used_note":  "arithmetic mean of raw V01-V09 λ_v (no floor)",
        },
        "baselines": {
            "B_sei":   {"lambda": LAMBDA_SEI_FROZEN, "source": "V01-V04 aggregate OLS, M2 (cell_to_field_report.json)"},
            "B_fleet": {"lambda": LAMBDA_FLEET_MED,  "source": "all-20 median, no floor (cell_to_field_temporal_report.json)"},
            "B_pool":  {"lambda": round(LAMBDA_POOL, 6), "source": "V01-V09 pooled mean, no floor (this module)"},
        },
        "tier3_evaluation": {
            "split": "test window: t > 0.50 * t_max per vehicle (same split as cell_to_field_temporal.py)",
            "metric": "trajectory RMSE on test-window ΔSOH timeseries",
            "per_vehicle": tier3_results,
            "fleet_mean_rmse": {
                "n_evaluable_vehicles": n_eval,
                "B_sei":   round(fleet_rmse_sei,   5),
                "B_fleet": round(fleet_rmse_fleet, 5),
                "B_pool":  round(fleet_rmse_pool,  5),
            },
        },
        "representativeness_note": (
            "Tier 2 (V05-V09) is not representative of the full fleet. "
            "V08 (λ=0.084) and V09 (λ=0.063) are extreme high-fade outliers; "
            "V05 and V07 are negative-λ. The V01-V09 pool mean (0.025) sits "
            "well above the all-20 fleet median (0.018) as a result. "
            "The all-20 fleet median is the better fleet-prior if Tier 3 coverage is available."
        ),
        "verdict": verdict,
    }

    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\n[Done] Report written → {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
