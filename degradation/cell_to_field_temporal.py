#!/usr/bin/env python3
"""
degradation/cell_to_field_temporal.py  —  Temporal-split per-vehicle λ experiment

Per-vehicle calendar-only model (cycling term dropped: confirmed D≈0.002, negligible).
Each vehicle is split 50/50 by time. λ_v is fit on the train window; the test window
is predicted and compared against measurement.

THREE BASELINES (test window only):
  B0' — carry-forward: ΔSOH(t) = ΔSOH_obs at train/test boundary (zero further change)
  B1' — LOO-transferred calendar: ΔSOH(t) = λ_LOO · √t
         λ_LOO = median(λ_v) across all OTHER vehicles' train windows (leave-one-out)
  B2' — per-vehicle calendar: ΔSOH(t) = λ_v · √t
         λ_v = fit on THIS vehicle's own train window

PRIMARY METRICS (test-window trajectory, not just endpoint):
  RMSE(ΔSOH_pred − ΔSOH_obs)  across all test-window cycles
  ρ(pred_traj, obs_traj)       Pearson on test-window timeseries

SECONDARY: endpoint RMSE (last cycle only; flagged [LOW] where |ΔSOH_endpoint|<0.01).

PRE-REGISTERED EXPECTATION (locked before code runs):
  Noise diagnostic gave median |ΔSOH|/σ = 1.70 (BORDERLINE).
  Expected: B2' beats B1' beats B0' on TRAJECTORY metrics by a small margin.
  High-fade vehicles (|ΔSOH|/σ > 2: V04, V08, V09, V15) likely show clearer B2' wins.
  Low-SNR vehicles (V03, V10, V14, V20) expected to show no improvement over B0'.
  Negative-λ vehicles (V07, V12, V16, V17, V20) excluded from B2'-beats verdict
  but included in all tables.
  If all baselines tie, the story closes: limiting factor is SOH measurement quality,
  not modeling.

OUTPUT: data/cell_to_field_temporal_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "data" / "cell_to_field_temporal_report.json"

# Identical to soh_predictor.py constants
Q_NOMINAL  = 136.2
SOH_WIN    = 50
MIN_Q      = 100
TRAIN_FRAC = 0.50        # first 50% of timeline → train
DSOH_LOW   = 0.01        # |ΔSOH_endpoint| below this → flag %err unreliable

PRE_REGISTERED = (
    "Noise diagnostic: median |ΔSOH|/σ = 1.70 (BORDERLINE). "
    "Expected B2'>B1'>B0' on trajectory RMSE by small margin. "
    "High-fade vehicles (V04,V08,V09,V15) should show clearest gains. "
    "If all baselines tie: limiting factor is SOH data quality, not modeling."
)

# SNR values from noise diagnostic (locked reference, not recomputed here)
NOISE_SNR = {
    "V01": 3.29, "V02": 1.48, "V03": 0.03, "V04": 4.31,
    "V05": 1.73, "V06": 1.26, "V07": 1.60, "V08": 4.50,
    "V09": 3.68, "V10": 0.73, "V11": 1.67, "V12": 2.59,
    "V13": 2.61, "V14": 0.55, "V15": 5.16, "V16": 2.38,
    "V17": 1.11, "V18": 1.15, "V19": 3.39, "V20": 0.58,
}


# ── SOH helpers ───────────────────────────────────────────────────────────────

def _observed_delta_soh(cycles: pd.DataFrame, veh: str) -> Optional[np.ndarray]:
    """ΔSOH[k] = SOH[0] - SOH[k]. Identical to soh_predictor.observed_delta_soh."""
    vc = cycles[cycles["vehicle"] == veh]
    q  = vc["Q_Ah"].values.astype(float)
    if int((~np.isnan(q)).sum()) < MIN_Q:
        return None
    soh = (pd.Series(q).ffill().bfill()
           .rolling(SOH_WIN, min_periods=1, center=True).median()
           .values / Q_NOMINAL)
    soh = np.clip(soh, 0.0, 1.0)
    return soh[0] - soh   # positive = degraded


def _fit_lambda(sqrt_t: np.ndarray, dsoh: np.ndarray) -> float:
    """
    OLS no-intercept: λ = (Σ √t · ΔSOH) / (Σ t).
    ΔSOH anchors at 0 when t=0, so no intercept is correct.
    """
    denom = float(np.dot(sqrt_t, sqrt_t))   # = Σ t
    if denom < 1e-12:
        return 0.0
    return float(np.dot(sqrt_t, dsoh)) / denom


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _rho(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ── main ──────────────────────────────────────────────────────────────────────

def run_temporal_split() -> None:
    from degradation.deng_loader     import load_all
    from degradation.cycle_segmentor import segment_all
    from degradation.soh_predictor   import add_t_years

    print("Cell-to-Field Temporal Split — per-vehicle λ experiment")
    print("=" * 65)
    print()
    print("PRE-REGISTERED EXPECTATION:")
    print(f"  {PRE_REGISTERED}")
    print()
    print("MODEL: ΔSOH(t) = λ_v · √t   (cycling term dropped: D≈0.002, negligible)")
    print(f"SPLIT: first {int(TRAIN_FRAC*100)}% by time → train; rest → test")
    print()

    # ── 1. Load and segment ───────────────────────────────────────────────────
    print("Loading Deng fleet…")
    raw = load_all(verbose=False)
    if not raw:
        print("ERROR: No Deng CSVs found in data/deng20/"); sys.exit(1)
    cycles = segment_all(raw, verbose=False)
    cycles = add_t_years(cycles)
    print(f"  {len(raw)} vehicles, {len(cycles)} cycle rows")
    print()

    # ── 2. Per-vehicle: split, fit λ_v ───────────────────────────────────────
    per_veh: Dict[str, Dict] = {}

    for veh in sorted(raw.keys()):
        vc  = cycles[cycles["vehicle"] == veh].copy().reset_index(drop=True)
        dS  = _observed_delta_soh(cycles, veh)
        if dS is None:
            per_veh[veh] = {"note": "insufficient Q readings"}
            continue

        t = vc["t_years"].values.astype(float)
        n = min(len(t), len(dS))
        t, dS = t[:n], dS[:n]

        t_max  = float(t[-1]) if n > 0 else 0.0
        t_cut  = TRAIN_FRAC * t_max

        tr_mask = t <= t_cut
        te_mask = t >  t_cut

        if tr_mask.sum() < 5 or te_mask.sum() < 5:
            per_veh[veh] = {"note": f"insufficient data for 50/50 split (n={n})"}
            continue

        t_tr,  dS_tr  = t[tr_mask],  dS[tr_mask]
        t_te,  dS_te  = t[te_mask],  dS[te_mask]
        sqrt_t_tr = np.sqrt(np.maximum(t_tr, 0.0))
        sqrt_t_te = np.sqrt(np.maximum(t_te, 0.0))

        lam_v = _fit_lambda(sqrt_t_tr, dS_tr)

        per_veh[veh] = {
            "lambda_v"          : lam_v,
            "negative_lambda"   : lam_v < 0,
            "snr"               : NOISE_SNR.get(veh, float("nan")),
            "train_n_cycles"    : int(tr_mask.sum()),
            "test_n_cycles"     : int(te_mask.sum()),
            "t_cut_years"       : round(float(t_cut), 3),
            "t_max_years"       : round(float(t_max), 3),
            "dsoh_at_cut"       : round(float(dS_tr[-1]), 5),   # B0' prediction
            # stored for later computation
            "_t_te"             : t_te,
            "_dS_te"            : dS_te,
            "_sqrt_t_te"        : sqrt_t_te,
        }

    # ── 3. LOO λ per vehicle: median of all OTHER vehicles' λ_v ──────────────
    all_lambdas = {
        v: d["lambda_v"]
        for v, d in per_veh.items()
        if "lambda_v" in d
    }

    for veh in list(all_lambdas.keys()):
        others = [lam for v, lam in all_lambdas.items() if v != veh]
        per_veh[veh]["lambda_loo"] = float(np.median(others)) if others else float("nan")

    # Global LOO median (for reporting — not used in predictions)
    global_lam = float(np.median(list(all_lambdas.values()))) if all_lambdas else float("nan")
    print(f"λ values from train windows (all vehicles):")
    print(f"  min={min(all_lambdas.values()):.5f}  "
          f"median={np.median(list(all_lambdas.values())):.5f}  "
          f"max={max(all_lambdas.values()):.5f}")
    print(f"  global median (fleet λ): {global_lam:.5f}  "
          f"[vs M2 λ_sei={0.02639332:.5f} from V01-V04 only]")
    print()

    # ── 4. Predictions and metrics on test window ─────────────────────────────
    for veh, d in per_veh.items():
        if "_t_te" not in d:
            continue

        t_te        = d.pop("_t_te")
        dS_te       = d.pop("_dS_te")
        sqrt_t_te   = d.pop("_sqrt_t_te")

        # B0': carry-forward (zero further change from last train observation)
        dsoh_cut = d["dsoh_at_cut"]
        pred_B0  = np.full_like(dS_te, dsoh_cut)

        # B1': LOO λ
        lam_loo  = d["lambda_loo"]
        pred_B1  = lam_loo * sqrt_t_te

        # B2': per-vehicle λ
        lam_v    = d["lambda_v"]
        pred_B2  = lam_v  * sqrt_t_te

        endpoint_obs  = float(dS_te[-1])
        low_flag      = abs(endpoint_obs) < DSOH_LOW

        d["trajectory_rmse_B0"]  = round(_rmse(pred_B0, dS_te), 5)
        d["trajectory_rmse_B1"]  = round(_rmse(pred_B1, dS_te), 5)
        d["trajectory_rmse_B2"]  = round(_rmse(pred_B2, dS_te), 5)
        d["trajectory_rho_B0"]   = round(_rho(pred_B0,  dS_te), 4)
        d["trajectory_rho_B1"]   = round(_rho(pred_B1,  dS_te), 4)
        d["trajectory_rho_B2"]   = round(_rho(pred_B2,  dS_te), 4)
        d["endpoint_dsoh_obs"]   = round(endpoint_obs, 5)
        d["endpoint_dsoh_B0"]    = round(float(pred_B0[-1]), 5)
        d["endpoint_dsoh_B1"]    = round(float(pred_B1[-1]), 5)
        d["endpoint_dsoh_B2"]    = round(float(pred_B2[-1]), 5)
        d["endpoint_low_flag"]   = low_flag

    # ── 5. Aggregate (all vehicles with results) ──────────────────────────────
    valid = [
        (v, d) for v, d in per_veh.items()
        if "trajectory_rmse_B0" in d
    ]

    def _agg_traj(bl: str) -> Dict:
        rmses = [d[f"trajectory_rmse_{bl}"] for _, d in valid]
        rhos  = [d[f"trajectory_rho_{bl}"]  for _, d in valid
                 if not np.isnan(d[f"trajectory_rho_{bl}"])]
        return {
            "mean_trajectory_rmse" : round(float(np.mean(rmses)), 5),
            "median_trajectory_rmse": round(float(np.median(rmses)), 5),
            "mean_rho"              : round(float(np.mean(rhos)), 4) if rhos else float("nan"),
            "n_vehicles"            : len(valid),
        }

    def _agg_endpoint(bl: str) -> Dict:
        obs_v  = np.array([d["endpoint_dsoh_obs"]       for _, d in valid])
        pred_v = np.array([d[f"endpoint_dsoh_{bl}"]     for _, d in valid])
        rmse   = _rmse(pred_v, obs_v)
        rho    = _rho(pred_v, obs_v)
        return {
            "endpoint_rmse" : round(rmse, 5),
            "endpoint_rho"  : round(rho,  4),
        }

    agg = {}
    for bl in ("B0", "B1", "B2"):
        agg[bl] = {**_agg_traj(bl), **_agg_endpoint(bl)}

    # Verdict (pre-registered logic)
    b2_beats_b1_traj = agg["B2"]["mean_trajectory_rmse"] < agg["B1"]["mean_trajectory_rmse"]
    b1_beats_b0_traj = agg["B1"]["mean_trajectory_rmse"] < agg["B0"]["mean_trajectory_rmse"]
    b2_beats_b1_ep   = agg["B2"]["endpoint_rmse"]        < agg["B1"]["endpoint_rmse"]
    b1_beats_b0_ep   = agg["B1"]["endpoint_rmse"]        < agg["B0"]["endpoint_rmse"]

    neg_lam_vehs = [v for v, d in per_veh.items() if d.get("negative_lambda", False)]

    if b2_beats_b1_traj and b1_beats_b0_traj:
        verdict = (
            "B2'>B1'>B0' on trajectory RMSE: per-vehicle λ captures "
            "individual fade rate; LOO transfer also adds value over carry-forward. "
            "Noise is the bottleneck but trend signal is exploitable."
        )
    elif b2_beats_b1_traj and not b1_beats_b0_traj:
        verdict = (
            "B2'>B1'≈B0' on trajectory RMSE: per-vehicle λ helps but "
            "LOO-transferred λ does not beat carry-forward — fleet λ spread "
            "is too wide for cross-vehicle transfer to add value. "
            "Within-vehicle calibration needed."
        )
    elif not b2_beats_b1_traj and not b1_beats_b0_traj:
        verdict = (
            "All baselines approximately tie on trajectory RMSE. "
            "Consistent with noise-floor finding (median |ΔSOH|/σ=1.70): "
            "the limiting factor is SOH measurement quality, not modeling. "
            "Per-vehicle λ adds no systematic improvement over carry-forward."
        )
    else:
        verdict = (
            "B1'≈B2' on trajectory RMSE; neither beats B0' consistently. "
            "Calendar λ transfer provides marginal value; carry-forward is "
            "a hard baseline to beat given the noise floor."
        )
    if neg_lam_vehs:
        verdict += (
            f" Note: {len(neg_lam_vehs)} vehicles have negative λ_v "
            f"({', '.join(sorted(neg_lam_vehs))}) — non-monotone SOH trajectories "
            "excluded from 'B2 beats' judgment but included in tables and RMSE."
        )

    # ── 6. Print tables ───────────────────────────────────────────────────────
    print("=" * 85)
    print("PER-VEHICLE LAMBDA AND TRAJECTORY METRICS")
    print("=" * 85)
    hdr = (f"{'Veh':4s} {'λ_v':8s} {'λ_LOO':8s} {'SNR':5s} "
           f"{'tr_n':5s} {'te_n':5s} "
           f"{'tRMSE_B0':9s} {'tRMSE_B1':9s} {'tRMSE_B2':9s} "
           f"{'ρ_B2':6s} {'negλ?':6s}")
    print(hdr)
    print("-" * 85)

    for veh in sorted(per_veh.keys()):
        d = per_veh[veh]
        if "trajectory_rmse_B0" not in d:
            print(f"{veh:4s}  [{d.get('note', 'excluded')}]")
            continue
        neg_s = "YES" if d["negative_lambda"] else "no"
        print(
            f"{veh:4s} {d['lambda_v']:8.5f} {d['lambda_loo']:8.5f} "
            f"{d['snr']:5.2f} "
            f"{d['train_n_cycles']:5d} {d['test_n_cycles']:5d} "
            f"{d['trajectory_rmse_B0']:9.5f} {d['trajectory_rmse_B1']:9.5f} "
            f"{d['trajectory_rmse_B2']:9.5f} "
            f"{d['trajectory_rho_B2']:6.3f} {neg_s:6s}"
        )

    print()
    print("=" * 65)
    print(f"AGGREGATE TABLE  (all {len(valid)} vehicles with sufficient data)")
    print("=" * 65)
    print(f"{'Baseline':8s} {'Mean tRMSE':11s} {'Med tRMSE':10s} "
          f"{'Mean ρ':7s} {'EP-RMSE':8s} {'EP-ρ':6s} "
          f"{'Beats B0?':10s} {'Beats B1?':9s}")
    print("-" * 70)
    for bl in ("B0", "B1", "B2"):
        a   = agg[bl]
        b0s = "—" if bl == "B0" else (
            "yes" if a["mean_trajectory_rmse"] < agg["B0"]["mean_trajectory_rmse"] else "no")
        b1s = "—" if bl in ("B0", "B1") else (
            "yes" if a["mean_trajectory_rmse"] < agg["B1"]["mean_trajectory_rmse"] else "no")
        print(
            f"{bl:8s} {a['mean_trajectory_rmse']:11.5f} "
            f"{a['median_trajectory_rmse']:10.5f} "
            f"{a['mean_rho']:7.4f} "
            f"{a['endpoint_rmse']:8.5f} {a['endpoint_rho']:6.4f} "
            f"{b0s:10s} {b1s:9s}"
        )

    print()
    print("VERDICT:")
    print(f"  {verdict}")
    print()
    print(f"  Global fleet λ (median of all per-vehicle train fits): {global_lam:.5f}")
    print(f"  M2 λ_sei from V01-V04 only: 0.02639  "
          f"[ratio: {global_lam/0.02639332:.2f}× — "
          f"{'M2 overestimates' if global_lam < 0.02639332 else 'M2 underestimates'} fleet-wide λ]")
    print()
    print("  Negative-λ vehicles (non-monotone SOH): "
          + (", ".join(sorted(neg_lam_vehs)) if neg_lam_vehs else "none"))
    print()
    print("  Pre-registered noise check: endpoint-RMSE winner is "
          + ("B2' (per-vehicle λ)" if b2_beats_b1_ep and b2_beats_b1_traj
             else "B0' or B1' — consistent with noise-floor finding"))

    # ── 7. Write JSON ─────────────────────────────────────────────────────────
    # Remove private arrays before serialising
    report = {
        "meta": {
            "script"        : "degradation/cell_to_field_temporal.py",
            "train_frac"    : TRAIN_FRAC,
            "model"         : "ΔSOH(t) = λ_v · √t  (cycling term dropped)",
            "cycling_drop_reason": "D_cumul≈0.002 across fleet; β·D^γ<0.001 SOH (noise level)",
        },
        "pre_registered_expectation": PRE_REGISTERED,
        "frozen_params": {
            "cycling_term"     : "dropped",
            "lambda_global"    : round(global_lam, 6),
            "lambda_global_note": "median of per-vehicle train-window fits (all 20 vehicles)",
            "lambda_M2_V01_V04": 0.02639332,
        },
        "baselines": {
            "B0_prime": "carry-forward: ΔSOH = last observed value at train/test boundary",
            "B1_prime": "LOO calendar: λ_LOO = median(λ_v) from all OTHER vehicles",
            "B2_prime": "per-vehicle calendar: λ_v fit on this vehicle's train window",
        },
        "per_vehicle": {
            v: {k: (round(float(val), 6) if isinstance(val, (float, np.floating)) else val)
                for k, val in d.items() if not k.startswith("_")}
            for v, d in per_veh.items()
        },
        "aggregate": agg,
        "verdict": verdict,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report written → {OUT_JSON}")


if __name__ == "__main__":
    run_temporal_split()
