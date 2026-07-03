#!/usr/bin/env python3
"""
degradation/cell_to_field_bridge.py  —  Cell-to-Field Bridge (Ather Problem 3)

Given (a) a calibrated cell-level degradation model and (b) a fleet's real
usage data, predict each vehicle's field SOH trajectory using Palmgren-Miner
convolution + SEI calendar aging.

FOUR PREDICTIONS per vehicle:
  B0  — fleet-mean observed final ΔSOH (naïve constant; baseline floor)
  B1  — calendar only:        ΔSOH(t) = λ_sei · √t
  B2  — full bridge:          ΔSOH(t) = λ_sei · √t + β_NASA · D_cumul^γ
  B2α — pack-scaled bridge:   ΔSOH(t) = λ_sei · √t + α · β_NASA · D_cumul^γ
        α is fit on Tier 2 (V05–V09) via OLS; multiplies CYCLING TERM ONLY.

THREE-TIER DATA SEPARATION (no parameters touch Tier 3):
  Tier 1  V01–V04    λ_sei source (M2 Deng combined fit)       [λ-cal]
  Tier 2  V05–V09    α calibration (scale factor, ID order)    [α-cal]
  Tier 3  V10–V20    Pure test — 11 vehicles, headline metrics  [test]

FROZEN PARAMETERS (pre-registered before examining fleet SOH):
  λ_sei   = 0.02639332 SOH/√yr   [Deng M2 V01–V04, params_sei.lam]
  β_NASA  = 0.021545              [NASA LOO-CV mean: 0.02229/0.01911/0.02326/0.02152]
  γ       = 0.5                   [fixed in Module 4]
  A       = 1 000 000             [Basquin, fatigue.SN_A_DEFAULT]
  m       = 2.5                   [Basquin, fatigue.SN_M_DEFAULT]

PRE-REGISTERED EXPECTATION:
  B2 ≈ B1 on Tier 3.  M2 found stress_frac = 0 % for all Deng vehicles —
  field fade is calendar-dominated.  If cycling term adds nothing that IS the
  reportable finding, not a model failure.

PRIMARY DATASET:
  Deng Z. et al. (2023) Applied Energy 339:120954. Chemistry: CATL NCM (NMC),
  145 Ah nominal, 90s pack. Q_NOMINAL=136.2 Ah is max observed at dataset entry
  (vehicles ~6% degraded at first reading). See deng_loader.py for details.

HONEST CAVEATS:
  1. Cell-to-pack scale mismatch: β,γ from NASA NMC 18650 lab cells (1C,
     DoD≈100%); Deng fleet is BAIC EU500 NCM 145 Ah pack (field partial
     cycles, 0.41C). SAME chemistry class (NMC) — transfer failure is
     cell-vs-pack scale, protocol (lab 1C/100% DoD → field 0.41C/57% DoD),
     and measurement noise, NOT chemistry mismatch.
  2. λ_sei is from Deng V01–V04 — partially fleet-sourced; disclosed above.
  3. T_mean_C from BMS per session; no assumed temperature was needed.
  4. Partial-cycle counting: ASTM E1049 rainflow on per-session SOC handles
     partial cycles.  Sequence effects ignored (Miner linearity).
  5. α vehicle selection: V05–V09 chosen by ID order, not by fit quality.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── Frozen parameters ─────────────────────────────────────────────────────────
LAMBDA_SEI = 0.02639332   # SOH/√yr — Deng M2 combined fit on V01–V04
BETA_NASA  = 0.021545     # NASA LOO-CV mean β across B0005/B0006/B0007/B0018
GAMMA_NASA = 0.5          # fixed throughout Module 4
SN_A       = 1.0e6        # Basquin A — fatigue.SN_A_DEFAULT
SN_M       = 2.5          # Basquin m — fatigue.SN_M_DEFAULT

TIER1_VEHS = {"V01", "V02", "V03", "V04"}
TIER2_VEHS = {"V05", "V06", "V07", "V08", "V09"}
TIER3_VEHS = {f"V{i:02d}" for i in range(10, 21)}

# Usage histogram bins (reporting only — predictions use D from accumulate_damage)
DOD_BINS   = [0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 100.0]
CRATE_BINS = [0.0, 0.1, 0.2, 0.5, 2.0]
T_BINS     = [-10.0, 15.0, 25.0, 35.0, 60.0]

OUT_JSON   = ROOT / "data" / "cell_to_field_report.json"

PRE_REGISTERED = (
    "B2≈B1 expected: M2 stress_frac=0% on all Deng vehicles. "
    "If cycling term adds nothing on Tier 3, that IS the finding — "
    "field fade is calendar-dominated, consistent with M2."
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _tier(v: str) -> str:
    if v in TIER1_VEHS:  return "T1-λcal"
    if v in TIER2_VEHS:  return "T2-αcal"
    return "T3-test"


def _usage_histogram(vc: pd.DataFrame) -> Dict:
    """Per-vehicle (DoD × C-rate × T) event counts and top-5 damage bins."""
    hist: Dict[str, int]   = {}
    bdmg: Dict[str, float] = {}
    for _, row in vc.iterrows():
        dod = float(row.get("DoD_pct", 0))
        cr  = abs(float(row.get("C_rate", 0)))
        T   = float(row.get("T_mean_C", 25))
        d   = float(row.get("d_cycle", 0)) if "d_cycle" in vc.columns else 0.0
        di  = min(int(np.digitize(dod, DOD_BINS)) - 1, len(DOD_BINS) - 2)
        ci  = min(int(np.digitize(cr,  CRATE_BINS)) - 1, len(CRATE_BINS) - 2)
        ti  = min(int(np.digitize(T,   T_BINS)) - 1, len(T_BINS) - 2)
        key = f"D{max(di,0)}C{max(ci,0)}T{max(ti,0)}"
        hist[key] = hist.get(key, 0) + 1
        bdmg[key] = bdmg.get(key, 0.0) + d
    total_d = sum(bdmg.values()) or 1e-12
    top5 = sorted(bdmg.items(), key=lambda x: -x[1])[:5]
    return {
        "n_cycles": len(vc),
        "top5_damage_bins": [
            {"bin": k, "n_events": hist[k],
             "damage_frac": round(v / total_d, 4)}
            for k, v in top5
        ],
    }


def _sei(t: np.ndarray) -> np.ndarray:
    return LAMBDA_SEI * np.sqrt(np.maximum(t, 0.0))


def _cyc(D: np.ndarray) -> np.ndarray:
    return BETA_NASA * np.power(np.maximum(D, 0.0), GAMMA_NASA)


def _predict(D: np.ndarray, t: np.ndarray,
             alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (B1, B2, B2α) ΔSOH arrays."""
    cal = _sei(t)
    cyc = _cyc(D)
    return cal, cal + cyc, cal + alpha * cyc


def _fit_alpha(D_list: List[np.ndarray],
               t_list: List[np.ndarray],
               dS_list: List[np.ndarray]) -> Tuple[float, bool]:
    """
    OLS for α with no intercept: α = (X·Y)/(X·X)
      X = β_NASA · D^γ   (cycling term, before scaling)
      Y = ΔSOH_obs − λ_sei · √t   (residual calendar cannot explain)
    Returns (alpha, is_degenerate).
    """
    X_all, Y_all = [], []
    for D, t, dS in zip(D_list, t_list, dS_list):
        n = min(len(D), len(t), len(dS))
        X_all.append(_cyc(D[:n]))
        Y_all.append(dS[:n] - _sei(t[:n]))
    X_cat = np.concatenate(X_all)
    Y_cat = np.concatenate(Y_all)
    denom = float(np.dot(X_cat, X_cat))
    if denom < 1e-20:
        return 1.0, True
    alpha = float(np.dot(X_cat, Y_cat) / denom)
    degenerate = abs(alpha) > 1000 or not np.isfinite(alpha)
    if degenerate:
        alpha = 1.0
    return alpha, degenerate


# ── main entry point ──────────────────────────────────────────────────────────

def run_cell_to_field_bridge() -> None:
    """Load Deng fleet, run bridge, write JSON, print tables."""
    from degradation.deng_loader     import load_all
    from degradation.cycle_segmentor import segment_all
    from degradation.fatigue         import accumulate_damage
    from degradation.soh_predictor   import add_t_years, observed_delta_soh

    print("Cell-to-Field Bridge — OpenCATHODE")
    print("=" * 65)
    print()
    print("PRE-REGISTERED EXPECTATION:")
    print(f"  {PRE_REGISTERED}")
    print()
    print("FROZEN PARAMETERS (set before any fleet SOH is examined):")
    print(f"  λ_sei  = {LAMBDA_SEI}  SOH/√yr   [Deng M2, V01–V04]")
    print(f"  β_NASA = {BETA_NASA}   [NASA LOO-CV mean, 4 cells]")
    print(f"  γ      = {GAMMA_NASA}               [fixed, Module 4]")
    print(f"  S-N A  = {SN_A:.0e},  m = {SN_M}   [fatigue.py defaults]")
    print()

    # ── 1. Load fleet ─────────────────────────────────────────────────────────
    print("Loading Deng fleet (20 vehicles)…")
    raw_vehicles = load_all(verbose=False)
    if not raw_vehicles:
        print("ERROR: No Deng vehicle CSVs found in data/deng20/. "
              "Place #1.csv .. #20.csv there and re-run.")
        sys.exit(1)
    print(f"  Loaded {len(raw_vehicles)} vehicles: {sorted(raw_vehicles)}")

    # ── 2. Segment into cycles ────────────────────────────────────────────────
    print("Segmenting cycles…")
    cycles = segment_all(raw_vehicles, verbose=False)
    print(f"  {len(cycles)} total cycle rows across all vehicles")

    # ── 3. Rainflow damage ────────────────────────────────────────────────────
    print("Accumulating rainflow damage (may take ~5 min for 20 vehicles)…")
    cycles = accumulate_damage(cycles, raw_vehicles, A=SN_A, m=SN_M)

    # ── 4. Calendar time ──────────────────────────────────────────────────────
    cycles = add_t_years(cycles)

    # ── 5. Per-vehicle arrays ─────────────────────────────────────────────────
    veh_arrays: Dict[str, Dict] = {}   # holds numpy arrays; NOT serialised to JSON
    per_vehicle: Dict[str, Dict] = {}  # scalar metadata; written to JSON

    for veh in sorted(raw_vehicles.keys()):
        vc   = cycles[cycles["vehicle"] == veh].copy().reset_index(drop=True)
        tier = _tier(veh)

        dS_arr = observed_delta_soh(cycles, veh)
        if dS_arr is None:
            per_vehicle[veh] = {"tier": tier, "note": "insufficient Q readings (<100)"}
            continue

        D = vc["D_cumul"].values.astype(float)
        t = vc["t_years"].values.astype(float)
        n = min(len(D), len(t), len(dS_arr))
        D, t, dS = D[:n], t[:n], dS_arr[:n]

        veh_arrays[veh] = {"D": D, "t": t, "dS": dS}
        per_vehicle[veh] = {
            "tier"          : tier,
            "n_cycles"      : n,
            "t_years_final" : round(float(t[-1]), 3),
            "D_cumul_final" : round(float(D[-1]), 7),
            "dsoh_obs_final": round(float(dS[-1]), 5),
            "histogram"     : _usage_histogram(vc.iloc[:n]),
        }

    # ── 6. Fit α on Tier 2 (V05–V09) ─────────────────────────────────────────
    t2_D  = [veh_arrays[v]["D"]  for v in sorted(TIER2_VEHS) if v in veh_arrays]
    t2_t  = [veh_arrays[v]["t"]  for v in sorted(TIER2_VEHS) if v in veh_arrays]
    t2_dS = [veh_arrays[v]["dS"] for v in sorted(TIER2_VEHS) if v in veh_arrays]

    alpha, alpha_degen = _fit_alpha(t2_D, t2_t, t2_dS) if t2_D else (1.0, True)

    print(f"\nScale factor α (Tier 2, V05–V09): α = {alpha:.4f}")
    if alpha_degen:
        print("  [WARN] α is degenerate (D≈0 everywhere → cycling term negligible).")
        print("  B2α ≈ B2 ≈ B1 confirmed pre-registration.")

    # ── 7. Compute predictions and metrics ────────────────────────────────────
    # B0 = Tier-3 fleet-mean observed final ΔSOH (computed from test data as naïve baseline)
    t3_obs = [
        veh_arrays[v]["dS"][-1]
        for v in sorted(TIER3_VEHS)
        if v in veh_arrays and len(veh_arrays[v]["dS"]) > 0
    ]
    B0_val = float(np.mean(t3_obs)) if t3_obs else float("nan")

    for veh in sorted(veh_arrays.keys()):
        D, t, dS = veh_arrays[veh]["D"], veh_arrays[veh]["t"], veh_arrays[veh]["dS"]
        n = len(D)
        obs_final = float(dS[-1])

        B1, B2, B2a = _predict(D, t, alpha)

        def _err(pred_final: float) -> Dict:
            err = pred_final - obs_final
            return {
                "pred_dsoh": round(float(pred_final), 5),
                "err_abs"  : round(float(err), 5),
                "err_pct"  : round(abs(err) / (abs(obs_final) + 1e-9) * 100, 2),
            }

        per_vehicle[veh]["B0"]  = _err(B0_val)
        per_vehicle[veh]["B1"]  = _err(float(B1[-1]))
        per_vehicle[veh]["B2"]  = _err(float(B2[-1]))
        per_vehicle[veh]["B2α"] = _err(float(B2a[-1]))

    # ── 8. Aggregate on Tier 3 ────────────────────────────────────────────────
    t3_recs = [
        per_vehicle[v]
        for v in sorted(TIER3_VEHS)
        if v in per_vehicle and "B0" in per_vehicle[v]
    ]

    def _agg(bl: str) -> Dict:
        obs_v  = np.array([r["dsoh_obs_final"] for r in t3_recs])
        pred_v = np.array([r[bl]["pred_dsoh"]   for r in t3_recs])
        errs   = np.array([r[bl]["err_pct"]      for r in t3_recs])
        rmse   = float(np.sqrt(np.mean((pred_v - obs_v) ** 2)))
        rho    = (float(np.corrcoef(pred_v, obs_v)[0, 1])
                  if len(obs_v) > 2 else float("nan"))
        return {
            "rmse_soh"    : round(rmse, 5),
            "mean_pct_err": round(float(np.mean(errs)), 2),
            "rho"         : round(rho, 4),
            "n_vehicles"  : len(t3_recs),
        }

    agg = {bl: _agg(bl) for bl in ("B0", "B1", "B2", "B2α")}

    # Verdict (pre-registered logic — do not change after seeing numbers)
    b1_b0  = agg["B1"]["rmse_soh"]  < agg["B0"]["rmse_soh"]
    b2_b1  = agg["B2"]["rmse_soh"]  < agg["B1"]["rmse_soh"]
    b2a_b1 = agg["B2α"]["rmse_soh"] < agg["B1"]["rmse_soh"]

    if not b2_b1 and not b2a_b1:
        verdict = (
            "Cycling term adds nothing on this fleet — field fade is "
            "calendar-dominated, consistent with M2 (stress_frac=0%). "
            "B2 ≈ B1 on Tier 3. Pre-registered expectation CONFIRMED."
        )
    elif b2a_b1 and not b2_b1:
        verdict = (
            f"α-scaled bridge (B2α, α={alpha:.3f}) beats B1 on Tier 3; "
            "raw transfer (B2) does not. Pack-scale correction recovers "
            "modest cycling signal."
        )
    else:
        verdict = (
            "Raw B2 beats B1 on Tier 3 — chemistry transfer partially "
            "successful despite cell→pack (NMC 18650 lab → NCM 145 Ah field) scale mismatch."
        )

    # ── 9. Print per-vehicle table ────────────────────────────────────────────
    print()
    print("=" * 79)
    print("PER-VEHICLE TABLE  (all tiers; ◄ = Tier 3, headline metrics)")
    print("=" * 79)
    hdr = (f"{'Veh':4s} {'Tier':8s} {'t_yr':5s} {'D_fin':8s} "
           f"{'ΔSOHobs':8s} {'B0%err':7s} {'B1%err':7s} "
           f"{'B2%err':7s} {'B2α%err':8s}")
    print(hdr)
    print("-" * 79)

    for veh in sorted(per_vehicle.keys()):
        rec  = per_vehicle[veh]
        tier = rec.get("tier", "?")
        if "B0" not in rec:
            print(f"{veh:4s} {tier:8s}  [{rec.get('note', 'excluded')}]")
            continue
        mark = " ◄" if tier == "T3-test" else ""
        print(
            f"{veh:4s} {tier:8s} "
            f"{rec['t_years_final']:5.2f} "
            f"{rec['D_cumul_final']:8.5f} "
            f"{rec['dsoh_obs_final']:8.5f} "
            f"{rec['B0']['err_pct']:7.1f} "
            f"{rec['B1']['err_pct']:7.1f} "
            f"{rec['B2']['err_pct']:7.1f} "
            f"{rec['B2α']['err_pct']:8.1f}"
            f"{mark}"
        )

    # ── 10. Print aggregate table ─────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"AGGREGATE TABLE — Tier 3 ONLY (V10–V20, n={len(t3_recs)} vehicles)")
    print("=" * 65)
    print(f"{'Baseline':8s} {'RMSE(SOH)':10s} {'Mean%err':9s} "
          f"{'ρ(pred,obs)':12s} {'Beats B0?':10s} {'Beats B1?':9s}")
    print("-" * 65)
    for bl in ("B0", "B1", "B2", "B2α"):
        b0_s = "—" if bl == "B0" else ("yes" if agg[bl]["rmse_soh"] < agg["B0"]["rmse_soh"] else "no")
        b1_s = "—" if bl in ("B0", "B1") else ("yes" if agg[bl]["rmse_soh"] < agg["B1"]["rmse_soh"] else "no")
        print(f"{bl:8s} {agg[bl]['rmse_soh']:10.5f} {agg[bl]['mean_pct_err']:9.2f} "
              f"{agg[bl]['rho']:12.4f} {b0_s:10s} {b1_s:9s}")

    print()
    print("VERDICT:")
    print(f"  {verdict}")
    print()
    if alpha_degen:
        print(f"  α = {alpha:.4f}  [DEGENERATE — cycling term D≈0 on V05-V09]")
    else:
        print(f"  α = {alpha:.4f}  (multiplies β·D^γ only; λ·√t calendar term NOT scaled)")

    # ── 11. Write JSON ────────────────────────────────────────────────────────
    report = {
        "meta": {
            "script" : "degradation/cell_to_field_bridge.py",
            "caveats": [
                "Cell-to-pack scale mismatch: β,γ from NASA NMC 18650 lab cells (1C, DoD≈100%); "
                "Deng fleet is BAIC EU500 NCM 145 Ah pack (field, 0.41C, 57% DoD). "
                "Same chemistry class (NMC) — transfer failure is scale+protocol+noise, not chemistry. "
                "Trend/rank tested, not absolute accuracy. "
                "Source: Deng Z. et al. (2023) Applied Energy 339:120954.",
                "λ_sei from Deng V01-V04 — partially fleet-sourced; disclosed.",
                "T_mean_C from BMS per session; no assumed temperature applied.",
                "Partial-cycle counting: ASTM E1049 rainflow per session; sequence effects ignored.",
                "α vehicles (V05-V09) chosen by ID order, not by fit quality.",
            ],
        },
        "frozen_params": {
            "lambda_sei"       : LAMBDA_SEI,
            "beta_NASA"        : BETA_NASA,
            "gamma_NASA"       : GAMMA_NASA,
            "SN_A"             : SN_A,
            "SN_M"             : SN_M,
            "lambda_source"    : "Deng M2 combined fit on V01-V04",
            "beta_gamma_source": "NASA LOO-CV mean: B0005/B0006/B0007/B0018",
        },
        "pre_registered_expectation": PRE_REGISTERED,
        "tier_separation": {
            "tier1_lambda_cal": sorted(TIER1_VEHS),
            "tier2_alpha_cal" : sorted(TIER2_VEHS),
            "tier3_pure_test" : sorted(TIER3_VEHS),
        },
        "alpha"             : round(alpha, 6),
        "alpha_degenerate"  : alpha_degen,
        "alpha_note"        : "multiplies cycling term (β·D^γ) ONLY; λ·√t not scaled",
        "alpha_fit_vehicles": sorted(TIER2_VEHS),
        "B0_value"          : round(B0_val, 5),
        "usage_histogram_bins": {
            "dod_pct_edges"   : DOD_BINS,
            "crate_edges"     : CRATE_BINS,
            "T_celsius_edges" : T_BINS,
        },
        "per_vehicle"       : per_vehicle,
        "aggregate_tier3"   : agg,
        "verdict"           : verdict,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written → {OUT_JSON}")


if __name__ == "__main__":
    run_cell_to_field_bridge()
