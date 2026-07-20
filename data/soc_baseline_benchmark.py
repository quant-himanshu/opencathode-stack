#!/usr/bin/env python3
"""
data/soc_baseline_benchmark.py
=================================
Apples-to-apples SOC-estimation benchmark: this project's Dual EKF vs two
GENERIC (chemistry/vendor-agnostic) baselines, on the SAME real held-out
fleet segments already used by validate_generic.py.

SCOPE NOTE (read before citing this anywhere): this does NOT benchmark
against TI's Impedance Track or ADI's ModelGauge m5 EZ. Those are patented,
closed-source commercial firmware; we do not have their exact algorithm or
their tuned calibration tables, and running their real silicon would require
physical eval-board hardware this project does not have. Naming them here
would misrepresent what is actually being tested. Instead this compares
against two textbook, unnamed baselines that represent the two extremes any
SOC estimator has to beat:

  1. PURE COULOMB COUNTING  -- integrate measured current from a true SOC
     anchor at segment start, apply ZERO voltage-based correction ever.
     This is the "what if you only counted amps" floor.
  2. PURE OCV LOOKUP        -- invert the fleet's own empirically-fitted
     OCV(SOC) curve against the RAW measured terminal voltage at every
     timestep, with NO filtering and NO IR-drop/dynamics compensation.
     This is the "what if you only trusted the voltmeter" floor.

Both baselines use the EXACT SAME held-out segments, SAME ground-truth SOC
(SOC_bms), SAME cell configuration (n_series/n_parallel/Q_cell_ah), and SAME
empirically-fitted OCV curve as this project's own Dual EKF (Mode B in
validate_generic.py) -- so the comparison is genuinely apples-to-apples on
data, even though it is NOT a comparison against any named commercial product.

Reuses validate_generic.py's existing loaders, calibration-vehicle split, and
run_mode_b_ekf() unmodified -- does not alter that file or any of its
existing published results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.validate_generic import (
    ValidationConfig, CellMode, config_from_cartridge,
    _split_by_vehicle, _build_calibration_for_fleet, run_mode_b_ekf,
)

OUT_JSON = ROOT / "data" / "soc_baseline_benchmark_report.json"
OUT_MD = ROOT / "docs" / "soc_baseline_benchmark.md"


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: pure coulomb counting
# ─────────────────────────────────────────────────────────────────────────────

def coulomb_counting_soc(seg_df: pd.DataFrame, cfg: ValidationConfig) -> np.ndarray:
    """
    SOC[0] = SOC_bms[0] + cfg.ekf_soc_offset -- the SAME deliberately-wrong
    starting condition run_mode_b_ekf() uses (validate_generic.py line ~291:
    soc_init_offset = soc_bms[0] + cfg.ekf_soc_offset). This is a FIX for an
    initial, biased version of this function that started coulomb counting
    from the TRUE SOC while the EKF was handicapped 20% away -- that gave
    coulomb counting a free unfair advantage and produced a misleading
    "coulomb counting beats EKF" result on the first run. Both methods must
    face the identical starting uncertainty for the comparison to mean
    anything. From this shared starting point: pure current integration,
    zero voltage-based correction ever.

    SIGN FIX 2026-07-20 (docs/SIGN_BUG_POSTMORTEM.md): the schema is
    DISCHARGE-NEGATIVE (common_schema.enforce_discharge_negative; verified
    empirically per segment on every dataset), so SOC evolves as
    soc0 + ∫I dt / (3600·Q): I < 0 while discharging lowers SOC. The
    original version used soc0 − ∫I dt (a discharge-positive formula,
    matching a wrong claim in this docstring), which INVERTED the coulomb
    baseline on every schema-conforming dataset (BMW/Deng/VED) while being
    accidentally correct on the two datasets whose loaders carried the
    opposite sign defect (CALCE/UMich). Both defects are now fixed and the
    convention is asserted at every load in make_schema_df.
    """
    t_s = seg_df["t_s"].values.astype(np.float64)
    I_cell = seg_df["I_A"].values.astype(np.float64) / cfg.n_parallel
    soc0 = float(np.clip(float(seg_df["SOC_bms"].iloc[0]) + cfg.ekf_soc_offset, 0.02, 0.98))

    dt = np.diff(t_s, prepend=t_s[0])
    dt[0] = 0.0
    charge_ah = np.cumsum(I_cell * dt) / 3600.0
    soc = soc0 + charge_ah / cfg.q_cell_ah   # discharge-negative schema
    return np.clip(soc, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: pure OCV lookup (naive, instantaneous, no filtering)
# ─────────────────────────────────────────────────────────────────────────────

def _invert_ocv(ocv_fn, n_grid: int = 400):
    """Numerically invert ocv_fn(soc)->V by dense sampling + interpolation.
    Returns v_to_soc(V) callable. V must be monotonic in ocv_fn for a clean
    inversion; ties/non-monotonicity are handled by sorting on V."""
    soc_grid = np.linspace(0.0, 1.0, n_grid)
    v_grid = np.array([ocv_fn(s) for s in soc_grid])
    order = np.argsort(v_grid)
    v_sorted = v_grid[order]
    soc_sorted = soc_grid[order]

    def v_to_soc(v):
        return np.interp(v, v_sorted, soc_sorted, left=soc_sorted[0], right=soc_sorted[-1])
    return v_to_soc


def ocv_lookup_soc(seg_df: pd.DataFrame, cfg: ValidationConfig, ocv_fn) -> np.ndarray:
    """
    Naive baseline: at every timestep, invert the RAW measured terminal
    voltage (which includes IR drop and RC dynamics, NOT true rest OCV)
    straight through the fleet's OCV(SOC) curve. No Kalman filtering, no
    current-based IR compensation, no smoothing -- deliberately naive, to
    show what "just read the voltmeter" gets you.
    """
    V_cell_meas = seg_df["V_V"].values.astype(np.float64) / cfg.n_series
    v_to_soc = _invert_ocv(ocv_fn)
    return np.clip(v_to_soc(V_cell_meas), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_segment(seg_df: pd.DataFrame, cfg: ValidationConfig,
                      calibration, ocv_fn) -> Dict:
    soc_bms = seg_df["SOC_bms"].values.astype(np.float64)

    result = {"n_rows": len(seg_df)}

    # Coulomb counting (always computable)
    soc_cc = coulomb_counting_soc(seg_df, cfg)
    result["soc_rmse_coulomb_pct"] = float(np.sqrt(np.mean((soc_cc - soc_bms) ** 2))) * 100.0

    # OCV lookup (needs a fitted ocv_fn from the calibration split)
    if ocv_fn is not None:
        try:
            soc_ocv = ocv_lookup_soc(seg_df, cfg, ocv_fn)
            result["soc_rmse_ocv_lookup_pct"] = float(np.sqrt(np.mean((soc_ocv - soc_bms) ** 2))) * 100.0
        except Exception as exc:
            result["soc_rmse_ocv_lookup_pct"] = None
            result["ocv_lookup_error"] = str(exc)
    else:
        result["soc_rmse_ocv_lookup_pct"] = None

    # EKF (this project's existing Mode B, unmodified)
    try:
        gamma = calibration.ekf_gamma if calibration is not None else 1.0
        R_meas_V2 = calibration.ekf_R_meas_V2 if calibration is not None else 4e-6
        soc_bms_out, soc_ekf, _, _ = run_mode_b_ekf(
            seg_df, cfg, ocv_fn=ocv_fn, calibration=calibration,
            gamma=gamma, R_meas_V2=R_meas_V2,
        )
        result["soc_rmse_ekf_pct"] = float(np.sqrt(np.mean((soc_ekf - soc_bms_out) ** 2))) * 100.0
    except Exception as exc:
        result["soc_rmse_ekf_pct"] = None
        result["ekf_error"] = str(exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-fleet runners (mirrors validate_generic.py's _run_* functions)
# ─────────────────────────────────────────────────────────────────────────────

def run_fleet_bmw_i3(max_trips=None) -> List[Dict]:
    from data.loaders.bmw_i3_loader import BMWI3Loader
    from data.loaders.pack_cartridge import BMW_I3_60AH

    loader = BMWI3Loader(max_trips=max_trips)
    all_pairs = list(loader.iter_segments())
    cal_pairs, eval_pairs = _split_by_vehicle(all_pairs)
    cfg = config_from_cartridge("BMW_i3", BMW_I3_60AH, CellMode.AVG_CELL)
    cal = _build_calibration_for_fleet(cal_pairs, cfg, "BMW_i3") if cal_pairs else None
    ocv_fn = cal.ocv_fn if cal else None

    results = []
    for idx, (seg_df, meta) in enumerate(eval_pairs):
        r = evaluate_segment(seg_df, cfg, cal, ocv_fn)
        r["vehicle_id"] = meta.vehicle_id
        results.append(r)
        if (idx + 1) % 10 == 0:
            print(f"  BMW i3: {idx + 1}/{len(eval_pairs)} held-out done")
    return results


def run_fleet_deng(max_vehicles=20, eval_sample_n=2000, rng_seed=42) -> List[Dict]:
    from data.loaders.deng_charging_loader import DengChargingLoader
    from data.loaders.pack_cartridge import BAIC_EU500_90S

    loader = DengChargingLoader(max_vehicles=max_vehicles)
    all_pairs = list(loader.iter_segments())
    cal_pairs, eval_pairs = _split_by_vehicle(all_pairs)
    cfg = config_from_cartridge("Deng_Charging", BAIC_EU500_90S, CellMode.AVG_CELL)
    cal = _build_calibration_for_fleet(cal_pairs, cfg, "Deng") if cal_pairs else None
    ocv_fn = cal.ocv_fn if cal else None

    rng = np.random.default_rng(rng_seed)
    if len(eval_pairs) > eval_sample_n:
        chosen = rng.choice(len(eval_pairs), size=eval_sample_n, replace=False)
        eval_sample = [eval_pairs[i] for i in sorted(chosen)]
    else:
        eval_sample = eval_pairs

    results = []
    for idx, (seg_df, meta) in enumerate(eval_sample):
        r = evaluate_segment(seg_df, cfg, cal, ocv_fn)
        r["vehicle_id"] = meta.vehicle_id
        results.append(r)
        if (idx + 1) % 200 == 0:
            print(f"  Deng: {idx + 1}/{len(eval_sample)} held-out done")
    return results


def run_fleet_ved(max_veh=None, max_trips=None) -> List[Dict]:
    from data.loaders.ved_loader import VEDLoader
    from data.loaders.pack_cartridge import lookup_ved_cartridge

    loader = VEDLoader(max_veh=max_veh, max_trips_per_veh=max_trips)
    all_pairs = list(loader.iter_segments())

    def _get_cfg(meta):
        cart = lookup_ved_cartridge(
            next((n.replace("vehicle=", "") for n in meta.notes
                  if n.startswith("vehicle=")), "")
        )
        return config_from_cartridge(
            "VED", cart, CellMode.AVG_CELL, dt_resample_s=20.0,
            min_duration_s=120.0, dt_short_s=5.0, dt_short_threshold_s=600.0,
        )

    sample_cfg = _get_cfg(all_pairs[0][1]) if all_pairs else None
    valid_pairs = [(s, m) for s, m in all_pairs
                   if (float(s["t_s"].iloc[-1]) - float(s["t_s"].iloc[0])) >= 120.0]
    cal_pairs, eval_pairs = _split_by_vehicle(valid_pairs)
    cal = _build_calibration_for_fleet(cal_pairs, sample_cfg, "VED") if (sample_cfg and cal_pairs) else None
    ocv_fn = cal.ocv_fn if cal else None

    results = []
    for idx, (seg_df, meta) in enumerate(eval_pairs):
        cfg = _get_cfg(meta)
        r = evaluate_segment(seg_df, cfg, cal, ocv_fn)
        r["vehicle_id"] = meta.vehicle_id
        results.append(r)
        if (idx + 1) % 50 == 0:
            print(f"  VED: {idx + 1}/{len(eval_pairs)} held-out done")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation + main
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(results: List[Dict]) -> Dict:
    def _mean(key):
        vals = [r[key] for r in results if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else None, len(vals)

    ekf_mean, ekf_n = _mean("soc_rmse_ekf_pct")
    cc_mean, cc_n = _mean("soc_rmse_coulomb_pct")
    ocv_mean, ocv_n = _mean("soc_rmse_ocv_lookup_pct")
    return {
        "n_segments": len(results),
        "ekf_soc_rmse_pct_mean": ekf_mean, "ekf_n_valid": ekf_n,
        "coulomb_only_soc_rmse_pct_mean": cc_mean, "coulomb_only_n_valid": cc_n,
        "ocv_lookup_only_soc_rmse_pct_mean": ocv_mean, "ocv_lookup_only_n_valid": ocv_n,
        "ekf_beats_coulomb_only": (ekf_mean is not None and cc_mean is not None and ekf_mean < cc_mean),
        "ekf_beats_ocv_lookup_only": (ekf_mean is not None and ocv_mean is not None and ekf_mean < ocv_mean),
    }


def main():
    print("=" * 78)
    print("SOC baseline benchmark -- EKF vs pure coulomb counting vs pure OCV lookup")
    print("Same held-out real fleet segments used throughout this project.")
    print("=" * 78)

    fleets: Dict[str, List[Dict]] = {}

    print("\nBMW i3...")
    fleets["BMW_i3"] = run_fleet_bmw_i3()

    print("\nDeng BAIC EU500...")
    fleets["Deng_BAIC_EU500"] = run_fleet_deng()

    print("\nVED (Michigan)...")
    fleets["VED"] = run_fleet_ved()

    report = {"meta": {
        "script": "data/soc_baseline_benchmark.py",
        "scope_note": (
            "Compares this project's Dual EKF against two generic, unnamed "
            "baselines (pure coulomb counting; pure naive OCV lookup) on the "
            "SAME held-out real fleet segments. Does NOT benchmark against "
            "any named commercial chip/vendor (TI, ADI, etc.) -- their exact "
            "firmware and tuned calibration are not available to this project."
        ),
    }}

    for fleet, results in fleets.items():
        agg = _aggregate(results)
        report[fleet] = {"per_segment": results, "aggregate": agg}
        print(f"\n--- {fleet} (n={agg['n_segments']} held-out segments) ---")
        print(f"  EKF SOC RMSE:              {agg['ekf_soc_rmse_pct_mean']:.2f}%  (n={agg['ekf_n_valid']})"
              if agg['ekf_soc_rmse_pct_mean'] is not None else "  EKF: no valid results")
        print(f"  Pure coulomb counting:     {agg['coulomb_only_soc_rmse_pct_mean']:.2f}%  (n={agg['coulomb_only_n_valid']})"
              if agg['coulomb_only_soc_rmse_pct_mean'] is not None else "  Coulomb: no valid results")
        print(f"  Pure OCV lookup (naive):   {agg['ocv_lookup_only_soc_rmse_pct_mean']:.2f}%  (n={agg['ocv_lookup_only_n_valid']})"
              if agg['ocv_lookup_only_soc_rmse_pct_mean'] is not None else "  OCV lookup: no valid results")
        print(f"  EKF beats coulomb-only: {agg['ekf_beats_coulomb_only']}   "
              f"EKF beats OCV-lookup-only: {agg['ekf_beats_ocv_lookup_only']}")

    def _serial(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, bool): return o
        if isinstance(o, dict): return {str(k): _serial(v) for k, v in o.items()}
        if isinstance(o, list): return [_serial(v) for v in o]
        return o

    OUT_JSON.write_text(json.dumps(_serial(report), indent=2))
    print(f"\nJSON report written to {OUT_JSON}")

    # Markdown summary
    lines = ["# SOC Baseline Benchmark\n",
             report["meta"]["scope_note"] + "\n",
             "| Fleet | n segments | EKF SOC RMSE | Coulomb-only | OCV-lookup-only | EKF beats both? |",
             "|---|---|---|---|---|---|"]
    for fleet in fleets:
        agg = report[fleet]["aggregate"]
        beats_both = agg["ekf_beats_coulomb_only"] and agg["ekf_beats_ocv_lookup_only"]
        lines.append(
            f"| {fleet} | {agg['n_segments']} | "
            f"{agg['ekf_soc_rmse_pct_mean']:.2f}% | "
            f"{agg['coulomb_only_soc_rmse_pct_mean']:.2f}% | "
            f"{agg['ocv_lookup_only_soc_rmse_pct_mean']:.2f}% | "
            f"{'Yes' if beats_both else 'No'} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Markdown summary written to {OUT_MD}")

    return report


if __name__ == "__main__":
    main()
