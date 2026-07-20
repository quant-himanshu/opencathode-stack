#!/usr/bin/env python3
"""
data/run_offset_sweep.py — Phases 2+3: scalar-bias baselines + initial-offset
sweep, one run.

Protocol identical to the headline benchmark (same loaders, splits, seeds,
fleet calibrations, native sampling — Phase 0 reproduced byte-identically),
with two additions:

  Phase 2: two ONLINE scalar-bias variants of the estimator
    rbc_dekf     — decoupled two-filter bias (RBC-DEKF analogue,
                   diagnosis/scalar_bias_dekf.py), Joseph-form updates
    rbc_coupled  — same bias AUGMENTED into one joint filter, full Jacobian
                   (run at the +20 pp headline offset only)
    Q_θ/R_θ tuned by grid search on the CALIBRATION split only (never on
    held-out trips), at the +20 pp headline offset, on native-sampling
    segments (same processing as evaluation); grid and chosen values logged
    to results/theta_tuning_<stamp>.json.

  Phase 3: initial-SOC-offset sweep over
    {−30, −20, −10, −5, 0, +5, +10, +20, +30} pp
    for my_ekf, rbc_dekf, coulomb (all offset-dependent) and ocv_lookup
    (init-independent: computed once, replicated across offset rows —
    flagged in the CSV footnote).

Protocol notes (disclosed):
  * Initial SOC is clipped to [0.02, 0.98] (the estimator's existing rule,
    validate_generic.py:291); at extreme offsets on trips starting near
    0%/100% the APPLIED offset is smaller than nominal — the per-trip
    applied offset is recorded and its mean reported per sweep cell.
  * The EKF-family initial covariance follows the project convention
    P0_soc = offset², floored at (2 pp)² so offset 0 does not degenerate to
    P0 = 0. The +20 pp headline cell is numerically unaffected
    (max(0.04, 4e-4) = 0.04).

Consistency guarantee: the (my_ekf, +20 pp) cell is cross-checked
per-segment against the committed data/soc_baseline_benchmark*_report.json
RMSEs (as are coulomb and ocv_lookup at +20); ABORTS on any mismatch
> 1e-6 pp. This proves the lean trajectory runner (no DFN co-simulation —
the DFN cell in run_mode_b_ekf never feeds the EKF) is trajectory-identical
to the headline path.

Outputs (results/): offset_sweep.csv, baseline_comparison.csv,
nominal_accuracy.csv + .md (offset-0 breakout), outcome_tiers.csv,
trip_durations.csv, per-trip dump + theta tuning log + meta (timestamped).

One command: venv/bin/python -u data/run_offset_sweep.py
Runtime: ~1–2 h wall on 8 workers (≈79k trajectory evaluations).
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loaders.common_schema import SegmentMeta
from data.soc_baseline_benchmark import coulomb_counting_soc, ocv_lookup_soc
from data.validate_generic import (
    CellMode, ValidationConfig, _build_calibration_for_fleet,
    _split_by_vehicle, config_from_cartridge,
)
from diagnosis.dual_ekf_lfp import DualEKF_LFP
from diagnosis.scalar_bias_dekf import CoupledBiasEKF, ScalarBiasDEKF
from validation.metrics import (
    OUTCOME_CONVERGED, OUTCOME_DIVERGED, OUTCOME_RECOVERED,
    aggregate_trips, footnote_lines, trip_metrics,
)

RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"

OFFSETS = [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30]
HEADLINE_OFFSET = 0.20
P0_FLOOR = 4e-4                    # (2 pp)² floor, see module docstring
CROSS_CHECK_TOL = 1e-6

# Wide adaptive tuning grid (2026-07-19 review: the original narrow grid
# {1e-10..1e-6}×{1e-6..1e-4} put 4/5 fleets on edges). Any selected edge is
# extended 2 decades repeatedly until the argmin is interior or the hard
# bounds are reached; Q_θ=~0/R_θ=~1 and Q_θ=~1 are semantic OFF/absorb-all
# endpoints, so a bound hit means the mechanism is saturated, not that the
# grid is too small.
THETA_GRID_Q = [1e-14, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
THETA_GRID_R = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
THETA_HARD_MIN, THETA_HARD_MAX = 1e-18, 1.0
THETA_EXTEND_DECADES = 2
THETA_TUNE_MAX_SEGS = 20                    # same cap as _tune_gamma

DENG_MAX_VEHICLES = 20
DENG_EVAL_SAMPLE_N = 2000
DENG_RNG_SEED = 42

SWEEP_METHODS = ("my_ekf", "rbc_dekf", "coulomb")   # + ocv_lookup (once)
ALL_METHODS = ("my_ekf", "rbc_dekf", "rbc_coupled", "coulomb", "ocv_lookup")

COMMITTED_REPORTS = {
    "BMW_i3":          (ROOT / "data" / "soc_baseline_benchmark_report.json", "BMW_i3"),
    "Deng_BAIC_EU500": (ROOT / "data" / "soc_baseline_benchmark_report.json", "Deng_BAIC_EU500"),
    "VED":             (ROOT / "data" / "soc_baseline_benchmark_report.json", "VED"),
    "CALCE_A123":      (ROOT / "data" / "soc_baseline_benchmark_calce_report.json", None),
    "Parallel_Module": (ROOT / "data" / "soc_baseline_benchmark_module_report.json", None),
}
_REPORT_RMSE_KEY = {"my_ekf": "soc_rmse_ekf_pct",
                    "coulomb": "soc_rmse_coulomb_pct",
                    "ocv_lookup": "soc_rmse_ocv_lookup_pct"}


# ─────────────────────────────────────────────────────────────────────────────
# Fleet preparation (mirrors run_main_table.py, additionally keeping the
# calibration-split segments for θ tuning)
# ─────────────────────────────────────────────────────────────────────────────

def _prep_bmw():
    from data.loaders.bmw_i3_loader import BMWI3Loader
    from data.loaders.pack_cartridge import BMW_I3_60AH
    all_pairs = list(BMWI3Loader(max_trips=None).iter_segments())
    cal_pairs, eval_pairs = _split_by_vehicle(all_pairs)
    cfg = config_from_cartridge("BMW_i3", BMW_I3_60AH, CellMode.AVG_CELL)
    cal = _build_calibration_for_fleet(cal_pairs, cfg, "BMW_i3") if cal_pairs else None
    return ([(s, m.vehicle_id, cfg) for s, m in eval_pairs],
            [(s, m.vehicle_id, cfg) for s, m in cal_pairs], cal)


def _prep_deng():
    from data.loaders.deng_charging_loader import DengChargingLoader
    from data.loaders.pack_cartridge import BAIC_EU500_90S
    loader = DengChargingLoader(max_vehicles=DENG_MAX_VEHICLES)
    all_pairs = list(loader.iter_segments())
    cal_pairs, eval_pairs = _split_by_vehicle(all_pairs)
    cfg = config_from_cartridge("Deng_Charging", BAIC_EU500_90S, CellMode.AVG_CELL)
    cal = _build_calibration_for_fleet(cal_pairs, cfg, "Deng") if cal_pairs else None
    rng = np.random.default_rng(DENG_RNG_SEED)
    if len(eval_pairs) > DENG_EVAL_SAMPLE_N:
        chosen = rng.choice(len(eval_pairs), size=DENG_EVAL_SAMPLE_N, replace=False)
        eval_pairs = [eval_pairs[i] for i in sorted(chosen)]
    return ([(s, m.vehicle_id, cfg) for s, m in eval_pairs],
            [(s, m.vehicle_id, cfg) for s, m in cal_pairs], cal)


def _prep_ved():
    from data.loaders.ved_loader import VEDLoader
    from data.loaders.pack_cartridge import lookup_ved_cartridge
    all_pairs = list(VEDLoader(max_veh=None, max_trips_per_veh=None).iter_segments())

    def _get_cfg(meta):
        cart = lookup_ved_cartridge(
            next((n.replace("vehicle=", "") for n in meta.notes
                  if n.startswith("vehicle=")), ""))
        return config_from_cartridge(
            "VED", cart, CellMode.AVG_CELL, dt_resample_s=20.0,
            min_duration_s=120.0, dt_short_s=5.0, dt_short_threshold_s=600.0)

    sample_cfg = _get_cfg(all_pairs[0][1]) if all_pairs else None
    valid = [(s, m) for s, m in all_pairs
             if (float(s["t_s"].iloc[-1]) - float(s["t_s"].iloc[0])) >= 120.0]
    cal_pairs, eval_pairs = _split_by_vehicle(valid)
    cal = (_build_calibration_for_fleet(cal_pairs, sample_cfg, "VED")
           if (sample_cfg and cal_pairs) else None)
    return ([(s, m.vehicle_id, _get_cfg(m)) for s, m in eval_pairs],
            [(s, m.vehicle_id, _get_cfg(m)) for s, m in cal_pairs], cal)


def _lab_meta_pairs(pairs, dataset, dt_median, has_T):
    out = []
    for seg_df, seg_id in pairs:
        out.append((seg_df, SegmentMeta(
            dataset=dataset, vehicle_id=seg_id, segment_id=seg_id,
            n_rows=len(seg_df), dt_s_median=dt_median,
            duration_s=float(seg_df["t_s"].iloc[-1]),
            soc_start=float(seg_df["SOC_bms"].iloc[0]),
            soc_end=float(seg_df["SOC_bms"].iloc[-1]),
            I_mean_A=float(seg_df["I_A"].mean()), V_mean_V=float(seg_df["V_V"].mean()),
            T_mean_degC=(float(seg_df["T_degC"].mean()) if has_T else float("nan")),
            has_temperature=has_T, notes=[])))
    return out


def _prep_calce():
    from data.soc_baseline_benchmark_calce import (
        CALCE_A123_CARTRIDGE, CALCE_DIR, CALCE_FILES, load_calce_file)
    all_pairs, q_effs = [], []
    for fname in CALCE_FILES:
        path = CALCE_DIR / fname
        if not path.exists():
            continue
        segs, q_eff = load_calce_file(path, fname.split("-")[0])
        q_effs.append(q_eff)
        all_pairs.extend(segs)
    if not all_pairs:
        return [], [], None
    cfg = config_from_cartridge("CALCE_A123", CALCE_A123_CARTRIDGE,
                                CellMode.AVG_CELL, dt_resample_s=5.0)
    cfg.q_cell_ah = float(np.mean(q_effs))
    by_cell: Dict[str, List] = {}
    for seg_df, seg_id in all_pairs:
        by_cell.setdefault(seg_id.split("_seg_")[0], []).append((seg_df, seg_id))
    cal_pairs, eval_pairs = [], []
    for cid, segs in by_cell.items():
        n_cal = max(1, int(len(segs) * 0.40))
        cal_pairs.extend(segs[:n_cal])
        eval_pairs.extend(segs[n_cal:])
    cal = _build_calibration_for_fleet(
        _lab_meta_pairs(cal_pairs, "calce_a123", 5.0, True), cfg, "CALCE_A123")
    return ([(s, sid, cfg) for s, sid in eval_pairs],
            [(s, sid, cfg) for s, sid in cal_pairs], cal)


def _prep_module():
    from data.soc_baseline_benchmark_module import (
        MODULE_CARTRIDGE, MODULE_FOLDERS, load_module_folder)
    all_pairs, q_effs = [], []
    for folder in MODULE_FOLDERS:
        segs, q_eff = load_module_folder(folder)
        if segs:
            q_effs.append(q_eff)
            all_pairs.extend(segs)
    if not all_pairs:
        return [], [], None
    cfg = config_from_cartridge("Parallel_Module", MODULE_CARTRIDGE,
                                CellMode.AVG_CELL, dt_resample_s=2.0)
    cfg.q_cell_ah = float(np.mean(q_effs))
    by_mod: Dict[str, List] = {}
    for seg_df, seg_id in all_pairs:
        by_mod.setdefault(seg_id.split("_seg_")[0], []).append((seg_df, seg_id))
    cal_pairs, eval_pairs = [], []
    for mid, segs in by_mod.items():
        n_cal = max(1, int(len(segs) * 0.30))
        cal_pairs.extend(segs[:n_cal])
        eval_pairs.extend(segs[n_cal:])
    cal = _build_calibration_for_fleet(
        _lab_meta_pairs(cal_pairs, "parallel_module", 2.0, False), cfg, "Parallel_Module")
    return ([(s, sid, cfg) for s, sid in eval_pairs],
            [(s, sid, cfg) for s, sid in cal_pairs], cal)


FLEET_PREPS = {
    "BMW_i3": _prep_bmw,
    "Deng_BAIC_EU500": _prep_deng,
    "VED": _prep_ved,
    "CALCE_A123": _prep_calce,
    "Parallel_Module": _prep_module,
}

# Global registry, populated in the parent BEFORE the fork-based pool is
# created; workers read it via copy-on-write fork memory.
_FLEETS: Dict[str, Dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Lean trajectory runners (no DFN co-simulation; see module docstring)
# ─────────────────────────────────────────────────────────────────────────────

def _make_filter(kind: str, cfg: ValidationConfig, cal, ocv_fn,
                 offset: float, theta_qr: Tuple[float, float]):
    gamma = cal.ekf_gamma if cal is not None else 1.0
    R_meas = cal.ekf_R_meas_V2 if cal is not None else 4e-6
    common = dict(Q_nom_Ah=cfg.q_cell_ah, R_int_ohm=cfg.r_ohm_cell,
                  ocv_fn=ocv_fn, R_meas_V2=R_meas,
                  P0_soc=max(offset * offset, P0_FLOOR), gamma=gamma)
    if kind == "my_ekf":
        return DualEKF_LFP(**common,
                           cal_soc_fn=(cal.soc_cal_fn() if cal is not None else None),
                           cal_dR0=(cal.delta_R0 if cal is not None else 0.0))
    if kind == "rbc_dekf":
        return ScalarBiasDEKF(**common, Q_theta_V2_per_s=theta_qr[0],
                              R_theta_V2=theta_qr[1])
    if kind == "rbc_coupled":
        return CoupledBiasEKF(**common, Q_theta_V2_per_s=theta_qr[0],
                              R_theta_V2=theta_qr[1])
    raise ValueError(kind)


def run_lean_traj(seg_df: pd.DataFrame, cfg: ValidationConfig, filt,
                  offset: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Replicates run_mode_b_ekf's EKF-relevant loop exactly (dt/T/sign
    conventions, [0.02, 0.98] init clip, per-step exception fallback)."""
    t_s = seg_df["t_s"].values.astype(np.float64)
    I_cell = seg_df["I_A"].values.astype(np.float64) / cfg.n_parallel
    V_cell = seg_df["V_V"].values.astype(np.float64) / cfg.n_series
    soc_bms = seg_df["SOC_bms"].values.astype(np.float64)
    T_arr = seg_df["T_degC"].values.astype(np.float64)

    soc_init = float(np.clip(float(soc_bms[0]) + offset, 0.02, 0.98))
    filt.set_soc(soc_init)

    soc_est = np.empty(len(t_s))
    for i in range(len(t_s)):
        dt = float(t_s[i] - t_s[i - 1]) if i > 0 else 1.0
        T = float(T_arr[i]) if np.isfinite(T_arr[i]) else 25.0
        try:
            r = filt.update(float(V_cell[i]), -float(I_cell[i]), dt, T)
            soc_est[i] = float(r.get("soc", filt.x1[0]))
        except Exception:
            soc_est[i] = float(filt.x1[0])
    applied_offset_pp = (soc_init - float(soc_bms[0])) * 100.0
    return t_s, soc_est, soc_bms, applied_offset_pp


def eval_one(fleet: str, seg_idx: int, method: str, offset: float,
             items_key: str = "eval") -> Optional[Dict]:
    fl = _FLEETS[fleet]
    seg_df, vid, cfg = fl[items_key][seg_idx]
    cal, ocv_fn, theta_qr = fl["cal"], fl["ocv_fn"], fl["theta_qr"]
    t_s = seg_df["t_s"].values.astype(np.float64)
    soc_true = seg_df["SOC_bms"].values.astype(np.float64)
    try:
        if method == "coulomb":
            soc_est = coulomb_counting_soc(seg_df, replace(cfg, ekf_soc_offset=offset))
            applied = (float(np.clip(soc_true[0] + offset, 0.02, 0.98))
                       - float(soc_true[0])) * 100.0
        elif method == "ocv_lookup":
            if ocv_fn is None:
                return None
            soc_est = ocv_lookup_soc(seg_df, cfg, ocv_fn)
            applied = None
        else:
            filt = _make_filter(method, cfg, cal, ocv_fn, offset, theta_qr)
            t_s, soc_est, soc_true, applied = run_lean_traj(seg_df, cfg, filt, offset)
        m = trip_metrics(t_s, soc_est, soc_true)
        m["applied_offset_pp"] = applied
        m["vehicle_id"] = vid
        return m
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "vehicle_id": vid}


def _worker(task):
    fleet, seg_idx, method, offset = task
    return (fleet, seg_idx, method, offset, eval_one(fleet, seg_idx, method, offset))


# ─────────────────────────────────────────────────────────────────────────────
# θ tuning (calibration split ONLY, +20 pp offset, native sampling)
# ─────────────────────────────────────────────────────────────────────────────

def _tune_combo(args) -> Tuple[float, float, Optional[float], int]:
    """Pool worker: mean cal-split RMSE for one (Q_θ, R_θ) combo."""
    fleet, q, r = args
    fl = _FLEETS[fleet]
    rmses = []
    for seg_df, vid, cfg in fl["cal_items"][:THETA_TUNE_MAX_SEGS]:
        try:
            filt = _make_filter("rbc_dekf", cfg, fl["cal"], fl["ocv_fn"],
                                HEADLINE_OFFSET, (q, r))
            t_s, est, tru, _ = run_lean_traj(seg_df, cfg, filt, HEADLINE_OFFSET)
            rmses.append(float(np.sqrt(np.mean((est - tru) ** 2))) * 100.0)
        except Exception:
            pass
    return q, r, (float(np.mean(rmses)) if rmses else None), len(rmses)


def _extend_grid(grid: List[float], direction: str) -> Tuple[List[float], bool]:
    g = sorted(grid)
    added, bounded = [], False
    for k in range(1, THETA_EXTEND_DECADES + 1):
        v = g[0] * 10.0 ** (-k) if direction == "down" else g[-1] * 10.0 ** k
        if v < THETA_HARD_MIN or v > THETA_HARD_MAX:
            bounded = True
            break
        added.append(v)
    return sorted(set(g + added)), bounded


def tune_theta(fleet: str, n_workers: int) -> Dict:
    """Wide ADAPTIVE grid search on the calibration split only (+20 pp),
    pooled across combos; ties keep smallest Q_θ then smallest R_θ."""
    qgrid, rgrid = list(THETA_GRID_Q), list(THETA_GRID_R)
    evaluated: Dict[Tuple[float, float], Tuple[Optional[float], int]] = {}
    bound_hits: List[str] = []
    ctx = mp.get_context("fork")
    rounds = 0
    while True:
        rounds += 1
        todo = [(fleet, q, r) for q in qgrid for r in rgrid
                if (q, r) not in evaluated]
        if todo:
            with ctx.Pool(n_workers) as pool:
                for q, r, m, n in pool.imap_unordered(_tune_combo, todo,
                                                      chunksize=1):
                    evaluated[(q, r)] = (m, n)
        valid = {k: v for k, v in evaluated.items() if v[0] is not None}
        if not valid:
            return {"fleet": fleet, "chosen_Q_theta_V2_per_s": 1e-8,
                    "chosen_R_theta_V2": 1e-5, "chosen_mean_cal_rmse_pct": None,
                    "error": "no valid combos", "grid": []}
        best_qr = min(valid, key=lambda k: (valid[k][0], k[0], k[1]))
        bq, br = best_qr
        grew = False
        for name, grid, val in (("Q", qgrid, bq), ("R", rgrid, br)):
            g = sorted(grid)
            for direction, edge in (("down", g[0]), ("up", g[-1])):
                if val == edge:
                    new_grid, bounded = _extend_grid(g, direction)
                    if bounded:
                        bound_hits.append(f"{name} {direction} bound at {edge:.0e}")
                    if len(new_grid) > len(g):
                        grew = True
                        if name == "Q":
                            qgrid = new_grid
                        else:
                            rgrid = new_grid
        if not grew or rounds > 8:
            break
    return {"fleet": fleet, "chosen_Q_theta_V2_per_s": bq,
            "chosen_R_theta_V2": br,
            "chosen_mean_cal_rmse_pct": valid[best_qr][0],
            "n_cal_segments_used": min(len(_FLEETS[fleet]["cal_items"]),
                                       THETA_TUNE_MAX_SEGS),
            "offset_used": HEADLINE_OFFSET, "rounds": rounds,
            "bound_hits": bound_hits,
            "interior": not bound_hits,
            "grid": [{"Q_theta_V2_per_s": q, "R_theta_V2": r,
                      "mean_cal_rmse_pct": m, "n_segs": n}
                     for (q, r), (m, n) in sorted(evaluated.items())]}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-check vs committed reports (my_ekf/coulomb/ocv_lookup @ +20)
# ─────────────────────────────────────────────────────────────────────────────

def cross_check(fleet: str, res: Dict) -> float:
    path, subkey = COMMITTED_REPORTS[fleet]
    doc = json.loads(path.read_text())
    ref = (doc[subkey] if subkey else doc)["per_segment"]
    n = len(_FLEETS[fleet]["eval"])
    if len(ref) != n:
        raise RuntimeError(f"{fleet}: {n} segments vs {len(ref)} committed — ABORT")
    max_diff = 0.0
    for i in range(n):
        for method, key in _REPORT_RMSE_KEY.items():
            old_v = ref[i].get(key)
            m = res.get((method, HEADLINE_OFFSET), {}).get(i)
            new_v = None if (m is None or "error" in m) else m["rmse_full_pct"]
            if old_v is None and new_v is None:
                continue
            if (old_v is None) != (new_v is None):
                raise RuntimeError(f"{fleet} seg {i} {method}: one side None — ABORT")
            d = abs(old_v - new_v)
            max_diff = max(max_diff, d)
            if d > CROSS_CHECK_TOL:
                raise RuntimeError(
                    f"{fleet} seg {i} {method}: {new_v:.8f} vs committed "
                    f"{old_v:.8f} (diff {d:.2e}) — ABORT")
    return max_diff


# ─────────────────────────────────────────────────────────────────────────────
# Output writing
# ─────────────────────────────────────────────────────────────────────────────

def _f(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else v)


def _write_csv(path: Path, header_note: List[str], cols: List[str],
               rows: List[Dict]) -> None:
    with path.open("w", newline="") as f:
        for line in header_note + footnote_lines("# "):
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    print(f"Wrote {path}")


def sweep_row(fleet: str, method: str, offset: float, trips: List[Dict],
              censor_t_s: Optional[float] = None) -> Dict:
    ok = [t for t in trips if t is not None and "error" not in t]
    agg = aggregate_trips(ok, censor_t_s=censor_t_s)
    applied = [t["applied_offset_pp"] for t in ok
               if t.get("applied_offset_pp") is not None]
    return {
        "dataset": fleet, "method": method, "offset_pp": round(offset * 100),
        "n_trips": agg["n_trips"],
        "n_errors": sum(1 for t in trips if t is None or "error" in t),
        "rmse_median": agg["rmse_full_pct"]["median"],
        "rmse_q25": agg["rmse_full_pct"]["q25"],
        "rmse_q75": agg["rmse_full_pct"]["q75"],
        "rmse_mean": agg["rmse_full_pct"]["mean"],
        "mae_median": agg["mae_full_pct"]["median"],
        "maxerr_median": agg["maxerr_full_pct"]["median"],
        "rmse_postconv_median": agg["rmse_postconv_pct"]["median"],
        "err_end_median": agg["err_end_pct"]["median"],
        "conv_rate_strict": agg["conv_rate_strict"],
        "conv_rate_hold": agg["conv_rate_hold"],
        "conv_rate_legacy": agg["conv_rate_legacy"],
        "rate_recovered": agg["rate_recovered"],
        "rate_diverged": agg["rate_diverged"],
        "t_conv_strict_median_s": agg["t_conv_strict_median_s"],
        "t_conv_hold_median_s": agg["t_conv_hold_median_s"],
        "t_conv_legacy_median_s": agg["t_conv_legacy_median_s"],
        "n_censored": agg.get("n_censored"),
        "conv_rate_strict_censaware": agg.get("conv_rate_strict_censaware"),
        "mean_applied_offset_pp": (float(np.mean(applied)) if applied else None),
    }


SWEEP_COLS = ["dataset", "method", "offset_pp", "n_trips", "n_errors",
              "rmse_median", "rmse_q25", "rmse_q75", "rmse_mean",
              "mae_median", "maxerr_median", "rmse_postconv_median",
              "err_end_median",
              "conv_rate_strict", "conv_rate_hold", "conv_rate_legacy",
              "rate_recovered", "rate_diverged",
              "t_conv_strict_median_s", "t_conv_hold_median_s",
              "t_conv_legacy_median_s",
              "n_censored", "conv_rate_strict_censaware",
              "mean_applied_offset_pp"]


def write_ved_breakdown(res_ved: Dict, censor_t: Optional[float], stamp: str) -> None:
    """docs/VED_BREAKDOWN.md — EKF @ +20 pp: why VED trips fail."""
    trips = [t for t in res_ved[("my_ekf", HEADLINE_OFFSET)].values()
             if t is not None and "error" not in t]
    conv = [t for t in trips if t["t_conv_strict_s"] is not None]
    nonconv = [t for t in trips if t["t_conv_strict_s"] is None]
    rediv = [t for t in nonconv if t["min_abs_err_pct"] < 5.0]
    never = [t for t in nonconv if t["min_abs_err_pct"] >= 5.0]

    def _short(ts):
        if censor_t is None:
            return []
        return [t for t in ts if t["duration_s"] < censor_t]

    def _rec(ts):
        return sum(1 for t in ts if t["outcome"] == OUTCOME_RECOVERED)

    def _med(ts, k):
        vals = [t[k] for t in ts if t.get(k) is not None]
        return float(np.median(vals)) if vals else None

    n = len(trips)
    lines = [
        "# VED failure-mode breakdown — Dual EKF, +20 pp protocol",
        "",
        f"Generated {stamp} by `data/run_offset_sweep.py`. n = {n} held-out "
        f"VED trips; strict convergence threshold 5 pp; 'short' = duration < "
        f"censoring threshold ({_f(censor_t, 0)} s = VED EKF median strict "
        f"t_conv); 'recovered' = trip-end error ≤ 10 pp without strict "
        f"convergence.",
        "",
        "| Tier | n | share | short trips | recovered-at-end | median duration (s) | median min\\|err\\| (pp) | median end-err (pp) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, ts in (("converged (strict)", conv),
                     ("re-diverging (entered 5 pp band, did not hold)", rediv),
                     ("never-approaching (never within 5 pp)", never)):
        lines.append(
            f"| {name} | {len(ts)} | {len(ts)/max(n,1)*100:.1f}% "
            f"| {len(_short(ts))} | {_rec(ts)} "
            f"| {_f(_med(ts, 'duration_s'), 0)} "
            f"| {_f(_med(ts, 'min_abs_err_pct'), 1)} "
            f"| {_f(_med(ts, 'err_end_pct'), 1)} |")
    lines += ["", *footnote_lines("> ")]
    (DOCS_DIR / "VED_BREAKDOWN.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {DOCS_DIR / 'VED_BREAKDOWN.md'}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, cwd=ROOT,
                                check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print("PHASES 2+3 — scalar-bias baselines + initial-offset sweep")
    print("=" * 78)

    # ── prep fleets (parent process) ────────────────────────────────────────
    for fleet, prep in FLEET_PREPS.items():
        print(f"\n[{fleet}] loading + calibration…")
        eval_items, cal_items, cal = prep()
        _FLEETS[fleet] = {
            "eval": eval_items, "cal_items": cal_items, "cal": cal,
            "ocv_fn": (cal.ocv_fn if cal else None), "theta_qr": (1e-8, 1e-5),
        }
        print(f"[{fleet}] {len(eval_items)} eval / {len(cal_items)} cal segments")

    # ── θ tuning on calibration split only (wide adaptive grid, pooled) ────
    n_workers_tune = max(1, (os.cpu_count() or 4) - 2)
    tuning = {}
    for fleet in FLEET_PREPS:
        print(f"[{fleet}] wide adaptive Q_θ/R_θ tuning on calibration split…")
        tr = tune_theta(fleet, n_workers_tune)
        tuning[fleet] = tr
        _FLEETS[fleet]["theta_qr"] = (tr["chosen_Q_theta_V2_per_s"],
                                      tr["chosen_R_theta_V2"])
        print(f"[{fleet}] chosen Q_θ={tr['chosen_Q_theta_V2_per_s']:.0e} V²/s, "
              f"R_θ={tr['chosen_R_theta_V2']:.0e} V² "
              f"(cal RMSE {(_f(tr['chosen_mean_cal_rmse_pct'], 2) or '?')}%, "
              f"rounds={tr.get('rounds')}, interior={tr.get('interior')})")
    (RESULTS_DIR / f"theta_tuning_wide_{stamp}.json").write_text(
        json.dumps(tuning, indent=2))

    # ── build tasks ─────────────────────────────────────────────────────────
    tasks = []
    for fleet in FLEET_PREPS:
        n = len(_FLEETS[fleet]["eval"])
        for i in range(n):
            for method in SWEEP_METHODS:
                for off in OFFSETS:
                    tasks.append((fleet, i, method, off))
            tasks.append((fleet, i, "ocv_lookup", HEADLINE_OFFSET))
            tasks.append((fleet, i, "rbc_coupled", HEADLINE_OFFSET))
    print(f"\n{len(tasks)} trajectory evaluations queued")

    # ── parallel execution (fork: workers inherit _FLEETS) ─────────────────
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Running on {n_workers} workers (fork)…")
    results: Dict[str, Dict[Tuple[str, float], Dict[int, Dict]]] = {
        fleet: {} for fleet in FLEET_PREPS}
    ctx = mp.get_context("fork")
    done = 0
    with ctx.Pool(n_workers) as pool:
        for fleet, idx, method, off, m in pool.imap_unordered(
                _worker, tasks, chunksize=16):
            results[fleet].setdefault((method, off), {})[idx] = m
            done += 1
            if done % 5000 == 0:
                print(f"  {done}/{len(tasks)} done "
                      f"({datetime.now(timezone.utc).strftime('%H:%M:%SZ')})")

    # ── cross-check @ +20 vs committed reports ─────────────────────────────
    diffs = {}
    for fleet in FLEET_PREPS:
        diffs[fleet] = cross_check(fleet, results[fleet])
        print(f"[{fleet}] cross-check vs committed: max |ΔRMSE| = "
              f"{diffs[fleet]:.2e} pct — OK")

    # ── replicate init-independent ocv_lookup across offsets ───────────────
    for fleet in FLEET_PREPS:
        base = results[fleet][("ocv_lookup", HEADLINE_OFFSET)]
        for off in OFFSETS:
            results[fleet][("ocv_lookup", off)] = base

    # ── censoring thresholds: per-dataset EKF median strict t_conv @ +20 ───
    censor_t: Dict[str, Optional[float]] = {}
    for fleet in FLEET_PREPS:
        ek = [t for t in results[fleet][("my_ekf", HEADLINE_OFFSET)].values()
              if t is not None and "error" not in t
              and t.get("t_conv_strict_s") is not None]
        censor_t[fleet] = (float(np.median([t["t_conv_strict_s"] for t in ek]))
                           if ek else None)

    def _trips(fleet, method, off):
        d = results[fleet].get((method, off), {})
        return [d[i] for i in sorted(d)]

    hdr = [f"# generated by data/run_offset_sweep.py {stamp} (git {commit}); "
           f"seeds: deng_eval_sample={DENG_RNG_SEED}; "
           f"P0_soc=max(offset^2,{P0_FLOOR}); init clipped to [0.02,0.98]",
           "# ocv_lookup is init-independent: identical values replicated "
           "across offset rows by construction",
           "# censoring threshold per dataset = EKF median strict t_conv at "
           "+20pp; n_censored/censaware only meaningful where computed"]

    # 1. offset sweep
    rows = []
    for fleet in FLEET_PREPS:
        for method in ("my_ekf", "rbc_dekf", "coulomb", "ocv_lookup"):
            for off in OFFSETS:
                rows.append(sweep_row(fleet, method, off, _trips(fleet, method, off),
                                      censor_t_s=censor_t[fleet]))
    _write_csv(RESULTS_DIR / "offset_sweep.csv", hdr, SWEEP_COLS, rows)

    # 2. baseline comparison @ +20 (Phase 2), incl. coupled variant
    rows20 = [sweep_row(fleet, method, HEADLINE_OFFSET,
                        _trips(fleet, method, HEADLINE_OFFSET),
                        censor_t_s=censor_t[fleet])
              for fleet in FLEET_PREPS for method in ALL_METHODS]
    _write_csv(RESULTS_DIR / "baseline_comparison.csv", hdr, SWEEP_COLS, rows20)

    # 3. nominal accuracy @ offset 0 (Phase 3 breakout)
    rows0 = [sweep_row(fleet, method, 0.0, _trips(fleet, method, 0.0),
                       censor_t_s=censor_t[fleet])
             for fleet in FLEET_PREPS
             for method in ("my_ekf", "rbc_dekf", "coulomb", "ocv_lookup")]
    _write_csv(RESULTS_DIR / "nominal_accuracy.csv",
               hdr + ["# NOMINAL protocol: correct initial SOC (offset 0), "
                      "P0_soc=(2pp)^2 floor — unlike every other table, which "
                      "uses the adversarial wrong-init stress protocol"],
               SWEEP_COLS, rows0)
    md = ["# Nominal accuracy (offset = 0, correct initialization)", "",
          f"Generated {stamp}. Median (IQR) primary, mean secondary — all pp.",
          "",
          "| Dataset | Method | n | RMSE med (IQR) | RMSE mean | MAE med | Conv% strict |",
          "|---|---|---|---|---|---|---|"]
    for r in rows0:
        md.append(f"| {r['dataset']} | {r['method']} | {r['n_trips']} "
                  f"| {_f(r['rmse_median'], 2)} ({_f(r['rmse_q25'], 2)}–{_f(r['rmse_q75'], 2)}) "
                  f"| {_f(r['rmse_mean'], 2)} | {_f(r['mae_median'], 2)} "
                  f"| {_f((r['conv_rate_strict'] or 0) * 100, 1)} |")
    md += ["", "> Nominal protocol: correct initial SOC; all other project "
           "tables use the +20 pp (or swept) adversarial wrong-init protocol.",
           *footnote_lines("> ")]
    (RESULTS_DIR / "nominal_accuracy.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {RESULTS_DIR / 'nominal_accuracy.md'}")

    # 4. outcome tiers @ +20 (Phase 1 extension)
    _write_csv(RESULTS_DIR / "outcome_tiers.csv", hdr,
               ["dataset", "method", "n_trips", "conv_rate_strict",
                "conv_rate_hold", "conv_rate_legacy", "rate_converged",
                "rate_recovered", "rate_diverged", "n_censored",
                "conv_rate_strict_censaware", "censor_t_s"],
               [{**sweep_row(fleet, method, HEADLINE_OFFSET,
                             _trips(fleet, method, HEADLINE_OFFSET),
                             censor_t_s=censor_t[fleet]),
                 "rate_converged": aggregate_trips(
                     [t for t in _trips(fleet, method, HEADLINE_OFFSET)
                      if t is not None and "error" not in t])["rate_converged"],
                 "censor_t_s": censor_t[fleet]}
                for fleet in FLEET_PREPS for method in ALL_METHODS])

    # 5. trip durations (Phase 1 extension)
    dur_rows = []
    for fleet in FLEET_PREPS:
        durs = [t["duration_s"] for t in _trips(fleet, "my_ekf", HEADLINE_OFFSET)
                if t is not None and "error" not in t]
        a = np.asarray(durs)
        dur_rows.append({"dataset": fleet, "n_trips": len(a),
                         "min_s": float(a.min()), "q25_s": float(np.percentile(a, 25)),
                         "median_s": float(np.median(a)),
                         "q75_s": float(np.percentile(a, 75)),
                         "max_s": float(a.max()), "mean_s": float(a.mean()),
                         "censor_t_s": censor_t[fleet]})
    _write_csv(RESULTS_DIR / "trip_durations.csv", hdr,
               ["dataset", "n_trips", "min_s", "q25_s", "median_s", "q75_s",
                "max_s", "mean_s", "censor_t_s"], dur_rows)

    # 6. VED breakdown (Phase 1 extension)
    write_ved_breakdown(results["VED"], censor_t["VED"], stamp)

    # 7. per-trip dump + meta
    dump_cols = ["dataset", "method", "offset_pp", "seg_index", "vehicle_id",
                 "rmse_full_pct", "mae_full_pct", "maxerr_full_pct",
                 "rmse_postconv_pct", "t_conv_strict_s", "t_conv_hold_s",
                 "t_conv_legacy_s", "err_end_pct", "min_abs_err_pct",
                 "outcome", "applied_offset_pp", "duration_s", "n_samples",
                 "error"]
    dump_path = RESULTS_DIR / f"offset_sweep_per_trip_{stamp}.csv"
    with dump_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dump_cols)
        w.writeheader()
        for fleet in FLEET_PREPS:
            for (method, off), d in sorted(results[fleet].items()):
                if method == "ocv_lookup" and off != HEADLINE_OFFSET:
                    continue  # replicated rows carry no extra information
                for i in sorted(d):
                    m = d[i] or {}
                    row = {"dataset": fleet, "method": method,
                           "offset_pp": round(off * 100), "seg_index": i}
                    row.update({k: m.get(k) for k in dump_cols if k in m})
                    w.writerow({k: ("" if row.get(k) is None else row.get(k))
                                for k in dump_cols})
    print(f"Wrote {dump_path}")

    meta = {"script": "data/run_offset_sweep.py", "utc": stamp,
            "git_commit": commit, "python": sys.version.split()[0],
            "n_workers": n_workers, "offsets": OFFSETS,
            "P0_floor": P0_FLOOR, "seeds": {"deng_eval_sample": DENG_RNG_SEED},
            "theta_tuning": {k: {kk: vv for kk, vv in v.items() if kk != "grid"}
                             for k, v in tuning.items()},
            "cross_check_max_diff_pct": diffs,
            "censor_t_s": censor_t}
    (RESULTS_DIR / f"offset_sweep_meta_{stamp}.json").write_text(
        json.dumps(meta, indent=2))
    print(f"Wrote {RESULTS_DIR / f'offset_sweep_meta_{stamp}.json'}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
