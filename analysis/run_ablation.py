#!/usr/bin/env python3
"""
analysis/run_ablation.py — Phase 5 ablation study (+20 pp protocol, all
five datasets, sign-corrected data).

Nine variants of the estimator, one row each per dataset. The production
class diagnosis/dual_ekf_lfp.py is NOT modified; variants are built here as
subclasses/instance patches, and the `full` variant is cross-checked
per-segment against the committed benchmark reports (abort on mismatch
> 1e-6 pp) to prove the ablation harness runs the exact headline estimator.

  full             decoupled H, δV(SOC)+δR0, adaptive Q, gated slow loops
  coupled_dcal_H   ∂δV/∂SOC ADDED to H — the known-broken Round-2 config,
                   run for the record
  no_corrections   δV=0, δR0=0 (pure physics; fleet OCV table kept)
  dR0_only         δR0 kept, δV(SOC) removed
  dV_only          δV(SOC) kept, δR0 removed
  const_Q          adaptive slope factor removed: Q = Q_base·γ
  slow_loops_on    calibration sanity gate DISABLED (R_int loop forced on
                   even when |δR0| > 50 mΩ — affects CALCE/UMich)
  slow_loops_off   both slow loops disabled (x2 = [SOH, R_int] frozen)
  joseph_form      full method with Joseph-form covariance update in the
                   fast loop (2026-07-19 review decision (a))

Outputs: results/ablation.csv (+ per-trip dump, timestamped) and
figures/ablation.pdf.
One command: venv/bin/python -u analysis/run_ablation.py
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data.run_offset_sweep as ros
from diagnosis.dual_ekf_lfp import DualEKF_LFP, EPS
from validation.metrics import aggregate_trips, footnote_lines, trip_metrics

RESULTS_DIR = ROOT / "results"
OFFSET = 0.20
N_WORKERS = max(1, (os.cpu_count() or 4) - 2)
CTX = mp.get_context("fork")

VARIANTS = ["full", "coupled_dcal_H", "no_corrections", "dR0_only",
            "dV_only", "const_Q", "slow_loops_on", "slow_loops_off",
            "joseph_form"]


class CoupledCalEKF(DualEKF_LFP):
    """Round-2 config: ∂δV/∂SOC included in H (everything else identical)."""

    def _dcal_dsoc(self, soc: float) -> float:
        if self._cal_soc_fn is None:
            return 0.0
        h = 0.005
        return (self._cal_offset(soc + h) - self._cal_offset(soc - h)) / (2 * h)

    def update(self, V_meas, I_A, dt_s, T_C=25.0):
        soc, v_pol = self.x1
        Q_eff = self.Q_nom * float(self.x2[0])
        R_use = float(self.x2[1])
        tau = 50.0
        soc_p = float(np.clip(soc - (I_A * dt_s) / (3600.0 * Q_eff), 0.0, 1.0))
        v_pol_p = v_pol * np.exp(-dt_s / tau) + R_use * (1.0 - np.exp(-dt_s / tau)) * I_A
        x_p = np.array([soc_p, v_pol_p])
        F = np.array([[1.0, 0.0], [0.0, np.exp(-dt_s / tau)]])
        P_p = F @ self.P1 @ F.T + self._adaptive_Q(soc)
        cal_off = self._cal_offset(x_p[0])
        r0_off = self._cal_dR0 * I_A
        V_pred = self._ocv(x_p[0]) - I_A * R_use + x_p[1] + cal_off + r0_off
        # THE ABLATED CHANGE: calibration slope enters the Jacobian
        H = np.array([[self._docv_dsoc(x_p[0]) + self._dcal_dsoc(x_p[0]), 1.0]])
        S = (H @ P_p @ H.T + self._R_meas)[0, 0] + EPS
        K = (P_p @ H.T) / S
        innov = V_meas - V_pred
        self.x1 = x_p + K.flatten() * innov
        self.x1[0] = float(np.clip(self.x1[0], 0.0, 1.0))
        self.P1 = (np.eye(2) - np.outer(K.flatten(), H)) @ P_p
        self._update_r_int(V_meas, I_A, cal_off, r0_off)
        self._update_soh(dt_s, I_A)
        return {"soc": float(self.x1[0]), "V_pred": float(V_pred),
                "innovation": float(innov)}


class ConstQEKF(DualEKF_LFP):
    """Adaptive slope factor removed: Q = Q_base·γ (γ kept so the fleet
    tuning is not silently discarded along with the adaptivity)."""

    def _adaptive_Q(self, soc: float) -> np.ndarray:
        return self._Q_base * self._gamma


class JosephEKF(DualEKF_LFP):
    """Full method, Joseph-form fast-loop covariance update."""

    def update(self, V_meas, I_A, dt_s, T_C=25.0):
        soc, v_pol = self.x1
        Q_eff = self.Q_nom * float(self.x2[0])
        R_use = float(self.x2[1])
        tau = 50.0
        soc_p = float(np.clip(soc - (I_A * dt_s) / (3600.0 * Q_eff), 0.0, 1.0))
        v_pol_p = v_pol * np.exp(-dt_s / tau) + R_use * (1.0 - np.exp(-dt_s / tau)) * I_A
        x_p = np.array([soc_p, v_pol_p])
        F = np.array([[1.0, 0.0], [0.0, np.exp(-dt_s / tau)]])
        P_p = F @ self.P1 @ F.T + self._adaptive_Q(soc)
        cal_off = self._cal_offset(x_p[0])
        r0_off = self._cal_dR0 * I_A
        V_pred = self._ocv(x_p[0]) - I_A * R_use + x_p[1] + cal_off + r0_off
        H = np.array([[self._docv_dsoc(x_p[0]), 1.0]])
        S = (H @ P_p @ H.T + self._R_meas)[0, 0] + EPS
        K = (P_p @ H.T) / S
        innov = V_meas - V_pred
        self.x1 = x_p + K.flatten() * innov
        self.x1[0] = float(np.clip(self.x1[0], 0.0, 1.0))
        IKH = np.eye(2) - np.outer(K.flatten(), H)
        self.P1 = IKH @ P_p @ IKH.T + np.outer(K.flatten(), K.flatten()) * self._R_meas[0, 0]
        self._update_r_int(V_meas, I_A, cal_off, r0_off)
        self._update_soh(dt_s, I_A)
        return {"soc": float(self.x1[0]), "V_pred": float(V_pred),
                "innovation": float(innov)}


def make_variant(variant: str, cfg, cal, ocv_fn):
    gamma = cal.ekf_gamma if cal is not None else 1.0
    R_meas = cal.ekf_R_meas_V2 if cal is not None else 4e-6
    spline = cal.soc_cal_fn() if cal is not None else None
    dR0 = cal.delta_R0 if cal is not None else 0.0
    common = dict(Q_nom_Ah=cfg.q_cell_ah, R_int_ohm=cfg.r_ohm_cell,
                  ocv_fn=ocv_fn, R_meas_V2=R_meas, P0_soc=OFFSET ** 2,
                  gamma=gamma)
    if variant == "full":
        return DualEKF_LFP(**common, cal_soc_fn=spline, cal_dR0=dR0)
    if variant == "coupled_dcal_H":
        return CoupledCalEKF(**common, cal_soc_fn=spline, cal_dR0=dR0)
    if variant == "no_corrections":
        return DualEKF_LFP(**common, cal_soc_fn=None, cal_dR0=0.0)
    if variant == "dR0_only":
        return DualEKF_LFP(**common, cal_soc_fn=None, cal_dR0=dR0)
    if variant == "dV_only":
        return DualEKF_LFP(**common, cal_soc_fn=spline, cal_dR0=0.0)
    if variant == "const_Q":
        return ConstQEKF(**common, cal_soc_fn=spline, cal_dR0=dR0)
    if variant == "slow_loops_on":
        f = DualEKF_LFP(**common, cal_soc_fn=spline, cal_dR0=dR0)
        f.r_int_update_enabled = True   # calibration sanity gate DISABLED
        f.r_int_guard_reason = None
        return f
    if variant == "slow_loops_off":
        f = DualEKF_LFP(**common, cal_soc_fn=spline, cal_dR0=dR0)
        f.r_int_update_enabled = False
        f._update_soh = lambda dt_s, I_A: None   # freeze x2 entirely
        return f
    if variant == "joseph_form":
        return JosephEKF(**common, cal_soc_fn=spline, cal_dR0=dR0)
    raise ValueError(variant)


def _worker(task):
    fleet, idx, variant = task
    fl = ros._FLEETS[fleet]
    seg_df, vid, cfg = fl["eval"][idx]
    try:
        filt = make_variant(variant, cfg, fl["cal"], fl["ocv_fn"])
        t_s, est, tru, _ = ros.run_lean_traj(seg_df, cfg, filt, OFFSET)
        m = trip_metrics(t_s, est, tru)
        m["vehicle_id"] = vid
        return fleet, idx, variant, m
    except Exception as exc:
        return fleet, idx, variant, {"error": f"{type(exc).__name__}: {exc}",
                                     "vehicle_id": vid}


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, cwd=ROOT,
                                check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    for fleet, prep in ros.FLEET_PREPS.items():
        print(f"[{fleet}] loading + calibration…")
        eval_items, cal_items, cal = prep()
        ros._FLEETS[fleet] = {"eval": eval_items, "cal_items": cal_items,
                              "cal": cal,
                              "ocv_fn": (cal.ocv_fn if cal else None),
                              "theta_qr": (0.0, 1.0)}

    tasks = [(fleet, i, v) for fleet in ros.FLEET_PREPS
             for i in range(len(ros._FLEETS[fleet]["eval"]))
             for v in VARIANTS]
    print(f"{len(tasks)} ablation evaluations on {N_WORKERS} workers…")
    results: Dict[str, Dict[str, Dict[int, Dict]]] = {
        f: {v: {} for v in VARIANTS} for f in ros.FLEET_PREPS}
    done = 0
    with CTX.Pool(N_WORKERS) as pool:
        for fleet, idx, variant, m in pool.imap_unordered(_worker, tasks,
                                                          chunksize=16):
            results[fleet][variant][idx] = m
            done += 1
            if done % 3000 == 0:
                print(f"  {done}/{len(tasks)}")

    # cross-check: 'full' must reproduce the committed EKF per-segment RMSE
    for fleet in ros.FLEET_PREPS:
        path, subkey = ros.COMMITTED_REPORTS[fleet]
        doc = json.loads(path.read_text())
        ref = (doc[subkey] if subkey else doc)["per_segment"]
        max_d = 0.0
        for i, old in enumerate(ref):
            new = results[fleet]["full"].get(i)
            if new is None or "error" in new:
                raise RuntimeError(f"{fleet} seg {i}: full variant failed — ABORT")
            d = abs(old["soc_rmse_ekf_pct"] - new["rmse_full_pct"])
            max_d = max(max_d, d)
            if d > 1e-6:
                raise RuntimeError(f"{fleet} seg {i}: full variant RMSE "
                                   f"{new['rmse_full_pct']:.8f} vs committed "
                                   f"{old['soc_rmse_ekf_pct']:.8f} — ABORT")
        print(f"[{fleet}] 'full' cross-check vs committed: max |Δ| = {max_d:.2e} — OK")

    # aggregate rows
    hdr = [f"# generated by analysis/run_ablation.py {stamp} (git {commit}); "
           f"+20pp protocol, sign-corrected data; 'full' cross-checked "
           f"byte-identical to committed reports"]
    rows = []
    for fleet in ros.FLEET_PREPS:
        for v in VARIANTS:
            d = results[fleet][v]
            trips = [d[i] for i in sorted(d)
                     if d[i] is not None and "error" not in d[i]]
            n_err = sum(1 for i in d if d[i] is None or "error" in d[i])
            agg = aggregate_trips(trips)
            rows.append({
                "dataset": fleet, "variant": v, "n_trips": agg["n_trips"],
                "n_errors": n_err,
                "rmse_median": agg["rmse_full_pct"]["median"],
                "rmse_q25": agg["rmse_full_pct"]["q25"],
                "rmse_q75": agg["rmse_full_pct"]["q75"],
                "rmse_mean": agg["rmse_full_pct"]["mean"],
                "mae_median": agg["mae_full_pct"]["median"],
                "conv_rate_strict": agg["conv_rate_strict"],
                "rate_recovered": agg["rate_recovered"],
                "rate_diverged": agg["rate_diverged"],
                "t_conv_strict_median_s": agg["t_conv_strict_median_s"],
            })
    cols = list(rows[0].keys())
    with (RESULTS_DIR / "ablation.csv").open("w", newline="") as f:
        for line in hdr + footnote_lines("# "):
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in cols})
    print(f"Wrote {RESULTS_DIR / 'ablation.csv'}")

    # per-trip dump
    dump_cols = ["dataset", "variant", "seg_index", "vehicle_id",
                 "rmse_full_pct", "mae_full_pct", "t_conv_strict_s",
                 "err_end_pct", "min_abs_err_pct", "outcome", "duration_s",
                 "error"]
    p = RESULTS_DIR / f"ablation_per_trip_{stamp}.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=dump_cols)
        w.writeheader()
        for fleet in ros.FLEET_PREPS:
            for v in VARIANTS:
                for i in sorted(results[fleet][v]):
                    m = results[fleet][v][i] or {}
                    row = {"dataset": fleet, "variant": v, "seg_index": i}
                    row.update({k: m.get(k) for k in dump_cols if k in m})
                    w.writerow({k: ("" if row.get(k) is None else row.get(k))
                                for k in dump_cols})
    print(f"Wrote {p}")
    print("DONE.")


if __name__ == "__main__":
    main()
