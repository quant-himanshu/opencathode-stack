#!/usr/bin/env python3
"""
analysis/diagnose_dataset.py — Phase 4 failure diagnosis on the datasets
where the Dual EKF loses to coulomb counting on sign-corrected data
(VED, CALCE; median full-trip RMSE at +20 pp).

Per held-out trip (my EKF, +20 pp, exactly the headline configuration):
  innovation statistics    mean (mV), std, lag-1 autocorrelation
                           (whiteness proxy: |ρ₁| ≈ 0 for a healthy filter)
  innovation-vs-SOC shape  binned medians → structured RMS (OCV-mismatch
                           proxy AFTER the δV/δR0 correction)
  flat-OCV exposure        fraction of samples with |∂OCV/∂SOC| < 0.1 V/SOC
                           at the true SOC (voltage-observability proxy)
  sensor characteristics   median dt, current quantization step estimate
  slow-loop state          R_int gate (calibration sanity guard) status,
                           SOH/R_int end values
  rmse / t_conv / outcome

Fleet-level: calibration provenance (δR0 vs physical range, δV(SOC) range,
OCV source incl. generic-table-fallback detection — hypothesis (d)),
γ, R_meas.

Outputs: results/diagnose_<dataset>.json + per-trip CSV (timestamped).
One command: venv/bin/python -u analysis/diagnose_dataset.py --dataset ved
             venv/bin/python -u analysis/diagnose_dataset.py --dataset calce
Read-only: no estimator or pipeline changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data.run_offset_sweep as ros
from diagnosis.dual_ekf_lfp import DualEKF_LFP

RESULTS_DIR = ROOT / "results"
OFFSET = 0.20
FLAT_SLOPE_V = 0.1
DATASET_KEY = {"ved": "VED", "calce": "CALCE_A123"}


def diagnose_segment(seg_df, cfg, cal, ocv_fn) -> Dict:
    from validation.metrics import trip_metrics

    gamma = cal.ekf_gamma if cal else 1.0
    R_meas = cal.ekf_R_meas_V2 if cal else 4e-6
    ekf = DualEKF_LFP(Q_nom_Ah=cfg.q_cell_ah, R_int_ohm=cfg.r_ohm_cell,
                      ocv_fn=ocv_fn, R_meas_V2=R_meas, P0_soc=OFFSET ** 2,
                      gamma=gamma,
                      cal_soc_fn=(cal.soc_cal_fn() if cal else None),
                      cal_dR0=(cal.delta_R0 if cal else 0.0))
    t_s = seg_df["t_s"].values.astype(np.float64)
    I_cell = seg_df["I_A"].values.astype(np.float64) / cfg.n_parallel
    V_cell = seg_df["V_V"].values.astype(np.float64) / cfg.n_series
    soc_bms = seg_df["SOC_bms"].values.astype(np.float64)
    T_arr = seg_df["T_degC"].values.astype(np.float64)

    ekf.set_soc(float(np.clip(soc_bms[0] + OFFSET, 0.02, 0.98)))
    est = np.empty(len(t_s))
    innov = np.full(len(t_s), np.nan)
    for i in range(len(t_s)):
        dt = float(t_s[i] - t_s[i - 1]) if i > 0 else 1.0
        T = float(T_arr[i]) if np.isfinite(T_arr[i]) else 25.0
        try:
            r = ekf.update(float(V_cell[i]), -float(I_cell[i]), dt, T)
            est[i] = float(r["soc"])
            innov[i] = float(r["innovation"])
        except Exception:
            est[i] = float(ekf.x1[0])

    m = trip_metrics(t_s, est, soc_bms)
    v = innov[np.isfinite(innov)]
    lag1 = (float(np.corrcoef(v[:-1], v[1:])[0, 1])
            if len(v) > 10 and np.std(v) > 0 else None)

    # innovation-vs-true-SOC structure (post-correction OCV mismatch proxy)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(soc_bms[np.isfinite(innov)], edges) - 1, 0, 9)
    meds = [float(np.median(v[idx == b])) for b in range(10)
            if (idx == b).sum() >= 10]
    struct_rms_mV = (float(np.sqrt(np.mean((np.array(meds)
                                            - np.mean(meds)) ** 2))) * 1000.0
                     if len(meds) >= 3 else None)

    # flat-OCV exposure at true SOC
    h = 0.005
    fn = ocv_fn if ocv_fn is not None else ekf._ocv
    sub = np.clip(soc_bms[:: max(1, len(soc_bms) // 300)], 0.01, 0.99)
    slopes = np.array([(float(fn(x + h)) - float(fn(x - h))) / (2 * h)
                       for x in sub])
    frac_flat = float(np.mean(np.abs(slopes) < FLAT_SLOPE_V))

    # current sensor characteristics
    dts = np.diff(t_s)
    iq = np.unique(np.round(np.abs(np.diff(np.sort(np.unique(I_cell)))), 6))
    iq = iq[iq > 1e-9]
    quant = float(iq.min()) if len(iq) else None

    return {
        **{k: m[k] for k in ("rmse_full_pct", "mae_full_pct",
                             "t_conv_strict_s", "err_end_pct",
                             "min_abs_err_pct", "outcome", "duration_s",
                             "n_samples")},
        "innov_mean_mV": float(np.mean(v)) * 1000.0 if len(v) else None,
        "innov_std_mV": float(np.std(v)) * 1000.0 if len(v) else None,
        "innov_lag1_autocorr": lag1,
        "innov_soc_struct_rms_mV": struct_rms_mV,
        "frac_flat_ocv": frac_flat,
        "dt_median_s": float(np.median(dts)) if len(dts) else None,
        "I_quant_step_A": quant,
        "r_int_gate_enabled": bool(ekf.r_int_update_enabled),
        "soh_end": float(ekf.x2[0]),
        "r_int_end_mOhm": float(ekf.x2[1]) * 1000.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_KEY), required=True)
    args = ap.parse_args()
    fleet = DATASET_KEY[args.dataset]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"[{fleet}] loading + calibration…")
    eval_items, cal_items, cal = ros.FLEET_PREPS[fleet]()
    ocv_fn = cal.ocv_fn if cal else None

    per_trip: List[Dict] = []
    for i, (seg_df, vid, cfg) in enumerate(eval_items):
        d = diagnose_segment(seg_df, cfg, cal, ocv_fn)
        d["vehicle_id"] = vid
        d["seg_index"] = i
        per_trip.append(d)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(eval_items)}")

    def _med(key):
        vals = [t[key] for t in per_trip if t.get(key) is not None]
        return float(np.median(vals)) if vals else None

    summary = {
        "dataset": fleet, "stamp": stamp, "n_trips": len(per_trip),
        "calibration": {
            "delta_R0_mOhm": cal.delta_R0 * 1000.0,
            "delta_R0_physical(|dR0|<50mOhm)": bool(abs(cal.delta_R0) < 0.05),
            "delta_V_OLS_mV": cal.delta_V * 1000.0,
            "dv_soc_range_mV": ([float(cal.dv_knots.min() * 1000),
                                 float(cal.dv_knots.max() * 1000)]
                                if cal.dv_knots is not None else None),
            "n_cal_segments": cal.n_cal_segments,
            "gamma": cal.ekf_gamma,
            "R_meas_mV": float(cal.ekf_R_meas_V2 ** 0.5 * 1000),
            "ocv_source": cal.ocv_source,
            "ocv_generic_fallback": ("generic" in cal.ocv_source.lower()),
            "r_int_gate_fired": not bool(
                abs(cal.delta_R0) < 0.05),
        },
        "medians": {k: _med(k) for k in (
            "rmse_full_pct", "innov_mean_mV", "innov_std_mV",
            "innov_lag1_autocorr", "innov_soc_struct_rms_mV",
            "frac_flat_ocv", "dt_median_s", "I_quant_step_A",
            "duration_s", "soh_end", "r_int_end_mOhm")},
        "outcome_counts": {o: sum(1 for t in per_trip if t["outcome"] == o)
                           for o in ("converged", "recovered", "diverged")},
    }
    out_json = RESULTS_DIR / f"diagnose_{args.dataset}.json"
    out_json.write_text(json.dumps({"summary": summary,
                                    "per_trip": per_trip}, indent=1,
                                   default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(f"Wrote {out_json}")

    cols = ["seg_index", "vehicle_id", "rmse_full_pct", "outcome",
            "t_conv_strict_s", "innov_mean_mV", "innov_std_mV",
            "innov_lag1_autocorr", "innov_soc_struct_rms_mV",
            "frac_flat_ocv", "dt_median_s", "I_quant_step_A",
            "duration_s", "soh_end", "r_int_end_mOhm"]
    p = RESULTS_DIR / f"diagnose_{args.dataset}_per_trip_{stamp}.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in per_trip:
            w.writerow({k: ("" if t.get(k) is None else t.get(k))
                        for k in cols})
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
