#!/usr/bin/env python3
"""
data/run_main_table.py — Phase 1: standardized-metrics main table.

Re-runs the EXACT headline benchmark protocol (same loaders, same
calibration split and fitting, same seeds, same +20%-offset wrong init,
same per-segment evaluation order) on all five datasets, but instead of a
single full-trip RMSE per segment it records the full standardized metric
bundle from validation/metrics.py for each of the three methods
(dual EKF, pure coulomb counting, naive OCV lookup).

Consistency guarantee: for every segment, the full-trip EKF/coulomb/OCV
RMSE computed here is cross-checked against the committed
data/soc_baseline_benchmark*_report.json per-segment values; the script
ABORTS if any value differs by more than CROSS_CHECK_TOL. The estimator and
protocol are untouched — only the metrics layer is new.

Outputs (under results/):
  main_table.csv                        — aggregate, one row per dataset×method
  main_table.md                         — human-readable version + footnotes
  main_table_per_segment_<UTC-stamp>.csv — full per-segment metric dump
  main_table_run_<UTC-stamp>.json       — everything incl. meta/environment

One command: venv/bin/python data/run_main_table.py
Runtime: ~25–30 min (dominated by Deng 2000 sessions + VED 408 trips).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
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
    _split_by_vehicle, config_from_cartridge, run_mode_b_ekf,
)
from validation.metrics import aggregate_trips, footnote_lines, trip_metrics

RESULTS_DIR = ROOT / "results"
CROSS_CHECK_TOL = 1e-6      # pct points; pipeline is deterministic (Phase 0)
DENG_MAX_VEHICLES = 20
DENG_EVAL_SAMPLE_N = 2000
DENG_RNG_SEED = 42          # identical to data/soc_baseline_benchmark.py

METHODS = ("ekf", "coulomb", "ocv_lookup")

COMMITTED_REPORTS = {
    "BMW_i3":          (ROOT / "data" / "soc_baseline_benchmark_report.json", "BMW_i3"),
    "Deng_BAIC_EU500": (ROOT / "data" / "soc_baseline_benchmark_report.json", "Deng_BAIC_EU500"),
    "VED":             (ROOT / "data" / "soc_baseline_benchmark_report.json", "VED"),
    "CALCE_A123":      (ROOT / "data" / "soc_baseline_benchmark_calce_report.json", None),
    "Parallel_Module": (ROOT / "data" / "soc_baseline_benchmark_module_report.json", None),
}
_REPORT_RMSE_KEY = {
    "ekf": "soc_rmse_ekf_pct",
    "coulomb": "soc_rmse_coulomb_pct",
    "ocv_lookup": "soc_rmse_ocv_lookup_pct",
}


# ─────────────────────────────────────────────────────────────────────────────
# Fleet iteration — mirrors the run_fleet_* functions of the benchmark
# scripts exactly (order, splits, seeds, calibration), yielding
# (seg_df, vehicle_id, cfg) per held-out segment plus the fleet calibration.
# ─────────────────────────────────────────────────────────────────────────────

def _prep_bmw():
    from data.loaders.bmw_i3_loader import BMWI3Loader
    from data.loaders.pack_cartridge import BMW_I3_60AH

    all_pairs = list(BMWI3Loader(max_trips=None).iter_segments())
    cal_pairs, eval_pairs = _split_by_vehicle(all_pairs)
    cfg = config_from_cartridge("BMW_i3", BMW_I3_60AH, CellMode.AVG_CELL)
    cal = _build_calibration_for_fleet(cal_pairs, cfg, "BMW_i3") if cal_pairs else None
    return [(s, m.vehicle_id, cfg) for s, m in eval_pairs], cal


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
    return [(s, m.vehicle_id, cfg) for s, m in eval_pairs], cal


def _prep_ved():
    from data.loaders.ved_loader import VEDLoader
    from data.loaders.pack_cartridge import lookup_ved_cartridge

    all_pairs = list(VEDLoader(max_veh=None, max_trips_per_veh=None).iter_segments())

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
    cal = (_build_calibration_for_fleet(cal_pairs, sample_cfg, "VED")
           if (sample_cfg and cal_pairs) else None)
    return [(s, m.vehicle_id, _get_cfg(m)) for s, m in eval_pairs], cal


def _lab_meta_pairs(pairs, dataset: str, dt_median: float, has_T: bool):
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
            has_temperature=has_T, notes=[],
        )))
    return out


def _prep_calce():
    from data.soc_baseline_benchmark_calce import (
        CALCE_A123_CARTRIDGE, CALCE_DIR, CALCE_FILES, load_calce_file,
    )

    all_pairs, q_effs = [], []
    for fname in CALCE_FILES:
        path = CALCE_DIR / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        segs, q_eff = load_calce_file(path, fname.split("-")[0])
        q_effs.append(q_eff)
        all_pairs.extend(segs)
    if not all_pairs:
        return [], None

    cfg = config_from_cartridge("CALCE_A123", CALCE_A123_CARTRIDGE,
                                CellMode.AVG_CELL, dt_resample_s=5.0)
    cfg.q_cell_ah = float(np.mean(q_effs))

    by_cell: Dict[str, List] = {}
    for seg_df, seg_id in all_pairs:
        by_cell.setdefault(seg_id.split("_seg_")[0], []).append((seg_df, seg_id))
    cal_pairs, eval_pairs = [], []
    for cid, segs in by_cell.items():
        n_cal = max(1, int(len(segs) * 0.40))   # 40% — disclosed exception
        cal_pairs.extend(segs[:n_cal])
        eval_pairs.extend(segs[n_cal:])

    cal = _build_calibration_for_fleet(
        _lab_meta_pairs(cal_pairs, "calce_a123", 5.0, True), cfg, "CALCE_A123")
    return [(s, sid, cfg) for s, sid in eval_pairs], cal


def _prep_module():
    from data.soc_baseline_benchmark_module import (
        MODULE_CARTRIDGE, MODULE_FOLDERS, load_module_folder,
    )

    all_pairs, q_effs = [], []
    for folder in MODULE_FOLDERS:
        segs, q_eff = load_module_folder(folder)
        if segs:
            q_effs.append(q_eff)
            all_pairs.extend(segs)
    if not all_pairs:
        return [], None

    cfg = config_from_cartridge("Parallel_Module", MODULE_CARTRIDGE,
                                CellMode.AVG_CELL, dt_resample_s=2.0)
    cfg.q_cell_ah = float(np.mean(q_effs))

    by_mod: Dict[str, List] = {}
    for seg_df, seg_id in all_pairs:
        by_mod.setdefault(seg_id.split("_seg_")[0], []).append((seg_df, seg_id))
    cal_pairs, eval_pairs = [], []
    for mid, segs in by_mod.items():
        n_cal = max(1, int(len(segs) * 0.30))   # 30% — disclosed exception
        cal_pairs.extend(segs[:n_cal])
        eval_pairs.extend(segs[n_cal:])

    cal = _build_calibration_for_fleet(
        _lab_meta_pairs(cal_pairs, "parallel_module", 2.0, False), cfg, "Parallel_Module")
    return [(s, sid, cfg) for s, sid in eval_pairs], cal


FLEETS = {
    "BMW_i3": _prep_bmw,
    "Deng_BAIC_EU500": _prep_deng,
    "VED": _prep_ved,
    "CALCE_A123": _prep_calce,
    "Parallel_Module": _prep_module,
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment trajectory evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_segment_trajectories(
    seg_df: pd.DataFrame, cfg: ValidationConfig, cal, ocv_fn,
) -> Dict[str, Optional[Dict]]:
    """Run all three methods on one segment; return per-method trip_metrics
    (None where a method failed, mirroring evaluate_segment's try/except)."""
    t_s = seg_df["t_s"].values.astype(np.float64)
    soc_true = seg_df["SOC_bms"].values.astype(np.float64)
    out: Dict[str, Optional[Dict]] = {}

    soc_cc = coulomb_counting_soc(seg_df, cfg)
    out["coulomb"] = trip_metrics(t_s, soc_cc, soc_true)

    if ocv_fn is not None:
        try:
            soc_ocv = ocv_lookup_soc(seg_df, cfg, ocv_fn)
            out["ocv_lookup"] = trip_metrics(t_s, soc_ocv, soc_true)
        except Exception:
            out["ocv_lookup"] = None
    else:
        out["ocv_lookup"] = None

    try:
        gamma = cal.ekf_gamma if cal is not None else 1.0
        R_meas_V2 = cal.ekf_R_meas_V2 if cal is not None else 4e-6
        soc_true_out, soc_ekf, _, _ = run_mode_b_ekf(
            seg_df, cfg, ocv_fn=ocv_fn, calibration=cal,
            gamma=gamma, R_meas_V2=R_meas_V2,
        )
        out["ekf"] = trip_metrics(t_s, soc_ekf, soc_true_out)
    except Exception:
        out["ekf"] = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-check against committed headline reports
# ─────────────────────────────────────────────────────────────────────────────

def _committed_per_segment(fleet: str) -> List[Dict]:
    path, subkey = COMMITTED_REPORTS[fleet]
    doc = json.loads(path.read_text())
    return (doc[subkey] if subkey else doc)["per_segment"]


def cross_check(fleet: str, per_seg: List[Dict[str, Optional[Dict]]]) -> float:
    ref = _committed_per_segment(fleet)
    if len(ref) != len(per_seg):
        raise RuntimeError(
            f"{fleet}: segment count mismatch vs committed report "
            f"({len(per_seg)} new vs {len(ref)} committed) — ABORT")
    max_diff = 0.0
    for i, (new, old) in enumerate(zip(per_seg, ref)):
        for method, key in _REPORT_RMSE_KEY.items():
            old_v = old.get(key)
            new_m = new.get(method)
            new_v = None if new_m is None else new_m["rmse_full_pct"]
            if old_v is None and new_v is None:
                continue
            if (old_v is None) != (new_v is None):
                raise RuntimeError(f"{fleet} seg {i} {method}: one side None — ABORT")
            d = abs(old_v - new_v)
            max_diff = max(max_diff, d)
            if d > CROSS_CHECK_TOL:
                raise RuntimeError(
                    f"{fleet} seg {i} {method}: RMSE {new_v:.6f} vs committed "
                    f"{old_v:.6f} (diff {d:.2e} > {CROSS_CHECK_TOL}) — ABORT")
    return max_diff


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

AGG_COLUMNS = [
    "dataset", "method", "n_trips",
    "rmse_full_median", "rmse_full_q25", "rmse_full_q75", "rmse_full_mean",
    "mae_full_median", "mae_full_mean",
    "maxerr_full_median",
    "rmse_postconv_median", "rmse_postconv_mean", "n_postconv",
    "conv_rate_strict", "t_conv_strict_median_s",
    "conv_rate_hold", "t_conv_hold_median_s",
    "conv_rate_legacy", "t_conv_legacy_median_s",
]


def _agg_row(dataset: str, method: str, agg: Dict) -> Dict:
    return {
        "dataset": dataset, "method": method, "n_trips": agg["n_trips"],
        "rmse_full_median": agg["rmse_full_pct"]["median"],
        "rmse_full_q25":    agg["rmse_full_pct"]["q25"],
        "rmse_full_q75":    agg["rmse_full_pct"]["q75"],
        "rmse_full_mean":   agg["rmse_full_pct"]["mean"],
        "mae_full_median":  agg["mae_full_pct"]["median"],
        "mae_full_mean":    agg["mae_full_pct"]["mean"],
        "maxerr_full_median": agg["maxerr_full_pct"]["median"],
        "rmse_postconv_median": agg["rmse_postconv_pct"]["median"],
        "rmse_postconv_mean":   agg["rmse_postconv_pct"]["mean"],
        "n_postconv":           agg["rmse_postconv_pct"]["n"],
        "conv_rate_strict":       agg["conv_rate_strict"],
        "t_conv_strict_median_s": agg["t_conv_strict_median_s"],
        "conv_rate_hold":         agg["conv_rate_hold"],
        "t_conv_hold_median_s":   agg["t_conv_hold_median_s"],
        "conv_rate_legacy":       agg["conv_rate_legacy"],
        "t_conv_legacy_median_s": agg["t_conv_legacy_median_s"],
    }


def _fmt(v, nd=2):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_outputs(rows: List[Dict], per_segment_rows: List[Dict],
                  meta: Dict, stamp: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # main_table.csv — footnotes + provenance as leading # comment lines
    csv_path = RESULTS_DIR / "main_table.csv"
    with csv_path.open("w", newline="") as f:
        for line in ([f"# generated by data/run_main_table.py {stamp} "
                      f"(git {meta['git_commit']}); seeds: deng_eval_sample={DENG_RNG_SEED}"]
                     + footnote_lines("# ")):
            f.write(line + "\n")
        w = csv.DictWriter(f, fieldnames=AGG_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in AGG_COLUMNS})

    # main_table.md — readable
    md = ["# Main Table — standardized metrics (Phase 1)", "",
          f"Generated {stamp} by `data/run_main_table.py` (git `{meta['git_commit']}`). "
          f"All numbers in percentage points. Median (IQR) primary; mean kept for "
          f"comparability with pre-Phase-1 headline numbers (which were means of "
          f"per-trip full-window RMSE — see docs/METRICS.md).", "",
          "| Dataset | Method | n | RMSE med (IQR) | RMSE mean | MAE med | MaxErr med "
          "| RMSE post-conv med | Conv% strict | t_conv strict med (s) "
          "| Conv% hold-600 | Conv% legacy | t_conv legacy med (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        iqr = (f"{_fmt(r['rmse_full_median'])} ({_fmt(r['rmse_full_q25'])}–"
               f"{_fmt(r['rmse_full_q75'])})")
        md.append(
            f"| {r['dataset']} | {r['method']} | {r['n_trips']} | {iqr} "
            f"| {_fmt(r['rmse_full_mean'])} | {_fmt(r['mae_full_median'])} "
            f"| {_fmt(r['maxerr_full_median'])} | {_fmt(r['rmse_postconv_median'])} "
            f"| {_fmt(100 * r['conv_rate_strict'], 1)} | {_fmt(r['t_conv_strict_median_s'], 0)} "
            f"| {_fmt(100 * r['conv_rate_hold'], 1)} "
            f"| {_fmt(100 * r['conv_rate_legacy'], 1)} | {_fmt(r['t_conv_legacy_median_s'], 0)} |")
    md += ["", *footnote_lines("> ")]
    (RESULTS_DIR / "main_table.md").write_text("\n".join(md) + "\n")

    # per-segment dump (timestamped)
    seg_cols = ["dataset", "method", "vehicle_id", "seg_index",
                "rmse_full_pct", "mae_full_pct", "maxerr_full_pct",
                "rmse_postconv_pct", "mae_postconv_pct", "maxerr_postconv_pct",
                "t_conv_strict_s", "t_conv_hold_s", "t_conv_legacy_s",
                "duration_s", "n_samples"]
    seg_path = RESULTS_DIR / f"main_table_per_segment_{stamp}.csv"
    with seg_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seg_cols)
        w.writeheader()
        for r in per_segment_rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in seg_cols})

    # full JSON (timestamped)
    (RESULTS_DIR / f"main_table_run_{stamp}.json").write_text(json.dumps({
        "meta": meta, "aggregate_rows": rows,
    }, indent=2))

    print(f"\nWrote {csv_path}")
    print(f"Wrote {RESULTS_DIR / 'main_table.md'}")
    print(f"Wrote {seg_path}")
    print(f"Wrote {RESULTS_DIR / f'main_table_run_{stamp}.json'}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, cwd=ROOT,
                                check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    meta = {
        "script": "data/run_main_table.py", "utc": stamp, "git_commit": commit,
        "python": sys.version.split()[0],
        "seeds": {"deng_eval_sample": DENG_RNG_SEED},
        "protocol": "identical to data/soc_baseline_benchmark*.py (Phase 0 "
                    "reproduced byte-identically); metrics layer standardized "
                    "per validation/metrics.py",
        "cross_check_tol_pct": CROSS_CHECK_TOL,
    }

    print("=" * 78)
    print("PHASE 1 — standardized-metrics main table (5 datasets × 3 methods)")
    print("=" * 78)

    agg_rows: List[Dict] = []
    per_segment_rows: List[Dict] = []
    cross_check_diffs: Dict[str, float] = {}

    for fleet, prep in FLEETS.items():
        print(f"\n[{fleet}] loading + calibration…")
        eval_items, cal = prep()
        ocv_fn = cal.ocv_fn if cal else None
        print(f"[{fleet}] {len(eval_items)} held-out segments")

        per_seg: List[Dict[str, Optional[Dict]]] = []
        for idx, (seg_df, vid, cfg) in enumerate(eval_items):
            per_seg.append(eval_segment_trajectories(seg_df, cfg, cal, ocv_fn))
            if (idx + 1) % 100 == 0:
                print(f"  [{fleet}] {idx + 1}/{len(eval_items)} done")

        max_diff = cross_check(fleet, per_seg)
        cross_check_diffs[fleet] = max_diff
        print(f"[{fleet}] cross-check vs committed report: "
              f"max |ΔRMSE| = {max_diff:.2e} pct — OK")

        for method in METHODS:
            trips = [s[method] for s in per_seg if s[method] is not None]
            agg_rows.append(_agg_row(fleet, method, aggregate_trips(trips)))

        for idx, ((seg_df, vid, cfg), s) in enumerate(zip(eval_items, per_seg)):
            for method in METHODS:
                if s[method] is None:
                    continue
                row = {"dataset": fleet, "method": method,
                       "vehicle_id": vid, "seg_index": idx}
                row.update(s[method])
                per_segment_rows.append(row)

    meta["cross_check_max_diff_pct"] = cross_check_diffs
    write_outputs(agg_rows, per_segment_rows, meta, stamp)

    # console summary
    print("\n" + "=" * 118)
    hdr = (f"{'dataset':18s} {'method':11s} {'n':>5} {'RMSEmed':>8} {'RMSEmean':>9} "
           f"{'MAEmed':>7} {'convS%':>7} {'tS_med':>7} {'convH%':>7} "
           f"{'convL%':>7} {'tL_med':>7}")
    print(hdr)
    print("-" * 118)
    for r in agg_rows:
        print(f"{r['dataset']:18s} {r['method']:11s} {r['n_trips']:>5d} "
              f"{_fmt(r['rmse_full_median']):>8} {_fmt(r['rmse_full_mean']):>9} "
              f"{_fmt(r['mae_full_median']):>7} "
              f"{_fmt(100*r['conv_rate_strict'],1):>7} {_fmt(r['t_conv_strict_median_s'],0):>7} "
              f"{_fmt(100*r['conv_rate_hold'],1):>7} "
              f"{_fmt(100*r['conv_rate_legacy'],1):>7} {_fmt(r['t_conv_legacy_median_s'],0):>7}")


if __name__ == "__main__":
    main()
