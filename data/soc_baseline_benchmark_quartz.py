#!/usr/bin/env python3
"""
data/soc_baseline_benchmark_quartz.py
========================================
Same fair EKF vs coulomb-counting-only vs naive-OCV-lookup-only benchmark,
run on the Quartz WLTP dataset: a REAL 36-cell (3 parallel x 12 series)
NMC811 pack under a WLTP (Worldwide Harmonised Light-vehicle Test
Procedure) driving cycle -- lab-controlled, multi-cell, dynamic real-driving
current profile. This is the genuinely representative "how EV packs
actually get used" test this project's headline MAE=18.6mV / R2=0.9217
result (data/validate_quartz.py) is already built on -- but that script
never compared against coulomb-counting-only or naive-OCV-lookup-only
baselines. This module adds that comparison, on the SAME data.

WHY THIS DATASET (vs CALCE, vs the abandoned iontech_lfp field data)
------------------------------------------------------------------------
- Multi-cell PACK (36 cells), not a single standalone cell -- addresses the
  concern that CALCE's single-cell result might not reflect how this
  project's calibration pipeline behaves on real packs (its actual design
  target, per validate_generic.py's n_series/n_parallel-aware config).
- Real WLTP driving-cycle current profile (not synthetic, not a simple
  constant-current discharge) -- genuinely representative of real-life
  battery usage, which is what was asked for.
- Lab-controlled, healthy pack (not a field-returned/flagged unit like the
  abandoned iontech_lfp dataset) -- ground truth (SoC_Actual_Battery) comes
  from bench instrumentation on a working pack, not a device that was
  returned for "unsatisfactory behavior."
- Already the primary validated dataset in this project (18.6mV MAE, R2=0.92
  headline figure) -- reusing it here is not a new download, and any result
  is directly comparable to that established figure.

ADAPTER, NOT A NEW LOADER
----------------------------
Reuses the parquet files in data/quartz_wltp/ directly (same files
validate_quartz.py already uses). Adapts them into this project's t_s/I_A/
V_V/SOC_bms schema at AVG_CELL granularity (matching the CellMode.AVG_CELL
convention already used for BMW i3/Deng/VED elsewhere in this project):
  I_A   = -Current_Actual_Battery / N_P     (per-string current; sign
          verified empirically -- Current_Actual_Battery > 0 during
          charging, opposite of this project's I_A>0=discharge convention)
  V_V   = mean of all 36 Voltage_Cell_P{p}S{s} columns  (avg cell voltage)
  SOC_bms = SoC_Actual_Battery [percent] / 100
Cartridge: n_series=12, n_parallel=3, chemistry=NMC, Q_cell_Ah=2.5 (Q_QUARTZ,
same constant validate_quartz.py itself uses).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loaders.common_schema import make_schema_df, split_segments, SegmentMeta
from data.loaders.pack_cartridge import PackCartridge
from data.validate_generic import (
    ValidationConfig, CellMode, config_from_cartridge, _build_calibration_for_fleet,
)
from data.soc_baseline_benchmark import evaluate_segment, _aggregate

QUARTZ_DIR = ROOT / "data" / "quartz_wltp"
N_P, N_S = 3, 12
Q_QUARTZ_AH = 2.5

QUARTZ_CARTRIDGE = PackCartridge(
    name="Quartz WLTP 36-cell NMC811 pack (3P12S)",
    n_series=N_S, n_parallel=N_P, chemistry="NMC",
    Q_cell_Ah=Q_QUARTZ_AH, R_ohm_cell=0.010, V_nom_pack=44.4,
    source="Same constants as data/validate_quartz.py (Q_QUARTZ=2.5 Ah).",
    topology_uncertain=False,
)


def load_quartz_file(parquet_path: Path, file_id: str,
                      window_s: float = 1800.0) -> List[Tuple[pd.DataFrame, str]]:
    df = pd.read_parquet(parquet_path)
    if "SoC_Actual_Battery [percent]" not in df.columns:
        return []

    v_cols = [f"Voltage_Cell_P{p}S{s} [V]" for p in range(1, N_P + 1) for s in range(1, N_S + 1)]
    v_cols = [c for c in v_cols if c in df.columns]
    if not v_cols:
        return []

    t0 = df["Timestamp"].iloc[0]
    t_s = (df["Timestamp"] - t0).dt.total_seconds().values.astype(np.float64)
    I_A = (-df["Current_Actual_Battery [A]"].values.astype(np.float64)) / N_P
    V_V = df[v_cols].mean(axis=1).values.astype(np.float64)
    SOC_bms = df["SoC_Actual_Battery [percent]"].values.astype(np.float64) / 100.0
    T_degC = np.full(len(df), np.nan)

    schema_df = make_schema_df(t_s, I_A, V_V, T_degC, SOC_bms)
    gap_free = list(split_segments(schema_df, dataset="quartz_wltp",
                                    vehicle_id=file_id, gap_thresh_s=120.0, min_rows=100))

    out: List[Tuple[pd.DataFrame, str]] = []
    for seg_df, meta in gap_free:
        t = seg_df["t_s"].values
        n_windows = max(1, int(np.ceil((t[-1] - t[0]) / window_s)))
        for w in range(n_windows):
            lo, hi = w * window_s, (w + 1) * window_s
            mask = (t >= lo) & (t < hi)
            if mask.sum() < 50:
                continue
            win_df = seg_df.iloc[mask].copy().reset_index(drop=True)
            win_df["t_s"] = win_df["t_s"] - win_df["t_s"].iloc[0]
            out.append((win_df, f"{file_id}_{meta.segment_id}_win{w:03d}"))
    return out


def run_fleet_quartz(max_files: int = 6) -> List[Dict]:
    files = sorted(QUARTZ_DIR.glob("*WLTP*.parquet"))[:max_files]
    if not files:
        print("  [ERROR] No WLTP parquet files found.")
        return []

    all_pairs: List[Tuple[pd.DataFrame, str]] = []
    for f in files:
        file_id = f.stem.replace("Qtzl_Cycle_", "cyc_").split("_WLTP")[0]
        segs = load_quartz_file(f, file_id)
        print(f"  {f.name}: {len(segs)} windows")
        all_pairs.extend(segs)

    if not all_pairs:
        return []

    cfg = config_from_cartridge("Quartz_WLTP", QUARTZ_CARTRIDGE, CellMode.AVG_CELL, dt_resample_s=1.0)

    by_file: Dict[str, List] = {}
    for seg_df, seg_id in all_pairs:
        fid = seg_id.split("_seg_")[0]
        by_file.setdefault(fid, []).append((seg_df, seg_id))

    cal_pairs, eval_pairs = [], []
    for fid, segs in by_file.items():
        n_cal = max(1, int(len(segs) * 0.20))
        cal_pairs.extend(segs[:n_cal])
        eval_pairs.extend(segs[n_cal:])
    print(f"  {len(cal_pairs)} calibration / {len(eval_pairs)} held-out windows")

    def _to_meta_pairs(pairs):
        out = []
        for seg_df, seg_id in pairs:
            meta = SegmentMeta(
                dataset="quartz_wltp", vehicle_id=seg_id, segment_id=seg_id,
                n_rows=len(seg_df), dt_s_median=1.0,
                duration_s=float(seg_df["t_s"].iloc[-1]),
                soc_start=float(seg_df["SOC_bms"].iloc[0]),
                soc_end=float(seg_df["SOC_bms"].iloc[-1]),
                I_mean_A=float(seg_df["I_A"].mean()), V_mean_V=float(seg_df["V_V"].mean()),
                T_mean_degC=float("nan"), has_temperature=False, notes=[],
            )
            out.append((seg_df, meta))
        return out

    cal_meta_pairs = _to_meta_pairs(cal_pairs)
    cal = _build_calibration_for_fleet(cal_meta_pairs, cfg, "Quartz_WLTP") if cal_meta_pairs else None
    ocv_fn = cal.ocv_fn if cal else None

    results = []
    for idx, (seg_df, seg_id) in enumerate(eval_pairs):
        r = evaluate_segment(seg_df, cfg, cal, ocv_fn)
        r["vehicle_id"] = seg_id
        results.append(r)
        if (idx + 1) % 20 == 0:
            print(f"  Quartz: {idx + 1}/{len(eval_pairs)} held-out done")
    return results


def main():
    print("=" * 78)
    print("SOC baseline benchmark -- Quartz WLTP 36-cell NMC811 pack (multi-cell, real driving cycle)")
    print("=" * 78)

    results = run_fleet_quartz()
    if not results:
        return None

    agg = _aggregate(results)
    print(f"\n--- Quartz WLTP (n={agg['n_segments']} held-out windows) ---")
    print(f"  EKF SOC RMSE:              {agg['ekf_soc_rmse_pct_mean']:.2f}%  (n={agg['ekf_n_valid']})")
    print(f"  Pure coulomb counting:     {agg['coulomb_only_soc_rmse_pct_mean']:.2f}%  (n={agg['coulomb_only_n_valid']})")
    print(f"  Pure OCV lookup (naive):   {agg['ocv_lookup_only_soc_rmse_pct_mean']:.2f}%  (n={agg['ocv_lookup_only_n_valid']})")
    print(f"  EKF beats coulomb-only: {agg['ekf_beats_coulomb_only']}   "
          f"EKF beats OCV-lookup-only: {agg['ekf_beats_ocv_lookup_only']}")

    import json
    def _serial(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, bool): return o
        if isinstance(o, dict): return {str(k): _serial(v) for k, v in o.items()}
        if isinstance(o, list): return [_serial(v) for v in o]
        return o

    report = {
        "meta": {
            "dataset": "Quartz WLTP 36-cell NMC811 pack (3P12S), real driving-cycle, lab-controlled",
            "note": "Multi-cell pack, addresses whether CALCE's single-cell "
                    "calibration instability is single-cell-specific.",
        },
        "aggregate": agg, "per_segment": results,
    }
    out_path = ROOT / "data" / "soc_baseline_benchmark_quartz_report.json"
    out_path.write_text(json.dumps(_serial(report), indent=2))
    print(f"\nJSON report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
