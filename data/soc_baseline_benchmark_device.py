#!/usr/bin/env python3
"""
data/soc_baseline_benchmark_device.py
========================================
Same fair EKF vs coulomb-counting-only vs naive-OCV-lookup-only benchmark as
soc_baseline_benchmark.py, but on a DEVICE dataset instead of a vehicle fleet:
portable LFP battery systems (24V, 8 prismatic LFP cells in series, ~160 Ah
nominal, used for recreational-vehicle/solar/off-grid power). This is the
closer analogue to what TI/ADI fuel-gauge chips actually target (portable,
non-automotive power systems) than the EV fleets tested elsewhere.

DATA SOURCE
-----------
28 systems, field-returned to the manufacturer for various reasons (dataset
is explicitly documented by its authors as NOT representative of the whole
population -- a known bias, stated here rather than glossed over). Source:
Zenodo record 13715694, "Lithium-Ion Battery Field Data: 28 LFP battery
systems with 8 cells in series, up to 5 years of operation." Already present
in this project at data/iontech_lfp/field_data.zip (downloaded 2026-07-07,
verified byte-for-byte against the Zenodo record's reported size before
reuse -- no re-download performed this session).

Two of the smaller systems (data_sys_28.csv, data_sys_26.csv -- 39 MB and 78
MB) were extracted for this benchmark; the other 26 systems (up to 2+ GB
each, ~20 GB uncompressed total) were NOT processed, for tractability. This
is a disclosed subsample, not the full dataset.

CURRENT SIGN CONVENTION -- VERIFIED, NOT ASSUMED
--------------------------------------------------
Checked empirically before writing the loader: I_Battery > 0 during rising
SOC (charging), I_Battery < 0 during falling SOC (discharging) -- the
OPPOSITE of this project's established vehicle-fleet convention (I_A > 0 =
discharge, per SOURCES.md's Deng documentation). Sign is flipped on load
(I_A = -I_Battery) so this dataset is directly comparable to the vehicle
benchmark's numbers.

CARTRIDGE
---------
n_series=8, n_parallel=1, chemistry=LFP, Q_cell_Ah=160 (README: "8
prismatic cells in series... approximately 160 Ah nominal capacity" -- for
series cells, capacity equals the single-cell/pack capacity, no scaling
needed). R_ohm_cell is a generic LFP prismatic-cell estimate, NOT fleet-
fitted -- flagged, same honesty standard as GENERIC_EV_PACK elsewhere in
this project.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loaders.common_schema import make_schema_df, split_segments, normalise_soc
from data.loaders.pack_cartridge import PackCartridge
from data.validate_generic import (
    ValidationConfig, CellMode, config_from_cartridge, _build_calibration_for_fleet,
)
from data.soc_baseline_benchmark import evaluate_segment, _aggregate

EXTRACTED_DIR = ROOT / "data" / "iontech_lfp" / "extracted"
# 8 of 28 systems (the 8 smallest files, ~1.0 GB total) -- picked purely by
# file size for tractability, not cherry-picked by result. Chosen specifically
# to give the near-rest OCV calibration enough real data to avoid the generic
# fallback table seen with only 2 systems.
SYSTEM_FILES = [
    "data_sys_28.csv", "data_sys_25.csv", "data_sys_26.csv", "data_sys_3.csv",
    "data_sys_24.csv", "data_sys_27.csv", "data_sys_12.csv", "data_sys_23.csv",
]

IONTECH_LFP_CARTRIDGE = PackCartridge(
    name="Portable LFP RV/Solar Pack (Zenodo 13715694)",
    n_series=8,
    n_parallel=1,
    chemistry="LFP",
    Q_cell_Ah=160.0,
    R_ohm_cell=0.008,   # generic LFP prismatic-cell estimate, NOT fleet-fitted -- flagged
    V_nom_pack=25.6,    # 8 * 3.2V nominal LFP
    source="Zenodo 13715694 README (8s, ~160Ah); R_ohm_cell is a generic "
           "estimate, not measured for this specific product.",
    topology_uncertain=False,
)


def load_system_segments(csv_path: Path, system_id: str,
                          gap_thresh_s: float = 600.0,
                          window_s: float = 86400.0) -> List[Tuple[pd.DataFrame, str]]:
    """
    Load one portable-LFP system CSV, split into gap-free segments, then
    further chop each gap-free segment into fixed-size windows (default 24h).
    This dataset logs continuously for months with almost no real gaps, so
    gap-based splitting alone yields only ~1-4 segments per system (too few
    to evaluate meaningfully) -- daily-window chopping gives a larger,
    still-realistic sample of independent evaluation units, the same way
    BMW i3/VED naturally split into per-trip segments.
    """
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    t0 = df["Timestamp"].iloc[0]
    t_s = (df["Timestamp"] - t0).dt.total_seconds().values

    I_A = -df["I_Battery"].values.astype(np.float64)   # sign-flipped, see module docstring
    V_V = df["U_Battery"].values.astype(np.float64)
    SOC_bms = normalise_soc(df["SOC_Battery"]).values
    T_degC = df[[c for c in df.columns if c.startswith("Temperature_")]].mean(axis=1).values

    schema_df = make_schema_df(t_s, I_A, V_V, T_degC, SOC_bms)
    gap_free = list(split_segments(schema_df, dataset="iontech_lfp",
                                    vehicle_id=system_id, gap_thresh_s=gap_thresh_s,
                                    min_rows=200))

    out: List[Tuple[pd.DataFrame, str]] = []
    for seg_df, meta in gap_free:
        t = seg_df["t_s"].values
        n_windows = max(1, int(np.ceil((t[-1] - t[0]) / window_s)))
        for w in range(n_windows):
            lo, hi = w * window_s, (w + 1) * window_s
            mask = (t >= lo) & (t < hi)
            if mask.sum() < 60:   # need at least ~1h of 60s-sampled data
                continue
            win_df = seg_df.iloc[mask].copy().reset_index(drop=True)
            win_df["t_s"] = win_df["t_s"] - win_df["t_s"].iloc[0]
            out.append((win_df, f"{system_id}_{meta.segment_id}_win{w:03d}"))
    return out


def run_fleet_iontech_lfp() -> List[Dict]:
    cfg = config_from_cartridge("iontech_LFP_device", IONTECH_LFP_CARTRIDGE,
                                 CellMode.AVG_CELL, dt_resample_s=60.0)

    all_pairs: List[Tuple[pd.DataFrame, str]] = []
    for fname in SYSTEM_FILES:
        path = EXTRACTED_DIR / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found at {path}")
            continue
        sys_id = fname.replace("data_sys_", "sys_").replace(".csv", "")
        segs = load_system_segments(path, sys_id)
        print(f"  {fname}: {len(segs)} gap-free segments")
        all_pairs.extend(segs)

    if not all_pairs:
        print("  [ERROR] No segments loaded.")
        return []

    # Calibration split: first segment of each system for calibration, rest held-out
    # (mirrors _split_by_vehicle's per-vehicle first-10% logic at a system level)
    by_system: Dict[str, List] = {}
    for seg_df, seg_id in all_pairs:
        sys_id = seg_id.split("_seg_")[0]
        by_system.setdefault(sys_id, []).append((seg_df, seg_id))

    cal_pairs, eval_pairs = [], []
    for sys_id, segs in by_system.items():
        n_cal = max(1, int(len(segs) * 0.10))
        cal_pairs.extend(segs[:n_cal])
        eval_pairs.extend(segs[n_cal:])
    print(f"  {len(cal_pairs)} calibration / {len(eval_pairs)} held-out segments")

    # _build_calibration_for_fleet expects (seg_df, meta)-like pairs with meta.notes;
    # build lightweight meta objects to reuse it unmodified.
    from data.loaders.common_schema import SegmentMeta

    def _to_meta_pairs(pairs):
        out = []
        for seg_df, seg_id in pairs:
            meta = SegmentMeta(
                dataset="iontech_lfp", vehicle_id=seg_id, segment_id=seg_id,
                n_rows=len(seg_df), dt_s_median=60.0,
                duration_s=float(seg_df["t_s"].iloc[-1]),
                soc_start=float(seg_df["SOC_bms"].iloc[0]),
                soc_end=float(seg_df["SOC_bms"].iloc[-1]),
                I_mean_A=float(seg_df["I_A"].mean()), V_mean_V=float(seg_df["V_V"].mean()),
                T_mean_degC=float(seg_df["T_degC"].mean()), has_temperature=True, notes=[],
            )
            out.append((seg_df, meta))
        return out

    cal_meta_pairs = _to_meta_pairs(cal_pairs)
    cal = _build_calibration_for_fleet(cal_meta_pairs, cfg, "iontech_LFP") if cal_meta_pairs else None
    ocv_fn = cal.ocv_fn if cal else None

    results = []
    for idx, (seg_df, seg_id) in enumerate(eval_pairs):
        r = evaluate_segment(seg_df, cfg, cal, ocv_fn)
        r["vehicle_id"] = seg_id
        results.append(r)
        if (idx + 1) % 20 == 0:
            print(f"  iontech_lfp: {idx + 1}/{len(eval_pairs)} held-out done")
    return results


def main():
    print("=" * 78)
    print("SOC baseline benchmark -- DEVICE dataset (portable LFP RV/solar systems)")
    print("Subsample: 2 of 28 systems (data_sys_28, data_sys_26)")
    print("=" * 78)

    results = run_fleet_iontech_lfp()
    if not results:
        return None

    agg = _aggregate(results)
    print(f"\n--- iontech_lfp device (n={agg['n_segments']} held-out segments) ---")
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
            "dataset": "Portable LFP RV/solar systems, Zenodo 13715694",
            "systems_used": SYSTEM_FILES,
            "systems_total_in_dataset": 28,
            "known_bias": "Dataset authors state all systems were field-returned "
                           "for unsatisfactory behavior -- not representative of "
                           "the full sold population.",
        },
        "aggregate": agg,
        "per_segment": results,
    }
    out_path = ROOT / "data" / "soc_baseline_benchmark_device_report.json"
    out_path.write_text(json.dumps(_serial(report), indent=2))
    print(f"\nJSON report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
