#!/usr/bin/env python3
"""
data/soc_baseline_benchmark_module.py
========================================
Same fair EKF vs coulomb-counting-only vs naive-OCV-lookup-only benchmark,
run on a GENUINELY NEW dataset (not previously in this project, downloaded
this session): "Experimental Characterization Data for Battery Modules with
Parallel-Connected Cells across Diverse Module-Level State of Health and
Cell-to-Cell Variations" -- Mendeley Data DOI 10.17632/ssrgfmb8vw.2,
descriptor paper arXiv:2604.16769 (University of Michigan + Ford Motor
Company). Real, multi-cell (3 parallel-connected cells per module), lab-
controlled, healthy modules (NOT field-returned/flagged units), purpose-
built for state-of-health/state estimation research.

WHY THIS DATASET
------------------
- Genuinely NEW to this project -- not Quartz WLTP (already used for the
  project's headline result) and not a variant of an existing dataset.
- Multi-cell (3 parallel-connected cells) -- real pack topology, addresses
  the same "is this project's calibration pipeline pack-specific" question
  the single-cell CALCE result raised, but on different, new data.
- 78 modules spanning M-SoH 80.98%-100% and a range of cell-to-cell
  variation -- realistic manufacturing/aging heterogeneity, not one
  idealised cell.
- Ground truth: Maccor S4000 (reference-grade lab cycler), CC-CV
  charge / CC discharge -- not an onboard BMS from a flagged/faulty field
  unit.
- Total dataset ~191 MB; 4 held-out module folders (16 files, ~11 MB)
  downloaded for this benchmark -- a disclosed subsample, not the full 78
  modules, for tractability.

DATA FORMAT
-----------
Each module folder: Cell_A.csv, Cell_B.csv, Cell_C.csv (the 3 parallel
cells, per-cell current/voltage), Module.csv (aggregate pack-level
current/voltage -- what feeds this benchmark, matching the AVG_CELL/pack
convention used elsewhere in this project). Columns: Current_A, Voltage_V,
Time_s, Charged_Capacity_Ah, Discharged_Capacity_Ah, Cycle_Index.

SOC derivation: same disclosed method as the CALCE benchmark --
    SOC(t) = 1.0 - (Discharged_Capacity(t) - Charged_Capacity(t) - offset) / Q_eff
Q_eff measured from the file's own net-capacity range (~8.3 Ah for a
3-parallel NCA-class module, consistent with ~2.5-3 Ah per cell x 3).

CURRENT SIGN -- VERIFIED, NOT ASSUMED
----------------------------------------
Current_A < 0 during discharge, > 0 during charge (checked via correlation
with which capacity column increases) -- flipped on load (I_A = -Current_A),
same as every other non-vehicle dataset checked this session.

CARTRIDGE
---------
n_series=1, n_parallel=3 (three parallel cells, no series stacking).
Chemistry set to NCA (nearest supported chemistry to the parallel-module
literature's stated cell type); NOT independently re-verified against this
specific dataset's own cell datasheet -- flagged, same honesty standard
applied to every cartridge in this project.
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

MODULE_DIR = ROOT / "data" / "parallel_module_dataset"
MODULE_FOLDERS = ["8e59a64e", "b1467b68", "6b39b077", "fc3de35f"]

MODULE_CARTRIDGE = PackCartridge(
    name="UMich/Ford 3-parallel-cell module (Mendeley ssrgfmb8vw)",
    n_series=1, n_parallel=3, chemistry="NCA",
    Q_cell_Ah=2.8,   # placeholder, overwritten per-file from observed Q_eff
    R_ohm_cell=0.006, V_nom_pack=3.6,
    source="Mendeley Data DOI 10.17632/ssrgfmb8vw.2, arXiv:2604.16769. "
           "Chemistry set to NCA per descriptor paper's stated cell family; "
           "not independently re-verified against this dataset's own "
           "datasheet -- flagged.",
    topology_uncertain=False,
)


def load_module_folder(folder_prefix: str, window_s: float = 900.0) -> List[Tuple[pd.DataFrame, str]]:
    path = MODULE_DIR / f"{folder_prefix}_Module.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)

    t_s = df["Time_s"].values.astype(np.float64)
    I_A = -df["Current_A"].values.astype(np.float64)   # sign-flipped, see module docstring
    V_V = df["Voltage_V"].values.astype(np.float64)

    net_discharged_ah = (df["Discharged_Capacity_Ah"] - df["Charged_Capacity_Ah"]).values
    q_eff = float(net_discharged_ah.max() - net_discharged_ah.min())
    if q_eff < 1e-6:
        return []
    soc = 1.0 - (net_discharged_ah - net_discharged_ah.min()) / q_eff
    soc = np.clip(soc, 0.0, 1.0)
    T_degC = np.full(len(df), np.nan)

    schema_df = make_schema_df(t_s, I_A, V_V, T_degC, soc)
    gap_free = list(split_segments(schema_df, dataset="parallel_module",
                                    vehicle_id=folder_prefix, gap_thresh_s=300.0, min_rows=100))

    out: List[Tuple[pd.DataFrame, str]] = []
    for seg_df, meta in gap_free:
        t = seg_df["t_s"].values
        n_windows = max(1, int(np.ceil((t[-1] - t[0]) / window_s)))
        for w in range(n_windows):
            lo, hi = w * window_s, (w + 1) * window_s
            mask = (t >= lo) & (t < hi)
            if mask.sum() < 30:
                continue
            win_df = seg_df.iloc[mask].copy().reset_index(drop=True)
            win_df["t_s"] = win_df["t_s"] - win_df["t_s"].iloc[0]
            out.append((win_df, f"{folder_prefix}_{meta.segment_id}_win{w:03d}"))
    return out, q_eff


def run_fleet_module() -> List[Dict]:
    all_pairs: List[Tuple[pd.DataFrame, str]] = []
    q_effs = []
    for folder in MODULE_FOLDERS:
        segs, q_eff = load_module_folder(folder)
        if segs:
            q_effs.append(q_eff)
            print(f"  {folder}: {len(segs)} windows, Q_eff={q_eff:.3f} Ah")
            all_pairs.extend(segs)

    if not all_pairs:
        print("  [ERROR] No data loaded.")
        return []

    q_eff_mean = float(np.mean(q_effs))
    cfg = config_from_cartridge("Parallel_Module", MODULE_CARTRIDGE, CellMode.AVG_CELL, dt_resample_s=2.0)
    cfg.q_cell_ah = q_eff_mean
    print(f"  Using Q_cell_Ah={q_eff_mean:.3f} Ah (mean across {len(q_effs)} modules)")

    by_mod: Dict[str, List] = {}
    for seg_df, seg_id in all_pairs:
        mid = seg_id.split("_seg_")[0]
        by_mod.setdefault(mid, []).append((seg_df, seg_id))

    cal_pairs, eval_pairs = [], []
    for mid, segs in by_mod.items():
        n_cal = max(1, int(len(segs) * 0.30))
        cal_pairs.extend(segs[:n_cal])
        eval_pairs.extend(segs[n_cal:])
    print(f"  {len(cal_pairs)} calibration / {len(eval_pairs)} held-out windows")

    def _to_meta_pairs(pairs):
        out = []
        for seg_df, seg_id in pairs:
            meta = SegmentMeta(
                dataset="parallel_module", vehicle_id=seg_id, segment_id=seg_id,
                n_rows=len(seg_df), dt_s_median=2.0,
                duration_s=float(seg_df["t_s"].iloc[-1]),
                soc_start=float(seg_df["SOC_bms"].iloc[0]),
                soc_end=float(seg_df["SOC_bms"].iloc[-1]),
                I_mean_A=float(seg_df["I_A"].mean()), V_mean_V=float(seg_df["V_V"].mean()),
                T_mean_degC=float("nan"), has_temperature=False, notes=[],
            )
            out.append((seg_df, meta))
        return out

    cal_meta_pairs = _to_meta_pairs(cal_pairs)
    cal = _build_calibration_for_fleet(cal_meta_pairs, cfg, "Parallel_Module") if cal_meta_pairs else None
    ocv_fn = cal.ocv_fn if cal else None

    results = []
    for idx, (seg_df, seg_id) in enumerate(eval_pairs):
        r = evaluate_segment(seg_df, cfg, cal, ocv_fn)
        r["vehicle_id"] = seg_id
        results.append(r)
        if (idx + 1) % 20 == 0:
            print(f"  Module: {idx + 1}/{len(eval_pairs)} held-out done")
    return results


def main():
    print("=" * 78)
    print("SOC baseline benchmark -- NEW dataset: 3-parallel-cell module (UMich/Ford)")
    print("=" * 78)

    results = run_fleet_module()
    if not results:
        return None

    agg = _aggregate(results)
    print(f"\n--- 3-parallel-cell module (n={agg['n_segments']} held-out windows) ---")
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
            "dataset": "3-parallel-cell battery modules, Mendeley DOI 10.17632/ssrgfmb8vw.2",
            "paper": "arXiv:2604.16769 (University of Michigan + Ford Motor Company)",
            "modules_used": MODULE_FOLDERS,
            "modules_total_in_dataset": 78,
        },
        "aggregate": agg, "per_segment": results,
    }
    out_path = ROOT / "data" / "soc_baseline_benchmark_module_report.json"
    out_path.write_text(json.dumps(_serial(report), indent=2))
    print(f"\nJSON report written to {out_path}")
    return report


if __name__ == "__main__":
    main()
