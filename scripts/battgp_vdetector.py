"""
BattGP V-detector study — pre-registered detection pipeline.

PRE-REGISTRATION: data/prototype_layer_validation.json
  battgp_vdetector.battgp_vdetector_preregistration
  Commits: 34d096f (initial lock), b89c635 (GP-confidence gating)

CALIBRATION/HELD-OUT SPLIT
  Calibration system: data_sys_17 (BURNED — bound-setting snooped which cell
    was faulty). Its result is shown as sanity-check only.
  Primary held-out validation: all other 28 systems (27 systems).

V_norm FORMULA (locked from data_sys_17 healthy-7 cells)
  clip((V - 3.140) / 0.243, 0.0, 1.0)

FOUR OUTCOMES (per held-out system, mutually exclusive)
  AGREE         — gate passes AND GP-confident AND V-detector top-1 == GP weak cell
  DISAGREE      — gate passes AND GP-confident AND V-detector top-1 != GP weak cell
  NO-CLEAR-FAULT— gate passes AND GP does not flag any cell above r0_upper_threshold
  GATE-FAIL     — power gate ratio < 2

TRIVIAL BASELINE
  argmin( mean V_norm per cell on high-current sub-segment )
  Evaluated against the same GP-confidence precondition.
  V-detector is only reported as adding value if its agreement rate >= baseline.

HIGH-CURRENT SUB-SEGMENT: |I_Batt| > 15 A (derived from data_sys_17 median).
  Re-derivation rule: after first 5 held-out systems, recompute pooled median;
  if it deviates from 15 A by > 5 A, update threshold and freeze for remainder.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import time
import zipfile
import zlib
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/private/tmp/BattGP")   # BattGP source (cloned for GP model)

from diagnosis.weakest_cell import VDetector

# BattGP imports (GP model for confidence gating)
from src.batt_data.data_utils import read_cell_characteristics
from src.batt_data.batt_data import BattData, SegmentCriteria
from src.batt_models.battgp_spatiotemporal import BattGP_SpatioTemporal
from src.batt_models.fault_probabilities import calc_fault_probabilities
from src.batt_models.ref_strategy import RefStrategy
from src.operating_point import Op
import src.config as battgp_cfg

# ---------------------------------------------------------------------------
# Pre-registered constants (DO NOT CHANGE after commit b89c635)
# ---------------------------------------------------------------------------
V_NORM_LOW   = 3.140   # V — healthy-7 p1 of data_sys_17 minus 50 mV
V_NORM_HIGH  = 3.383   # V — healthy-7 p99 of data_sys_17 plus 50 mV
V_NORM_SPAN  = V_NORM_HIGH - V_NORM_LOW                   # 0.243 V

T_NORM_CENTER = 25.0   # °C
T_NORM_SCALE  = 50.0   # °C

HIGH_CURRENT_THRESHOLD_A = 15.0   # |I_Batt| > this value (derived from data_sys_17 median)

# BattGP GP fault line (from example_usage_battgp.py)
R0_UPPER_THRESHOLD = 2.0e-3   # Ohm — fault line
R0_BAND           = 0.55e-3   # Ohm — band for fault probability

CALIBRATION_SYSTEM = "17"     # data_sys_17 — burned, sanity only
POWER_GATE_SEEDS   = [42, 43, 44, 45]
POWER_GATE_MIN_RATIO = 2.0

# GP INPUT CAP — pre-registered before any system 7-28 is processed.
# Discharge-segment rows fed to BattGP capped at this value per system.
# Uniform stride sampling (time-ordered) preserves temporal coverage.
# 40,000 rows ≈ 5,000 per cell × 8 cells; leaves systems 1-4 unchanged.
GP_MAX_DISCHARGE_ROWS = 40_000

# Borderline self-check: if capped GP yields max_band_prob in [LOW, HIGH],
# re-run at GP_RECHECK_ROWS. If result changes, both are recorded and flagged.
# Clearly-confident (>HIGH) and clear-no-fault (<LOW) skip the recheck.
GP_RECHECK_ROWS      = 100_000
GP_BORDERLINE_LOW    = 0.50    # = GP_BAND_PROB_THRESHOLD — the decision boundary
GP_BORDERLINE_HIGH   = 0.95    # above this: clearly confident, cap dilution irrelevant

ZIP_PATH = ROOT / "data" / "iontech_lfp" / "field_data.zip"
PARTIAL_RESULTS_PATH = ROOT / "data" / "battgp_results_partial.json"

BATTGP_CONFIG_OVERRIDES = {
    "PATH_FIELDDATA_DATA": str(ZIP_PATH),
    "PATH_FIELDDATA_CELL_CHARACTERISTIC": "/private/tmp/BattGP/data/ocv_linear_approx.csv",
    "PATH_DATA_CACHE": None,  # disable cache for reproducibility
}

# ---------------------------------------------------------------------------
# V_norm / T_norm
# ---------------------------------------------------------------------------

def v_norm(v: np.ndarray) -> np.ndarray:
    return np.clip((v - V_NORM_LOW) / V_NORM_SPAN, 0.0, 1.0)

def t_norm(t: np.ndarray) -> np.ndarray:
    return (t - T_NORM_CENTER) / T_NORM_SCALE

# ---------------------------------------------------------------------------
# Zip streaming loader
# ---------------------------------------------------------------------------

def _load_csv_from_zip(zip_path: Path, system_id: str) -> Optional[pd.DataFrame]:
    """
    Stream-decompress data_sys_{system_id}.csv from the (possibly partial) zip.
    Returns None if the file is not yet present / incomplete.
    """
    target_name = f"field_data/data_sys_{system_id}.csv"
    with open(zip_path, "rb") as f:
        raw = f.read()

    pos = 0
    while True:
        idx = raw.find(b"PK\x03\x04", pos)
        if idx == -1:
            return None
        if idx + 30 > len(raw):
            return None
        fname_len  = struct.unpack_from("<H", raw, idx + 26)[0]
        extra_len  = struct.unpack_from("<H", raw, idx + 28)[0]
        if idx + 30 + fname_len > len(raw):
            return None
        fname = raw[idx+30:idx+30+fname_len].decode("utf-8", errors="replace")
        data_start = idx + 30 + fname_len + extra_len
        if fname == target_name:
            try:
                decompressed = zlib.decompressobj(wbits=-15).decompress(raw[data_start:])
                return pd.read_csv(BytesIO(decompressed), parse_dates=["Timestamp"])
            except (zlib.error, Exception):
                return None
        pos = idx + 4
    return None

# ---------------------------------------------------------------------------
# Segment selection (mirrors BattGP SegmentCriteria)
# ---------------------------------------------------------------------------

def apply_segment_criteria(df: pd.DataFrame) -> pd.DataFrame:
    cnv_cols  = [f"I_CNV_Cell_{i}" for i in range(1, 9)]
    temp_cols = ["Temperature_1", "Temperature_2", "Temperature_3", "Temperature_4"]

    mask = (
        (df["I_Battery"] >= -80.0) & (df["I_Battery"] <= -5.0) &
        (df["SOC_Battery"] > 40.0) & (df["SOC_Battery"] < 95.0)
    )
    for tc in temp_cols:
        mask &= (df[tc] > 10.0) & (df[tc] < 100.0)
    for cc in cnv_cols:
        mask &= (df[cc].abs() < 20.0)
    return df[mask].copy()

def apply_high_current(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["I_Battery"].abs() > HIGH_CURRENT_THRESHOLD_A].copy()

# ---------------------------------------------------------------------------
# GP input cap: write subsampled temp zip for BattGP
# ---------------------------------------------------------------------------

def _write_capped_zip(df_seg: pd.DataFrame, system_id: str,
                      cap_rows: int = GP_MAX_DISCHARGE_ROWS) -> tuple[Path, int]:
    """
    Write a stride-subsampled (≤ cap_rows rows) temp zip for BattGP.
    Timestamp is set as the index to match the original CSV format BattGP expects.
    Returns (path_to_temp_zip, actual_rows_used).
    """
    tmp_dir = Path(tempfile.gettempdir()) / "battgp_cap"
    tmp_dir.mkdir(exist_ok=True)
    zip_path = tmp_dir / f"sys_{system_id}_cap{cap_rows}.zip"

    if len(df_seg) > cap_rows:
        stride = max(1, len(df_seg) // cap_rows)
        df_cap = df_seg.iloc[::stride].copy()
    else:
        df_cap = df_seg.copy()

    csv_buf = BytesIO()
    df_cap.set_index("Timestamp").to_csv(csv_buf, index=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"field_data/data_sys_{system_id}.csv", csv_buf.getvalue())

    return zip_path, len(df_cap)


# ---------------------------------------------------------------------------
# Incremental save / resume helpers
# ---------------------------------------------------------------------------

def _serial(obj):
    if isinstance(obj, bool): return bool(obj)
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _serial(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_serial(x) for x in obj]
    return obj


def _load_partial_results() -> tuple[list[dict], set[str]]:
    """Return (results_list, set_of_completed_system_ids)."""
    if PARTIAL_RESULTS_PATH.exists():
        try:
            with open(PARTIAL_RESULTS_PATH) as f:
                d = json.load(f)
            systems = d.get("systems", [])
            completed = {str(s["system_id"]) for s in systems}
            return systems, completed
        except Exception:
            pass
    return [], set()


def _save_partial_results(results: list[dict]) -> None:
    """Overwrite partial results file with current list (crash-safe after each system)."""
    out = {
        "note": "Incremental crash-safe results. gp_input_cap applied to all systems.",
        "preregistration_commits": ["34d096f", "b89c635", "12668d0", "5a515bf"],
        "gp_input_cap": {
            "max_discharge_rows_per_system": GP_MAX_DISCHARGE_ROWS,
            "method": "uniform_stride_time_ordered",
            "locked_before_system": "7",
        },
        "systems": results,
    }
    with open(PARTIAL_RESULTS_PATH, "w") as f:
        json.dump(_serial(out), f, indent=2)


# ---------------------------------------------------------------------------
# Feature extraction: (V_norm, T_norm) per cell
# ---------------------------------------------------------------------------

TEMP_MAP = {1: "Temperature_1", 2: "Temperature_1",
            3: "Temperature_2", 4: "Temperature_2",
            5: "Temperature_3", 6: "Temperature_3",
            7: "Temperature_4", 8: "Temperature_4"}

def cell_features(df: pd.DataFrame, cell_nr: int) -> np.ndarray:
    """Return (N, 2) array of (V_norm, T_norm) for cell_nr on high-current sub-segment."""
    v = v_norm(df[f"U_Cell_{cell_nr}"].to_numpy(np.float64))
    t = t_norm(df[TEMP_MAP[cell_nr]].to_numpy(np.float64))
    return np.column_stack([v, t])

# ---------------------------------------------------------------------------
# Trivial baseline
# ---------------------------------------------------------------------------

def trivial_baseline(df_hc: pd.DataFrame) -> dict:
    """
    Pre-registered trivial baseline: argmin of per-cell mean V_norm on high-current segment.
    Returns dict with scores and predicted weak cell (1-indexed).
    """
    scores = {}
    for i in range(1, 9):
        scores[i] = float(v_norm(df_hc[f"U_Cell_{i}"].to_numpy(np.float64)).mean())
    weak_cell = min(scores, key=scores.get)
    return {"scores": scores, "weak_cell": weak_cell}

# ---------------------------------------------------------------------------
# V-detector LOO (8 iterations, one seed)
# ---------------------------------------------------------------------------

def vdetector_loo_single_seed(
    df_hc: pd.DataFrame,
    seed: int,
) -> dict[int, float]:
    """
    LOO V-detector: for each cell i, train on the other 7 cells' high-current features,
    then return PRIMARY score (mean activation fraction) for cell i.
    Returns dict {cell_nr: primary_score}.
    """
    all_features = {i: cell_features(df_hc, i) for i in range(1, 9)}
    scores = {}
    for target in range(1, 9):
        self_arr = np.vstack([all_features[j] for j in range(1, 9) if j != target])
        vd = VDetector(rng_seed=seed)
        vd.observe_normal_array(self_arr)
        vd.train()
        fracs = vd.activation_fraction_array(all_features[target])
        scores[target] = float(fracs.mean())
    return scores

# ---------------------------------------------------------------------------
# Power gate (4 seeds)
# ---------------------------------------------------------------------------

def power_gate(df_hc: pd.DataFrame) -> dict:
    """
    Pre-registered power gate with seeds {42,43,44,45}.
    Returns ratio, noise_ceiling, per-seed signal gaps, and gate decision.
    """
    per_seed_gaps = {}
    for seed in POWER_GATE_SEEDS:
        scores = vdetector_loo_single_seed(df_hc, seed)
        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - np.mean(sorted_scores[1:])
        per_seed_gaps[seed] = {"scores": scores, "gap": gap}

    # Noise ceiling: p95 of per-cell max-pairwise |Δ gap| across 4 seeds
    # Here gap is pack-level (top-1 minus mean); compute per-seed gaps array
    gap_values = np.array([per_seed_gaps[s]["gap"] for s in POWER_GATE_SEEDS])
    pairwise_deltas = [abs(gap_values[i] - gap_values[j])
                       for i in range(4) for j in range(i+1, 4)]
    noise_ceiling = float(np.percentile(pairwise_deltas, 95))
    median_gap    = float(np.median(gap_values))
    ratio         = median_gap / noise_ceiling if noise_ceiling > 0 else float("inf")

    # Prediction from seed 42 (primary)
    primary_scores = per_seed_gaps[42]["scores"]
    vdetector_pred = max(primary_scores, key=primary_scores.get)

    return {
        "ratio": ratio,
        "noise_ceiling": noise_ceiling,
        "median_gap": median_gap,
        "gate_pass": ratio >= POWER_GATE_MIN_RATIO,
        "per_seed_gaps": {s: v["gap"] for s, v in per_seed_gaps.items()},
        "primary_scores": primary_scores,
        "vdetector_pred": vdetector_pred,
    }

# ---------------------------------------------------------------------------
# GP confidence gate
# ---------------------------------------------------------------------------

def gp_confidence(system_id: str, df_seg: pd.DataFrame,
                  cap_rows: int = GP_MAX_DISCHARGE_ROWS) -> dict:
    """
    Run BattGP spatiotemporal GP on this system and check fault confidence.
    df_seg is the already-filtered discharge-segment DataFrame (from run_system).
    If len(df_seg) > cap_rows, writes a stride-subsampled temp zip so BattGP
    processes ≤ cap_rows rows instead of the full file.
    Returns gp_confident bool, gp_weak_cell (1-indexed or None), R0 estimates,
    and cap metadata.
    """
    battgp_cfg.PATH_FIELDDATA_CELL_CHARACTERISTIC = (
        BATTGP_CONFIG_OVERRIDES["PATH_FIELDDATA_CELL_CHARACTERISTIC"]
    )
    battgp_cfg.PATH_DATA_CACHE = None

    # Write capped temp zip; BattGP reads from this instead of the full zip.
    cap_zip, n_used = _write_capped_zip(df_seg, system_id, cap_rows)
    battgp_cfg.PATH_FIELDDATA_DATA = str(cap_zip)
    cap_fraction = n_used / max(len(df_seg), 1)
    print(f"  GP input: {n_used:,} / {len(df_seg):,} discharge rows "
          f"(cap_fraction={cap_fraction:.3f}"
          f"{', stride-capped' if len(df_seg) > cap_rows else ', under cap'})")

    try:
        cell_char = read_cell_characteristics(
            path=BATTGP_CONFIG_OVERRIDES["PATH_FIELDDATA_CELL_CHARACTERISTIC"]
        )
        batt_data = BattData(system_id, cell_char)
        cell_nrs  = batt_data.cell_nrs   # [1..8]

        bgp = BattGP_SpatioTemporal(
            batt_data,
            sampling_time_sec=3600,
            ref_strategy=RefStrategy(Op(-15, 90, 25)),
            max_batch_size=1000,
            basis_vector_strategy="kmeans",
            nbasis=[60],
        )
        gp_res = bgp.predict_cell_r0_op(smooth=True)

        df_faults = calc_fault_probabilities(
            gp_res,
            causal=True,
            r0_band=R0_BAND,
            r0_upper_threshold=R0_UPPER_THRESHOLD,
        )

        # GP-confidence uses the BAND-based upper fault probability (relative to pack mean),
        # NOT the absolute r0_upper_threshold. At this battery age, all cells exceed 2 mΩ
        # absolute, making that threshold useless. The band-based P(R0 > mean_pack + r0_band)
        # is what BattGP's own fault detection uses and correctly identifies relative outliers.
        #
        # GP-confident: max R_upper{i}_band_i_fault_prob over the LAST 24 time steps
        #               (last ~24 hours of hourly GP predictions) >= GP_BAND_PROB_THRESHOLD.
        # GP weak cell: argmax of per-cell mean R_upper band prob over the same window.
        GP_BAND_PROB_THRESHOLD = 0.5   # pre-registered threshold for GP confidence

        r0_df = gp_res.get_cell_data(cell_nrs, ["t", "r0"], causal=True)
        r0_final = {}
        for cell_nr in cell_nrs:
            col = f"r0_c{cell_nr}"
            if col in r0_df.columns:
                series = r0_df[col].dropna()
                r0_final[cell_nr] = float(series.iloc[-1]) if len(series) > 0 else float("nan")
            else:
                r0_final[cell_nr] = float("nan")

        # Band probability from final window
        window = df_faults.tail(24)
        band_prob_final = {}
        for cell_nr in cell_nrs:
            col = f"R_upper{cell_nr} band_i fault prob"
            if col in window.columns:
                band_prob_final[cell_nr] = float(window[col].mean())
            else:
                band_prob_final[cell_nr] = float("nan")

        valid_bp = {k: v for k, v in band_prob_final.items() if not np.isnan(v)}
        max_prob  = max(valid_bp.values()) if valid_bp else 0.0
        gp_confident = max_prob >= GP_BAND_PROB_THRESHOLD
        gp_weak_cell = (max(valid_bp, key=valid_bp.get) if gp_confident else None)

        return {
            "gp_confident": gp_confident,
            "gp_weak_cell": gp_weak_cell,
            "r0_final": r0_final,
            "band_prob_final": band_prob_final,
            "max_band_prob": max_prob,
            "gp_band_prob_threshold": GP_BAND_PROB_THRESHOLD,
            "fault_probs_computed": True,
            "cap_rows_used": cap_rows,
            "n_seg_rows": len(df_seg),
            "n_gp_rows": n_used,
            "cap_fraction": round(cap_fraction, 4),
        }

    except Exception as e:
        return {
            "gp_confident": None,
            "gp_weak_cell": None,
            "r0_final": {},
            "error": str(e),
            "fault_probs_computed": False,
            "cap_rows_used": cap_rows,
            "n_seg_rows": len(df_seg),
            "n_gp_rows": n_used,
            "cap_fraction": round(cap_fraction, 4),
        }

# ---------------------------------------------------------------------------
# Single-system runner
# ---------------------------------------------------------------------------

def run_system(system_id: str, is_calibration: bool = False) -> dict:
    """
    Full pipeline for one system. Returns structured result dict.
    """
    label = f"[{'CALIBRATION/SNOOPED' if is_calibration else 'HELD-OUT'}] data_sys_{system_id}"
    print(f"\n{'='*65}")
    print(f"{label}")
    print("=" * 65)
    t0 = time.time()

    # --- Load raw data ---
    df_raw = _load_csv_from_zip(ZIP_PATH, system_id)
    if df_raw is None:
        print(f"  SKIP: data_sys_{system_id}.csv not available in zip (download incomplete)")
        return {"system_id": system_id, "status": "DATA_UNAVAILABLE"}

    print(f"  Raw rows: {len(df_raw):,}  "
          f"({df_raw['Timestamp'].min()} → {df_raw['Timestamp'].max()})")

    # --- Segment selection ---
    df_seg = apply_segment_criteria(df_raw)
    print(f"  Discharge-segment rows: {len(df_seg):,} ({len(df_seg)/len(df_raw):.1%})")

    df_hc = apply_high_current(df_seg)
    print(f"  High-current rows (|I|>{HIGH_CURRENT_THRESHOLD_A}A): "
          f"{len(df_hc):,} ({len(df_hc)/max(len(df_seg),1):.1%} of discharge-segment)")

    if len(df_hc) < 100:
        print(f"  SKIP: insufficient high-current rows ({len(df_hc)} < 100)")
        return {"system_id": system_id, "status": "INSUFFICIENT_DATA",
                "n_seg": len(df_seg), "n_hc": len(df_hc)}

    # --- Trivial baseline (pre-registered comparator) ---
    baseline = trivial_baseline(df_hc)
    print(f"\n  TRIVIAL BASELINE:")
    print(f"  Per-cell mean V_norm: " +
          "  ".join(f"Cell{i}={baseline['scores'][i]:.4f}" for i in range(1, 9)))
    print(f"  Trivial weak cell: Cell {baseline['weak_cell']}")

    # --- Power gate ---
    print(f"\n  POWER GATE (seeds {POWER_GATE_SEEDS}):")
    gate = power_gate(df_hc)
    print(f"  Per-seed gaps: " +
          "  ".join(f"s{s}={gate['per_seed_gaps'][s]:.3e}" for s in POWER_GATE_SEEDS))
    print(f"  Noise ceiling (p95 pairwise Δgap): {gate['noise_ceiling']:.3e}")
    print(f"  Median gap: {gate['median_gap']:.3e}")
    print(f"  Ratio: {gate['ratio']:.2f}  → {'PASS' if gate['gate_pass'] else 'FAIL (< 2.0)'}")

    if not gate["gate_pass"]:
        outcome = "GATE-FAIL"
        print(f"\n  OUTCOME: {outcome}")
        return {"system_id": system_id, "status": "OK", "outcome": outcome,
                "n_seg": len(df_seg), "n_hc": len(df_hc),
                "baseline": baseline, "gate": gate,
                "elapsed_s": time.time() - t0}

    # --- V-detector prediction (primary seed=42 already in gate) ---
    print(f"\n  V-DETECTOR (primary seed=42 from gate):")
    primary_scores = gate["primary_scores"]
    vdet_pred = gate["vdetector_pred"]
    for i in range(1, 9):
        marker = " ← PREDICTED WEAK" if i == vdet_pred else ""
        print(f"    Cell {i}: PRIMARY={primary_scores[i]:.4e}{marker}")

    # --- GP confidence gate (standard cap) ---
    print(f"\n  GP CONFIDENCE GATE (BattGP spatiotemporal GP):")
    gp = gp_confidence(system_id, df_seg)
    gp_recheck = None
    cap_flag = "ok"

    if not gp["fault_probs_computed"]:
        print(f"  GP ERROR: {gp.get('error', 'unknown')}")
        gp_status = "GP_ERROR"
    else:
        print(f"  R0 final per cell (mOhm): " +
              "  ".join(f"Cell{k}={v*1000:.2f}" if not np.isnan(v) else f"Cell{k}=NaN"
                        for k, v in sorted(gp["r0_final"].items())))
        print(f"  Band P(above) per cell: " +
              "  ".join(f"Cell{k}={v:.3f}" if not np.isnan(v) else f"Cell{k}=NaN"
                        for k, v in sorted(gp.get("band_prob_final", {}).items())))
        mp = gp.get("max_band_prob", 0.0)
        print(f"  Max band prob: {mp:.3f}  threshold: 0.5  "
              f"cap_fraction: {gp.get('cap_fraction', 1.0):.3f}")
        print(f"  GP-confident: {gp['gp_confident']}  GP weak cell: {gp['gp_weak_cell']}")
        gp_status = "OK"

        # --- Borderline self-check: re-run at 100k if band_prob in [0.5, 0.95] ---
        if GP_BORDERLINE_LOW <= mp <= GP_BORDERLINE_HIGH and len(df_seg) > GP_MAX_DISCHARGE_ROWS:
            print(f"\n  BORDERLINE band_prob ({mp:.3f} in [{GP_BORDERLINE_LOW},{GP_BORDERLINE_HIGH}]) "
                  f"AND cap bound — re-running GP at {GP_RECHECK_ROWS:,} rows to check stability...")
            gp_recheck = gp_confidence(system_id, df_seg, cap_rows=GP_RECHECK_ROWS)
            if gp_recheck["fault_probs_computed"]:
                same_cell = gp_recheck["gp_weak_cell"] == gp["gp_weak_cell"]
                same_conf = gp_recheck["gp_confident"] == gp["gp_confident"]
                cap_flag = "recheck_same" if (same_cell and same_conf) else "recheck_differs"
                print(f"  RECHECK (100k): GP weak cell={gp_recheck['gp_weak_cell']}  "
                      f"confident={gp_recheck['gp_confident']}  "
                      f"max_prob={gp_recheck['max_band_prob']:.3f}  → {cap_flag.upper()}")
            else:
                cap_flag = "recheck_gp_error"
                print(f"  RECHECK failed: {gp_recheck.get('error', 'unknown')}")

    # --- Outcome (from standard-cap GP) ---
    if gp_status == "GP_ERROR" or gp["gp_confident"] is None:
        outcome = "GP_ERROR"
    elif not gp["gp_confident"]:
        outcome = "NO-CLEAR-FAULT"
    elif vdet_pred == gp["gp_weak_cell"]:
        outcome = "AGREE"
    else:
        outcome = "DISAGREE"

    # Baseline outcome (same preconditions)
    if gp_status == "GP_ERROR" or gp["gp_confident"] is None:
        baseline_outcome = "GP_ERROR"
    elif not gp["gp_confident"]:
        baseline_outcome = "NO-CLEAR-FAULT"
    elif baseline["weak_cell"] == gp["gp_weak_cell"]:
        baseline_outcome = "BASELINE-AGREE"
    else:
        baseline_outcome = "BASELINE-DISAGREE"

    # --- Fault severity: GP weak cell R0 vs pack-mean R0 ---
    severity_ratio = None
    if gp_status == "OK" and gp["gp_confident"] and gp["gp_weak_cell"] is not None:
        r0 = gp["r0_final"]
        weak = gp["gp_weak_cell"]
        pack_vals = [v for k, v in r0.items() if k != weak and not np.isnan(v)]
        if pack_vals and not np.isnan(r0.get(weak, float("nan"))):
            severity_ratio = round(r0[weak] / np.mean(pack_vals), 3)

    print(f"\n  V-DETECTOR OUTCOME:  {outcome}")
    print(f"  BASELINE OUTCOME:    {baseline_outcome}")
    if severity_ratio is not None:
        print(f"  FAULT SEVERITY:      R0_weak/R0_pack_mean = {severity_ratio:.3f}")
    if cap_flag != "ok":
        print(f"  CAP FLAG:            {cap_flag.upper()}")
    print(f"  Elapsed: {time.time()-t0:.0f}s")
    sys.stdout.flush()

    return {
        "system_id": system_id,
        "status": "OK",
        "is_calibration": is_calibration,
        "outcome": outcome,
        "baseline_outcome": baseline_outcome,
        "n_raw": len(df_raw),
        "n_seg": len(df_seg),
        "n_hc": len(df_hc),
        "baseline": baseline,
        "gate": {k: (v if not isinstance(v, dict) else
                     {str(kk): float(vv) for kk, vv in v.items()})
                 for k, v in gate.items()},
        "gp": gp,
        "gp_recheck": gp_recheck,
        "cap_flag": cap_flag,
        "vdetector_pred": vdet_pred,
        "fault_severity_ratio": severity_ratio,
        "elapsed_s": round(time.time() - t0, 1),
    }

# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all(systems: list[str], output_path: Path) -> None:
    # --- RESUME: load already-completed systems from partial file ---
    results, completed_ids = _load_partial_results()
    skipped = [sid for sid in systems if sid in completed_ids]
    pending = [sid for sid in systems if sid not in completed_ids]
    if skipped:
        print(f"\nRESUME: skipping {len(skipped)} already-completed systems: {skipped}")
    print(f"Processing {len(pending)} remaining systems.\n")

    for sid in pending:
        is_cal = (sid == CALIBRATION_SYSTEM)
        result = run_system(sid, is_calibration=is_cal)
        results.append(result)
        # --- INCREMENTAL SAVE: write after every system ---
        _save_partial_results(results)
        outcome = result.get("outcome", result.get("status", "UNKNOWN"))
        print(f"  → Saved to partial file ({len(results)} systems total so far)")
        sys.stdout.flush()

    # --- Final summary (held-out only) ---
    held_out = [r for r in results if str(r.get("system_id")) != CALIBRATION_SYSTEM]
    print(f"\n{'='*65}")
    print("FULL BREAKDOWN (held-out systems only)")
    print("=" * 65)
    n_total     = len(held_out)
    n_gate_pass = sum(1 for r in held_out
                      if r.get("outcome") in ("AGREE","DISAGREE","NO-CLEAR-FAULT","GP_ERROR"))
    n_gp_conf   = sum(1 for r in held_out if r.get("outcome") in ("AGREE","DISAGREE"))
    n_agree     = sum(1 for r in held_out if r.get("outcome") == "AGREE")
    n_disagree  = sum(1 for r in held_out if r.get("outcome") == "DISAGREE")
    n_no_fault  = sum(1 for r in held_out if r.get("outcome") == "NO-CLEAR-FAULT")
    n_gate_fail = sum(1 for r in held_out if r.get("outcome") == "GATE-FAIL")
    n_gp_err    = sum(1 for r in held_out if r.get("outcome") == "GP_ERROR")
    b_agree     = sum(1 for r in held_out if r.get("baseline_outcome") == "BASELINE-AGREE")

    print(f"  N_held_out       = {n_total}")
    print(f"  N_gate_pass      = {n_gate_pass}")
    print(f"  N_gp_confident   = {n_gp_conf}")
    print(f"  N_agree          = {n_agree}  (V-detector == GP weak cell)")
    print(f"  N_disagree       = {n_disagree}  (V-detector != GP weak cell)")
    print(f"  N_no_fault       = {n_no_fault}  (GP flags no cell above threshold)")
    print(f"  N_gate_fail      = {n_gate_fail}  (power gate ratio < 2)")
    print(f"  N_gp_error       = {n_gp_err}  (GP pipeline failed)")
    if n_gp_conf > 0:
        print(f"\n  V-detector agreement rate: {n_agree}/{n_gp_conf} = {n_agree/n_gp_conf:.0%}")
        print(f"  Trivial baseline agreement: {b_agree}/{n_gp_conf} = {b_agree/n_gp_conf:.0%}")

    # Write final results file
    out_data = {
        "preregistration_commits": ["34d096f", "b89c635", "12668d0", "5a515bf"],
        "calibration_system": CALIBRATION_SYSTEM,
        "gp_input_cap": {"max_discharge_rows": GP_MAX_DISCHARGE_ROWS,
                          "method": "uniform_stride_time_ordered"},
        "breakdown": {
            "n_held_out": n_total, "n_gate_pass": n_gate_pass,
            "n_gp_confident": n_gp_conf, "n_agree": n_agree,
            "n_disagree": n_disagree, "n_no_fault": n_no_fault,
            "n_gate_fail": n_gate_fail, "n_gp_error": n_gp_err,
            "baseline_agree": b_agree,
        },
        "systems": results,
    }
    with open(output_path, "w") as f:
        json.dump(_serial(out_data), f, indent=2)
    print(f"\nResults written to {output_path}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def smoke_test():
    """Run calibration system (data_sys_17) as pipeline smoke-test."""
    print("=" * 65)
    print("SMOKE TEST — calibration system data_sys_17 (SNOOPED, sanity only)")
    print("Pre-registered: should detect U_Cell_3 (known degraded)")
    print("=" * 65)
    result = run_system(CALIBRATION_SYSTEM, is_calibration=True)
    out_path = ROOT / "data" / "battgp_smoke_test.json"
    with open(out_path, "w") as f:
        json.dump(_serial(result), f, indent=2)
    print(f"\nSmoke-test result written to data/battgp_smoke_test.json")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BattGP V-detector pipeline")
    parser.add_argument("--smoke", action="store_true",
                        help="Run calibration-system smoke test only (data_sys_17)")
    parser.add_argument("--systems", nargs="+", type=str,
                        help="System IDs to process (default: all 27 held-out)")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
    else:
        all_ids = [str(i) for i in range(1, 29)]
        held_out = [sid for sid in all_ids if sid != CALIBRATION_SYSTEM]
        target = args.systems if args.systems else held_out
        out = ROOT / "data" / "battgp_vdetector_results.json"
        run_all(target, out)
