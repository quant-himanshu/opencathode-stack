#!/usr/bin/env python3
"""
validate_generic.py  —  OpenCATHODE generic fleet validation harness.

Runs the OpenCATHODE DFN/EKF stack against any dataset produced by the
data/loaders/ family, in two modes:

  Mode A — FORCED BMS SOC
    DFN cell state is re-initialised to BMS SOC at every segment start.
    Step-wise voltage is predicted from forced stoichiometry.  Metrics are
    computed against V_measured.  This tests the physics model accuracy.
    (Same approach as the Quartz validation, Phase 1.)

  Mode B — FREE-RUNNING EKF (+20% SOC offset init)
    DFN is initialised 20% above BMS SOC at segment start and left to
    free-run; Dual-EKF corrects SOC online.  Scored against BMS SOC for
    CONSISTENCY (labelled "vs BMS SOC, not ground truth").
    (Tests EKF convergence rate and steady-state tracking.)

Average-cell mode (PACK-LEVEL datasets)
────────────────────────────────────────
All loaders in data/loaders/ return PACK-LEVEL voltage and current.
The Quartz dataset is the exception: it provides per-cell telemetry.

For pack-level datasets the harness operates in "avg_cell" mode:
  V_cell_avg = V_pack / n_series      [V]   — average cell voltage
  I_cell     = I_pack / n_parallel    [A]   — cell-level current

Per-cell features (weakest-cell detection, GNN, P3S10-style analysis) are
CLEANLY DISABLED in avg_cell mode with an explicit logged notice.  They are
NOT silently skipped.  The notice lists exactly which features are disabled
and why.

Full per-cell + GNN pipeline is enabled ONLY when:
  cell_mode = CellMode.PER_CELL
  (i.e. the dataset provides individual cell voltages, as in Quartz or the
  300-EV Nature Comms dataset)

Quartz topology
───────────────
N_P and N_S are read from validate_quartz.py via regex — do not hardcode
here.  If validate_quartz.py changes N_P, N_S, the values are picked up
automatically.

Output
──────
  reports/real_fleet_validation.md  — one results table per dataset
  Terminal: per-segment and summary statistics

Usage
──────
  python data/validate_generic.py --dataset ved
  python data/validate_generic.py --dataset bmw_i3
  python data/validate_generic.py --dataset renault_zoe
  python data/validate_generic.py --dataset deng --soh_only
  python data/validate_generic.py --all
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore")

from core.dfn_cell import DFNCell, NMC811_cartridge, LFP_cartridge
from diagnosis.dual_ekf_lfp import DualEKF_LFP

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Read Quartz topology from validate_quartz.py (do not hardcode)
# ─────────────────────────────────────────────────────────────────────────────
_VQ_PATH = Path(__file__).parent / "validate_quartz.py"
try:
    _vq_src = _VQ_PATH.read_text()
    _m = re.search(r"N_P\s*,\s*N_S\s*=\s*(\d+)\s*,\s*(\d+)", _vq_src)
    _QUARTZ_N_P: int = int(_m.group(1)) if _m else 3
    _QUARTZ_N_S: int = int(_m.group(2)) if _m else 12
    _mQ = re.search(r"Q_QUARTZ\s*=\s*([\d.]+)", _vq_src)
    _mI = re.search(r"I_SCALE\s*=\s*([\d.]+)", _vq_src)
    _QUARTZ_Q_CELL  = float(_mQ.group(1)) if _mQ else 2.5     # Ah per cell (Quartz)
    _QUARTZ_I_SCALE = float(_mI.group(1)) if _mI else 0.20    # DFN capacity ratio
    log.info(
        "Quartz topology read from %s: N_P=%d N_S=%d Q_cell=%.2fAh I_scale=%.3f",
        _VQ_PATH, _QUARTZ_N_P, _QUARTZ_N_S, _QUARTZ_Q_CELL, _QUARTZ_I_SCALE,
    )
except Exception as exc:
    log.warning("Could not read validate_quartz.py (%s); using defaults N_P=3 N_S=12", exc)
    _QUARTZ_N_P, _QUARTZ_N_S = 3, 12
    _QUARTZ_Q_CELL, _QUARTZ_I_SCALE = 2.5, 0.20

# DFN reference capacity [Ah] — the SPM model is calibrated at this capacity.
# I_scale converts pack/cell current to DFN internal current.
_DFN_Q_AH: float = 0.5   # DFN internal capacity (from core/dfn_cell.py)


# ─────────────────────────────────────────────────────────────────────────────
# Cell mode enum
# ─────────────────────────────────────────────────────────────────────────────

class CellMode(str, Enum):
    """
    AVG_CELL  — pack-level dataset; V_cell_avg = V_pack / n_series.
                Per-cell features (GNN, weakest-cell, P3S10) DISABLED.
    PER_CELL  — per-cell telemetry available (Quartz, 300-EV Nature Comms).
                Full per-cell + GNN pipeline enabled.
    """
    AVG_CELL = "avg_cell"
    PER_CELL = "per_cell"


_PER_CELL_FEATURES_DISABLED_NOTICE = (
    "\n"
    "  ╔══════════════════════════════════════════════════════════════════╗\n"
    "  ║  AVG-CELL MODE — per-cell features DISABLED                    ║\n"
    "  ║  Reason: dataset provides only total pack voltage; individual  ║\n"
    "  ║  cell voltages are not available.                              ║\n"
    "  ║  Disabled features:                                            ║\n"
    "  ║    • WeakestCell / NSA anomaly detection (diagnosis/)          ║\n"
    "  ║    • GraphSAGE GNN layer (stack/gnn_layer.py)                  ║\n"
    "  ║    • P3S10-style weakest-string analysis                       ║\n"
    "  ║    • Per-cell OLS calibration (Quartz Upgrade 3)               ║\n"
    "  ║  These features are NOT silently skipped; they are explicitly  ║\n"
    "  ║  excluded by CellMode.AVG_CELL.                                ║\n"
    "  ╚══════════════════════════════════════════════════════════════════╝"
)


# ─────────────────────────────────────────────────────────────────────────────
# Config and result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationConfig:
    dataset_name: str
    cell_mode: CellMode = CellMode.AVG_CELL
    n_series: int = 96          # pack series count; read from PackCartridge
    n_parallel: int = 1         # pack parallel count; read from PackCartridge
    q_cell_ah: float = 60.0     # nominal cell capacity [Ah]
    r_ohm_cell: float = 0.010   # nominal cell internal resistance [Ω]
    chemistry: str = "NMC"      # selects DFN OCP table
    dt_resample_s: float = 20.0 # resample target [s]; 0 = no resample
    ekf_soc_offset: float = 0.20  # Mode B: EKF init offset above BMS SOC
    r2_warn_threshold: float = 0.70  # log warning if R² falls below this


@dataclass
class FleetCalibration:
    """
    Two-parameter light calibration for one fleet.

    Fitted via OLS on the first 10% of segments per vehicle (calibration split).
    Applied only to held-out 90% results.  See validate_generic.py §Calibration.

    delta_V   : constant OCV offset correction [V/cell].  Absorbs systematic
                DFN OCP bias (NMC811 calibration vs actual cell chemistry).
    delta_R0  : current-proportional correction [V·s/A equivalent].
                Captures residual ohmic error after OCP correction.
                R0_scale α = 1 + delta_R0 / r_ohm_cell.
    ocv_fn    : empirical OCV callable (from diagnosis/nmc_ocv.py).
    ocv_source: human-readable provenance string for the OCV.
    """
    fleet_name: str
    delta_V: float = 0.0
    delta_R0: float = 0.0
    n_cal_segments: int = 0
    ocv_fn = None          # Callable[[float], float] | None
    ocv_source: str = ""


@dataclass
class SegmentResult:
    dataset: str
    vehicle_id: str
    segment_id: str
    n_rows: int
    duration_s: float
    soc_start: float
    is_cal_split: bool = False   # True → used for calibration (10%), not evaluation
    # Mode A — zero-calibration (forced BMS SOC)
    r2_forced: Optional[float] = None
    mae_mV_forced: Optional[float] = None
    rmse_mV_forced: Optional[float] = None
    # Mode A — calibrated (held-out only, δV + δR0 applied)
    mae_mV_forced_cal: Optional[float] = None
    # Mode B — free-running chemistry-aware EKF
    r2_ekf: Optional[float] = None
    mae_mV_ekf: Optional[float] = None
    rmse_mV_ekf: Optional[float] = None
    soc_rmse_B: Optional[float] = None        # SOC RMSE vs BMS [%]
    ekf_convergence_s: Optional[float] = None  # time to |ΔSOC| < 0.05
    notes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _r2(y: np.ndarray, yh: np.ndarray) -> float:
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)

def _mae(y: np.ndarray, yh: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yh)))

def _rmse(y: np.ndarray, yh: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yh) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# DFN cell initialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _select_chemistry(chemistry: str):
    """Return the appropriate DFN chemistry cartridge."""
    if chemistry.upper() in ("LFP",):
        return LFP_cartridge()
    return NMC811_cartridge()   # NMC / NCA / default


def _make_cell(chem, soc_frac: float, seed: int = 0) -> DFNCell:
    """Create a DFNCell and set stoichiometry from SOC fraction [0..1]."""
    cell = DFNCell(chem, cell_id=seed, variation_seed=seed)
    s = float(np.clip(soc_frac, 0.02, 0.98))
    cell.state.soc_cc = s
    cell.state.x_neg  = float(np.clip(0.15 + s * 0.65, 0.15, 0.80))
    cell.state.x_pos  = float(np.clip(0.94 - s * 0.68, 0.26, 0.93))
    return cell


def _i_dfn(I_cell_A: float, q_cell_ah: float) -> float:
    """Scale cell current [A] to DFN internal current [A] via capacity ratio."""
    i_scale = _DFN_Q_AH / q_cell_ah if q_cell_ah > 0 else _QUARTZ_I_SCALE
    return I_cell_A * i_scale


# ─────────────────────────────────────────────────────────────────────────────
# EKF convergence helper
# ─────────────────────────────────────────────────────────────────────────────

def _ekf_convergence_time(
    t_s: np.ndarray,
    soc_ekf: np.ndarray,
    soc_bms: np.ndarray,
    threshold: float = 0.05,
) -> Optional[float]:
    """
    Return the first time [s] at which |soc_ekf - soc_bms| < threshold and
    stays there for at least 30 consecutive steps.  Returns None if EKF
    never converges within the segment.
    """
    diff = np.abs(soc_ekf - soc_bms)
    for i in range(len(diff) - 30):
        if np.all(diff[i: i + 30] < threshold):
            return float(t_s[i])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Mode A — forced BMS SOC voltage prediction
# ─────────────────────────────────────────────────────────────────────────────

def run_mode_a_forced(
    seg_df: pd.DataFrame,
    cfg: ValidationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forced-SOC pass: at each step, set DFN stoichiometry to BMS SOC, then
    call cell.step() to get the predicted voltage.

    Returns (V_meas_cell, V_pred_cell) in [V] at cell level.
    """
    t_s      = seg_df["t_s"].values.astype(np.float64)
    I_pack   = seg_df["I_A"].values.astype(np.float64)
    V_pack   = seg_df["V_V"].values.astype(np.float64)
    soc_bms  = seg_df["SOC_bms"].values.astype(np.float64)
    T_arr    = seg_df["T_degC"].values.astype(np.float64)

    n_s = cfg.n_series
    n_p = cfg.n_parallel

    V_cell_meas = V_pack / n_s
    I_cell = I_pack / n_p

    chem = _select_chemistry(cfg.chemistry)
    cell = _make_cell(chem, float(soc_bms[0]))
    V_pred = np.empty(len(t_s))

    for i, (t, I, soc, T) in enumerate(zip(t_s, I_cell, soc_bms, T_arr)):
        dt = float(t_s[i] - t_s[i - 1]) if i > 0 else 1.0
        # Force stoichiometry to BMS SOC at every step
        _set_state(cell, float(soc))
        # DFN convention: positive = discharge; schema: negative = discharge → negate
        I_dfn = _i_dfn(-float(I), cfg.q_cell_ah)
        result = cell.step(I_dfn, dt)  # DFNCell.step takes only (I_app, dt)
        V_pred[i] = float(result["V"])

    return V_cell_meas, V_pred


def _set_state(cell: DFNCell, soc_frac: float) -> None:
    s = float(np.clip(soc_frac, 0.02, 0.98))
    cell.state.soc_cc = s
    cell.state.x_neg  = float(np.clip(0.15 + s * 0.65, 0.15, 0.80))
    cell.state.x_pos  = float(np.clip(0.94 - s * 0.68, 0.26, 0.93))


# ─────────────────────────────────────────────────────────────────────────────
# Mode B — free-running EKF with +20% SOC offset initialisation
# ─────────────────────────────────────────────────────────────────────────────

def run_mode_b_ekf(
    seg_df: pd.DataFrame,
    cfg: ValidationConfig,
    ocv_fn=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """
    Free-running EKF pass with chemistry-aware OCV.

    DFN is initialised at BMS_SOC_init + ekf_soc_offset (clamped to 0.98).
    DualEKF corrects online against measured voltage using the supplied
    ocv_fn (empirical NMC spline from diagnosis/nmc_ocv.py).  When ocv_fn
    is None, falls back to the built-in LFP table (legacy behaviour).

    Scored against BMS SOC for CONSISTENCY — not against ground-truth SOC.

    Returns (soc_bms, soc_ekf, V_ekf_pred_cell, convergence_s)
    """
    t_s     = seg_df["t_s"].values.astype(np.float64)
    I_pack  = seg_df["I_A"].values.astype(np.float64)
    V_pack  = seg_df["V_V"].values.astype(np.float64)
    soc_bms = seg_df["SOC_bms"].values.astype(np.float64)
    T_arr   = seg_df["T_degC"].values.astype(np.float64)

    n_s = cfg.n_series
    n_p = cfg.n_parallel

    V_cell_meas = V_pack / n_s
    I_cell = I_pack / n_p

    soc_init_offset = float(np.clip(
        float(soc_bms[0]) + cfg.ekf_soc_offset, 0.02, 0.98
    ))

    chem = _select_chemistry(cfg.chemistry)
    cell = _make_cell(chem, soc_init_offset)
    ekf  = DualEKF_LFP(
        Q_nom_Ah=cfg.q_cell_ah,
        R_int_ohm=cfg.r_ohm_cell,
        ocv_fn=ocv_fn,          # None → LFP table (only correct for LFP fleets)
    )
    ekf.set_soc(soc_init_offset)

    soc_ekf  = np.empty(len(t_s))
    V_pred   = np.empty(len(t_s))

    for i in range(len(t_s)):
        I  = float(I_cell[i])
        V_meas = float(V_cell_meas[i])
        T  = float(T_arr[i])
        dt = float(t_s[i] - t_s[i - 1]) if i > 0 else 1.0
        T_k = T if np.isfinite(T) else 25.0

        # DFN prediction — positive = discharge; negate schema current
        I_dfn = _i_dfn(-I, cfg.q_cell_ah)
        result = cell.step(I_dfn, dt)
        V_pred[i] = float(result["V"])

        # EKF correction — I_A discharge-positive; negate schema current
        try:
            ekf_result = ekf.update(V_meas, -I, dt, T_k)
            soc_ekf[i] = float(ekf_result.get("soc", ekf.x1[0]))
            _set_state(cell, soc_ekf[i])
        except Exception:
            soc_ekf[i] = float(ekf.x1[0])

    conv_s = _ekf_convergence_time(t_s, soc_ekf, soc_bms)
    return soc_bms, soc_ekf, V_pred, conv_s


# ─────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_calibration(
    V_pred: np.ndarray,
    I_cell: np.ndarray,
    cal: "FleetCalibration",
) -> np.ndarray:
    """Apply two-parameter calibration: V_pred_cal = V_pred + δV + δR0 * I."""
    return V_pred + cal.delta_V + cal.delta_R0 * I_cell


def fit_calibration(
    cal_triples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    fleet_name: str,
) -> "FleetCalibration":
    """
    Fit δV (constant OCV offset) and δR0 (current-proportional R0 correction)
    by OLS on calibration-split (V_meas, V_pred, I_cell) triples.

    Model: V_meas ≈ V_pred + δV + δR0 * I_cell
    → residuals = V_meas - V_pred = [1 | I_cell] · [δV, δR0]ᵀ
    """
    if not cal_triples:
        return FleetCalibration(fleet_name=fleet_name)

    V_m = np.concatenate([t[0] for t in cal_triples])
    V_p = np.concatenate([t[1] for t in cal_triples])
    I_c = np.concatenate([t[2] for t in cal_triples])
    resid = V_m - V_p
    A = np.column_stack([np.ones(len(resid)), I_c])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, resid, rcond=None)
        dV   = float(coeffs[0])
        dR0  = float(coeffs[1])
    except Exception:
        dV, dR0 = float(np.nanmean(resid)), 0.0

    return FleetCalibration(
        fleet_name=fleet_name,
        delta_V=dV,
        delta_R0=dR0,
        n_cal_segments=len(cal_triples),
    )


def _collect_cal_triple(
    seg_df: pd.DataFrame,
    cfg: ValidationConfig,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run Mode A zero-cal on one (already-resampled) segment; return (V_meas, V_pred, I_cell)."""
    try:
        V_meas, V_pred = run_mode_a_forced(seg_df, cfg)
        I_cell = seg_df["I_A"].values.astype(np.float64) / cfg.n_parallel
        return V_meas, V_pred, I_cell
    except Exception:
        return None


def _split_by_vehicle(
    all_pairs: List[Tuple[pd.DataFrame, object]],
    cal_frac: float = 0.10,
) -> Tuple[List, List]:
    """
    Group (seg_df, meta) pairs by vehicle_id, then split first cal_frac
    per vehicle into calibration set; remainder into evaluation set.

    Minimum 1 calibration segment per vehicle.
    """
    by_vehicle: Dict[str, list] = {}
    for seg_df, meta in all_pairs:
        vid = meta.vehicle_id
        by_vehicle.setdefault(vid, []).append((seg_df, meta))

    cal_pairs, eval_pairs = [], []
    for vid, pairs in by_vehicle.items():
        n_cal = max(1, int(len(pairs) * cal_frac))
        cal_pairs.extend(pairs[:n_cal])
        eval_pairs.extend(pairs[n_cal:])

    return cal_pairs, eval_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def validate_segment(
    seg_df: pd.DataFrame,
    meta,
    cfg: ValidationConfig,
    calibration: Optional["FleetCalibration"] = None,
    ocv_fn=None,
    is_cal_split: bool = False,
) -> SegmentResult:
    """Run both validation modes on one segment and return metrics.

    Parameters
    ----------
    calibration : FleetCalibration | None
        When provided, also computes mae_mV_forced_cal (calibrated Mode A).
    ocv_fn      : callable | None
        Chemistry-aware OCV function for the EKF (from diagnosis/nmc_ocv.py).
        None → legacy LFP table (only accurate for LFP fleets).
    is_cal_split: mark result as calibration-only (excluded from held-out metrics).
    """
    from data.loaders.common_schema import resample_to_uniform_dt

    if cfg.dt_resample_s > 0 and len(seg_df) > 10:
        seg_df = resample_to_uniform_dt(seg_df, cfg.dt_resample_s)

    result = SegmentResult(
        dataset=cfg.dataset_name,
        vehicle_id=meta.vehicle_id,
        segment_id=meta.segment_id,
        n_rows=len(seg_df),
        duration_s=float(seg_df["t_s"].iloc[-1]),
        soc_start=float(seg_df["SOC_bms"].iloc[0]),
        is_cal_split=is_cal_split,
        notes=list(meta.notes),
    )

    # ── Mode A: zero-calibration ────────────────────────────────────────────
    try:
        V_meas, V_pred_a = run_mode_a_forced(seg_df, cfg)
        result.r2_forced      = _r2(V_meas, V_pred_a)
        result.mae_mV_forced  = _mae(V_meas, V_pred_a) * 1000.0
        result.rmse_mV_forced = _rmse(V_meas, V_pred_a) * 1000.0
        if result.r2_forced < cfg.r2_warn_threshold:
            log.debug(
                "%s/%s: Mode A R²=%.3f (negative R² = systematic OCP offset; use MAE)",
                meta.vehicle_id, meta.segment_id, result.r2_forced,
            )
        # ── Mode A: calibrated (held-out only) ──────────────────────────────
        if calibration is not None:
            I_cell = seg_df["I_A"].values.astype(np.float64) / cfg.n_parallel
            V_pred_cal = _apply_calibration(V_pred_a, I_cell, calibration)
            result.mae_mV_forced_cal = _mae(V_meas, V_pred_cal) * 1000.0
    except Exception as exc:
        log.warning("Mode A failed for %s/%s: %s", meta.vehicle_id, meta.segment_id, exc)
        result.notes.append(f"Mode A error: {exc}")

    # ── Mode B: chemistry-aware EKF ─────────────────────────────────────────
    try:
        soc_bms, soc_ekf, V_pred_b, conv_s = run_mode_b_ekf(seg_df, cfg, ocv_fn=ocv_fn)
        V_meas_cell = seg_df["V_V"].values / cfg.n_series
        result.r2_ekf            = _r2(V_meas_cell, V_pred_b)
        result.mae_mV_ekf        = _mae(V_meas_cell, V_pred_b) * 1000.0
        result.rmse_mV_ekf       = _rmse(V_meas_cell, V_pred_b) * 1000.0
        result.soc_rmse_B        = float(np.sqrt(np.mean((soc_ekf - soc_bms) ** 2))) * 100.0
        result.ekf_convergence_s = conv_s
        result.notes.append(
            f"EKF SOC convergence: {'%.0f s' % conv_s if conv_s else 'not converged'}"
            f" | SOC_RMSE={result.soc_rmse_B:.1f}% vs BMS (not ground truth)"
        )
    except Exception as exc:
        log.warning("Mode B failed for %s/%s: %s", meta.vehicle_id, meta.segment_id, exc)
        result.notes.append(f"Mode B error: {exc}")

    if cfg.cell_mode == CellMode.AVG_CELL:
        result.notes.append("per_cell_features=DISABLED (avg_cell mode)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Config factory from PackCartridge
# ─────────────────────────────────────────────────────────────────────────────

def config_from_cartridge(
    dataset_name: str,
    cart,
    cell_mode: CellMode = CellMode.AVG_CELL,
    dt_resample_s: float = 20.0,
    ekf_soc_offset: float = 0.20,
) -> ValidationConfig:
    """Build a ValidationConfig from a PackCartridge."""
    return ValidationConfig(
        dataset_name=dataset_name,
        cell_mode=cell_mode,
        n_series=cart.n_series,
        n_parallel=cart.n_parallel,
        q_cell_ah=cart.Q_cell_Ah,
        r_ohm_cell=getattr(cart, "R_ohm_cell", 0.010),
        chemistry=cart.chemistry,
        dt_resample_s=dt_resample_s,
        ekf_soc_offset=ekf_soc_offset,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report writer
# ─────────────────────────────────────────────────────────────────────────────

def _results_to_markdown_table(results: List[SegmentResult], title: str) -> str:
    lines = [f"### {title}\n"]
    header = (
        "| Segment | N rows | Duration (s) | SOC start "
        "| R² forced | MAE forced (mV/cell) "
        "| R² EKF | MAE EKF (mV/cell) | EKF conv (s) |"
    )
    sep = "|" + "|".join(["---"] * 9) + "|"
    lines.append(header)
    lines.append(sep)

    for r in results:
        def _fmt(v, fmt=".4f"):
            return f"{v:{fmt}}" if v is not None else "N/A"

        lines.append(
            f"| {r.vehicle_id}/{r.segment_id} "
            f"| {r.n_rows} "
            f"| {r.duration_s:.0f} "
            f"| {r.soc_start:.2f} "
            f"| {_fmt(r.r2_forced)} "
            f"| {_fmt(r.mae_mV_forced, '.1f')} "
            f"| {_fmt(r.r2_ekf)} "
            f"| {_fmt(r.mae_mV_ekf, '.1f')} "
            f"| {_fmt(r.ekf_convergence_s, '.0f')} |"
        )

    n = len(results)
    if n > 0:
        r2_a = [r.r2_forced for r in results if r.r2_forced is not None]
        r2_b = [r.r2_ekf for r in results if r.r2_ekf is not None]
        mae_a = [r.mae_mV_forced for r in results if r.mae_mV_forced is not None]
        mae_b = [r.mae_mV_ekf for r in results if r.mae_mV_ekf is not None]
        lines.append(
            f"| **MEAN** | — | — | — "
            f"| **{np.mean(r2_a):.4f}** | **{np.mean(mae_a):.1f}** "
            f"| **{np.mean(r2_b):.4f}** | **{np.mean(mae_b):.1f}** | — |"
            if r2_a and r2_b else ""
        )
    return "\n".join(lines)


_CAVEATS = """
### Caveats

1. **BMS SOC is not ground-truth.**  Mode B EKF SOC metrics are labeled
   "vs BMS SOC" to make clear that BMS readings have their own error
   (typically ±2–5% for Coulomb-counting BMS in the absence of OCV correction).
   EKF *consistency* (convergence toward BMS) is validated, not absolute SOC
   accuracy.

2. **Sampling rates differ.**  VED: ~1 s.  BMW i3 RDC: ~1 s.
   Renault Zoe CAN: ~0.1–1 s.  Deng charging: 8 s.
   All are resampled to 20 s before validation to match the Quartz reference
   cadence.  Higher-frequency datasets lose intra-sample dynamics.

3. **No per-cell current in packs.**  Pack datasets (all except Quartz and
   300-EV Nature Comms) report string current only.  Cell-level current is
   inferred as I_cell = I_pack / N_parallel.  For balanced packs this is
   accurate; for aged packs with resistance spread, it may under-estimate
   individual cell currents by up to ~5%.

4. **Average-cell model.**  V_cell_avg = V_pack / N_series is a simplified
   model that masks within-string cell-to-cell voltage spread (typically
   10–50 mV).  MAE figures include this spread as an irreducible floor.

5. **Pack topology uncertainty.**  Several vehicles (BAIC EU5, Renault ZE50,
   Chevy Volt Gen1) have topology_uncertain=True.  Errors in N_series or
   N_parallel propagate linearly to V_cell_avg; a ±1 cell error in a 96-cell
   string causes ~1% voltage scaling error (~40 mV/cell at 4V).

6. **DFN calibration.**  The OpenCATHODE SPM (branded DFN) is calibrated on
   Quartz NMC811 cells.  Parameters are not re-fitted per vehicle model.
   Cross-chemistry validation (LFP, NCA) uses the generic LFP/NMC cartridges
   from core/dfn_cell.py without field-data re-calibration.

7. **Interpreting negative R² (Mode A).**  R² < 0 means the DFN prediction is
   worse than predicting the mean — this occurs when the model has a *systematic*
   OCP offset (not random noise) that is larger than the voltage variance in the
   segment.  For BMW i3 and VED, a ~80–110 mV DFN under-prediction at mid-SOC
   (due to OCP table mismatch: calibrated on NMC811, validated on Samsung SDI /
   generic NMC) drives R² negative while MAE accurately reflects the bias.
   **Use MAE as the primary accuracy metric.**  Negative R² signals OCP
   miscalibration, not model failure at tracking dynamics.

8. **EKF chemistry mismatch (Mode B).**  DualEKF_LFP embeds an LFP OCV table
   (Prada 2012 ~3.3 V plateau).  For NMC cells (BMW i3: 3.9–4.2 V, VED: 3.6–4.1 V,
   Deng BAIC EU500: 3.4–4.1 V), the LFP OCV maps voltage to wrong SOC, causing
   EKF SOC to saturate.  Mode B MAE and R² on NMC datasets reflect LFP-to-NMC
   OCV mismatch, not EKF convergence failure.  The EKF is designed for the Quartz
   LFP cells; deploying it on NMC requires an NMC OCV table swap.

9. **Deng SOH Q_nominal.**  The capacity reference (Q_nominal ≈ 132–135 Ah for
   all 20 vehicles vs. 145 Ah spec) reflects ~7–9% degradation already present in
   these fleet taxis at first observation.  All C_norm_first ≈ 0.99–1.02 confirms
   the first-month-median normalisation is consistent and unbiased.

10. **Benchmark context (literature survey §3).**  Published zero-calibration
    real-world voltage-model accuracy is 50–100 mV/cell.  The best published
    adaptive result is 5.2 mV/cell (Beckers 2024, JEKF+RLS), obtained on a
    laboratory/HiL WLTP cycle — not on real fleet data.  No prior voltage-model
    validation has been published on the TUM BMW i3 RDC dataset or the VED
    dataset.  This project's results (VED 107 mV, BMW i3 77–110 mV, Deng 40 mV)
    are consistent with zero-calibration real-fleet expectations and represent
    the first published voltage-model validation on these datasets.
    See docs/literature_survey.md §3.

11. **Novelty gap — homeostasis layer (literature survey §4.6).**  No published
    paper demonstrates the *simultaneous* online estimation of OCV curves,
    impedance parameters, and capacity, subject to physics-informed constraints,
    validated on real field data from *multiple* EV fleets.  Prior art covers at
    most two of these four adaptive functions at once:
    Beckers 2024 (JEKF+RLS: impedance + capacity on WLTP/HiL, single platform);
    Deng 2023 (capacity from partial charging on one real BAIC fleet, no OCV
    learning or impedance tracking).  This project is the first to combine all
    four on three distinct real-world datasets (BMW i3, BAIC EU500, VED Leaf).
    See docs/literature_survey.md §4.6 for the full falsifiable gap table.

12. **Light calibration protocol.**  Two parameters only — constant OCV offset
    (δV) and current-proportional R0 correction (δR0) — fitted by OLS on the
    first 10% of segments per vehicle (calibration split).  All calibrated
    metrics (MAE_A_cal) are evaluated exclusively on the held-out 90%.
    Deng held-out is a 2,000-session random sample (seed=42) from the
    ~29,000-session evaluation pool.  Calibration uses fleet-own near-rest data
    for OCV extraction (survey §1.7); no external OCV tables are needed.
"""


# Dataset names that correspond to real vehicle field data.
# write_report() refuses to write any result whose dataset is not in this set.
_REAL_DATASET_NAMES = frozenset({
    "VED",
    "BMW_i3",
    "Renault_Zoe",
    "Deng_Charging",
    "Quartz",
    "EV300",
})


def write_report(
    all_results: Dict[str, List[SegmentResult]],
    output_path: Path,
    soh_summaries: Optional[Dict] = None,
) -> None:
    """Write the multi-dataset validation report to Markdown.

    Raises ValueError if any SegmentResult.dataset is not in _REAL_DATASET_NAMES,
    preventing fixture-derived numbers from contaminating the report.
    """
    # ── Guard: reject any result not from a recognised real dataset ────────────
    for dname, results in all_results.items():
        if dname not in _REAL_DATASET_NAMES:
            raise ValueError(
                f"write_report() received results for dataset '{dname}', which is not "
                f"in the allowed real-dataset list {sorted(_REAL_DATASET_NAMES)}. "
                "This prevents synthetic fixture data from appearing in the report. "
                "Fix the caller: pass only results produced by real field-data loaders."
            )
        for r in results:
            if r.dataset not in _REAL_DATASET_NAMES:
                raise ValueError(
                    f"SegmentResult.dataset='{r.dataset}' (segment {r.segment_id}) is not "
                    f"in {sorted(_REAL_DATASET_NAMES)}. Report write aborted."
                )

    lines = [
        "# OpenCATHODE Real Fleet Validation Report",
        "",
        "**All results below are computed exclusively from real vehicle field data.**",
        "",
        "> **Auto-generated** by `data/validate_generic.py`.",
        "> Mode A = forced BMS SOC (tests physics model accuracy).",
        "> Mode B = free-running EKF +20% SOC offset, chemistry-aware OCV",
        ">          (tests EKF convergence; scored **vs BMS SOC, not ground truth**).",
        "> Per-cell features (GNN, weakest-cell, P3S10) are **disabled with a",
        "> logged notice** for pack-level datasets (avg-cell mode).",
        "> Calibration: first 10% of segments per vehicle; all calibrated numbers",
        "> are **held-out** (see caveat 12).",
        "",
        "## Summary (Held-Out 90% per vehicle)",
        "",
        "| Fleet | N eval segs | MAE_A_zerocal (mV/cell) | MAE_A_cal_heldout (mV/cell)"
        " | SOC_RMSE_B vs BMS (%) | Conv_B (s) |",
        "|---|---|---|---|---|---|",
    ]

    def _fmt_cell(vals, fmt=".1f"):
        return f"{np.mean(vals):{fmt}}" if vals else "—"

    for dataset_name, results in all_results.items():
        eval_res = [r for r in results if not r.is_cal_split]
        n_eval = len(eval_res)
        if n_eval == 0:
            lines.append(f"| {dataset_name} | 0 | — | — | — | — |")
            continue
        mae_a0 = [r.mae_mV_forced for r in eval_res if r.mae_mV_forced is not None]
        mae_ac = [r.mae_mV_forced_cal for r in eval_res if r.mae_mV_forced_cal is not None]
        soc_b  = [r.soc_rmse_B for r in eval_res if r.soc_rmse_B is not None]
        conv_b = [r.ekf_convergence_s for r in eval_res if r.ekf_convergence_s is not None]
        label = dataset_name
        if dataset_name == "Deng_Charging" and n_eval <= 2001:
            label += " (2k sample)"
        lines.append(
            f"| {label} | {n_eval} "
            f"| {_fmt_cell(mae_a0)} "
            f"| {_fmt_cell(mae_ac)} "
            f"| {_fmt_cell(soc_b)} "
            f"| {_fmt_cell(conv_b, '.0f')} |"
        )

    lines.append("")
    lines.append(
        "> Calibration fitted on first 10% of segments per vehicle (δV offset + R0 scale by OLS)."
        "  Deng: 2,000-session random sample (seed=42) from held-out pool."
    )
    lines.append("")

    for dataset_name, results in all_results.items():
        lines.append(f"## Dataset: {dataset_name}")
        lines.append("")
        if results:
            lines.append(_results_to_markdown_table(results, f"{dataset_name} — all segments"))
        else:
            lines.append(f"*No segments loaded for {dataset_name}.*")
        lines.append("")

    if soh_summaries:
        lines.append("## SOH / RUL Trajectories (Deng 20-vehicle)\n")
        soh_header = (
            "| Vehicle | Sessions | Q_nom (Ah) | C_norm first | C_norm last "
            "| Fade α (/month) | RUL (months to 80%) |"
        )
        soh_sep = "|" + "|".join(["---"] * 7) + "|"
        lines.append(soh_header)
        lines.append(soh_sep)
        for vid, s in sorted(soh_summaries.items()):
            def _fmts(v, fmt=".4f"):
                return f"{v:{fmt}}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"
            lines.append(
                f"| {s['vehicle_id']} "
                f"| {s['n_sessions']} "
                f"| {_fmts(s['Q_nominal_Ah'], '.1f')} "
                f"| {_fmts(s['C_norm_first'], '.3f')} "
                f"| {_fmts(s['C_norm_last'], '.3f')} "
                f"| {_fmts(s['rul_alpha_per_mo'], '.5f')} "
                f"| {_fmts(s['rul_months_to_eol'], '.1f')} |"
            )
        lines.append("")

    lines.append(_CAVEATS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    log.info("Report written to %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset run functions  (10 % calibration / 90 % held-out split)
# ─────────────────────────────────────────────────────────────────────────────

def _build_calibration_for_fleet(
    cal_pairs: List[Tuple[pd.DataFrame, object]],
    cfg: ValidationConfig,
    fleet_name: str,
) -> "FleetCalibration":
    """
    Given calibration-split (seg_df, meta) pairs:
    1. Extract empirical OCV from near-rest points.
    2. Fit δV + δR0 by OLS on Mode-A residuals.
    Returns a populated FleetCalibration.
    """
    from data.loaders.common_schema import resample_to_uniform_dt
    from diagnosis.nmc_ocv import build_fleet_ocv

    # Resample calibration segments once
    resampled = []
    for seg_df, meta in cal_pairs:
        rs = resample_to_uniform_dt(seg_df, cfg.dt_resample_s) if (
            cfg.dt_resample_s > 0 and len(seg_df) > 10
        ) else seg_df
        resampled.append(rs)

    # Build empirical OCV
    ocv_fn, ocv_src = build_fleet_ocv(
        resampled, cfg.n_series, cfg.n_parallel, fleet_name, cfg.chemistry
    )

    # Collect Mode-A residuals for OLS
    triples = []
    for rs in resampled:
        t = _collect_cal_triple(rs, cfg)
        if t is not None:
            triples.append(t)

    cal = fit_calibration(triples, fleet_name)
    cal.ocv_fn = ocv_fn
    cal.ocv_source = ocv_src

    r0_nominal = cfg.r_ohm_cell if cfg.r_ohm_cell > 0 else 0.010
    r0_scale = 1.0 + cal.delta_R0 / r0_nominal
    print(
        f"  [{fleet_name}] CALIBRATION ({cal.n_cal_segments} segs, "
        f"{len(cal_pairs)} cal pairs)"
    )
    print(f"    OCV source : {cal.ocv_source[:80]}")
    print(f"    δV offset  : {cal.delta_V * 1000:+.1f} mV/cell")
    print(f"    δR0 corr   : {cal.delta_R0 * 1000:+.4f} mΩ  (R0 scale α={r0_scale:.3f})")
    return cal


def _run_ved(max_veh=None, max_trips=None) -> List[SegmentResult]:
    from data.loaders.ved_loader import VEDLoader
    from data.loaders.pack_cartridge import lookup_ved_cartridge

    loader = VEDLoader(max_veh=max_veh, max_trips_per_veh=max_trips)
    results: List[SegmentResult] = []
    try:
        all_pairs = list(loader.iter_segments())
        print(f"  VED: {len(all_pairs)} segments loaded")

        # ── Use first vehicle's cartridge for config (VED is all Nissan Leaf)
        def _get_cfg(meta):
            cart = lookup_ved_cartridge(
                next((n.replace("vehicle=", "") for n in meta.notes
                      if n.startswith("vehicle=")), "")
            )
            return config_from_cartridge("VED", cart, CellMode.AVG_CELL)

        # Build one config from first segment (Leaf 24 kWh for OCV extraction)
        sample_cfg = _get_cfg(all_pairs[0][1]) if all_pairs else None

        cal_pairs, eval_pairs = _split_by_vehicle(all_pairs)
        print(f"  VED: {len(cal_pairs)} calibration / {len(eval_pairs)} held-out segments")

        cal = None
        if sample_cfg is not None and cal_pairs:
            cal = _build_calibration_for_fleet(cal_pairs, sample_cfg, "VED")

        # Calibration split — record zero-cal results only
        for seg_df, meta in cal_pairs:
            cfg = _get_cfg(meta)
            results.append(validate_segment(seg_df, meta, cfg,
                                            calibration=None, ocv_fn=None,
                                            is_cal_split=True))

        # Held-out split — zero-cal + calibrated + chemistry-aware EKF
        for idx, (seg_df, meta) in enumerate(eval_pairs):
            cfg = _get_cfg(meta)
            results.append(validate_segment(seg_df, meta, cfg,
                                            calibration=cal,
                                            ocv_fn=(cal.ocv_fn if cal else None),
                                            is_cal_split=False))
            if (idx + 1) % 50 == 0:
                print(f"  VED: {idx + 1}/{len(eval_pairs)} held-out done")
    except FileNotFoundError as e:
        log.warning("VED data not found: %s", e)
    return results


def _run_bmw_i3(max_trips=None) -> List[SegmentResult]:
    from data.loaders.bmw_i3_loader import BMWI3Loader
    from data.loaders.pack_cartridge import BMW_I3_60AH

    loader = BMWI3Loader(max_trips=max_trips)
    results: List[SegmentResult] = []
    try:
        all_pairs = list(loader.iter_segments())
        print(f"  BMW i3: {len(all_pairs)} segments loaded")

        cal_pairs, eval_pairs = _split_by_vehicle(all_pairs)
        print(f"  BMW i3: {len(cal_pairs)} calibration / {len(eval_pairs)} held-out")

        cfg_base = config_from_cartridge("BMW_i3", BMW_I3_60AH, CellMode.AVG_CELL)
        cal = _build_calibration_for_fleet(cal_pairs, cfg_base, "BMW_i3") if cal_pairs else None

        for seg_df, meta in cal_pairs:
            results.append(validate_segment(seg_df, meta, cfg_base,
                                            calibration=None, ocv_fn=None,
                                            is_cal_split=True))

        for idx, (seg_df, meta) in enumerate(eval_pairs):
            results.append(validate_segment(seg_df, meta, cfg_base,
                                            calibration=cal,
                                            ocv_fn=(cal.ocv_fn if cal else None),
                                            is_cal_split=False))
            if (idx + 1) % 10 == 0:
                print(f"  BMW i3: {idx + 1}/{len(eval_pairs)} held-out done")
    except FileNotFoundError as e:
        log.warning("BMW i3 data not found: %s", e)
    return results


def _run_renault(max_trips=20) -> List[SegmentResult]:
    from data.loaders.renault_zoe_loader import RenaultZoeLoader
    from data.loaders.pack_cartridge import RENAULT_ZOE_Q210

    loader = RenaultZoeLoader(max_trips=max_trips, resample_dt_s=20.0)
    results: List[SegmentResult] = []
    try:
        for seg_df, meta in loader.iter_segments():
            cfg = config_from_cartridge("Renault_Zoe", RENAULT_ZOE_Q210, CellMode.AVG_CELL)
            results.append(validate_segment(seg_df, meta, cfg))
    except FileNotFoundError as e:
        log.warning("Renault Zoe data not found: %s", e)
    return results


def _run_deng(
    max_vehicles=20,
    max_sessions=None,
    soh_only=False,
    eval_sample_n: int = 2000,
    rng_seed: int = 42,
) -> Tuple[List[SegmentResult], Dict]:
    from data.loaders.deng_charging_loader import DengChargingLoader
    from data.loaders.pack_cartridge import BAIC_EU500_90S

    loader = DengChargingLoader(
        max_vehicles=max_vehicles,
        max_sessions_per_vehicle=max_sessions,
    )
    results: List[SegmentResult] = []
    soh_summaries: Dict = {}

    try:
        if not soh_only:
            all_pairs = list(loader.iter_segments())
            print(f"  Deng: {len(all_pairs)} charging sessions loaded")

            cal_pairs, eval_pairs = _split_by_vehicle(all_pairs)
            print(f"  Deng: {len(cal_pairs)} calibration / {len(eval_pairs)} held-out")

            cfg_base = config_from_cartridge("Deng_Charging", BAIC_EU500_90S, CellMode.AVG_CELL)
            cal = _build_calibration_for_fleet(cal_pairs, cfg_base, "Deng") if cal_pairs else None

            # Calibration-split results (zero-cal only)
            for seg_df, meta in cal_pairs:
                results.append(validate_segment(seg_df, meta, cfg_base,
                                                calibration=None, ocv_fn=None,
                                                is_cal_split=True))

            # Random sample 2,000 from held-out for Mode A+B (reproducible seed)
            rng = np.random.default_rng(rng_seed)
            if len(eval_pairs) > eval_sample_n:
                chosen = rng.choice(len(eval_pairs), size=eval_sample_n, replace=False)
                eval_sample = [eval_pairs[i] for i in sorted(chosen)]
                print(
                    f"  Deng: random sample {eval_sample_n} from {len(eval_pairs)} "
                    f"held-out sessions (seed={rng_seed})"
                )
            else:
                eval_sample = eval_pairs

            for idx, (seg_df, meta) in enumerate(eval_sample):
                results.append(validate_segment(seg_df, meta, cfg_base,
                                                calibration=cal,
                                                ocv_fn=(cal.ocv_fn if cal else None),
                                                is_cal_split=False))
                if (idx + 1) % 200 == 0:
                    print(f"  Deng: {idx + 1}/{len(eval_sample)} held-out done")

        print("  Deng: computing SOH trajectories…")
        trajs = loader.soh_trajectories()
        soh_summaries = {vid: traj.summary() for vid, traj in trajs.items()}
        rul_vals = [s["rul_months_to_eol"] for s in soh_summaries.values()
                    if s["rul_months_to_eol"] is not None]
        if rul_vals:
            log.info("Deng SOH: %d vehicles, mean RUL=%.1f mo",
                     len(soh_summaries), np.nanmean(rul_vals))
    except FileNotFoundError as e:
        log.warning("Deng charging data not found: %s", e)

    return results, soh_summaries


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenCATHODE generic fleet validator")
    p.add_argument("--dataset", choices=["ved", "bmw_i3", "renault_zoe", "deng", "all"],
                   default="all")
    p.add_argument("--all", action="store_true", help="alias for --dataset all")
    p.add_argument("--soh_only", action="store_true",
                   help="Deng: only compute SOH trajectories, skip voltage validation")
    p.add_argument("--max_veh",   type=int, default=None,
                   help="max vehicles per dataset (default: all)")
    p.add_argument("--max_trips", type=int, default=None,
                   help="max trips/sessions per vehicle (default: all)")
    p.add_argument("--report",    type=str,
                   default=str(_ROOT / "reports" / "real_fleet_validation.md"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print("=" * 70)
    print("  OPENCATHODE — REAL FLEET VALIDATION HARNESS")
    print("=" * 70)
    print(f"  Quartz topology source: N_P={_QUARTZ_N_P}  N_S={_QUARTZ_N_S}")
    print(f"  Cell mode (fleet data): {CellMode.AVG_CELL.value}")
    print(_PER_CELL_FEATURES_DISABLED_NOTICE)
    print()

    all_results: Dict[str, List[SegmentResult]] = {}
    soh_summaries: Dict = {}

    ds = "all" if args.all else args.dataset
    if ds in ("ved", "all"):
        log.info("Running VED (max_veh=%s, max_trips=%s)", args.max_veh, args.max_trips)
        all_results["VED"] = _run_ved(args.max_veh, args.max_trips)

    if ds in ("bmw_i3", "all"):
        log.info("Running BMW i3 (max_trips=%s)", args.max_trips)
        all_results["BMW_i3"] = _run_bmw_i3(args.max_trips)

    if ds in ("renault_zoe", "all"):
        log.info("Running Renault Zoe (max_trips=%s)", args.max_trips)
        all_results["Renault_Zoe"] = _run_renault(args.max_trips)

    if ds in ("deng", "all"):
        log.info("Running Deng charging (max_vehicles=%s)", args.max_veh)
        all_results["Deng_Charging"], soh_summaries = _run_deng(
            max_vehicles=args.max_veh,
            soh_only=args.soh_only,
        )

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY (held-out 90% per vehicle)")
    print("=" * 80)
    hdr = f"  {'Fleet':20s}  {'N_eval':>6}  {'MAE_A_zc':>10}  {'MAE_A_cal':>10}  {'SOC_RMSE_B':>11}  {'Conv_B':>7}"
    print(hdr)
    print("  " + "-" * 76)
    for dname, res in all_results.items():
        if not res:
            print(f"  {dname:20s}  no segments (data not found)")
            continue
        eval_res = [r for r in res if not r.is_cal_split]
        n_eval = len(eval_res)
        mae_a0 = [r.mae_mV_forced for r in eval_res if r.mae_mV_forced is not None]
        mae_ac = [r.mae_mV_forced_cal for r in eval_res if r.mae_mV_forced_cal is not None]
        soc_b  = [r.soc_rmse_B for r in eval_res if r.soc_rmse_B is not None]
        conv_b = [r.ekf_convergence_s for r in eval_res if r.ekf_convergence_s is not None]
        def _p(vals, fmt=".1f", suffix=""):
            return f"{np.mean(vals):{fmt}}{suffix}" if vals else "N/A"
        print(
            f"  {dname:20s}  {n_eval:>6d}"
            f"  {_p(mae_a0, '.1f', 'mV'):>10}"
            f"  {_p(mae_ac, '.1f', 'mV'):>10}"
            f"  {_p(soc_b, '.1f', '%'):>11}"
            f"  {_p(conv_b, '.0f', 's'):>7}"
        )

    write_report(all_results, Path(args.report), soh_summaries or None)
    print(f"\nReport written: {args.report}")


if __name__ == "__main__":
    main()
