#!/usr/bin/env python3
"""
compare_pybamm_all.py — Multi-dataset PyBaMM vs OpenCATHODE EKF comparison.

Runs ONE representative real trip per dataset:
  VED          — NISSAN Leaf driving  (LMO-NMC, 96s2p, 33.1 Ah/cell)
  BMW i3       — TUM RDC driving      (NMC111, 96s1p, 60 Ah/cell)
  Deng/BAIC    — BAIC EU500 charging  (NCM, 90s1p, 145 Ah/cell)
  Quartz WLTP  — NMC811 WLTP lab     (NMC811, 3P×12S, 2.5 Ah/cell)
  NASA B0018   — lab 18650 discharge  (NMC811/graphite, 1 cell ~3 Ah)

PyBaMM: Chen2020 (LG M50 5Ah NMC811), same current C-rate, NO V_meas feedback.
EKF   : DualEKF_LFP, per-dataset OCV, gamma from fleet sweep, reads V_meas.
Both  : +10% init SOC offset (configurable).

If a dataset's raw files are absent, the row is marked "DATA MISSING" — no
synthetic substitution.

Usage:
    python scripts/compare_pybamm_all.py
    python scripts/compare_pybamm_all.py --soc-offset 0.10
    python scripts/compare_pybamm_all.py --max-steps 150
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── optional ──────────────────────────────────────────────────────────────────
try:
    import pybamm
    _HAVE_PYBAMM = True
except ImportError:
    _HAVE_PYBAMM = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Trip record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trip:
    """Per-dataset trip, already scaled to per-cell values."""
    dataset:    str
    chemistry:  str          # for table display
    cart_name:  str          # pack cartridge label
    Q_cell_Ah:  float
    t_s:        np.ndarray   # elapsed time [s]
    I_cell_A:   np.ndarray   # discharge-negative convention [A]
    V_cell:     np.ndarray   # per-cell voltage [V]
    soc_bms:    np.ndarray   # BMS SOC [0..1]
    ocv_fn:     object       # callable soc→V_cell
    ekf_gamma:  float = 1.0
    ekf_R_meas: float = 1e-6
    ekf_R_int:  float = 0.010
    notes:      List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# OCV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nmc_generic_ocv():
    from diagnosis.nmc_ocv import _NMC_SOC, _NMC_OCV
    soc_t, ocv_t = _NMC_SOC.copy(), _NMC_OCV.copy()
    def fn(soc: float) -> float:
        return float(np.interp(np.clip(soc, 0.0, 1.0), soc_t, ocv_t))
    return fn


def _lmonmc_ocv():
    from diagnosis.nmc_ocv import _LMONMC_SOC, _LMONMC_OCV
    soc_t, ocv_t = _LMONMC_SOC.copy(), _LMONMC_OCV.copy()
    def fn(soc: float) -> float:
        return float(np.interp(np.clip(soc, 0.0, 1.0), soc_t, ocv_t))
    return fn


def _nmc811_ocv():
    """Build NMC811 full-cell OCV table from DFN model (sampled at I=0)."""
    from core.dfn_cell import DFNCell, NMC811_cartridge
    chem = NMC811_cartridge()

    def _cell(soc_frac):
        s = float(np.clip(soc_frac, 0.02, 0.98))
        cell = DFNCell(chem, cell_id=0, variation_seed=0)
        cell.state.soc_cc = s
        cell.state.x_neg = float(np.clip(0.15 + s * 0.65, 0.15, 0.80))
        cell.state.x_pos = float(np.clip(0.94 - s * 0.68, 0.26, 0.93))
        return cell

    soc_pts = np.linspace(0.02, 0.98, 30)
    ocv_pts = np.array([_cell(s).step(0.0, 0.001)["V"] for s in soc_pts])

    def fn(soc: float) -> float:
        return float(np.interp(np.clip(soc, 0.02, 0.98), soc_pts, ocv_pts))
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders — each returns Trip or raises FileNotFoundError
# ─────────────────────────────────────────────────────────────────────────────

def _clip_trip(trip: Trip, max_steps: int) -> Trip:
    """Limit trip to max_steps timesteps."""
    n = min(max_steps, len(trip.t_s))
    trip.t_s      = trip.t_s[:n]
    trip.I_cell_A = trip.I_cell_A[:n]
    trip.V_cell   = trip.V_cell[:n]
    trip.soc_bms  = trip.soc_bms[:n]
    return trip


def load_ved(max_steps: int = 300) -> Trip:
    from data.loaders.ved_loader import VEDLoader
    from data.loaders.pack_cartridge import lookup_ved_cartridge
    from data.loaders.common_schema import resample_to_uniform_dt

    loader = VEDLoader(max_veh=5, max_trips_per_veh=20)
    for seg_df, meta in loader.iter_segments():
        dur = float(seg_df["t_s"].iloc[-1] - seg_df["t_s"].iloc[0])
        if dur < 600:
            continue
        vid = next((n.replace("vehicle=", "") for n in meta.notes if n.startswith("vehicle=")), "")
        cart = lookup_ved_cartridge(vid)
        seg_r = resample_to_uniform_dt(seg_df, 20.0)
        t   = seg_r["t_s"].values.astype(np.float64)
        I_p = seg_r["I_A"].values.astype(np.float64) / cart.n_parallel
        V_c = seg_r["V_V"].values.astype(np.float64) / cart.n_series
        soc = seg_r["SOC_bms"].values.astype(np.float64)
        print(f"[VED] {meta.vehicle_id}/{meta.segment_id}: n={len(t)}, "
              f"dur={t[-1]/60:.1f} min, SOC {soc[0]:.2%}→{soc[-1]:.2%}, "
              f"cart={cart.name}")
        # R_int=0.002 Ω (2 mΩ/cell) is data-implied for this ~40 Ah EV cell.
        # cart.R_ohm_cell=0.015 Ω gives ±600 mV V_pred swings at ±40 A
        # bipolar driving currents → inflated V_MAE with no SOC benefit.
        trip = Trip(
            dataset="VED", chemistry="LMO-NMC", cart_name=cart.name,
            Q_cell_Ah=cart.Q_cell_Ah, t_s=t, I_cell_A=I_p, V_cell=V_c, soc_bms=soc,
            ocv_fn=_lmonmc_ocv(), ekf_gamma=2.0, ekf_R_meas=1e-6,
            ekf_R_int=0.002,
            notes=[f"vehicle={meta.vehicle_id}", f"segment={meta.segment_id}",
                   "R_int=2mOhm (data-implied; generic pack 15mOhm too large)"],
        )
        return _clip_trip(trip, max_steps)
    raise FileNotFoundError("VED: no trip ≥ 600 s found in data/ved/")


def load_bmw_i3(max_steps: int = 300) -> Trip:
    from data.loaders.bmw_i3_loader import BMWI3Loader
    from data.loaders.pack_cartridge import BMW_I3_60AH
    from data.loaders.common_schema import resample_to_uniform_dt

    # Require ≥ 25 min so EKF has time to converge from +10% init offset.
    # R_int_ohm=0.001 Ω (1 mΩ/cell) is the data-implied dynamic resistance;
    # the cartridge value 0.040 Ω is a DC/SOH parameter that blows up
    # EKF V_pred at transient peak currents of ±265 A (10.6 V/cell ohmic drop).
    loader = BMWI3Loader(max_trips=10)
    for seg_df, meta in loader.iter_segments():
        dur = float(seg_df["t_s"].iloc[-1] - seg_df["t_s"].iloc[0])
        if dur < 1500:
            continue
        cart = BMW_I3_60AH
        seg_r = resample_to_uniform_dt(seg_df, 20.0)
        t   = seg_r["t_s"].values.astype(np.float64)
        I_c = seg_r["I_A"].values.astype(np.float64) / cart.n_parallel
        V_c = seg_r["V_V"].values.astype(np.float64) / cart.n_series
        soc = seg_r["SOC_bms"].values.astype(np.float64)
        print(f"[BMW] {meta.segment_id}: n={len(t)}, "
              f"dur={t[-1]/60:.1f} min, SOC {soc[0]:.2%}→{soc[-1]:.2%}, "
              f"I_cell=[{I_c.min():.1f},{I_c.max():.1f}]A")
        trip = Trip(
            dataset="BMW i3", chemistry="NMC111", cart_name=cart.name,
            Q_cell_Ah=cart.Q_cell_Ah, t_s=t, I_cell_A=I_c, V_cell=V_c, soc_bms=soc,
            ocv_fn=_nmc_generic_ocv(), ekf_gamma=2.0, ekf_R_meas=1e-6,
            ekf_R_int=0.001,
            notes=[f"segment={meta.segment_id}", "R_int=1mOhm (data-implied dynamic)"],
        )
        return _clip_trip(trip, max_steps)
    raise FileNotFoundError("BMW i3: no trip ≥ 1500 s found in data/bmw_i3/")


def load_deng(max_steps: int = 300) -> Trip:
    from data.loaders.deng_charging_loader import DengChargingLoader
    from data.loaders.pack_cartridge import BAIC_EU500_90S
    from data.loaders.common_schema import resample_to_uniform_dt

    cart = BAIC_EU500_90S
    loader = DengChargingLoader(max_vehicles=3, max_sessions_per_vehicle=5)
    for seg_df, meta in loader.iter_segments():
        dur = float(seg_df["t_s"].iloc[-1] - seg_df["t_s"].iloc[0])
        # Deng: ~8 s sampling. Need ≥ 600 s and meaningful SOC swing.
        if dur < 600:
            continue
        soc_arr = seg_df["SOC_bms"].values.astype(np.float64)
        if abs(soc_arr[-1] - soc_arr[0]) < 0.05:
            continue
        seg_r = resample_to_uniform_dt(seg_df, 20.0)
        t   = seg_r["t_s"].values.astype(np.float64)
        I_c = seg_r["I_A"].values.astype(np.float64) / cart.n_parallel
        V_c = seg_r["V_V"].values.astype(np.float64) / cart.n_series
        soc = seg_r["SOC_bms"].values.astype(np.float64)
        print(f"[Deng] {meta.vehicle_id}/{meta.segment_id}: n={len(t)}, "
              f"dur={t[-1]/60:.1f} min, SOC {soc[0]:.2%}→{soc[-1]:.2%} "
              f"(CHARGING: I_schema>0)")
        # Deng is a charging dataset: I_schema > 0 (loader negated)
        trip = Trip(
            dataset="Deng/BAIC", chemistry="NCM (CATL)", cart_name=cart.name,
            Q_cell_Ah=cart.Q_cell_Ah, t_s=t, I_cell_A=I_c, V_cell=V_c, soc_bms=soc,
            ocv_fn=_nmc_generic_ocv(), ekf_gamma=1.0, ekf_R_meas=2.5e-7,
            ekf_R_int=cart.R_ohm_cell,
            notes=[f"vehicle={meta.vehicle_id}", "charging_session"],
        )
        return _clip_trip(trip, max_steps)
    raise FileNotFoundError("Deng: no charging session ≥ 600 s found in data/deng20/")


def load_quartz(max_steps: int = 300) -> Trip:
    """Load one WLTP cycle from Quartz WLTP parquet files."""
    wltp_dir = ROOT / "data" / "quartz_wltp"
    wltp_files = sorted(wltp_dir.glob("Qtzl_Cycle_0*_WLTP*.parquet"))
    if not wltp_files:
        raise FileNotFoundError(
            f"Quartz WLTP: no Qtzl_Cycle_*_WLTP*.parquet found in {wltp_dir}"
        )

    N_P, N_S = 3, 12
    Q_CELL_AH = 2.5
    R_OHM = 0.005

    for pq_path in wltp_files:
        df = pd.read_parquet(pq_path)
        soc_full = df["SoC_Actual_Battery [percent]"].values.astype(np.float64) / 100.0

        # Find a window with ≥ 15% SOC swing (active WLTP cycling)
        window = 500  # raw rows at ~0.4 s sampling ≈ 200 s
        found_start = None
        for start in range(0, len(df) - window, window // 2):
            soc_win = soc_full[start:start + window]
            if soc_win.max() - soc_win.min() >= 0.10:
                found_start = start
                break
        if found_start is None:
            continue

        # Take up to 60 min from that window, resample to 20 s
        t_raw = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds().values
        end_t = t_raw[found_start] + 3600.0  # 60 min window
        mask = (t_raw >= t_raw[found_start]) & (t_raw <= end_t)
        df_win = df.loc[mask].copy().reset_index(drop=True)

        # Build per-cell arrays from per-branch data
        t_abs = (df_win["Timestamp"] - df_win["Timestamp"].iloc[0]).dt.total_seconds().values
        # Average 3 branch currents → per-branch (= per series string) current
        I_branches = np.stack([
            df_win[f"Current_Actual_P{p} [A]"].values.astype(np.float64)
            for p in range(1, N_P + 1)
        ], axis=0)
        I_cell_raw = np.mean(I_branches, axis=0)  # discharge-negative per schema

        # Average cell voltage from pre-computed column
        V_cell_raw = df_win["Voltage_Avg_Cell [V]"].values.astype(np.float64)
        soc_raw = df_win["SoC_Actual_Battery [percent]"].values.astype(np.float64) / 100.0

        # Build a simple DataFrame and resample to 20 s
        df_schema = pd.DataFrame({
            "t_s": t_abs, "I_A": I_cell_raw, "V_V": V_cell_raw,
            "T_degC": np.full(len(t_abs), 25.0), "SOC_bms": soc_raw,
        })
        from data.loaders.common_schema import resample_to_uniform_dt
        df_r = resample_to_uniform_dt(df_schema, 20.0)

        t   = df_r["t_s"].values.astype(np.float64)
        I_c = df_r["I_A"].values.astype(np.float64)
        V_c = df_r["V_V"].values.astype(np.float64)
        soc = df_r["SOC_bms"].values.astype(np.float64)

        if len(t) < 20:
            continue

        print(f"[Quartz] {pq_path.name}: n={len(t)}, "
              f"dur={t[-1]/60:.1f} min, SOC {soc[0]:.2%}→{soc[-1]:.2%}")
        trip = Trip(
            dataset="Quartz WLTP", chemistry="NMC811", cart_name="Quartz 3P×12S 2.5Ah NMC811",
            Q_cell_Ah=Q_CELL_AH, t_s=t, I_cell_A=I_c, V_cell=V_c, soc_bms=soc,
            ocv_fn=_nmc811_ocv(), ekf_gamma=1.0, ekf_R_meas=1e-6, ekf_R_int=R_OHM,
            notes=[f"file={pq_path.name}", f"N_P={N_P}", f"N_S={N_S}"],
        )
        return _clip_trip(trip, max_steps)

    raise FileNotFoundError(
        "Quartz WLTP: no suitable window with ≥ 10% SOC swing found."
    )


def load_nasa() -> Trip:
    """
    NASA B0018 real data.  The repository contains nasa_battery.zip and
    nasa_battery_data.zip but no extracted MAT/CSV files with a current
    profile usable for EKF comparison.  The nasa_validator.py in this repo
    uses SYNTHETIC data (DFN-generated) that matches NASA B0018 statistics.
    Real extracted NASA profiles require MATLAB or scipy.io.loadmat.
    """
    nasa_dir = ROOT / "data" / "nasa"
    mat_files = list(nasa_dir.glob("*.mat")) + list(nasa_dir.glob("*.csv"))
    if not mat_files:
        raise FileNotFoundError(
            "NASA: no .mat or .csv files in data/nasa/. "
            "The nasa_battery.zip needs to be extracted and parsed "
            "with scipy.io.loadmat. Only a synthetic validator is present."
        )
    raise FileNotFoundError("NASA: real current profiles not available (see notes).")


# ─────────────────────────────────────────────────────────────────────────────
# Coulomb-counting ground truth
# ─────────────────────────────────────────────────────────────────────────────

def _coulomb_counting(t_s, I_cell_A, soc0, Q_cell_Ah):
    """discharge-negative convention: I<0 → soc decreases."""
    soc = np.empty(len(t_s))
    soc[0] = soc0
    for i in range(1, len(t_s)):
        dt = float(t_s[i] - t_s[i-1])
        soc[i] = np.clip(soc[i-1] + I_cell_A[i-1] * dt / (3600.0 * Q_cell_Ah), 0.0, 1.0)
    return soc


# ─────────────────────────────────────────────────────────────────────────────
# EKF pass
# ─────────────────────────────────────────────────────────────────────────────

def run_ekf(
    trip: Trip, soc_init: float, current_bias: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (soc_ekf, V_pred, µs_per_step).

    current_bias: fractional bias applied to the measured current before the
    EKF prediction step (simulates a biased sensor).  V_meas is unchanged —
    the EKF still reads the true terminal voltage, so Kalman updates can
    partially correct the drift.  Ground-truth SOC uses unbiased current.
    """
    from diagnosis.dual_ekf_lfp import DualEKF_LFP

    ekf = DualEKF_LFP(
        Q_nom_Ah=trip.Q_cell_Ah,
        R_int_ohm=trip.ekf_R_int,
        ocv_fn=trip.ocv_fn,
        R_meas_V2=trip.ekf_R_meas,
        P0_soc=(0.10) ** 2,
        gamma=trip.ekf_gamma,
        cal_soc_fn=None,
        cal_dR0=0.0,
    )
    ekf.set_soc(soc_init)

    soc_ekf = np.empty(len(trip.t_s))
    V_pred  = np.empty(len(trip.t_s))
    dt0 = float(trip.t_s[1] - trip.t_s[0]) if len(trip.t_s) > 1 else 20.0
    t0 = time.perf_counter()

    for i in range(len(trip.t_s)):
        # Apply sensor bias to current; V_meas is unaffected (voltage sensor ok)
        I_neg = float(trip.I_cell_A[i]) * (1.0 + current_bias)
        V_m   = float(trip.V_cell[i])
        dt    = float(trip.t_s[i] - trip.t_s[i-1]) if i > 0 else dt0
        res = ekf.update(V_m, -I_neg, dt)   # EKF expects discharge-positive
        soc_ekf[i] = float(res["soc"])
        V_pred[i]  = float(res["V_pred"])

    return soc_ekf, V_pred, (time.perf_counter() - t0) / len(trip.t_s) * 1e6


# ─────────────────────────────────────────────────────────────────────────────
# PyBaMM pass
# ─────────────────────────────────────────────────────────────────────────────

Q_CHEN = 5.0   # Chen2020 LG M50 nominal capacity [Ah]
PYBAMM_PARAM_SET = "Chen2020"


def run_pybamm(
    trip: Trip, soc_init: float, current_bias: float = 0.0,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Returns (soc_pybamm, V_pybamm, µs_per_step) or None on failure.

    current_bias: fractional bias on measured current.  PyBaMM is open-loop
    (no V_meas feedback), so the bias causes a monotone SOC drift that
    accumulates unboundedly over the trip.
    """
    if not _HAVE_PYBAMM:
        return None

    # C-rate scaling: discharge-negative schema → discharge-positive C-rate.
    # Bias is applied before clamping (matches what a biased sensor would supply).
    C_rate   = (-trip.I_cell_A / trip.Q_cell_Ah) * (1.0 + current_bias)
    I_pb     = np.clip(C_rate * Q_CHEN, -1.5 * Q_CHEN, 1.5 * Q_CHEN)
    t_data   = trip.t_s.copy()

    try:
        model = pybamm.lithium_ion.SPMe()
        param = pybamm.ParameterValues(PYBAMM_PARAM_SET)

        try:
            param.update({"Initial SoC": soc_init}, check_already_exists=False)
        except TypeError:
            param.update({"Initial SoC": soc_init})
        except Exception:
            pass

        # PyBaMM 26.x passes a symbolic Time object to callables,
        # so use pybamm.Interpolant instead of a Python lambda with float(t).
        param["Current function [A]"] = pybamm.Interpolant(
            t_data.copy(), I_pb.copy(), pybamm.t
        )

        solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-5)
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

        n_eval = min(len(trip.t_s), 200)
        t_eval = np.linspace(float(trip.t_s[0]), float(trip.t_s[-1]), n_eval)

        t0 = time.perf_counter()
        sol = sim.solve(t_eval)
        elapsed = time.perf_counter() - t0

        # sol.t is the time array directly in PyBaMM 26.x
        t_sol = np.asarray(sol.t, dtype=float)
        V_sol = np.asarray(sol["Terminal voltage [V]"].entries, dtype=float)
        # "State of charge" not present in Chen2020/26.x — use discharge capacity
        try:
            soc_sol = np.asarray(sol["State of charge"].entries, dtype=float)
        except KeyError:
            Q_dis = np.asarray(sol["Discharge capacity [A.h]"].entries, dtype=float)
            soc_sol = np.clip(soc_init - Q_dis / Q_CHEN, 0.0, 1.0)

        V_out   = np.interp(trip.t_s, t_sol, V_sol)
        soc_out = np.interp(trip.t_s, t_sol, soc_sol)
        return soc_out, V_out, elapsed / len(trip.t_s) * 1e6

    except Exception as exc:
        print(f"  [PyBaMM] {trip.dataset} failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(V_meas, V_pred, soc_truth, soc_est) -> dict:
    return {
        "V_MAE_mV":          float(np.mean(np.abs(V_meas - V_pred))) * 1000,
        "SOC_RMSE_%":        float(np.sqrt(np.mean((soc_truth - soc_est) ** 2))) * 100,
        "SOC_final_err_%":   float(abs(soc_truth[-1] - soc_est[-1])) * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined plot
# ─────────────────────────────────────────────────────────────────────────────

def save_plot(
    results: list,
    out_path: Path,
    current_bias: float = 0.0,
    soc_offset: float = 0.0,
) -> None:
    """3-row grid per dataset: (0) SOC absolute, (1) SOC error vs time, (2) voltage."""
    if not _HAVE_MPL:
        print("[PLOT] matplotlib not available — skipping.")
        return

    active = [r for r in results if r.get("status") == "ok"]
    n = len(active)
    if n == 0:
        print("[PLOT] No successful datasets to plot.")
        return

    fig, axes = plt.subplots(3, n, figsize=(5 * n, 10), squeeze=False)

    for col, r in enumerate(active):
        t_min     = r["t_s"] / 60.0
        soc_truth = r["soc_truth"]
        soc_ekf   = r["soc_ekf"]
        soc_pb    = r.get("soc_pybamm")

        # ── row 0: SOC absolute ───────────────────────────────────────────
        ax0 = axes[0][col]
        ax0.plot(t_min, soc_truth * 100, "k-",  lw=2,   label="CC truth")
        ax0.plot(t_min, soc_ekf   * 100, "b-",  lw=1.6, label="EKF")
        if soc_pb is not None:
            ax0.plot(t_min, soc_pb * 100, "r--", lw=1.5, label="PyBaMM")
        ax0.set_title(f"{r['dataset']}\n{r['chemistry']}", fontsize=9)
        ax0.set_ylabel("SOC [%]" if col == 0 else "")
        ax0.legend(fontsize=7)
        ax0.grid(True, alpha=0.3)

        # ── row 1: SOC error vs time (signed: est − truth) ───────────────
        ax1 = axes[1][col]
        err_ekf = (soc_ekf - soc_truth) * 100          # signed %, + = over-estimate
        ax1.plot(t_min, err_ekf, "b-", lw=1.6, label="EKF error")
        ax1.axhline(0, color="k", lw=0.8, ls="--")
        if soc_pb is not None:
            err_pb = (soc_pb - soc_truth) * 100
            ax1.plot(t_min, err_pb, "r--", lw=1.5, label="PyBaMM error")
        ax1.set_ylabel("SOC error [pp]" if col == 0 else "")
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        # Shade monotone drift region for PyBaMM if available
        if soc_pb is not None and len(err_pb) > 3:
            ax1.fill_between(t_min, err_pb, 0,
                             where=np.abs(err_pb) > np.abs(err_ekf),
                             color="red", alpha=0.08,
                             label="_nolegend_")

        # ── row 2: terminal voltage ───────────────────────────────────────
        ax2 = axes[2][col]
        ax2.plot(t_min, r["V_meas"] * 1000, "k-",  lw=1.5, label="V meas")
        ax2.plot(t_min, r["V_ekf"]  * 1000, "b-",  lw=1.2, alpha=0.85, label="EKF pred")
        if r.get("V_pybamm") is not None:
            ax2.plot(t_min, r["V_pybamm"] * 1000, "r--", lw=1.2, alpha=0.85,
                     label="PyBaMM pred")
        ax2.set_xlabel("Time [min]")
        ax2.set_ylabel("V cell [mV]" if col == 0 else "")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    bias_str   = f"  |  current bias {current_bias*100:.1f}%" if current_bias else ""
    offset_str = f"+{soc_offset*100:.0f}% SOC init offset" if soc_offset else "no SOC init offset"
    fig.suptitle(
        f"PyBaMM ({PYBAMM_PARAM_SET}) vs OpenCATHODE EKF — {offset_str}{bias_str}\n"
        f"Row 1: SOC absolute  |  Row 2: SOC error (signed pp, + = over-estimate)  "
        f"|  Row 3: terminal voltage",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: list, soc_offset: float) -> None:
    hdr = (
        f"{'Dataset':<16} {'Chem':<14} {'Real?':>5} "
        f"{'EKF SOC RMSE':>13} {'EKF SOC fin':>12} {'EKF V MAE':>10} "
        f"{'PB SOC RMSE':>12} {'PB SOC fin':>11} {'PB V MAE':>9} "
        f"{'EKF µs/step':>12}"
    )
    sep = "─" * len(hdr)
    print()
    print("=" * len(hdr))
    print(f"  CONSOLIDATED COMPARISON  —  +{soc_offset*100:.0f}% init SOC offset, "
          f"PyBaMM param set: {PYBAMM_PARAM_SET}")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    for r in results:
        ds  = r["dataset"][:15]
        ch  = r.get("chemistry", "—")[:13]
        real = "YES" if r.get("status") == "ok" else "NO"

        if r["status"] == "ok":
            ekf  = r["ekf_m"]
            pb   = r.get("pb_m")
            rt   = r.get("ekf_us", float("nan"))

            def _f(d, k, fmt=".1f"):
                if d is None:
                    return "N/A".rjust(len(fmt) + 4)
                return f"{d[k]:{fmt}}"

            pb_ran = "YES" if pb else "N/A"
            row = (
                f"{ds:<16} {ch:<14} {real:>5} "
                f"{_f(ekf, 'SOC_RMSE_%'):>13} "
                f"{_f(ekf, 'SOC_final_err_%'):>12} "
                f"{_f(ekf, 'V_MAE_mV'):>10} "
                f"{_f(pb,  'SOC_RMSE_%'):>12} "
                f"{_f(pb,  'SOC_final_err_%'):>11} "
                f"{_f(pb,  'V_MAE_mV'):>9} "
                f"{rt:>12.1f}"
            )
        else:
            msg = r.get("error", "unknown error")[:55]
            row = (
                f"{ds:<16} {ch:<14} {real:>5}  "
                f"DATA MISSING — {msg}"
            )
        print(row)
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Bias-sensitivity comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_bias_table(
    results_b0: list,
    results_bx: list,
    bias_x: float,
    soc_offset: float,
) -> None:
    """Side-by-side SOC RMSE and final-error at bias=0 and bias=X."""
    # Column widths
    W = 8

    def _v(d, k):
        """Format metric value or 'N/A' / 'FAIL'."""
        if d is None:
            return "  N/A  "
        if d == "fail":
            return " FAIL  "
        v = d.get(k)
        if v is None:
            return "  N/A  "
        return f"{v:>{W}.1f}"

    def _delta(d0, dx, k):
        """Signed delta dx − d0, or blank if either missing."""
        if d0 is None or dx is None or d0 == "fail" or dx == "fail":
            return "  —    "
        v0, vx = d0.get(k), dx.get(k)
        if v0 is None or vx is None:
            return "  —    "
        delta = vx - v0
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:>{W-1}.1f}"

    bx_pct = f"{bias_x*100:.0f}%"
    hdr1 = (
        f"{'':17} {'─'*8} SOC RMSE [%] {'─'*33}  {'─'*8} Final SOC error [%] {'─'*28}"
    )
    hdr2 = (
        f"{'Dataset':<16} {'C':1}  "
        f"{'EKF@0%':>{W}}  {'EKF@'+bx_pct:>{W}}  {'ΔEKF':>{W}}  "
        f"{'PB@0%':>{W}}  {'PB@'+bx_pct:>{W}}  {'ΔPB':>{W}}    "
        f"{'EKF@0%':>{W}}  {'EKF@'+bx_pct:>{W}}  {'ΔEKF':>{W}}  "
        f"{'PB@0%':>{W}}  {'PB@'+bx_pct:>{W}}  {'ΔPB':>{W}}"
    )
    sep = "─" * len(hdr2)

    print()
    print("=" * len(hdr2))
    print(f"  BIAS SENSITIVITY  —  +{soc_offset*100:.0f}% init offset  |  "
          f"current-sensor bias 0% → {bx_pct}  |  PyBaMM: {PYBAMM_PARAM_SET}")
    print("=" * len(hdr2))
    print(hdr1)
    print(hdr2)
    print(sep)

    # Build lookup by dataset name for both bias runs
    def _lookup(results, ds):
        for r in results:
            if r["dataset"] == ds:
                return r
        return None

    datasets = [r["dataset"] for r in results_b0]
    for ds in datasets:
        r0 = _lookup(results_b0, ds)
        rx = _lookup(results_bx, ds)

        if r0 is None or r0.get("status") != "ok":
            chem = (r0 or {}).get("chemistry", "?")[:1]
            print(f"{ds:<16} {chem}  DATA MISSING")
            continue

        chem = r0.get("chemistry", "?")[:1]
        ekf0 = r0.get("ekf_m")
        ekfx = rx.get("ekf_m") if rx else None
        pb0  = r0.get("pb_m")
        pbx  = rx.get("pb_m") if rx else None
        # Mark solver failure explicitly
        if r0.get("pb_failed"):  pb0  = "fail"
        if rx and rx.get("pb_failed"): pbx = "fail"

        print(
            f"{ds:<16} {chem}  "
            f"{_v(ekf0,'SOC_RMSE_%')}  {_v(ekfx,'SOC_RMSE_%')}  {_delta(ekf0,ekfx,'SOC_RMSE_%')}  "
            f"{_v(pb0,'SOC_RMSE_%')}  {_v(pbx,'SOC_RMSE_%')}  {_delta(pb0,pbx,'SOC_RMSE_%')}    "
            f"{_v(ekf0,'SOC_final_err_%')}  {_v(ekfx,'SOC_final_err_%')}  {_delta(ekf0,ekfx,'SOC_final_err_%')}  "
            f"{_v(pb0,'SOC_final_err_%')}  {_v(pbx,'SOC_final_err_%')}  {_delta(pb0,pbx,'SOC_final_err_%')}"
        )

    print(sep)
    print("  ΔEKF / ΔPB = (bias run) − (no-bias run).  Positive = degraded.")
    print("  FAIL = PyBaMM solver aborted (event violation).  N/A = PyBaMM not run / dataset missing.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Print assumptions
# ─────────────────────────────────────────────────────────────────────────────

def print_assumptions(results: list, soc_offset: float, max_steps: int,
                      current_bias: float = 0.0) -> None:
    print()
    print("ASSUMPTIONS AND DESIGN CHOICES")
    print("─" * 70)
    blk = [
        f"1. SOC OFFSET: Both EKF and PyBaMM initialised at BMS_SOC_0 +{soc_offset*100:.0f}%.",
        "   Coulomb-counting ground-truth starts at BMS_SOC_0 (trusted start).",
        "",
        f"2. PYBAMM MODEL: SPMe with parameter set '{PYBAMM_PARAM_SET}' (LG M50",
        "   NMC811 18650, Q=5 Ah).  NOT the actual fleet cell in any row.",
        f"   Current is C-rate scaled: I_pybamm = (I_meas/Q_fleet) × {Q_CHEN} Ah.",
        "   PyBaMM has NO access to V_meas — open-loop forward simulation only.",
        "",
        "3. EKF CONFIG (per dataset, single-trip mode — no fleet calibration):",
        "   VED:        LMO-NMC PCHIP OCV, γ=2.0, R_meas=(1 mV)², Q=40 Ah/cell (generic)",
        "               R_int=2 mΩ/cell (data-implied; generic pack 15 mΩ gives ±600 mV",
        "               V_pred swings at ±40 A bipolar driving currents).",
        "   BMW i3:     generic NMC OCV,    γ=2.0, R_meas=(1 mV)², Q=60 Ah/cell",
        "               R_int=1 mΩ/cell (data-implied; cartridge 40 mΩ is DC/SOH",
        "               parameter — blows up EKF at ±265 A transient peak currents).",
        "   Deng/BAIC:  generic NMC OCV,    γ=1.0, R_meas=(0.5mV)², Q=145 Ah/cell",
        "   Quartz:     NMC811 OCV (DFN),   γ=1.0, R_meas=(1 mV)², Q=2.5 Ah/cell",
        "   Full fleet validation adds 12-bin PCHIP δV(SOC) cal on 10% cal split.",
        "",
        "4. GROUND-TRUTH SOC: Coulomb counting from BMS SOC_0.  BMS accuracy ±2–5%.",
        "   For Deng (charging), truth SOC rises; EKF and PyBaMM both start high.",
        "",
        f"5. TRIP LENGTH: capped at {max_steps} timesteps × 20 s = {max_steps*20/60:.0f} min.",
        "   Full trips used where available; capped for PyBaMM speed.",
        "",
        "6. DENG NOTE: charging dataset (SOC rises). EKF handles bidirectional",
        "   current correctly. PyBaMM Chen2020 simulates charging via I < 0 in",
        "   its discharge-positive convention. V_meas is CC charging voltage.",
        "",
        "7. QUARTZ NOTE: per-branch current averaged over 3 strings (P1–P3).",
        "   Per-cell voltage from 'Voltage_Avg_Cell [V]' column (pack mean).",
        "   n_series=12 and n_parallel=3 used only to interpret raw columns;",
        "   EKF operates on already-scaled per-cell I and V arrays.",
        "",
        "8. NASA NOTE: data/nasa/ contains only zip archives. Extraction and",
        "   scipy.io.loadmat parsing not implemented. The nasa_validator.py in",
        "   this repo uses synthetic DFN-generated profiles, not real lab data.",
        "   Marked DATA MISSING in this comparison.",
        "",
        "9. ASYMMETRIC TASK: EKF reads V_meas at every step → closed-loop SOC",
        "   correction. PyBaMM uses only I(t) → SOC init error propagates.",
        "   The comparison shows WHY closed-loop estimation exists for BMS.",
        "",
        f"10. CURRENT-SENSOR BIAS: {current_bias*100:.1f}% multiplicative bias applied to",
        "    I_meas seen by BOTH EKF and PyBaMM.  V_meas is unaffected (voltage",
        "    sensor assumed accurate).  Ground-truth SOC uses unbiased current.",
        "    PyBaMM is open-loop, so bias accumulates monotonically into SOC error.",
        "    EKF reads V_meas at each step, so Kalman updates partially cancel",
        "    the bias-driven SOC drift — quantified in the BIAS SENSITIVITY table.",
    ]
    for ln in blk:
        print("  " + ln)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

LOADERS = [
    ("VED",           "LMO-NMC",       load_ved),
    ("BMW i3",        "NMC111",        load_bmw_i3),
    ("Deng/BAIC",     "NCM (CATL)",    load_deng),
    ("Quartz WLTP",   "NMC811",        load_quartz),
    ("NASA B0018",    "NMC811/gr",     load_nasa),
]


def _run_one_bias(
    offset: float,
    max_steps: int,
    current_bias: float,
    trips_cache: dict,
    verbose: bool = True,
) -> list:
    """Run all datasets at a single current_bias level and return results list.

    trips_cache: populated on first call; subsequent calls reuse loaded data
    so datasets are loaded exactly once across multiple bias levels.
    """
    results = []

    for ds_name, chem, loader_fn in LOADERS:
        if verbose:
            print(f"{'─'*50}")
            print(f"  {ds_name}  ({chem})  [bias={current_bias*100:.1f}%]")

        r: dict = {"dataset": ds_name, "chemistry": chem}

        # ── load trip (cached after first bias level) ──────────────────────
        if ds_name not in trips_cache:
            try:
                trip = (loader_fn(max_steps=max_steps)
                        if ds_name != "NASA B0018" else loader_fn())
                trips_cache[ds_name] = trip
            except FileNotFoundError as exc:
                trips_cache[ds_name] = exc

        cached = trips_cache[ds_name]
        if isinstance(cached, Exception):
            r["status"] = "missing"
            r["error"]  = str(cached)[:100]
            if verbose:
                print(f"  → DATA MISSING: {cached}")
            results.append(r)
            continue

        trip      = cached
        soc_init  = float(np.clip(float(trip.soc_bms[0]) + offset, 0.02, 0.98))
        # Ground truth always uses unbiased current — bias is a sensor artefact
        soc_truth = _coulomb_counting(
            trip.t_s, trip.I_cell_A, float(trip.soc_bms[0]), trip.Q_cell_Ah
        )

        if verbose and current_bias == 0.0:
            print(f"  BMS SOC_0={trip.soc_bms[0]:.3f}  init={soc_init:.3f} "
                  f"(+{offset*100:.0f}%)  Q_cell={trip.Q_cell_Ah:.1f} Ah")
            print(f"  OCV: fn  γ={trip.ekf_gamma}  "
                  f"R_meas=({(trip.ekf_R_meas**0.5)*1000:.2f} mV)²")

        # ── EKF ───────────────────────────────────────────────────────────
        soc_ekf, V_ekf, ekf_us = run_ekf(trip, soc_init, current_bias)
        ekf_m = _metrics(trip.V_cell, V_ekf, soc_truth, soc_ekf)
        if verbose:
            print(f"  EKF → RMSE={ekf_m['SOC_RMSE_%']:.1f}%  "
                  f"fin={ekf_m['SOC_final_err_%']:.1f}%  "
                  f"V_MAE={ekf_m['V_MAE_mV']:.1f} mV  {ekf_us:.0f} µs/step")

        # ── PyBaMM ────────────────────────────────────────────────────────
        pb_result = None
        pb_m      = None
        pb_failed = False
        if _HAVE_PYBAMM:
            if verbose:
                print(f"  PyBaMM ({PYBAMM_PARAM_SET}) running …")
            pb_result = run_pybamm(trip, soc_init, current_bias)
            if pb_result is not None:
                soc_pb, V_pb, pb_us = pb_result
                pb_m = _metrics(trip.V_cell, V_pb, soc_truth, soc_pb)
                if verbose:
                    print(f"  PyBaMM → RMSE={pb_m['SOC_RMSE_%']:.1f}%  "
                          f"fin={pb_m['SOC_final_err_%']:.1f}%  "
                          f"V_MAE={pb_m['V_MAE_mV']:.1f} mV  {pb_us:.0f} µs/step")
            else:
                pb_failed = True
                if verbose:
                    print(f"  PyBaMM → solver failed (error shown above)")
        else:
            soc_pb = V_pb = None
            pb_us  = float("nan")

        r.update({
            "status":     "ok",
            "trip":       trip,
            "t_s":        trip.t_s,
            "soc_truth":  soc_truth,
            "V_meas":     trip.V_cell,
            "soc_ekf":    soc_ekf,
            "V_ekf":      V_ekf,
            "ekf_m":      ekf_m,
            "ekf_us":     ekf_us,
            "soc_pybamm": soc_pb if pb_result else None,
            "V_pybamm":   V_pb   if pb_result else None,
            "pb_m":       pb_m,
            "pb_failed":  pb_failed,
            "pb_us":      pb_us if pb_result else float("nan"),
        })
        results.append(r)

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soc-offset", type=float, default=0.10,
                    help="Deliberate SOC init error (default 0.10 = +10%%)")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="Max timesteps per trip (default 300 × 20s = 100 min)")
    ap.add_argument("--current-bias", type=float, default=0.0,
                    help="Fractional current-sensor bias applied to both methods "
                         "(e.g. 0.02 = 2%% over-read). Default 0.")
    args = ap.parse_args()
    offset       = float(args.soc_offset)
    max_steps    = int(args.max_steps)
    current_bias = float(args.current_bias)

    run_bias_compare = current_bias != 0.0

    print("=" * 70)
    print("  PyBaMM vs OpenCATHODE EKF — Multi-dataset comparison")
    print("=" * 70)
    print(f"  SOC init offset  : +{offset*100:.0f}%")
    print(f"  max_steps        : {max_steps} (~{max_steps*20//60} min at 20 s)")
    print(f"  current_bias     : {current_bias*100:.1f}%"
          + ("  (+ baseline at 0% for comparison)" if run_bias_compare else ""))
    print(f"  PyBaMM           : "
          f"{'installed (' + pybamm.__version__ + ')' if _HAVE_PYBAMM else 'NOT installed'}")
    print()

    trips_cache: dict = {}   # loaded once, shared across both bias runs

    # ── PASS 1: bias = 0% (baseline) ─────────────────────────────────────────
    print("═" * 70)
    print(f"  PASS 1 / {'2' if run_bias_compare else '1'}  —  current bias = 0%  (baseline)")
    print("═" * 70)
    results_b0 = _run_one_bias(offset, max_steps, 0.0, trips_cache, verbose=True)
    print_table(results_b0, offset)

    # ── PASS 2: bias = X% ────────────────────────────────────────────────────
    if run_bias_compare:
        print()
        print("═" * 70)
        print(f"  PASS 2 / 2  —  current bias = {current_bias*100:.1f}%")
        print("═" * 70)
        results_bx = _run_one_bias(offset, max_steps, current_bias, trips_cache, verbose=True)
        print_table(results_bx, offset)
        print_bias_table(results_b0, results_bx, current_bias, offset)
    else:
        results_bx = results_b0

    print_assumptions(results_b0, offset, max_steps, current_bias)

    # ── Plots ─────────────────────────────────────────────────────────────────
    out_b0 = Path(__file__).parent / "pybamm_vs_opencathode_all.png"
    save_plot(results_b0, out_b0, current_bias=0.0, soc_offset=offset)
    print(f"[DONE] Baseline plot → {out_b0}")
    if run_bias_compare:
        out_bx = (Path(__file__).parent
                  / f"pybamm_vs_opencathode_bias{int(current_bias*100)}pct.png")
        save_plot(results_bx, out_bx, current_bias=current_bias, soc_offset=offset)
        print(f"[DONE] Bias={current_bias*100:.0f}% plot → {out_bx}")


if __name__ == "__main__":
    main()
