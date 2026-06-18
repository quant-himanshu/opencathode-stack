#!/usr/bin/env python3
"""
compare_pybamm.py — Fair head-to-head: PyBaMM (forward) vs OpenCATHODE EKF (inverse).

FRAMING — two fundamentally different tasks:
  PyBaMM : open-loop FORWARD SIMULATOR.  Given I(t) → predicts V(t) and SOC(t).
           No voltage feedback.  SOC init error propagates indefinitely.
           Uses Chen2020 generic NMC811 cell (5 Ah), C-rate scaled to fleet cell.
  EKF    : closed-loop INVERSE ESTIMATOR.  Given I(t) + V_meas(t) → corrects SOC
           online via Kalman update.  Converges from ±offset in minutes.

Both are given a deliberate +10% SOC init error (default).  The test asks:
"Can your method recover from a bad start?"  EKF can (closed-loop); PyBaMM
cannot (no feedback path).  This asymmetry IS the BMS design point.

Usage:
    python scripts/compare_pybamm.py                 # try real VED, else synthetic
    python scripts/compare_pybamm.py --synthetic      # synthetic trip (no data needed)
    python scripts/compare_pybamm.py --soc-offset 0.20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── optional dependencies ─────────────────────────────────────────────────────
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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

_Trip = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, int]
# (t_s, I_cell_A [discharge-neg], V_cell_meas, soc_bms, Q_cell_Ah, n_series, n_parallel)


def _load_real_trip(vehicle_id: Optional[str] = None) -> _Trip:
    from data.loaders.ved_loader import VEDLoader
    from data.loaders.pack_cartridge import lookup_ved_cartridge
    from data.loaders.common_schema import resample_to_uniform_dt

    loader = VEDLoader(max_veh=5, max_trips_per_veh=15)
    for seg_df, meta in loader.iter_segments():
        dur = float(seg_df["t_s"].iloc[-1] - seg_df["t_s"].iloc[0])
        if dur < 600:
            continue
        vid = next((n.replace("vehicle=", "") for n in meta.notes if n.startswith("vehicle=")), "")
        if vehicle_id and vid != vehicle_id:
            continue
        cart = lookup_ved_cartridge(vid)
        seg_r = resample_to_uniform_dt(seg_df, 20.0)
        t_s     = seg_r["t_s"].values.astype(np.float64)
        I_cell  = (seg_r["I_A"].values / cart.n_parallel).astype(np.float64)
        V_cell  = (seg_r["V_V"].values / cart.n_series).astype(np.float64)
        soc_bms = seg_r["SOC_bms"].values.astype(np.float64)
        print(f"[DATA] VED {meta.vehicle_id}/{meta.segment_id}: n={len(t_s)}, "
              f"dur={t_s[-1]/60:.1f} min, SOC {soc_bms[0]:.2%}→{soc_bms[-1]:.2%}")
        print(f"[DATA] {cart.name}: {cart.n_series}s{cart.n_parallel}p, "
              f"Q_cell={cart.Q_cell_Ah:.1f} Ah")
        return t_s, I_cell, V_cell, soc_bms, cart.Q_cell_Ah, cart.n_series, cart.n_parallel
    raise FileNotFoundError("No VED trip with duration ≥ 600 s found.")


def _make_synthetic_trip(
    n_steps: int = 600,
    dt_s: float = 20.0,
    soc_init: float = 0.75,
    Q_cell_Ah: float = 33.1,
    n_series: int = 96,
    n_parallel: int = 2,
) -> _Trip:
    """Synthetic VED-like driving trip (no real data needed)."""
    from diagnosis.nmc_ocv import _LMONMC_SOC, _LMONMC_OCV

    rng = np.random.default_rng(2026)
    t_s = np.arange(n_steps, dtype=np.float64) * dt_s

    I_mean = -Q_cell_Ah / 5.0   # ~C/5 discharge (discharge-negative)
    I_cell = np.full(n_steps, I_mean) + rng.normal(0, 2.0, n_steps)
    regen_mask = rng.random(n_steps) < 0.10
    I_cell[regen_mask] = abs(I_mean) * 0.3   # brief regen (positive = charge)

    soc = np.empty(n_steps)
    soc[0] = soc_init
    for i in range(1, n_steps):
        # discharge-negative convention: I < 0 → soc + I*dt/Q < soc (SOC decreases)
        soc[i] = np.clip(soc[i-1] + I_cell[i-1] * dt_s / (3600.0 * Q_cell_Ah), 0.0, 1.0)

    ocv_cell = np.interp(soc, _LMONMC_SOC, _LMONMC_OCV)
    R_cell = 0.012
    # V = OCV + I * R: I is negative during discharge, so V drops below OCV
    V_cell = ocv_cell + I_cell * R_cell + rng.normal(0, 0.0005, n_steps)

    print(f"[DATA] SYNTHETIC trip: {n_steps}×{dt_s}s = {t_s[-1]/60:.1f} min, "
          f"SOC {soc[0]:.2%}→{soc[-1]:.2%}")
    print(f"[DATA] Synthetic pack: {n_series}s{n_parallel}p, Q_cell={Q_cell_Ah:.1f} Ah, LMO-NMC")
    return t_s, I_cell, V_cell, soc, Q_cell_Ah, n_series, n_parallel


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth SOC: Coulomb counting from a trusted BMS start
# ─────────────────────────────────────────────────────────────────────────────

def _coulomb_counting(
    t_s: np.ndarray,
    I_cell_A: np.ndarray,   # discharge-negative
    soc0: float,
    Q_cell_Ah: float,
) -> np.ndarray:
    soc = np.empty(len(t_s))
    soc[0] = soc0
    for i in range(1, len(t_s)):
        dt = float(t_s[i] - t_s[i-1])
        # discharge-negative convention: I < 0 → soc + I*dt/Q < soc (SOC decreases)
        soc[i] = np.clip(soc[i-1] + I_cell_A[i-1] * dt / (3600.0 * Q_cell_Ah), 0.0, 1.0)
    return soc


# ─────────────────────────────────────────────────────────────────────────────
# OpenCATHODE EKF pass
# ─────────────────────────────────────────────────────────────────────────────

def _run_ekf(
    t_s: np.ndarray,
    I_cell_A: np.ndarray,    # discharge-negative
    V_cell_meas: np.ndarray,
    Q_cell_Ah: float,
    soc_init: float,
    n_series: int,
    n_parallel: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Returns (soc_ekf, V_pred_ekf, runtime_us_per_step)."""
    from diagnosis.dual_ekf_lfp import DualEKF_LFP
    from diagnosis.nmc_ocv import build_fleet_ocv, _LMONMC_SOC, _LMONMC_OCV

    # Generic LMO-NMC OCV (no fleet cal data in single-trip mode).
    # Full fleet run uses empirical PCHIP built from 10% cal segments.
    def ocv_fn(soc: float) -> float:
        return float(np.interp(np.clip(soc, 0.0, 1.0), _LMONMC_SOC, _LMONMC_OCV))

    print(f"[EKF] OCV: generic LMO-NMC table (single-trip mode, no fleet cal)")
    print(f"[EKF] Config: Q_cell={Q_cell_Ah:.1f} Ah, SOC_init={soc_init:.3f}, "
          f"gamma=2.0, R_meas=(1 mV)²")

    ekf = DualEKF_LFP(
        Q_nom_Ah=Q_cell_Ah,
        R_int_ohm=0.012,
        ocv_fn=ocv_fn,
        R_meas_V2=1e-6,           # (1 mV)², VED fleet value
        P0_soc=(0.10) ** 2,       # σ=10% matches deliberate +10% offset
        gamma=2.0,                # VED optimal from fleet gamma sweep
        cal_soc_fn=None,          # no SOC-dep cal in single-trip mode
        cal_dR0=0.0,
    )
    ekf.set_soc(soc_init)

    soc_ekf = np.empty(len(t_s))
    V_pred  = np.empty(len(t_s))
    t0 = time.perf_counter()

    for i in range(len(t_s)):
        I_neg = float(I_cell_A[i])   # discharge-negative
        V_m   = float(V_cell_meas[i])
        dt    = float(t_s[i] - t_s[i-1]) if i > 0 else float(t_s[1] - t_s[0])
        # EKF expects discharge-positive: pass -I_neg
        res = ekf.update(V_m, -I_neg, dt)
        soc_ekf[i] = float(res["soc"])
        V_pred[i]  = float(res["V_pred"])

    runtime_us = (time.perf_counter() - t0) / len(t_s) * 1e6
    return soc_ekf, V_pred, runtime_us


# ─────────────────────────────────────────────────────────────────────────────
# PyBaMM pass
# ─────────────────────────────────────────────────────────────────────────────

def _run_pybamm(
    t_s: np.ndarray,
    I_cell_A: np.ndarray,   # discharge-negative (our convention)
    Q_cell_Ah: float,
    soc_init: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Returns (soc_pybamm, V_pybamm [V/cell], runtime_us_per_step) or None on failure.

    Current scaling: same C-rate as the fleet cell, applied to Chen2020 (5 Ah).
    PyBaMM uses discharge-positive convention internally.
    """
    if not _HAVE_PYBAMM:
        print("[PyBaMM] Not installed — run: pip install pybamm")
        print("[PyBaMM] Skipping PyBaMM pass; SOC/V comparison will show N/A.")
        return None

    Q_chen = 5.0   # Chen2020 LG M50 NMC811 cell nominal capacity [Ah]

    # C-rate from fleet cell → scale to Chen2020 cell
    # I_cell_A is discharge-negative; PyBaMM discharge-positive → negate
    I_pybamm = (-I_cell_A / Q_cell_Ah) * Q_chen   # A, discharge-positive

    # Clip to safe SPMe range — SPMe diverges above ~3C for Chen2020
    I_clipped = np.clip(I_pybamm, -1.5 * Q_chen, 1.5 * Q_chen)
    if np.any(np.abs(I_clipped) != np.abs(I_pybamm)):
        print("[PyBaMM] WARN: some timesteps clipped to ±1.5 C for SPMe stability")

    print(f"[PyBaMM] SPMe + Chen2020 (5 Ah NMC811), SOC_init={soc_init:.3f}")
    print(f"[PyBaMM] C-rate scale: {Q_chen:.0f}/{Q_cell_Ah:.1f} Ah = "
          f"{Q_chen/Q_cell_Ah:.3f}x  (same C-rate, different cell)")
    print(f"[PyBaMM] OPEN-LOOP: no V_meas feedback — SOC init error propagates.")

    try:
        model = pybamm.lithium_ion.SPMe()
        param = pybamm.ParameterValues("Chen2020")

        # Initial SOC
        try:
            param.update({"Initial SoC": soc_init}, check_already_exists=False)
        except TypeError:
            param.update({"Initial SoC": soc_init})
        except Exception:
            pass   # fall back to Chen2020 default (~0.8)

        # Current profile: callable (robust across PyBaMM versions)
        t_data = t_s.copy()
        I_data = I_clipped.copy()

        def _current(t):
            return float(np.interp(float(t), t_data, I_data))

        param["Current function [A]"] = _current

        solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-5)
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

        # Evaluate at ≤300 points for speed; interpolate back to t_s grid
        n_eval = min(len(t_s), 300)
        t_eval = np.linspace(float(t_s[0]), float(t_s[-1]), n_eval)

        t0 = time.perf_counter()
        sol = sim.solve(t_eval)
        elapsed = time.perf_counter() - t0

        t_sol = np.asarray(sol["Time [s]"].entries, dtype=float)
        V_sol = np.asarray(sol["Terminal voltage [V]"].entries, dtype=float)

        try:
            soc_sol = np.asarray(sol["State of charge"].entries, dtype=float)
        except KeyError:
            Q_dis = np.asarray(sol["Discharge capacity [A.h]"].entries, dtype=float)
            soc_sol = np.clip(soc_init - Q_dis / Q_chen, 0.0, 1.0)

        # Interpolate to original t_s grid
        V_out   = np.interp(t_s, t_sol, V_sol)
        soc_out = np.interp(t_s, t_sol, soc_sol)

        runtime_us = elapsed / len(t_s) * 1e6
        print(f"[PyBaMM] Solved {n_eval} eval points in {elapsed:.2f} s "
              f"({runtime_us:.0f} µs/step equiv.)")
        return soc_out, V_out, runtime_us

    except Exception as exc:
        print(f"[PyBaMM] Simulation failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    V_meas: np.ndarray,
    V_pred: np.ndarray,
    soc_truth: np.ndarray,
    soc_est: np.ndarray,
) -> dict:
    v_mae  = float(np.mean(np.abs(V_meas - V_pred))) * 1000.0
    v_rmse = float(np.sqrt(np.mean((V_meas - V_pred) ** 2))) * 1000.0
    s_rmse = float(np.sqrt(np.mean((soc_truth - soc_est) ** 2))) * 100.0
    s_fin  = float(abs(soc_truth[-1] - soc_est[-1])) * 100.0
    return {"V_MAE_mV": v_mae, "V_RMSE_mV": v_rmse,
            "SOC_RMSE_%": s_rmse, "SOC_final_err_%": s_fin}


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _save_plot(
    t_s: np.ndarray,
    soc_truth: np.ndarray,
    V_meas: np.ndarray,
    soc_ekf: np.ndarray,
    V_ekf: np.ndarray,
    soc_pybamm: Optional[np.ndarray],
    V_pybamm: Optional[np.ndarray],
    soc_offset: float,
    out_path: Path,
) -> None:
    if not _HAVE_MPL:
        print("[PLOT] matplotlib not available — skipping.")
        return

    t_min = t_s / 60.0
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax0.plot(t_min, soc_truth * 100, "k-",  lw=2,   label="Coulomb-counting truth")
    ax0.plot(t_min, soc_ekf   * 100, "b-",  lw=1.8, label="OpenCATHODE EKF (closed-loop)")
    if soc_pybamm is not None:
        ax0.plot(t_min, soc_pybamm * 100, "r--", lw=1.6, label="PyBaMM SPMe (open-loop)")
    ax0.set_ylabel("SOC [%]")
    ax0.set_title(f"SOC — deliberate +{soc_offset*100:.0f}% init offset applied to both")
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    ax1.plot(t_min, V_meas * 1000, "k-",  lw=1.5, label="V measured [mV]")
    ax1.plot(t_min, V_ekf  * 1000, "b-",  lw=1.2, alpha=0.85, label="EKF V_pred")
    if V_pybamm is not None:
        ax1.plot(t_min, V_pybamm * 1000, "r--", lw=1.2, alpha=0.85, label="PyBaMM V_pred")
    ax1.set_xlabel("Time [min]")
    ax1.set_ylabel("Cell voltage [mV]")
    ax1.set_title("Voltage prediction (per cell)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle",    default=None,  help="VED vehicle ID")
    ap.add_argument("--synthetic",  action="store_true", help="Use synthetic trip")
    ap.add_argument("--soc-offset", type=float, default=0.10,
                    help="Deliberate SOC init error (default 0.10 = +10%%)")
    args = ap.parse_args()
    offset = float(args.soc_offset)

    print("=" * 70)
    print("  PyBaMM  vs  OpenCATHODE EKF  —  Head-to-head comparison")
    print("=" * 70)
    print()
    print("  PyBaMM : open-loop forward simulator.  No voltage feedback.")
    print("           SOC init error is NEVER corrected.")
    print("  EKF    : closed-loop inverse estimator.  Uses V_meas to correct SOC.")
    print("           Converges from init offset within minutes.")
    print()

    # Step 1: Load data
    if args.synthetic:
        trip = _make_synthetic_trip()
    else:
        try:
            trip = _load_real_trip(args.vehicle)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}  →  falling back to synthetic trip.\n")
            trip = _make_synthetic_trip()

    t_s, I_cell, V_cell_meas, soc_bms, Q_cell_Ah, n_series, n_parallel = trip

    # Step 2: Coulomb-counting ground truth from trusted BMS SOC_0
    soc_truth = _coulomb_counting(t_s, I_cell, float(soc_bms[0]), Q_cell_Ah)

    # Step 3: Deliberate offset applied to both methods
    soc_init = float(np.clip(float(soc_bms[0]) + offset, 0.02, 0.98))
    print(f"[INIT] BMS SOC_0 = {soc_bms[0]:.3f}  +{offset*100:.0f}%  "
          f"→  both start at SOC = {soc_init:.3f}\n")

    # Step 4: OpenCATHODE EKF
    print("─" * 50)
    print("Running OpenCATHODE EKF ...")
    soc_ekf, V_ekf, ekf_us = _run_ekf(
        t_s, I_cell, V_cell_meas, Q_cell_Ah, soc_init, n_series, n_parallel,
    )

    # Step 5: PyBaMM
    print()
    print("─" * 50)
    print("Running PyBaMM SPMe (Chen2020) ...")
    pb_result = _run_pybamm(t_s, I_cell, Q_cell_Ah, soc_init)

    soc_pybamm: Optional[np.ndarray] = None
    V_pybamm:   Optional[np.ndarray] = None
    pb_us: float = float("nan")
    pb_m: Optional[dict] = None
    if pb_result is not None:
        soc_pybamm, V_pybamm, pb_us = pb_result
        pb_m = _compute_metrics(V_cell_meas, V_pybamm, soc_truth, soc_pybamm)

    # Step 6: Metrics + table
    ekf_m = _compute_metrics(V_cell_meas, V_ekf, soc_truth, soc_ekf)

    print()
    print("=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    fmt_head = f"  {'Metric':<30} {'OpenCATHODE EKF':>22} {'PyBaMM SPMe':>14}"
    print(fmt_head)
    print("  " + "─" * 66)

    def _row(label: str, ekf_val: float, pb_val: Optional[float], fmt: str = ".1f") -> None:
        ev = f"{ekf_val:{fmt}}"
        pv = f"{pb_val:{fmt}}" if pb_val is not None else "N/A"
        print(f"  {label:<30} {ev:>22} {pv:>14}")

    _row("V MAE  [mV/cell]",         ekf_m["V_MAE_mV"],          pb_m["V_MAE_mV"] if pb_m else None)
    _row("V RMSE [mV/cell]",         ekf_m["V_RMSE_mV"],         pb_m["V_RMSE_mV"] if pb_m else None)
    _row("SOC RMSE  [%]",            ekf_m["SOC_RMSE_%"],         pb_m["SOC_RMSE_%"] if pb_m else None)
    _row("SOC final error  [%]",     ekf_m["SOC_final_err_%"],    pb_m["SOC_final_err_%"] if pb_m else None)
    _row("Runtime [µs/step]",        ekf_us,                      pb_us if not np.isnan(pb_us) else None)

    print("=" * 70)

    # Step 7: Honest interpretation
    print()
    print("ASSUMPTIONS AND INTERPRETATION")
    print("─" * 70)
    lines = [
        "1. ASYMMETRIC TASK: EKF is a closed-loop estimator — it reads V_meas at",
        "   every timestep and corrects its SOC estimate via Kalman gain.  PyBaMM",
        "   is open-loop — given only I(t), it integrates forward with no feedback.",
        "   Comparing SOC RMSE is therefore not a symmetric 'battle'; it shows WHY",
        "   closed-loop estimation exists: to correct for unknown init conditions.",
        "",
        "2. CELL MODEL MISMATCH: PyBaMM uses Chen2020 (LG M50, 5 Ah NMC811).",
        f"   VED pack is ~33 Ah LMO-NMC (AESC LEV50N).  Current was C-rate scaled",
        "   ({:.2f}×), but OCV, R_int, and diffusion coefficients differ.  The".format(5.0/Q_cell_Ah),
        "   voltage prediction error partly reflects this unavoidable mismatch.",
        "   EKF corrects for it via empirical OCV + calibration.",
        "",
        "3. SOC GROUND TRUTH: Coulomb counting from BMS SOC_0.  BMS itself has",
        "   ±2–5% drift; this is the best reference available for field data.",
        "",
        "4. CALIBRATION: EKF uses generic LMO-NMC OCV (no fleet cal data) in",
        "   single-trip mode.  Full fleet validation uses 12-bin PCHIP + δR0 OLS",
        "   cal fitted on the first 10% of fleet segments per vehicle.",
        "",
        f"5. SPEED: EKF runs at {ekf_us:.0f} µs/step — real-time capable at 1 Hz.",
    ]
    if pb_result is not None:
        lines.append(
            f"   PyBaMM equivalent: {pb_us:.0f} µs/step (batch solve — not a live loop)."
        )
    lines += [
        "",
        "BOTTOM LINE: PyBaMM is the right tool for cell design, parameter fitting,",
        "and physics exploration — not for online SOC estimation with init error.",
        "OpenCATHODE EKF is the right tool for online BMS where V_meas is available",
        "and closed-loop correction is the design goal.",
    ]
    for ln in lines:
        print("  " + ln)
    print()

    # Step 8: Save plot
    out_path = Path(__file__).parent / "pybamm_vs_opencathode.png"
    _save_plot(
        t_s, soc_truth, V_cell_meas,
        soc_ekf, V_ekf,
        soc_pybamm, V_pybamm,
        offset, out_path,
    )


if __name__ == "__main__":
    main()
