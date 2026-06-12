#!/usr/bin/env python3
"""
scripts/dashboard_fleet.py — OpenCATHODE real-fleet validation dashboard.

Generates three plots:
  1. Mode A: predicted vs measured voltage (1 example segment per dataset)
  2. Mode B: EKF SOC convergence (BMW i3 example)
  3. Deng: per-vehicle capacity-fade trajectories with linear fit

Usage:
    python scripts/dashboard_fleet.py
    python scripts/dashboard_fleet.py --outdir docs/fleet_plots/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_C = {
    "ved":    "#2196F3",   # blue
    "bmw":    "#4CAF50",   # green
    "deng":   "#FF9800",   # orange
    "bms":    "#9E9E9E",   # grey
    "ekf":    "#E91E63",   # pink
    "pred":   "#F44336",   # red
    "meas":   "#212121",   # near-black
}


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Mode A voltage overlay (1 segment per dataset)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_mode_a_overlay(ax, seg_df, cfg, dataset_label: str, color: str) -> None:
    from data.validate_generic import run_mode_a_forced
    from data.loaders.common_schema import resample_to_uniform_dt

    if len(seg_df) > 10:
        seg_df = resample_to_uniform_dt(seg_df, 20.0)
    V_meas, V_pred = run_mode_a_forced(seg_df, cfg)
    t = seg_df["t_s"].values / 60.0  # minutes

    mae = float(np.mean(np.abs(V_meas - V_pred))) * 1000
    r2 = float(1 - np.sum((V_meas - V_pred)**2) / (np.sum((V_meas - np.mean(V_meas))**2) + 1e-12))
    ax.plot(t, V_meas * 1000, color=_C["meas"], lw=1.2, label="Measured", alpha=0.9)
    ax.plot(t, V_pred * 1000, color=color, lw=1.0, ls="--", label=f"DFN predicted (MAE={mae:.0f}mV)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Cell voltage (mV)")
    ax.set_title(f"{dataset_label} — Mode A (R²={r2:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Mode B EKF SOC convergence
# ─────────────────────────────────────────────────────────────────────────────

def _plot_mode_b_conv(ax, seg_df, cfg, dataset_label: str, ocv_fn=None) -> None:
    from data.validate_generic import run_mode_b_ekf
    from data.loaders.common_schema import resample_to_uniform_dt

    if len(seg_df) > 10:
        seg_df = resample_to_uniform_dt(seg_df, 20.0)
    soc_bms, soc_ekf, V_pred, conv_s = run_mode_b_ekf(seg_df, cfg, ocv_fn=ocv_fn)
    t = seg_df["t_s"].values / 60.0

    ax.plot(t, soc_bms * 100, color=_C["bms"], lw=1.2, label="BMS SOC (reference)")
    ax.plot(t, soc_ekf * 100, color=_C["ekf"], lw=1.0, ls="--",
            label="EKF SOC (+20% init, NMC OCV)")
    if conv_s is not None:
        ax.axvline(conv_s / 60, color="green", ls=":", alpha=0.7,
                   label=f"Convergence: {conv_s:.0f}s")
    rmse = float(np.sqrt(np.mean((soc_ekf - soc_bms)**2))) * 100
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("SOC (%)")
    ax.set_title(f"{dataset_label} — Mode B (chemistry-aware EKF, RMSE={rmse:.1f}%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.02, "vs BMS SOC, not ground truth",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="grey", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Deng capacity fade
# ─────────────────────────────────────────────────────────────────────────────

def _plot_deng_soh(ax, max_vehicles: int = 5) -> None:
    from data.loaders.deng_charging_loader import DengChargingLoader

    loader = DengChargingLoader(max_vehicles=max_vehicles)
    trajs = loader.soh_trajectories()

    for vid, traj in sorted(trajs.items()):
        if len(traj.sessions) < 3:
            continue
        traj.sessions.sort(key=lambda x: x[0])
        t0 = traj.sessions[0][0]
        t_mo = np.array([(s[0] - t0) / 30.44 for s in traj.sessions])
        c = np.array([s[1] for s in traj.sessions])
        valid = np.isfinite(c)
        if valid.sum() < 3:
            continue
        label = vid + ("*" if traj.q_nominal_flagged else "")
        ax.scatter(t_mo[valid], c[valid] * 100, s=6, alpha=0.5)
        if traj.rul_alpha is not None:
            t_fit = np.linspace(0, t_mo[valid].max(), 50)
            c_fit = (traj.sessions[0][1] if traj.sessions else 1.0) - traj.rul_alpha * t_fit
            ax.plot(t_fit, c_fit * 100, lw=1.0, alpha=0.7, label=label)

    ax.axhline(80, color="red", ls="--", lw=0.8, label="EOL (80%)")
    ax.set_xlabel("Months since first session")
    ax.set_ylabel("Capacity (% of Q_nominal)")
    ax.set_title(f"Deng BAIC EU500 — Capacity fade ({max_vehicles} vehicles)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(60, 110)
    if any(t.q_nominal_flagged for t in trajs.values()):
        ax.text(0.01, 0.02, "* Q_nominal deviates >15% from spec (partial first charges)",
                transform=ax.transAxes, fontsize=7, color="grey")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(outdir: Path = _ROOT / "docs") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Generating fleet dashboard → {outdir}/")

    from data.validate_generic import config_from_cartridge, CellMode

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.38)
    ax_a_bmw  = fig.add_subplot(gs[0, 0])
    ax_a_ved  = fig.add_subplot(gs[0, 1])
    ax_a_deng = fig.add_subplot(gs[0, 2])
    ax_b_bmw  = fig.add_subplot(gs[1, 0])
    ax_soh    = fig.add_subplot(gs[1, 1:])

    # ── Mode A: BMW i3 ───────────────────────────────────────────────────────
    try:
        from data.loaders.bmw_i3_loader import BMWI3Loader
        from data.loaders.pack_cartridge import BMW_I3_60AH
        loader = BMWI3Loader(max_trips=1)
        segs, metas = loader.load_all()
        if segs:
            cfg = config_from_cartridge("BMW_i3", BMW_I3_60AH, CellMode.AVG_CELL)
            _plot_mode_a_overlay(ax_a_bmw, segs[0], cfg, "BMW i3 (NMC, 96S1P)", _C["bmw"])
    except Exception as e:
        ax_a_bmw.set_title(f"BMW i3 — error: {e}")

    # ── Mode A: VED ──────────────────────────────────────────────────────────
    try:
        from data.loaders.ved_loader import VEDLoader
        from data.loaders.pack_cartridge import lookup_ved_cartridge
        loader = VEDLoader(max_veh=1, max_trips_per_veh=1)
        segs, metas = loader.load_all()
        if segs:
            cart = lookup_ved_cartridge(
                next((n.replace("vehicle=", "") for n in metas[0].notes if n.startswith("vehicle=")), "")
            )
            cfg = config_from_cartridge("VED", cart, CellMode.AVG_CELL)
            _plot_mode_a_overlay(ax_a_ved, segs[0], cfg, "VED (NMC, avg-cell)", _C["ved"])
    except Exception as e:
        ax_a_ved.set_title(f"VED — error: {e}")

    # ── Mode A: Deng ─────────────────────────────────────────────────────────
    try:
        from data.loaders.deng_charging_loader import DengChargingLoader
        from data.loaders.pack_cartridge import BAIC_EU500_90S
        loader = DengChargingLoader(max_vehicles=1, max_sessions_per_vehicle=1)
        segs, metas = loader.load_all()
        if segs:
            cfg = config_from_cartridge("Deng_Charging", BAIC_EU500_90S, CellMode.AVG_CELL)
            _plot_mode_a_overlay(ax_a_deng, segs[0], cfg, "Deng BAIC EU500 (NMC, 90S1P)", _C["deng"])
    except Exception as e:
        ax_a_deng.set_title(f"Deng — error: {e}")

    # ── Mode B: BMW i3 with chemistry-aware NMC OCV ──────────────────────────
    try:
        from data.loaders.bmw_i3_loader import BMWI3Loader
        from data.loaders.pack_cartridge import BMW_I3_60AH
        from data.validate_generic import _split_by_vehicle, _build_calibration_for_fleet
        loader = BMWI3Loader(max_trips=10)
        all_bmw = list(loader.iter_segments())
        cfg = config_from_cartridge("BMW_i3", BMW_I3_60AH, CellMode.AVG_CELL)
        cal_p, eval_p = _split_by_vehicle(all_bmw)
        cal = _build_calibration_for_fleet(cal_p, cfg, "BMW_i3") if cal_p else None
        ocv_fn = cal.ocv_fn if cal else None
        if eval_p:
            _plot_mode_b_conv(ax_b_bmw, eval_p[0][0], cfg,
                              "BMW i3 — NMC EKF (empirical OCV)", ocv_fn=ocv_fn)
    except Exception as e:
        ax_b_bmw.set_title(f"BMW i3 Mode B — error: {e}")

    # ── Deng SOH ─────────────────────────────────────────────────────────────
    try:
        _plot_deng_soh(ax_soh, max_vehicles=20)
    except Exception as e:
        ax_soh.set_title(f"Deng SOH — error: {e}")

    fig.suptitle("OpenCATHODE — Real Fleet Validation Dashboard\n"
                 "Mode A: forced BMS SOC  |  Mode B: EKF +20% SOC offset (vs BMS, not ground truth)",
                 fontsize=11)

    outpath = outdir / "fleet_validation_dashboard.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(_ROOT / "docs"))
    args = parser.parse_args()
    main(Path(args.outdir))
