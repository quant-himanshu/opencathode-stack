#!/usr/bin/env python3
"""
analysis/ved_hypothesis_test.py — VED failure-mode hypothesis tests
(2026-07-19 review; read-only diagnostics + one DIAGNOSTIC calibration
variant; the estimator and every headline table stay untouched).

Tests, per VED vehicle (VehId 0010 / 0455 / 0541):

  H-TOPO  Pack-topology sanity: the cartridge the benchmark actually used
          (lookup_ved_cartridge on the static-file VehicleName), the
          loader's own inferred n_series (from meta notes, when present),
          and the implied per-cell voltage window V_pack/n_series across
          all trips, compared against plausible chemistry windows
          (NMC/NCA ≈ 3.0–4.2 V/cell, LFP ≈ 2.5–3.65 V/cell).
          Also: median per-trip drift of anchored coulomb SOC vs BMS SOC
          (a direct probe of Q/topology error).

  H-OCV   OCV-table residual test on near-rest samples (|I_cell| < 2 A):
          V_cell_meas − table(SOC_coulomb) for every chemistry table in
          the repo — generic NMC, generic LMO-NMC (diagnosis/nmc_ocv.py),
          Prada-2012 LFP (diagnosis/dual_ekf_lfp.py), NMC811 at zero
          current (core DFN model), plus the fleet-fitted empirical OCV
          the benchmark actually used. Reports mean bias, the SOC-binned
          residual shape, and the structured-residual RMS (RMS of binned
          medians about their own mean — insensitive to a pure constant
          offset). SOC_coulomb = SOC_bms[0] − ∫I dt / (3600·Q_pack).

  H-CAL   Per-vehicle calibration probe (DIAGNOSTIC VARIANT ONLY): δV(SOC)
          + δR0 + OCV + γ fitted on each vehicle's own first-10% split
          (identical split discipline and identical held-out trips as the
          fleet-level benchmark), then the +20 pp protocol re-run with the
          per-vehicle calibration. Reports per-vehicle δR0 (vs physical
          range), outcome tiers, and median RMSE, side by side with the
          fleet-level rows from the cross-checked sweep dump.

Outputs: results/ved_hypothesis_<stamp>.json (all numbers) + evidence
tables appended to docs/VED_BREAKDOWN.md. Verdict prose is written
separately from the JSON — numbers only from this run.

One command: venv/bin/python -u analysis/ved_hypothesis_test.py
"""
from __future__ import annotations

import glob
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

from data.loaders.pack_cartridge import lookup_ved_cartridge
from data.loaders.ved_loader import VEDLoader
from data.validate_generic import (
    CellMode, _build_calibration_for_fleet, _select_chemistry, _make_cell,
    _set_state, config_from_cartridge,
)
from diagnosis.dual_ekf_lfp import DualEKF_LFP
from diagnosis.nmc_ocv import _LMONMC_OCV, _LMONMC_SOC, _NMC_OCV, _NMC_SOC
from validation.metrics import trip_metrics

RESULTS_DIR = ROOT / "results"
DOCS = ROOT / "docs"
I_REST_A = 2.0
N_BINS = 10
HEADLINE_OFFSET = 0.20
CHEM_WINDOWS = {"NMC/NCA": (3.0, 4.2), "LFP": (2.5, 3.65)}


def _get_cfg(meta):
    """Identical to the benchmark's per-segment config path."""
    name = next((n.replace("vehicle=", "") for n in meta.notes
                 if n.startswith("vehicle=")), "")
    cart = lookup_ved_cartridge(name)
    cfg = config_from_cartridge(
        "VED", cart, CellMode.AVG_CELL, dt_resample_s=20.0,
        min_duration_s=120.0, dt_short_s=5.0, dt_short_threshold_s=600.0)
    return cfg, cart, name


def _dfn_nmc811_ocv_table(n_pts: int = 41) -> Tuple[np.ndarray, np.ndarray]:
    chem = _select_chemistry("NMC")
    socs = np.linspace(0.02, 0.98, n_pts)
    vs = []
    for s in socs:
        cell = _make_cell(chem, float(s))
        _set_state(cell, float(s))
        vs.append(float(cell.step(0.0, 1.0)["V"]))
    return socs, np.asarray(vs)


def _soc_coulomb(seg_df: pd.DataFrame, cfg) -> np.ndarray:
    """Anchored coulomb SOC, DISCHARGE-NEGATIVE schema (sign fix 2026-07-20:
    the first version used the discharge-positive formula, which inverted
    the SOC axis of the H-OCV test and produced the '+40 pp/h drift'
    artifact that led to the sign-bug discovery)."""
    t = seg_df["t_s"].values.astype(np.float64)
    I_cell = seg_df["I_A"].values.astype(np.float64) / cfg.n_parallel
    dt = np.diff(t, prepend=t[0]); dt[0] = 0.0
    ah = np.cumsum(I_cell * dt) / 3600.0
    return float(seg_df["SOC_bms"].iloc[0]) + ah / cfg.q_cell_ah


def structured_rms(soc: np.ndarray, resid: np.ndarray) -> Tuple[Optional[float], List]:
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    idx = np.clip(np.digitize(soc, edges) - 1, 0, N_BINS - 1)
    meds, centers = [], []
    for b in range(N_BINS):
        m = idx == b
        if m.sum() >= 10:
            meds.append(float(np.median(resid[m])))
            centers.append(float(0.5 * (edges[b] + edges[b + 1])))
    if len(meds) < 3:
        return None, []
    meds_a = np.asarray(meds)
    return (float(np.sqrt(np.mean((meds_a - meds_a.mean()) ** 2))),
            [{"soc_bin": c, "median_resid_mV": m * 1000.0}
             for c, m in zip(centers, meds)])


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("Loading VED (all vehicles, all trips)…")
    all_pairs = list(VEDLoader(max_veh=None, max_trips_per_veh=None).iter_segments())
    valid = [(s, m) for s, m in all_pairs
             if (float(s["t_s"].iloc[-1]) - float(s["t_s"].iloc[0])) >= 120.0]
    print(f"{len(all_pairs)} segments loaded, {len(valid)} ≥120 s")

    by_veh: Dict[str, List] = {}
    for s, m in valid:
        by_veh.setdefault(m.vehicle_id, []).append((s, m))

    report: Dict = {"meta": {"stamp": stamp, "n_valid_segments": len(valid)}}

    # ── H-TOPO ──────────────────────────────────────────────────────────────
    print("\nH-TOPO: pack-topology sanity check")
    topo = {}
    for vid, pairs in sorted(by_veh.items()):
        cfg, cart, name = _get_cfg(pairs[0][1])
        V_all = np.concatenate([s["V_V"].values for s, _ in pairs]).astype(float)
        v1, v99 = np.percentile(V_all, [1, 99])
        inferred = None
        for _, m in pairs[:3]:
            for n in m.notes:
                if "n_series" in n and "inferred" in n:
                    inferred = n
                    break
        window = {"V_pack_min": float(V_all.min()), "V_pack_max": float(V_all.max()),
                  "V_pack_p1": float(v1), "V_pack_p99": float(v99)}
        per_cell = {k: v / cart.n_series for k, v in window.items()}
        consistency = {
            chem: bool(per_cell["V_pack_p1"] >= lo - 0.05
                       and per_cell["V_pack_p99"] <= hi + 0.05)
            for chem, (lo, hi) in CHEM_WINDOWS.items()}
        # coulomb-vs-BMS SOC drift per trip (Q/topology probe)
        drifts = []
        for s, m in pairs:
            sc = _soc_coulomb(s, cfg)
            sb = s["SOC_bms"].values.astype(float)
            dur_h = (float(s["t_s"].iloc[-1]) - float(s["t_s"].iloc[0])) / 3600.0
            if dur_h > 0.05:
                drifts.append((float(sc[-1] - sb[-1]) * 100.0, dur_h))
        drift_pp = [d for d, h in drifts]
        drift_pp_per_h = [d / h for d, h in drifts]
        topo[vid] = {
            "vehicle_name_static": name,
            "cartridge_used_by_benchmark": cart.name,
            "n_series": cart.n_series, "n_parallel": cart.n_parallel,
            "chemistry_assumed": cart.chemistry, "Q_cell_Ah": cart.Q_cell_Ah,
            "topology_uncertain_flag": cart.topology_uncertain,
            "loader_inference_note": inferred,
            "pack_window_V": window,
            "per_cell_window_V": per_cell,
            "window_consistent_with": consistency,
            "n_trips": len(pairs),
            "coulomb_vs_bms_drift_end_pp_median": float(np.median(drift_pp)) if drift_pp else None,
            "coulomb_vs_bms_drift_pp_per_hour_median": float(np.median(drift_pp_per_h)) if drift_pp_per_h else None,
        }
        print(f"  {vid} ({name!r}): cart={cart.name} {cart.n_series}s{cart.n_parallel}p "
              f"{cart.chemistry} Q={cart.Q_cell_Ah}Ah | per-cell p1–p99 "
              f"[{per_cell['V_pack_p1']:.3f},{per_cell['V_pack_p99']:.3f}]V | "
              f"consistent: {consistency} | drift {topo[vid]['coulomb_vs_bms_drift_pp_per_hour_median']:.2f} pp/h")
    report["H_TOPO"] = topo

    # ── H-OCV ───────────────────────────────────────────────────────────────
    print("\nH-OCV: OCV-table residual test (near-rest, coulomb-derived SOC)")
    ekf_tbl = DualEKF_LFP()
    dfn_soc, dfn_v = _dfn_nmc811_ocv_table()
    tables = {
        "generic_NMC": lambda s: float(np.interp(np.clip(s, 0, 1), _NMC_SOC, _NMC_OCV)),
        "generic_LMO-NMC": lambda s: float(np.interp(np.clip(s, 0, 1), _LMONMC_SOC, _LMONMC_OCV)),
        "LFP_Prada2012": lambda s: float(np.interp(np.clip(s, 0, 1),
                                                   ekf_tbl._soc_pts, ekf_tbl._ocv_pts)),
        "NMC811_DFN_rest": lambda s: float(np.interp(np.clip(s, 0, 1), dfn_soc, dfn_v)),
    }
    # the fleet-fitted OCV the benchmark actually used (same split)
    from data.validate_generic import _split_by_vehicle
    cal_pairs, eval_pairs = _split_by_vehicle(valid)
    fleet_cfg = _get_cfg(valid[0][1])[0]
    fleet_cal = _build_calibration_for_fleet(cal_pairs, fleet_cfg, "VED")
    tables["fleet_fitted_empirical"] = lambda s: float(fleet_cal.ocv_fn(s))

    hocv = {}
    for vid, pairs in sorted(by_veh.items()):
        cfg, cart, _ = _get_cfg(pairs[0][1])
        soc_l, v_l = [], []
        for s, m in pairs:
            I_cell = s["I_A"].values.astype(float) / cfg.n_parallel
            rest = np.abs(I_cell) < I_REST_A
            if rest.sum() == 0:
                continue
            soc_cc = np.clip(_soc_coulomb(s, cfg), 0.0, 1.0)
            v_cell = s["V_V"].values.astype(float) / cfg.n_series
            soc_l.append(soc_cc[rest]); v_l.append(v_cell[rest])
        soc = np.concatenate(soc_l); vc = np.concatenate(v_l)
        hocv[vid] = {"n_rest_samples": int(len(soc)), "tables": {}}
        best = None
        for name, fn in tables.items():
            pred = np.array([fn(x) for x in soc])
            resid = vc - pred
            srms, shape = structured_rms(soc, resid)
            entry = {"bias_mV_mean": float(resid.mean() * 1000.0),
                     "bias_mV_median": float(np.median(resid) * 1000.0),
                     "structured_rms_mV": (srms * 1000.0 if srms is not None else None),
                     "shape_binned": shape}
            hocv[vid]["tables"][name] = entry
            if srms is not None and (best is None or srms * 1000.0 < best[1]):
                best = (name, srms * 1000.0)
        hocv[vid]["best_table_by_structured_rms"] = best[0] if best else None
        print(f"  {vid}: best table = {best[0] if best else '?'} "
              f"({best[1]:.1f} mV structured RMS)" if best else f"  {vid}: insufficient")
        for name in tables:
            e = hocv[vid]["tables"][name]
            print(f"     {name:24s} bias {e['bias_mV_mean']:+7.1f} mV  "
                  f"structRMS {e['structured_rms_mV'] if e['structured_rms_mV'] is None else round(e['structured_rms_mV'],1)} mV")
    report["H_OCV"] = hocv

    # ── H-CAL ───────────────────────────────────────────────────────────────
    print("\nH-CAL: per-vehicle calibration probe (DIAGNOSTIC variant)")
    # fleet-level per-vehicle baseline from the cross-checked sweep dump
    dump = sorted(glob.glob(str(RESULTS_DIR / "offset_sweep_per_trip_2*.csv")))[-1]
    dd = pd.read_csv(dump)
    fleet_rows = dd[(dd.dataset == "VED") & (dd.method == "my_ekf")
                    & (dd.offset_pp == 20)]

    def _tiers(df_or_list) -> Dict:
        if isinstance(df_or_list, pd.DataFrame):
            conv = df_or_list["t_conv_strict_s"].notna()
            near = df_or_list["min_abs_err_pct"] < 5.0
            rmse = df_or_list["rmse_full_pct"]
            n = len(df_or_list)
        else:
            conv = np.array([t["t_conv_strict_s"] is not None for t in df_or_list])
            near = np.array([t["min_abs_err_pct"] < 5.0 for t in df_or_list])
            rmse = np.array([t["rmse_full_pct"] for t in df_or_list])
            n = len(df_or_list)
        return {"n": int(n),
                "converged": int(conv.sum()),
                "rediverging": int((~conv & near).sum()),
                "never_approach": int((~conv & ~near).sum()),
                "median_rmse_pp": float(np.median(rmse))}

    hcal = {}
    for vid, pairs in sorted(by_veh.items()):
        cfg, cart, _ = _get_cfg(pairs[0][1])
        n_cal = max(1, int(len(pairs) * 0.10))
        veh_cal_pairs, veh_eval_pairs = pairs[:n_cal], pairs[n_cal:]
        cal_v = _build_calibration_for_fleet(veh_cal_pairs, cfg, "VED")
        trips = []
        for s, m in veh_eval_pairs:
            try:
                ekf = DualEKF_LFP(
                    Q_nom_Ah=cfg.q_cell_ah, R_int_ohm=cfg.r_ohm_cell,
                    ocv_fn=cal_v.ocv_fn, R_meas_V2=cal_v.ekf_R_meas_V2,
                    P0_soc=HEADLINE_OFFSET ** 2, gamma=cal_v.ekf_gamma,
                    cal_soc_fn=cal_v.soc_cal_fn(), cal_dR0=cal_v.delta_R0)
                t_s = s["t_s"].values.astype(np.float64)
                I_cell = s["I_A"].values.astype(np.float64) / cfg.n_parallel
                V_cell = s["V_V"].values.astype(np.float64) / cfg.n_series
                soc_bms = s["SOC_bms"].values.astype(np.float64)
                T_arr = s["T_degC"].values.astype(np.float64)
                ekf.set_soc(float(np.clip(soc_bms[0] + HEADLINE_OFFSET, 0.02, 0.98)))
                est = np.empty(len(t_s))
                for i in range(len(t_s)):
                    dt = float(t_s[i] - t_s[i - 1]) if i > 0 else 1.0
                    T = float(T_arr[i]) if np.isfinite(T_arr[i]) else 25.0
                    try:
                        est[i] = float(ekf.update(float(V_cell[i]),
                                                  -float(I_cell[i]), dt, T)["soc"])
                    except Exception:
                        est[i] = float(ekf.x1[0])
                trips.append(trip_metrics(t_s, est, soc_bms))
            except Exception as exc:
                print(f"    {vid} segment failed: {exc}")
        fl = fleet_rows[fleet_rows.vehicle_id == vid]
        hcal[vid] = {
            "n_cal_segments": n_cal,
            "delta_R0_mOhm": cal_v.delta_R0 * 1000.0,
            "delta_R0_physical": bool(abs(cal_v.delta_R0) < 0.05),
            "delta_V_OLS_mV": cal_v.delta_V * 1000.0,
            "gamma": cal_v.ekf_gamma,
            "ocv_source": cal_v.ocv_source[:100],
            "fleet_level": _tiers(fl),
            "per_vehicle_cal": _tiers(trips),
        }
        print(f"  {vid}: δR0={hcal[vid]['delta_R0_mOhm']:+.2f} mΩ "
              f"(physical={hcal[vid]['delta_R0_physical']}), γ={cal_v.ekf_gamma} | "
              f"fleet {hcal[vid]['fleet_level']} → per-veh {hcal[vid]['per_vehicle_cal']}")
    report["H_CAL"] = hcal

    out = RESULTS_DIR / f"ved_hypothesis_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out}")

    # ── evidence tables → docs/VED_BREAKDOWN.md ────────────────────────────
    lines = ["", f"## Hypothesis-test evidence (2026-07-19, `analysis/ved_hypothesis_test.py`, run {stamp})", "",
             "### H-TOPO — topology / chemistry window", "",
             "| VehId | static name | cartridge used | topology | chem assumed | per-cell p1–p99 (V) | NMC-window OK | LFP-window OK | coulomb-vs-BMS drift (pp/h, median) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for vid, t in sorted(topo.items()):
        pc = t["per_cell_window_V"]
        lines.append(
            f"| {vid} | {t['vehicle_name_static']} | {t['cartridge_used_by_benchmark']} "
            f"| {t['n_series']}s{t['n_parallel']}p, Q={t['Q_cell_Ah']}Ah "
            f"| {t['chemistry_assumed']} | {pc['V_pack_p1']:.3f}–{pc['V_pack_p99']:.3f} "
            f"| {t['window_consistent_with']['NMC/NCA']} "
            f"| {t['window_consistent_with']['LFP']} "
            f"| {t['coulomb_vs_bms_drift_pp_per_hour_median']:+.2f} |")
    lines += ["", "### H-OCV — table residuals (near-rest, coulomb SOC)", "",
              "| VehId | table | bias (mV) | structured RMS (mV) |",
              "|---|---|---|---|"]
    for vid, h in sorted(hocv.items()):
        for name, e in h["tables"].items():
            star = " **← best**" if name == h["best_table_by_structured_rms"] else ""
            srms = e["structured_rms_mV"]
            lines.append(f"| {vid} | {name}{star} | {e['bias_mV_mean']:+.1f} "
                         f"| {srms if srms is None else round(srms, 1)} |")
    lines += ["", "### H-CAL — fleet-level vs per-vehicle calibration (+20 pp, DIAGNOSTIC variant — headline tables stay fleet-level)", "",
              "| VehId | δR0 (mΩ) | physical? | γ | cal | n | conv | re-div | never | median RMSE (pp) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for vid, h in sorted(hcal.items()):
        for label, k in (("fleet", "fleet_level"), ("per-veh", "per_vehicle_cal")):
            d = h[k]
            dr = f"{h['delta_R0_mOhm']:+.2f}" if label == "per-veh" else "(fleet fit)"
            ph = h["delta_R0_physical"] if label == "per-veh" else ""
            gm = h["gamma"] if label == "per-veh" else ""
            lines.append(f"| {vid} | {dr} | {ph} | {gm} | {label} | {d['n']} "
                         f"| {d['converged']} | {d['rediverging']} "
                         f"| {d['never_approach']} | {d['median_rmse_pp']:.1f} |")
    p = DOCS / "VED_BREAKDOWN.md"
    p.write_text(p.read_text() + "\n".join(lines) + "\n")
    print(f"Appended evidence tables to {p}")


if __name__ == "__main__":
    main()
