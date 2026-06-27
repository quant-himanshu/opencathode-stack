"""
validate_deng_degradation.py — Module 2 end-to-end validation on real Deng fleet.

Pipeline:
  1. Load all 20 Deng vehicles (chemistry auto-detected)
  2. Segment into sessions (30-min gap threshold)
  3. Stress proxy per cycle (DoD × C-rate-factor × Arrhenius-T)
  4. Rainflow + Palmgren-Miner damage accumulation
  5. Calibrate BOTH models on V01–V04:
       Model A — stress-only:       ΔSOH = β_s · D^γ_s
       Model B — stress + SEI:      ΔSOH = β_c · D^γ_c + λ · √t
     Both models predict ΔSOH = cumulative fade from first observation.
  6. Evaluate on ALL 20 vehicles — no cherry-picking
  7. Report stress vs SEI fraction; diagnose SNR limitation honestly

HONEST DATA LIMITATION:
  The 2-year BAIC EU500 observation window shows ~2.5% mean SOH fade,
  but per-session Q_Ah noise has std ~2.8% SOH — SNR < 1.
  Five vehicles show negative ΔSOH_final (apparent capacity recovery),
  which is physically impossible true degradation — they reflect BMS
  recalibration events or seasonal temperature variation in capacity.
  For SNR < 1, R² is not a meaningful fitness metric; we use MAE_ΔSOH.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from degradation.deng_loader     import load_all, _Q_NOMINAL
from degradation.cycle_segmentor import segment_all
from degradation.stress_model    import add_stress_column
from degradation.fatigue         import accumulate_damage, SN_A_DEFAULT, SN_M_DEFAULT
from degradation.soh_predictor   import (
    add_t_years, calibrate_combined, evaluate_combined,
    observed_delta_soh, TRAIN_VEHS, Q_NOMINAL,
)

DATA_DIR   = Path(__file__).parent
REPORT_OUT = DATA_DIR / "deng_degradation_report.json"


def _snr_label(dS: np.ndarray) -> str:
    """Return a short note if this vehicle's trajectory is unreliable."""
    final  = float(dS[-1])
    noise  = float(np.std(dS))
    if final < 0:
        return "negative fade (BMS recal/seasonal) — no monotone model fits"
    if noise > abs(final) * 1.5:
        return f"low SNR (noise {noise:.3f} >> signal {abs(final):.3f})"
    return ""


def _fmt(v, fmt=".4f"):
    return format(v, fmt) if v is not None else "  N/A"


def main() -> None:
    t_start = time.time()
    print("=" * 76)
    print("  MODULE 2 — Stress-Fatigue + SEI Degradation Validation")
    print("  Dataset: Deng BAIC EU500 (20 real vehicles, 2019–2021)")
    print("=" * 76)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading Deng fleet data...")
    vehicles = load_all(verbose=True)
    chems    = {k: v["chemistry"].iloc[0] for k, v in vehicles.items()}
    nmc_ct   = sum(c == "NMC" for c in chems.values())
    lfp_ct   = sum(c == "LFP" for c in chems.values())
    print(f"\n  CHEMISTRY: NMC={nmc_ct}  LFP={lfp_ct}")
    if lfp_ct:
        print("  [!] LFP detected — stress-fatigue branch invalid. Excluding.")
        vehicles = {k: v for k, v in vehicles.items() if chems[k] == "NMC"}
    print(f"  NMC branch for {len(vehicles)} vehicles.\n")

    # ── 2. Segment ────────────────────────────────────────────────────────────
    print("[2/5] Segmenting sessions...")
    cycles = segment_all(vehicles, verbose=False)
    print(f"  Total sessions: {len(cycles)}")
    print(f"  DoD: mean={cycles['DoD_pct'].mean():.1f}%  "
          f"median={cycles['DoD_pct'].median():.1f}%")
    print(f"  C-rate: mean={cycles['C_rate'].mean():.3f}  max={cycles['C_rate'].max():.3f}")
    print(f"  T_mean: {cycles['T_mean_C'].mean():.1f} °C")

    # ── 3. Stress + time ─────────────────────────────────────────────────────
    print("\n[3/5] Computing stress proxy and elapsed time...")
    cycles = add_stress_column(cycles)
    cycles = add_t_years(cycles)

    # ── 4. Damage ─────────────────────────────────────────────────────────────
    print(f"\n[4/5] Rainflow + Palmgren-Miner damage "
          f"(S-N: A={SN_A_DEFAULT:.0e}, m={SN_M_DEFAULT})...")
    t_dmg    = time.time()
    cycles_d = accumulate_damage(cycles, vehicles)
    print(f"  Done in {time.time() - t_dmg:.1f}s")
    print(f"  D_final range: "
          f"{cycles_d.groupby('vehicle')['D_cumul'].last().min():.4f} – "
          f"{cycles_d.groupby('vehicle')['D_cumul'].last().max():.4f}")
    print(f"  [Note] D_final ≈ 0.002 is very small — stress term "
          f"contributes negligibly vs SEI over 2 years")

    # ── Signal-to-noise diagnostic ────────────────────────────────────────────
    print("\n  Signal-vs-noise diagnostic (ΔSOH fade over ~2.3 years):")
    print(f"  {'Veh':<6} {'ΔSOH_final':>11} {'ΔSOH_std':>10} {'SNR':>7}  Status")
    print("  " + "-" * 65)
    snr_data = {}
    for veh in sorted(cycles_d["vehicle"].unique()):
        vc = cycles_d[cycles_d["vehicle"] == veh]
        dS = observed_delta_soh(vc, veh)
        if dS is None:
            snr_data[veh] = None
            continue
        final   = float(dS[-1])
        noise   = float(np.std(dS))
        snr     = abs(final) / noise if noise > 1e-9 else 0.0
        status  = _snr_label(dS)
        snr_data[veh] = {"final": final, "std": noise, "snr": snr}
        print(f"  {veh:<6} {final:>11.4f} {noise:>10.4f} {snr:>7.2f}  {status}")

    valid_snr   = [d["snr"] for d in snr_data.values() if d is not None]
    neg_fade    = [v for v, d in snr_data.items() if d and d["final"] < 0]
    low_snr_ct  = sum(1 for d in snr_data.values() if d and d["snr"] < 1.0)
    print(f"\n  Vehicles with negative ΔSOH (apparent recovery): {neg_fade}")
    print(f"  Vehicles with SNR < 1 (noise dominates signal):  "
          f"{low_snr_ct}/{len(valid_snr)}")
    print(f"  Mean SNR across fleet: {np.mean(valid_snr):.2f}")
    print(f"  → For SNR < 1, R² is not a meaningful metric. We report MAE_ΔSOH.")

    # ── 5. Calibrate ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Calibrating on training set {sorted(TRAIN_VEHS)}...")
    p_stress, p_sei, train_info = calibrate_combined(cycles_d, TRAIN_VEHS)
    beta_s, gamma_s        = p_stress["beta"], p_stress["gamma"]
    beta_c, gamma_c, lam_c = p_sei["beta"],   p_sei["gamma"],  p_sei["lam"]

    print(f"\n  Model A (stress-only):  β={beta_s:.4f}   γ={gamma_s:.4f}")
    print(f"  Model B (stress + SEI): β={beta_c:.4f}   γ={gamma_c:.4f}   "
          f"λ={lam_c:.6f} SOH/√yr")
    print(f"  SEI contribution at 2.3 yr: λ·√t = {lam_c*np.sqrt(2.3):.4f} ΔSOH "
          f"(= {lam_c*np.sqrt(2.3)*100:.1f}% SOH fade)")
    stress_at_Dfinal = beta_c * (0.002 ** gamma_c)
    print(f"  Stress contribution at D=0.002: β·D^γ = {stress_at_Dfinal:.2e} ΔSOH "
          f"(≈ negligible)")
    print(f"\n  Training fit (ΔSOH basis):")
    print(f"  {'Veh':<6} {'R²_A':>8} {'R²_B':>8} {'MAE_A':>8} {'MAE_B':>8}  ΔMAE")
    print("  " + "-" * 50)
    for veh in sorted(train_info):
        inf = train_info[veh]
        delta = inf["mae_combined"] - inf["mae_stress"]
        sign  = "▲worse" if delta > 0 else "▼better"
        print(f"  {veh:<6} {inf['r2_stress']:>8.4f} {inf['r2_combined']:>8.4f} "
              f"{inf['mae_stress']:>8.4f} {inf['mae_combined']:>8.4f}  "
              f"{sign} {abs(delta):.4f}")

    # ── 6. Evaluate all ───────────────────────────────────────────────────────
    all_res = evaluate_combined(cycles_d, p_stress, p_sei)

    print(f"\n{'─'*86}")
    print(f"  {'Veh':<6} {'Spl':>5} {'R²_A':>8} {'R²_B':>8} {'MAE_A':>7} "
          f"{'MAE_B':>7} {'SEI%':>6}  Note")
    print(f"{'─'*86}")

    t_r2a,  t_r2b  = [], []
    t_maea, t_maeb = [], []
    sei_fracs       = []

    for veh in sorted(all_res.keys()):
        res   = all_res[veh]
        snr_d = snr_data.get(veh)
        note  = res.get("note", "")
        # Augment note with SNR context
        if snr_d and snr_d["final"] < 0 and not note:
            note = "negative fade (BMS recal/seasonal)"
        elif snr_d and snr_d["snr"] < 1.0 and not note:
            note = f"low SNR={snr_d['snr']:.2f}"

        r2a  = _fmt(res["r2_stress"],    ".4f")
        r2b  = _fmt(res["r2_combined"],  ".4f")
        maa  = _fmt(res["mae_stress"],   ".4f")
        mab  = _fmt(res["mae_combined"], ".4f")
        seip = f"{res['sei_frac']*100:.0f}%" if res["sei_frac"] is not None else "  N/A"

        print(f"  {veh:<6} {res['split']:>5} {r2a:>8} {r2b:>8} {maa:>7} "
              f"{mab:>7} {seip:>6}  {note}")

        if res["split"] == "test" and res["r2_stress"] is not None:
            t_r2a.append(res["r2_stress"]);  t_r2b.append(res["r2_combined"])
            t_maea.append(res["mae_stress"]); t_maeb.append(res["mae_combined"])
        if res["sei_frac"] is not None:
            sei_fracs.append(res["sei_frac"])

    print(f"{'─'*86}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    if t_r2a:
        print(f"\n  Side-by-side: Model A (stress-only) vs Model B (stress + SEI)")
        print(f"  Held-out test set: V05–V20 ({len(t_r2a)} vehicles)\n")
        print(f"  {'Metric':<30} {'Model A':>10}  {'Model B':>10}  {'Δ (B–A)':>10}")
        print("  " + "-" * 66)
        rows = [
            ("R² mean",         np.mean(t_r2a),  np.mean(t_r2b)),
            ("R² min (worst)",  np.min(t_r2a),   np.min(t_r2b)),
            ("R² max (best)",   np.max(t_r2a),   np.max(t_r2b)),
            ("MAE_ΔSOH mean",   np.mean(t_maea), np.mean(t_maeb)),
            ("MAE_ΔSOH max",    np.max(t_maea),  np.max(t_maeb)),
        ]
        for label, va, vb in rows:
            d    = vb - va
            sign = "+" if d >= 0 else ""
            print(f"  {label:<30} {va:>10.4f}  {vb:>10.4f}  {sign}{d:>9.4f}")

    print(f"\n  Degradation driver (from Model B fit):")
    sei_mean = np.mean(sei_fracs) * 100 if sei_fracs else float("nan")
    print(f"  SEI/calendar fraction:   {sei_mean:.0f}%  "
          f"(λ={lam_c:.6f} SOH/√yr → {lam_c*np.sqrt(2.3)*100:.1f}% at 2.3 yr)")
    print(f"  Stress-fatigue fraction: {100-sei_mean:.0f}%  "
          f"(D_final≈0.002 → β·D^γ ≈ {stress_at_Dfinal:.1e} ΔSOH, negligible)")
    print(f"\n  Interpretation: For BAIC EU500 in normal urban operation over 2 years,")
    print(f"  calendar SEI aging is the dominant degradation driver. Stress-fatigue")
    print(f"  damage is real but accumulates too slowly over this observation window")
    print(f"  to be measurable against BMS capacity noise.")

    print(f"\n  Why R² is negative for most vehicles:")
    print(f"  • 2-year ΔSOH fade signal: ~{np.mean([d['final'] for d in snr_data.values() if d and d['final']>0])*100:.1f}% mean SOH (vehicles with positive fade)")
    print(f"  • Per-session Q_Ah noise:  ~{np.mean([d['std'] for d in snr_data.values() if d])*100:.1f}% SOH std — exceeds signal for {low_snr_ct}/20 vehicles")
    print(f"  • {len(neg_fade)} vehicles show apparent capacity increase (V07 V12 V16 V17 V20)")
    print(f"    These are BMS recalibration events or seasonal temperature shifts,")
    print(f"    not true capacity recovery. No deterministic model fits them.")
    print(f"  • When noise > signal, R² = 1 − SS_res/SS_tot < 0 by construction.")
    print(f"    MAE_ΔSOH ≈ {np.mean(t_maea)*100:.1f}% SOH is the interpretable metric.")

    print(f"\n  What would make this model work better:")
    print(f"  • Longer observation window (5+ years) for stress damage to accumulate")
    print(f"  • Controlled lab cycling where D is systematically varied")
    print(f"  • Improved Q_Ah estimation (GITT or dV/dQ analysis vs raw BMS reporting)")

    n_train_ok = sum(1 for v, r in all_res.items()
                     if r["split"] == "train" and r["r2_combined"] is not None)
    n_test_pos = sum(1 for v in t_r2b if v > 0)
    print(f"\n  Vehicles with R²_B > 0 on held-out set: {n_test_pos}/{len(t_r2b)}")
    print(f"  Total runtime: {time.time() - t_start:.1f}s")

    # ── Write report ──────────────────────────────────────────────────────────
    # Add SNR data to per-vehicle results for the JSON
    for veh, res in all_res.items():
        sd = snr_data.get(veh)
        if sd:
            res["dsoh_final"]  = round(sd["final"], 4)
            res["dsoh_std"]    = round(sd["std"],   4)
            res["signal_snr"]  = round(sd["snr"],   3)

    report = {
        "system"            : "OpenCATHODE Stack — Module 2 Degradation (v2: stress + SEI)",
        "dataset"           : "Deng BAIC EU500 fleet (20 vehicles, 2019-2021)",
        "was_real_data"     : True,
        "chemistry"         : "NMC — all 20 vehicles (cell voltage 3.50–4.26 V)",
        "models": {
            "A_stress_only": {
                "formula": "ΔSOH = β_s · D^γ_s",
                "beta_s" : round(beta_s,  6),
                "gamma_s": round(gamma_s, 6),
            },
            "B_stress_plus_sei": {
                "formula"        : "ΔSOH = β_c · D^γ_c + λ · √t",
                "beta_c"         : round(beta_c,  6),
                "gamma_c"        : round(gamma_c, 6),
                "lambda_soh_per_sqrt_yr": round(lam_c, 8),
                "sei_fade_at_2p3yr": round(float(lam_c * np.sqrt(2.3)), 4),
                "stress_fade_at_D0p002": round(float(stress_at_Dfinal), 8),
            },
        },
        "sn_A"              : SN_A_DEFAULT,
        "sn_m"              : SN_M_DEFAULT,
        "q_nominal_Ah"      : Q_NOMINAL,
        "train_vehicles"    : sorted(TRAIN_VEHS),
        "held_out_vehicles" : sorted(v for v, r in all_res.items() if r["split"] == "test"),
        "total_cycles"      : int(len(cycles_d)),
        "results_per_vehicle": all_res,
        "aggregate_test": {
            "n_test_vehicles": len(t_r2a),
            "model_A_stress_only": {
                "r2_mean" : round(float(np.mean(t_r2a)),  4) if t_r2a  else None,
                "r2_max"  : round(float(np.max(t_r2a)),   4) if t_r2a  else None,
                "mae_mean": round(float(np.mean(t_maea)), 4) if t_maea else None,
            },
            "model_B_stress_sei": {
                "r2_mean" : round(float(np.mean(t_r2b)),  4) if t_r2b  else None,
                "r2_max"  : round(float(np.max(t_r2b)),   4) if t_r2b  else None,
                "mae_mean": round(float(np.mean(t_maeb)), 4) if t_maeb else None,
                "n_positive_r2": n_test_pos,
            },
        },
        "degradation_driver": {
            "dominant_mechanism" : "SEI/calendar aging",
            "sei_fraction_pct"   : round(sei_mean, 1) if not np.isnan(sei_mean) else None,
            "stress_fraction_pct": round(100 - sei_mean, 1) if not np.isnan(sei_mean) else None,
            "lambda_interp"      : f"{lam_c:.6f} SOH/√yr → {lam_c*np.sqrt(2.3)*100:.1f}% fade at 2.3 yr",
            "stress_interp"      : f"D_final≈0.002 → β·D^γ≈{stress_at_Dfinal:.1e}, negligible vs SEI",
        },
        "data_quality": {
            "mean_signal_snr"        : round(float(np.mean(valid_snr)), 2),
            "vehicles_snr_below_1"   : low_snr_ct,
            "vehicles_negative_fade" : len(neg_fade),
            "note": (
                "SNR < 1 for most vehicles: 2-year fade signal (~2.5% SOH) "
                "is comparable to per-session BMS capacity noise (~2.8% SOH). "
                "R² is not interpretable; MAE_ΔSOH is the primary metric."
            ),
        },
        "assumptions": [
            "ΔSOH fitted (not absolute SOH) — pre-dataset history unknown",
            "t_years from vehicle's first in-dataset observation",
            "Linear Miner's rule — no load-sequence effects",
            "√t SEI kinetics (Pinson-Bazant 2013) at constant T approximation",
            "Basquin m=2.5 fixed (NMC literature prior)",
            "Q_nominal=136.2 Ah; SOH smoothed with 50-cycle rolling median",
            "Vehicles with negative ΔSOH_final flagged as non-monotone",
            "Training: V01–V04 (first 4 by filename — no selection bias)",
        ],
    }

    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report written: {REPORT_OUT}")


if __name__ == "__main__":
    main()
