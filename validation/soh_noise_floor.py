#!/usr/bin/env python3
"""
validation/soh_noise_floor.py  —  SOH measurement noise-floor diagnostic

For each vehicle: fit a local-linear smoothed trend to its SOH(t) timeseries,
compute std of residuals around that trend, compare to the magnitude of the
measured net ΔSOH.

If std(residuals) ≈ |ΔSOH|, then per-vehicle endpoint SOH is noise-dominated
and no model can beat B0 on this metric — the data itself is the ceiling.

Outputs:
  data/soh_noise_floor_report.json — per-vehicle and aggregate results
  Printed tables and conclusion

NOTE on mean% error: mean%err explodes for near-zero true ΔSOH (e.g. V03:
2686%, V14: 547%). RMSE and ρ are the meaningful aggregates; %err is flagged
unreliable for vehicles where |ΔSOH| < 0.01.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "data" / "soh_noise_floor_report.json"

Q_NOMINAL  = 136.2   # Ah — from soh_predictor.py
SOH_WIN    = 50      # rolling window, same as soh_predictor.py
MIN_Q      = 100     # minimum valid Q readings
# Trend smoothing: local median with wider window than SOH_WIN
TREND_WIN  = 150     # wider window captures the true slow trend
DSOH_NOISY_THRESH = 0.01   # |ΔSOH| below this makes %err unreliable


def _raw_soh(cycles: pd.DataFrame, veh: str) -> Optional[np.ndarray]:
    """Rolling-median SOH, identical to soh_predictor.observed_soh."""
    vc  = cycles[cycles["vehicle"] == veh]
    q   = vc["Q_Ah"].values.astype(float)
    if int((~np.isnan(q)).sum()) < MIN_Q:
        return None
    soh = (pd.Series(q).ffill().bfill()
           .rolling(SOH_WIN, min_periods=1, center=True).median()
           .values / Q_NOMINAL)
    return np.clip(soh, 0.0, 1.0)


def _trend_and_noise(soh: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Fit a wider rolling-median trend to capture the true slow degradation.
    residual = soh - trend
    Returns (trend, sigma_residual, sigma_detrended_std).
    """
    trend = (pd.Series(soh)
             .rolling(TREND_WIN, min_periods=max(1, TREND_WIN // 3), center=True)
             .median().values)
    # Fill edges where rolling window has no data
    trend = pd.Series(trend).ffill().bfill().values
    residuals = soh - trend
    sigma = float(np.std(residuals))
    return trend, sigma, float(np.mean(np.abs(residuals)))


def run_noise_diagnostic() -> None:
    from degradation.deng_loader     import load_all
    from degradation.cycle_segmentor import segment_all

    print("SOH Noise-Floor Diagnostic")
    print("=" * 65)
    print()

    raw = load_all(verbose=False)
    if not raw:
        print("ERROR: No Deng CSVs found in data/deng20/")
        sys.exit(1)

    cycles = segment_all(raw, verbose=False)

    records: List[Dict] = []

    for veh in sorted(raw.keys()):
        soh = _raw_soh(cycles, veh)
        if soh is None:
            records.append({"vehicle": veh, "note": "insufficient Q readings"})
            continue

        n          = len(soh)
        soh_start  = float(soh[0])
        soh_end    = float(soh[-1])
        net_dsoh   = soh_start - soh_end        # positive = degraded
        negative   = net_dsoh < 0               # non-physical increase

        trend, sigma, mae_trend = _trend_and_noise(soh)
        trend_start = float(trend[0])
        trend_end   = float(trend[-1])
        trend_dsoh  = trend_start - trend_end   # ΔSOH from smoothed trend

        # SNR: ratio of trend_dsoh signal to noise
        snr = abs(trend_dsoh) / (sigma + 1e-9)

        noisy_pct_err = abs(net_dsoh) < DSOH_NOISY_THRESH

        records.append({
            "vehicle"      : veh,
            "n_cycles"     : n,
            "soh_start"    : round(soh_start,  4),
            "soh_end"      : round(soh_end,    4),
            "net_dsoh"     : round(net_dsoh,   5),
            "trend_dsoh"   : round(trend_dsoh, 5),
            "sigma_resid"  : round(sigma,      5),
            "mae_trend"    : round(mae_trend,  5),
            "snr"          : round(snr,        3),
            "negative_fade": negative,
            "noisy_pct_err": noisy_pct_err,
            "note"         : "",
        })

    # ── Summary stats ─────────────────────────────────────────────────────────
    valid = [r for r in records if "sigma_resid" in r]
    n_neg  = sum(1 for r in valid if r["negative_fade"])
    n_low  = sum(1 for r in valid if r["noisy_pct_err"])
    sigmas = np.array([r["sigma_resid"] for r in valid])
    dsoh_v = np.array([r["net_dsoh"]    for r in valid])
    snrs   = np.array([r["snr"]         for r in valid])

    # Key question: is |ΔSOH| typically comparable to σ?
    ratio_dsoh_to_sigma = np.abs(dsoh_v) / (sigmas + 1e-9)

    # ── Print per-vehicle table ───────────────────────────────────────────────
    print(f"{'Veh':4s} {'n_cyc':6s} {'SOH_0':6s} {'SOH_f':6s} "
          f"{'ΔSOHmeas':9s} {'ΔSOHtrend':10s} {'σ_resid':8s} "
          f"{'|ΔSOH|/σ':9s} {'SNR':5s} {'neg?':5s} {'flag':6s}")
    print("-" * 80)

    for r in records:
        if "sigma_resid" not in r:
            print(f"{r['vehicle']:4s}  [{r['note']}]")
            continue
        neg_s  = "YES" if r["negative_fade"] else "no"
        flag_s = "LOW" if r["noisy_pct_err"] else "ok"
        ratio  = abs(r["net_dsoh"]) / (r["sigma_resid"] + 1e-9)
        print(f"{r['vehicle']:4s} {r['n_cycles']:6d} {r['soh_start']:6.3f} "
              f"{r['soh_end']:6.3f} {r['net_dsoh']:+9.5f} {r['trend_dsoh']:+10.5f} "
              f"{r['sigma_resid']:8.5f} {ratio:9.2f} {r['snr']:5.2f} "
              f"{neg_s:5s} {flag_s:6s}")

    print()
    print("=" * 65)
    print("NOISE FLOOR SUMMARY")
    print("=" * 65)
    print(f"  Vehicles with sufficient Q data: {len(valid)}/20")
    print(f"  Vehicles with negative net fade (non-physical): {n_neg}/{len(valid)}")
    print(f"  Vehicles with |ΔSOH| < {DSOH_NOISY_THRESH} (%%err unreliable): {n_low}/{len(valid)}")
    print()
    print(f"  σ_resid (per-vehicle measurement noise):")
    print(f"    min={sigmas.min():.5f}  median={np.median(sigmas):.5f}  max={sigmas.max():.5f}")
    print()
    print(f"  |ΔSOH_meas| (per-vehicle net fade):")
    print(f"    min={np.abs(dsoh_v).min():.5f}  median={np.median(np.abs(dsoh_v)):.5f}  "
          f"max={np.abs(dsoh_v).max():.5f}")
    print()
    print(f"  |ΔSOH| / σ_resid  (signal-to-noise ratio per vehicle):")
    print(f"    min={ratio_dsoh_to_sigma.min():.2f}  "
          f"median={np.median(ratio_dsoh_to_sigma):.2f}  "
          f"max={ratio_dsoh_to_sigma.max():.2f}")
    print()
    print(f"  Trend-based SNR (|ΔSOHtrend| / σ_resid):")
    print(f"    min={snrs.min():.2f}  median={np.median(snrs):.2f}  "
          f"max={snrs.max():.2f}")
    print()

    # ── Noise-floor conclusion ────────────────────────────────────────────────
    med_ratio = float(np.median(ratio_dsoh_to_sigma))
    med_snr   = float(np.median(snrs))

    if med_ratio < 1.5:
        noise_conclusion = (
            "NOISE-DOMINATED: median |ΔSOH|/σ = {:.2f} — "
            "the vehicle-level endpoint SOH variation is comparable to "
            "measurement noise. No deterministic model can reliably beat "
            "B0 on per-vehicle endpoint ΔSOH. The limiting factor for "
            "field SOH prediction on this fleet is SOH measurement quality, "
            "not the physics model.".format(med_ratio)
        )
    elif med_ratio < 3.0:
        noise_conclusion = (
            "BORDERLINE: median |ΔSOH|/σ = {:.2f} — "
            "weak signal above noise. Some vehicles show genuine fade "
            "signal (SNR>2), but the fleet average is near the noise floor. "
            "A per-vehicle model may work for high-fade vehicles; "
            "fleet-aggregate metrics will remain poor.".format(med_ratio)
        )
    else:
        noise_conclusion = (
            "SIGNAL-DOMINATED: median |ΔSOH|/σ = {:.2f} — "
            "genuine fade signal above noise. Modeling improvements "
            "should yield measurable gains over B0.".format(med_ratio)
        )

    print("NOISE-FLOOR CONCLUSION:")
    print(f"  {noise_conclusion}")
    print()
    print("NOTE on mean%err metric:")
    print(f"  {n_low} vehicles have |ΔSOH| < {DSOH_NOISY_THRESH}: V03 (0.00048), "
          "V14 (0.00620), V20 (-0.00927) etc.")
    print("  For these, any finite prediction → mean%err > 100%.")
    print("  RMSE and ρ are the meaningful aggregates; %err is flagged [LOW]")
    print("  for |ΔSOH| < 0.01 and should not drive model selection.")

    # ── Write JSON ────────────────────────────────────────────────────────────
    report = {
        "meta": {
            "script"      : "validation/soh_noise_floor.py",
            "trend_window": TREND_WIN,
            "soh_window"  : SOH_WIN,
            "noisy_thresh": DSOH_NOISY_THRESH,
        },
        "per_vehicle": {r["vehicle"]: {k: v for k, v in r.items() if k != "vehicle"}
                        for r in records},
        "aggregate": {
            "n_valid"               : len(valid),
            "n_negative_fade"       : n_neg,
            "n_noisy_pct_err"       : n_low,
            "sigma_resid_min"       : round(float(sigmas.min()),  5),
            "sigma_resid_median"    : round(float(np.median(sigmas)), 5),
            "sigma_resid_max"       : round(float(sigmas.max()),  5),
            "dsoh_abs_min"          : round(float(np.abs(dsoh_v).min()), 5),
            "dsoh_abs_median"       : round(float(np.median(np.abs(dsoh_v))), 5),
            "dsoh_abs_max"          : round(float(np.abs(dsoh_v).max()), 5),
            "ratio_dsoh_sigma_min"  : round(float(ratio_dsoh_to_sigma.min()), 3),
            "ratio_dsoh_sigma_median": round(float(np.median(ratio_dsoh_to_sigma)), 3),
            "ratio_dsoh_sigma_max"  : round(float(ratio_dsoh_to_sigma.max()), 3),
            "snr_trend_median"      : round(float(np.median(snrs)), 3),
        },
        "noise_conclusion": noise_conclusion,
        "pct_err_note": (
            f"{n_low} vehicles have |ΔSOH|<{DSOH_NOISY_THRESH}; "
            "mean%err unreliable for these — use RMSE and ρ as primary aggregates."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written → {OUT_JSON}")


if __name__ == "__main__":
    run_noise_diagnostic()
