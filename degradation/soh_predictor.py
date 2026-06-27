"""
soh_predictor.py — Capacity fade models: stress-only and stress + SEI calendar.

TWO MODELS (both fit ΔSOH = fade from first observation, not absolute SOH):

  Model A — Stress-only:
    ΔSOH(D) = β_s × D^γ_s
    where D = cumulative Miner's damage (from fatigue.py)

  Model B — Combined stress + SEI calendar:
    ΔSOH(D, t) = β_c × D^γ_c + λ × √t
    where t = years elapsed from first observation (standard SEI √t kinetics,
    Pinson & Bazant 2013 J. Electrochem. Soc. 160:A243)

DESIGN CHOICE — why ΔSOH (not absolute SOH):
  All 20 Deng vehicles entered the dataset already degraded (SOH ≈ 0.62–0.71).
  We have at most 2 years of in-service data per vehicle, not the full lifecycle.
  Fitting absolute SOH(D, t) would require knowing the pre-dataset history.
  Fitting ΔSOH anchors to the first observed state and models the RATE of
  further fade — which is what we can actually measure and compare across vehicles.

ASSUMPTIONS:
  - Q_nominal = 136.2 Ah (max observed, used as 100% SOH reference)
  - Q_Ah per cycle is session-median available_capacity (BMS reported)
  - SOH smoothed with 50-cycle rolling median to reduce BMS quantisation noise
  - Training set: V01–V04; held-out: V05–V20 (no selection bias — first 4 by filename)
  - t_years is measured from each vehicle's FIRST observation in the dataset
    (not from battery manufacture date — that is unknown)
  - MIN_Q_COUNT = 100 valid readings required; below this, vehicle is excluded
  - Flat/non-monotone capacity trajectories (V03, V10) cannot be fit by any
    monotone model; poor R² for these is expected and reported honestly
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

Q_NOMINAL    = 136.2   # Ah — nameplate capacity (100% SOH)
SOH_WIN      = 50      # rolling window for SOH smoothing
MIN_Q_COUNT  = 100     # minimum valid Q readings to include vehicle in evaluation
TRAIN_VEHS   = {"V01", "V02", "V03", "V04"}


def soh_model(D: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """SOH(D) = clip(1 - beta * D^gamma, 0, 1)."""
    return np.clip(1.0 - beta * np.power(np.maximum(D, 0.0), gamma), 0.0, 1.0)


def observed_soh(cycles: pd.DataFrame, vehicle: str) -> Optional[np.ndarray]:
    """
    Compute rolling-median SOH from Q_Ah readings for one vehicle.
    Returns None if fewer than MIN_Q_COUNT valid readings.
    """
    vc = cycles[cycles["vehicle"] == vehicle].copy()
    q = vc["Q_Ah"].values.astype(float)
    valid = ~np.isnan(q)
    if int(valid.sum()) < MIN_Q_COUNT:
        return None
    # Use forward-fill then rolling median to handle isolated NaNs
    q_series = pd.Series(q).ffill().bfill()
    soh_raw = (q_series.rolling(SOH_WIN, min_periods=1, center=True).median()
               .values / Q_NOMINAL)
    return np.clip(soh_raw, 0.0, 1.0)


def calibrate(
    cycles_with_damage: pd.DataFrame,
    train_vehicles: Optional[set] = None,
) -> Tuple[float, float, Dict]:
    """
    Calibrate β and γ from training vehicles.

    Returns:
        beta, gamma: fitted parameters
        fit_info   : dict with per-vehicle training metrics
    """
    tveh = train_vehicles or TRAIN_VEHS

    D_all, S_all = [], []
    for veh in sorted(tveh):
        vc = cycles_with_damage[cycles_with_damage["vehicle"] == veh]
        if len(vc) == 0:
            continue
        soh_obs = observed_soh(vc, veh)
        if soh_obs is None:
            continue
        D = vc["D_cumul"].values.astype(float)
        # Align lengths (soh_obs derived from vc)
        n = min(len(D), len(soh_obs))
        D_all.append(D[:n])
        S_all.append(soh_obs[:n])

    if not D_all:
        raise ValueError("No training data available for calibration")

    D_cat = np.concatenate(D_all)
    S_cat = np.concatenate(S_all)

    # Remove NaNs and ensure D > 0 for power law
    mask = np.isfinite(D_cat) & np.isfinite(S_cat) & (D_cat > 0)
    D_fit = D_cat[mask]
    S_fit = S_cat[mask]

    # Fit with bounds: beta in (0, 1000), gamma in (0.1, 5)
    try:
        (beta, gamma), _ = curve_fit(
            soh_model, D_fit, S_fit,
            p0=[500.0, 0.5],
            bounds=([0.01, 0.05], [10000.0, 5.0]),
            maxfev=5000,
        )
    except Exception as e:
        # Fall back to defaults if optimisation fails
        beta, gamma = 500.0, 0.5
        print(f"  [WARN] curve_fit failed ({e}); using defaults beta={beta}, gamma={gamma}")

    # Per-vehicle training metrics
    fit_info: Dict[str, Dict] = {}
    for veh, D_v, S_v in zip(sorted(tveh), D_all, S_all):
        n = min(len(D_v), len(S_v))
        S_pred = soh_model(D_v[:n], beta, gamma)
        residuals = S_v[:n] - S_pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((S_v[:n] - S_v[:n].mean()) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        mae = float(np.mean(np.abs(residuals)))
        fit_info[veh] = {"r2": r2, "mae_soh": mae, "n_cycles": n}

    return float(beta), float(gamma), fit_info


def evaluate_all(
    cycles_with_damage: pd.DataFrame,
    beta: float,
    gamma: float,
) -> Dict[str, Dict]:
    """
    Evaluate SOH predictions on ALL vehicles (train + held-out).
    Returns per-vehicle dict with r2, mae_soh, n_cycles, split ('train'/'test').
    """
    results = {}
    for veh in sorted(cycles_with_damage["vehicle"].unique()):
        vc = cycles_with_damage[cycles_with_damage["vehicle"] == veh]
        soh_obs = observed_soh(vc, veh)
        split = "train" if veh in TRAIN_VEHS else "test"

        if soh_obs is None:
            results[veh] = {
                "split": split, "r2": None, "mae_soh": None,
                "n_cycles": len(vc), "note": "insufficient Q readings"
            }
            continue

        D = vc["D_cumul"].values.astype(float)
        n = min(len(D), len(soh_obs))
        S_pred = soh_model(D[:n], beta, gamma)
        S_obs  = soh_obs[:n]

        residuals = S_obs - S_pred
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((S_obs - S_obs.mean()) ** 2))
        r2  = float(1.0 - ss_res / (ss_tot + 1e-12))
        mae = float(np.mean(np.abs(residuals)))

        results[veh] = {
            "split"   : split,
            "r2"      : round(r2,  4),
            "mae_soh" : round(mae, 4),
            "n_cycles": n,
            "soh_start": round(float(S_obs[0]),  3),
            "soh_end"  : round(float(S_obs[-1]), 3),
            "D_final"  : round(float(D[n-1]),    6),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model B helpers — Combined stress + SEI calendar aging
# ─────────────────────────────────────────────────────────────────────────────

def add_t_years(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    Add t_years column: elapsed time in years from each vehicle's first observation.
    t_years = 0 at first cycle, grows to ~2.0 for a 2-year vehicle history.
    """
    cycles = cycles.copy()
    parts = []
    for veh, grp in cycles.groupby("vehicle", sort=False):
        grp = grp.copy()
        t0 = grp["cycle_date"].min()
        grp["t_years"] = (grp["cycle_date"] - t0).dt.total_seconds() / (365.25 * 86400.0)
        parts.append(grp)
    return pd.concat(parts).sort_values(["vehicle", "cycle_date"]).reset_index(drop=True)


def observed_delta_soh(cycles: pd.DataFrame, vehicle: str) -> Optional[np.ndarray]:
    """
    Cumulative SOH fade from first observation: ΔSOH[i] = soh_obs[0] - soh_obs[i].
    Positive = fade from initial state. Returns None if insufficient data.
    """
    vc = cycles[cycles["vehicle"] == vehicle].copy()
    q  = vc["Q_Ah"].values.astype(float)
    if int((~np.isnan(q)).sum()) < MIN_Q_COUNT:
        return None
    soh_raw = (pd.Series(q).ffill().bfill()
               .rolling(SOH_WIN, min_periods=1, center=True).median()
               .values / Q_NOMINAL)
    soh_s = np.clip(soh_raw, 0.0, 1.0)
    return soh_s[0] - soh_s   # positive = degraded below starting point


def _stress_delta(D: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    return beta * np.power(np.maximum(D, 0.0), gamma)


def _sei_term(t: np.ndarray, lam: float) -> np.ndarray:
    return lam * np.sqrt(np.maximum(t, 0.0))


def _r2_mae(obs: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    res    = obs - pred
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2     = float(1.0 - ss_res / (ss_tot + 1e-12))
    mae    = float(np.mean(np.abs(res)))
    return r2, mae


def calibrate_combined(
    cycles_d: pd.DataFrame,
    train_vehicles: Optional[set] = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    Fit stress-only (Model A) and stress+SEI (Model B) on training vehicles.
    Both fit ΔSOH so the comparison is apples-to-apples.

    Returns:
        params_stress : {"beta": ..., "gamma": ...}
        params_sei    : {"beta": ..., "gamma": ..., "lam": ...}
        fit_info      : per-vehicle training metrics for both models
    """
    tveh = train_vehicles or TRAIN_VEHS

    D_all, t_all, dS_all, veh_list = [], [], [], []
    for veh in sorted(tveh):
        vc = cycles_d[cycles_d["vehicle"] == veh]
        if len(vc) == 0:
            continue
        dS = observed_delta_soh(vc, veh)
        if dS is None:
            continue
        D  = vc["D_cumul"].values.astype(float)
        t  = vc["t_years"].values.astype(float) if "t_years" in vc.columns else np.zeros(len(vc))
        n  = min(len(D), len(t), len(dS))
        D_all.append(D[:n]);  t_all.append(t[:n]);  dS_all.append(dS[:n])
        veh_list.append(veh)

    if not D_all:
        raise ValueError("No training data for combined calibration")

    D_cat  = np.concatenate(D_all)
    t_cat  = np.concatenate(t_all)
    dS_cat = np.concatenate(dS_all)
    mask   = np.isfinite(D_cat) & np.isfinite(t_cat) & np.isfinite(dS_cat)
    D_f, t_f, dS_f = D_cat[mask], t_cat[mask], dS_cat[mask]

    # Model A: stress-only
    def _model_a(D, beta, gamma):
        return np.clip(_stress_delta(D, beta, gamma), 0.0, 1.0)

    try:
        (beta_s, gamma_s), _ = curve_fit(
            _model_a, D_f, dS_f,
            p0=[200.0, 0.4],
            bounds=([0.001, 0.05], [1e7, 5.0]),
            maxfev=8000,
        )
    except Exception as e:
        print(f"  [WARN] stress-only fit failed ({e}); using defaults")
        beta_s, gamma_s = 200.0, 0.4

    # Model B: stress + SEI (initialise lambda from calendar fraction of observed fade)
    def _model_b(X, beta, gamma, lam):
        D_x, t_x = X
        return np.clip(_stress_delta(D_x, beta, gamma) + _sei_term(t_x, lam), 0.0, 1.0)

    lam_init = float(np.mean(np.abs(dS_f))) / max(float(np.mean(np.sqrt(t_f + 1e-9))), 0.01)
    try:
        (beta_c, gamma_c, lam_c), _ = curve_fit(
            _model_b,
            np.vstack([D_f, t_f]),
            dS_f,
            p0=[max(beta_s * 0.5, 0.01), gamma_s, lam_init],
            bounds=([0.001, 0.05, 0.0], [1e7, 5.0, 0.5]),
            maxfev=8000,
        )
    except Exception as e:
        print(f"  [WARN] combined fit failed ({e}); using stress-only + lam=0")
        beta_c, gamma_c, lam_c = beta_s, gamma_s, 0.0

    # Per-vehicle training breakdown
    fit_info: Dict[str, Dict] = {}
    for veh, D_v, t_v, dS_v in zip(veh_list, D_all, t_all, dS_all):
        n = min(len(D_v), len(t_v), len(dS_v))
        pred_a = _model_a(D_v[:n], beta_s, gamma_s)
        pred_b = _model_b(np.vstack([D_v[:n], t_v[:n]]), beta_c, gamma_c, lam_c)
        r2_a, mae_a = _r2_mae(dS_v[:n], pred_a)
        r2_b, mae_b = _r2_mae(dS_v[:n], pred_b)
        fit_info[veh] = {
            "r2_stress"   : round(r2_a,  4),
            "r2_combined" : round(r2_b,  4),
            "mae_stress"  : round(mae_a, 4),
            "mae_combined": round(mae_b, 4),
            "n_cycles"    : n,
        }

    return (
        {"beta": float(beta_s), "gamma": float(gamma_s)},
        {"beta": float(beta_c), "gamma": float(gamma_c), "lam": float(lam_c)},
        fit_info,
    )


def evaluate_combined(
    cycles_d: pd.DataFrame,
    params_stress: Dict,
    params_sei: Dict,
) -> Dict[str, Dict]:
    """
    Evaluate both models on ALL vehicles.

    Returns per-vehicle dict with:
      r2_stress / mae_stress  — Model A (stress-only, ΔSOH)
      r2_combined / mae_combined — Model B (stress + SEI, ΔSOH)
      sei_frac / stress_frac  — fraction of total modelled fade at end of vehicle history
      note                    — reason for exclusion / non-monotone flag
    """
    beta_s, gamma_s = params_stress["beta"], params_stress["gamma"]
    beta_c, gamma_c, lam_c = params_sei["beta"], params_sei["gamma"], params_sei["lam"]

    results: Dict[str, Dict] = {}
    for veh in sorted(cycles_d["vehicle"].unique()):
        vc    = cycles_d[cycles_d["vehicle"] == veh]
        split = "train" if veh in TRAIN_VEHS else "test"
        dS    = observed_delta_soh(vc, veh)

        if dS is None:
            results[veh] = {
                "split": split, "note": "insufficient Q readings",
                "r2_stress": None, "r2_combined": None,
                "mae_stress": None, "mae_combined": None,
                "sei_frac": None, "n_cycles": len(vc),
            }
            continue

        D = vc["D_cumul"].values.astype(float)
        t = vc["t_years"].values.astype(float) if "t_years" in vc.columns else np.zeros(len(vc))
        n = min(len(D), len(t), len(dS))

        pred_a = np.clip(_stress_delta(D[:n], beta_s, gamma_s), 0.0, 1.0)
        pred_b = np.clip(_stress_delta(D[:n], beta_c, gamma_c) + _sei_term(t[:n], lam_c), 0.0, 1.0)

        r2_a,  mae_a  = _r2_mae(dS[:n], pred_a)
        r2_b,  mae_b  = _r2_mae(dS[:n], pred_b)

        # Stress vs SEI fraction at final cycle (using combined model params)
        D_end = float(D[n - 1])
        t_end = float(t[n - 1])
        s_contrib   = float(_stress_delta(np.array([D_end]), beta_c, gamma_c)[0])
        sei_contrib = float(_sei_term(np.array([t_end]), lam_c)[0])
        total       = s_contrib + sei_contrib
        sei_frac    = (sei_contrib / total) if total > 1e-9 else float("nan")

        # Flag non-monotone: observed fade range tiny relative to noise
        total_fade    = float(dS[:n].max() - dS[:n].min())
        non_monotone  = bool(total_fade < 0.02)

        note = ""
        if non_monotone:
            note = "non-monotone/flat capacity — monotone model inapplicable"

        results[veh] = {
            "split"         : split,
            "r2_stress"     : round(r2_a,     4),
            "r2_combined"   : round(r2_b,     4),
            "mae_stress"    : round(mae_a,    4),
            "mae_combined"  : round(mae_b,    4),
            "sei_frac"      : round(sei_frac, 3) if not np.isnan(sei_frac) else None,
            "stress_frac"   : round(1.0 - sei_frac, 3) if not np.isnan(sei_frac) else None,
            "n_cycles"      : n,
            "note"          : note,
        }
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from degradation.deng_loader import load_all
    from degradation.cycle_segmentor import segment_all
    from degradation.fatigue import accumulate_damage

    print("=== SOH Predictor — Quick Check (V01–V04 only) ===")
    print("Loading all vehicles...")
    vehicles = load_all(verbose=False)

    # Limit to V01–V04 for quick test
    small = {k: v for k, v in vehicles.items() if k in {"V01", "V02", "V03", "V04"}}
    cycles = segment_all(small, verbose=False)
    print(f"Segmented {len(cycles)} cycles")
    print("Computing rainflow damage (this takes ~30 s for 4 vehicles)...")
    cycles_d = accumulate_damage(cycles, small)
    print(f"Calibrating...")
    beta, gamma, train_info = calibrate(cycles_d, TRAIN_VEHS & set(small))
    print(f"Fitted: beta={beta:.2f}  gamma={gamma:.4f}")
    for veh, info in train_info.items():
        print(f"  {veh}: R²={info['r2']:.4f}  MAE_SOH={info['mae_soh']:.4f}  N={info['n_cycles']}")
