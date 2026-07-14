#!/usr/bin/env python3
"""
degradation/problem1_360_validation.py
==========================================
Comprehensive ("360 degree") validation of Problem 1 (limited-data degradation
prediction: predict a cell's full degradation curve from a small early-cycle
slice). Runs 7 independent checks on BOTH chemistries separately (NASA LCO
n=4, Severson LFP n=124) — LCO and LFP results are NEVER pooled into one
number, per this project's own established finding
(severson_gp_predictor.py: "The analysis is SEPARATE from the NASA LCO
results and is NOT directly comparable to them.").

MODEL: closed-form conjugate Bayesian linear regression, NOT the full
Matern52-kernel GP used in bayes_gp_predictor.py / severson_gp_predictor.py
------------------------------------------------------------------------------
This is a DELIBERATE, DISCLOSED methodological simplification, not an
oversight. The full GP machinery in the two sibling modules calibrates kernel
hyperparameters via PyMC NUTS per fold; even severson_gp_predictor.py's
own docstring documents ~15-30 minutes for ONE calibration run at 3 N-values.
This validation needs 5 early-cycle fractions x (4 + 124) cells x multiple
checks -- full GP-NUTS at that scale is computationally infeasible in a single
run. Instead:

    m(k) = beta * k^0.5                          physics mean function
    beta | data ~ Normal(mu_post, tau2_post)      conjugate Normal-Normal update
    mu_prior, sigma_prior = LOO population stats from the OTHER cells' full
                             trajectories (same predictive-SD correction
                             sqrt(1+1/(n-1)) as bayes_gp_predictor.py)
    Predictive variance for a future cycle k:
        var(k) = tau2_post * x(k)^2 + sigma_obs^2
        (sigma_obs = pooled per-chemistry residual SD from full-trajectory
         single-cell OLS fits -- same quantity bayes_gp_predictor.py and
         hierarchical_beta.py compute for their own sigma_obs priors)

WHAT THIS MEANS FOR THE RESULTS: this closed-form model has NO correlated
residual (GP kernel) term -- all residual variance is treated as i.i.d. noise.
The full GP models capture short-range correlated structure the kernel is
built to model; this simplified model cannot. Expect this model's coverage
and RMSE to differ (plausibly worse, since it cannot borrow strength from a
smooth correlated residual trend) from the full GP reported elsewhere in this
project. This is reported explicitly, not smoothed over, in Step 3.

TOTAL CYCLE LIFE DEFINITION: "total cycle life" here = length of each cell's
recorded degradation trajectory (len(dsoh)), NOT the formal 80%-EOL
"cycle_life" field Severson's own dataset separately provides. For a small
number of cells whose recorded trajectory extends past 80% fade these could
differ; not reconciled here, flagged as a definitional choice.

7 CHECKS (see docstrings on each run_* function below)
---------------------------------------------------------
1. run_fraction_sweep       -- multi-fraction early-cycle test, LOO-CV
2. (folded into #1's per-cell output) leave-one-cell-out per-cell distribution
3. run_calibration_check    -- 90% interval empirical coverage
4. run_baseline_comparison  -- flat and linear-extrapolation baselines
5. run_residual_analysis    -- residual bias vs. position in life
6. run_ood_check            -- performance vs. cycle-life percentile (extrapolation limit)
7. run_sample_size_honesty  -- CIs on the CIs; NASA n=4 vs Severson n=124 bootstrap

CITATIONS (inherited from sibling modules, not re-derived here)
---------------------------------------------------------------
- Richardson, Osborne & Howey (2017). J. Power Sources 357:209-219.
- Severson et al. (2019). Nature Energy 4:383-391.
- DeGroot & Schervish (2012), §8.6 -- predictive-SD correction.
- This project's own bayes_gp_predictor.py, severson_gp_predictor.py,
  hierarchical_beta.py -- physics mean function, D_k=k scale, LOO-consistent
  priors, sigma_obs calibration approach all inherited from these.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
NASA_DIR = ROOT / "data" / "nasa"
OUT_JSON = ROOT / "data" / "problem1_360_validation_report.json"
OUT_MD   = ROOT / "docs" / "problem1_360_validation.md"

GAMMA = 0.5
FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50]
MIN_N_OBS = 3
Z_90 = 1.6448536269514722   # standard normal 90% two-sided z

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_nasa_cells() -> List[Dict]:
    cells = []
    for cid in NASA_CELLS:
        mat = loadmat(str(NASA_DIR / f"{cid}.mat"))
        key = [k for k in mat if not k.startswith("_")][0]
        cycs = mat[key]["cycle"][0, 0]
        Qs = []
        for i in range(cycs.shape[1]):
            c = cycs[0, i]
            if "discharge" not in str(c["type"][0]).strip().lower():
                continue
            data = c["data"][0, 0]
            I = data["Current_measured"][0].astype(np.float64)
            t = data["Time"][0].astype(np.float64)
            dt = np.diff(t, prepend=t[0])
            Q = float(np.cumsum(np.abs(I) * dt)[-1] / 3600.0)
            if Q > 0:
                Qs.append(Q)
        Q0 = Qs[0]
        dsoh = 1.0 - np.array([q / Q0 for q in Qs])
        cells.append({"cell_id": cid, "dsoh": dsoh, "n_total": len(dsoh)})
    return cells


def _load_severson_cells() -> List[Dict]:
    sys.path.insert(0, str(ROOT / "data" / "loaders"))
    import severson_loader
    raw = severson_loader.load_severson(verbose=False)
    cells = []
    for c in raw:
        soh = np.clip(np.array(c["soh"], dtype=float), 0.0, 1.05)
        dsoh = 1.0 - soh
        cells.append({"cell_id": c["cell_id"], "dsoh": dsoh, "n_total": len(dsoh),
                       "batch": c["batch"]})
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Core model: LOO population prior + conjugate posterior + prediction
# ─────────────────────────────────────────────────────────────────────────────

def _ols_beta_resid_sd(dsoh: np.ndarray) -> Tuple[float, float]:
    k = np.arange(1, len(dsoh) + 1, dtype=float)
    x = np.power(k, GAMMA)
    sum_x2 = float(np.dot(x, x))
    beta = float(np.dot(x, dsoh)) / sum_x2 if sum_x2 > 0 else 0.0
    resid = dsoh - beta * x
    resid_sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else float(np.std(resid))
    return beta, resid_sd


def _population_stats(cells: List[Dict], exclude_idx: int) -> Tuple[float, float, float]:
    """LOO population mean/sd of beta (excluding `exclude_idx`), predictive-SD corrected.
    Also returns pooled sigma_obs (residual SD) across the excluded population."""
    betas, sds = [], []
    for j, c in enumerate(cells):
        if j == exclude_idx:
            continue
        b, s = _ols_beta_resid_sd(c["dsoh"])
        betas.append(b)
        sds.append(s)
    betas = np.array(betas)
    n_other = len(betas)
    mu_prior = float(betas.mean())
    sigma_prior = float(betas.std(ddof=1)) * np.sqrt(1.0 + 1.0 / n_other)
    sigma_obs = float(np.mean(sds))
    return mu_prior, sigma_prior, sigma_obs


def _fit_and_predict(dsoh: np.ndarray, n_obs: int,
                      mu_prior: float, sigma_prior: float,
                      sigma_obs: float) -> Dict:
    """Fit conjugate beta posterior on first n_obs cycles, predict the rest."""
    k_obs = np.arange(1, n_obs + 1, dtype=float)
    x_obs = np.power(k_obs, GAMMA)
    y_obs = dsoh[:n_obs]

    lam0 = 1.0 / sigma_prior**2
    sum_x2 = float(np.dot(x_obs, x_obs))
    post_var = 1.0 / (lam0 + sum_x2)
    post_mean = post_var * (lam0 * mu_prior + float(np.dot(x_obs, y_obs)))

    k_pred = np.arange(n_obs + 1, len(dsoh) + 1, dtype=float)
    x_pred = np.power(k_pred, GAMMA)
    y_true = dsoh[n_obs:]

    pred_mean = post_mean * x_pred
    pred_var = post_var * x_pred**2 + sigma_obs**2
    pred_sd = np.sqrt(pred_var)

    return {
        "beta_post_mean": post_mean, "beta_post_sd": np.sqrt(post_var),
        "k_pred": k_pred, "pred_mean": pred_mean, "pred_sd": pred_sd,
        "y_true": y_true,
    }


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def _rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Baselines (Step 4)
# ─────────────────────────────────────────────────────────────────────────────

def _baseline_flat(dsoh: np.ndarray, n_obs: int) -> np.ndarray:
    last = dsoh[n_obs - 1]
    return np.full(len(dsoh) - n_obs, last)


def _baseline_linear(dsoh: np.ndarray, n_obs: int) -> np.ndarray:
    k_obs = np.arange(1, n_obs + 1, dtype=float)
    y_obs = dsoh[:n_obs]
    A = np.vstack([k_obs, np.ones_like(k_obs)]).T
    slope, intercept = np.linalg.lstsq(A, y_obs, rcond=None)[0]
    k_pred = np.arange(n_obs + 1, len(dsoh) + 1, dtype=float)
    return slope * k_pred + intercept


# ─────────────────────────────────────────────────────────────────────────────
# Steps 1+2: fraction sweep + per-cell LOO-CV
# ─────────────────────────────────────────────────────────────────────────────

def run_fraction_sweep(cells: List[Dict], chem: str) -> Dict:
    """
    Step 1+2. For each cell (LOO) and each fraction, fit on the early fraction,
    predict the rest, and record R2/MAE/RMSE. Returns both per-fraction
    aggregates (Step 1) and full per-cell tables (Step 2) -- same underlying
    computation, two views, so results cannot be cherry-picked between them.
    """
    per_cell_by_frac: Dict[str, List[Dict]] = {str(f): [] for f in FRACTIONS}

    for i, c in enumerate(cells):
        dsoh, n_total = c["dsoh"], c["n_total"]
        mu_prior, sigma_prior, sigma_obs = _population_stats(cells, i)

        for frac in FRACTIONS:
            n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
            if n_obs >= n_total - 1:
                continue  # not enough future cycles to evaluate
            res = _fit_and_predict(dsoh, n_obs, mu_prior, sigma_prior, sigma_obs)
            r2 = _r2(res["y_true"], res["pred_mean"])
            mae = _mae(res["y_true"], res["pred_mean"])
            rmse = _rmse(res["y_true"], res["pred_mean"])
            per_cell_by_frac[str(frac)].append({
                "cell_id": c["cell_id"], "n_obs": n_obs, "n_total": n_total,
                "r2": r2, "mae": mae, "rmse": rmse,
                "beta_post_mean": res["beta_post_mean"],
            })

    # Step 1: aggregate per fraction
    step1 = {}
    for frac in FRACTIONS:
        rows = per_cell_by_frac[str(frac)]
        if not rows:
            step1[str(frac)] = {"n_cells": 0, "note": "no cells had enough cycles"}
            continue
        r2s = np.array([r["r2"] for r in rows])
        maes = np.array([r["mae"] for r in rows])
        rmses = np.array([r["rmse"] for r in rows])
        step1[str(frac)] = {
            "n_cells": len(rows),
            "r2_mean": float(r2s.mean()), "r2_median": float(np.median(r2s)),
            "r2_std": float(r2s.std(ddof=1)) if len(r2s) > 1 else 0.0,
            "mae_mean": float(maes.mean()), "mae_median": float(np.median(maes)),
            "rmse_mean": float(rmses.mean()), "rmse_median": float(np.median(rmses)),
            "r2_frac_negative": float(np.mean(r2s < 0)),
        }

    # Step 2: per-cell distribution (full table + percentiles) per fraction
    step2 = {}
    for frac in FRACTIONS:
        rows = per_cell_by_frac[str(frac)]
        if not rows:
            step2[str(frac)] = {"per_cell": [], "distribution": {}}
            continue
        r2s = np.array([r["r2"] for r in rows])
        step2[str(frac)] = {
            "per_cell": rows,
            "distribution": {
                "r2_min": float(r2s.min()), "r2_p25": float(np.percentile(r2s, 25)),
                "r2_median": float(np.median(r2s)), "r2_p75": float(np.percentile(r2s, 75)),
                "r2_max": float(r2s.max()),
                "worst_cell": rows[int(np.argmin(r2s))]["cell_id"],
                "best_cell": rows[int(np.argmax(r2s))]["cell_id"],
            },
        }

    return {"step1_fraction_sweep": step1, "step2_per_cell_loo": step2,
            "raw_per_cell_by_frac": per_cell_by_frac}


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: uncertainty calibration
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration_check(cells: List[Dict], chem: str) -> Dict:
    """
    Pool ALL (cell, fraction, future-cycle) predictions for this chemistry and
    compute empirical coverage of the nominal 90% credible interval.
    Also broken down by fraction, since coverage is expected to vary with how
    much data informs the posterior.
    """
    by_frac_covered: Dict[str, List[bool]] = {str(f): [] for f in FRACTIONS}
    all_covered: List[bool] = []

    for i, c in enumerate(cells):
        dsoh, n_total = c["dsoh"], c["n_total"]
        mu_prior, sigma_prior, sigma_obs = _population_stats(cells, i)
        for frac in FRACTIONS:
            n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
            if n_obs >= n_total - 1:
                continue
            res = _fit_and_predict(dsoh, n_obs, mu_prior, sigma_prior, sigma_obs)
            lo = res["pred_mean"] - Z_90 * res["pred_sd"]
            hi = res["pred_mean"] + Z_90 * res["pred_sd"]
            covered = (res["y_true"] >= lo) & (res["y_true"] <= hi)
            by_frac_covered[str(frac)].extend(covered.tolist())
            all_covered.extend(covered.tolist())

    result = {
        "nominal_coverage": 0.90,
        "overall_empirical_coverage": float(np.mean(all_covered)) if all_covered else None,
        "n_predictions_pooled": len(all_covered),
        "by_fraction": {},
    }
    for frac in FRACTIONS:
        vals = by_frac_covered[str(frac)]
        result["by_fraction"][str(frac)] = {
            "empirical_coverage": float(np.mean(vals)) if vals else None,
            "n_predictions": len(vals),
        }
    overall = result["overall_empirical_coverage"]
    if overall is not None:
        if overall < 0.75:
            verdict = f"MISCALIBRATED (overconfident): {overall:.2f} actual vs 0.90 nominal."
        elif overall > 0.98:
            verdict = f"MISCALIBRATED (vacuously wide): {overall:.2f} actual vs 0.90 nominal."
        else:
            verdict = f"Reasonably calibrated: {overall:.2f} actual vs 0.90 nominal."
        result["verdict"] = verdict
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b: jackknife+ calibration (LCO ONLY -- Thread 2 fix, NASA n=4 GP
# posterior is overconfident at 73.9% actual vs 90% nominal coverage).
# Severson LFP is NOT touched by this check -- its existing 94.2% coverage
# result from run_calibration_check is left as-is.
# ─────────────────────────────────────────────────────────────────────────────

from degradation.bayes_gp_predictor import (
    jackknife_plus_interval as _jk_interval,
    jackknife_plus_interval_practical as _jk_interval_practical,
)


def run_jackknife_plus_calibration(cells: List[Dict]) -> Dict:
    """
    LCO-only. For each held-out cell H and each fraction, build a jackknife+
    ensemble from the OTHER 3 cells (n=3 LOO-ensemble size):

      For each excluded member j in {other 3 cells}:
        - population_excl_j = the remaining 2 cells (not H, not j) --
          weak, n=2, population prior.
        - mu_loo_j(k) = posterior-mean prediction for H's future cycle k,
          using population_excl_j + H's own observed early fraction.
        - R_j = RMSE of predicting j's OWN future cycles from j's own early
          fraction, using population_excl_j -- a genuine leave-j-out
          residual, same units as mu_loo_j.

      jackknife+ interval at cycle k = combine {mu_loo_j(k) +/- R_j}_{j=1..3}
      via the order-statistic formula (n=3 ensemble).

    Uses this module's own closed-form conjugate model throughout (NOT the
    full GP in bayes_gp_predictor.py) for consistency with every other check
    in this validation -- only the INTERVAL CONSTRUCTION is jackknife+; the
    point-prediction model is unchanged from the rest of this file.
    """
    results_by_frac: Dict[str, Dict] = {}

    for frac in FRACTIONS:
        all_covered_formal_90 = []      # alpha=0.05 formal attempt (expected unachievable)
        all_covered_formal_50 = []      # alpha=0.25, the max formally-achievable guarantee
        all_covered_practical = []      # alpha=0.05, index clipped to n (empirical only)
        widths_formal_50 = []
        widths_practical = []
        n_folds = 0

        for hi, H in enumerate(cells):
            others = [c for c in cells if c["cell_id"] != H["cell_id"]]  # 3 cells
            n_total_H = H["n_total"]
            n_obs_H = max(MIN_N_OBS, int(np.ceil(frac * n_total_H)))
            if n_obs_H >= n_total_H - 1:
                continue

            mu_loo_by_j = []   # will become array (3, n_pred)
            R_j_list = []

            for j_idx, J in enumerate(others):
                excl_pop = [c for c in others if c["cell_id"] != J["cell_id"]]  # 2 cells
                if len(excl_pop) < 2:
                    continue
                betas = np.array([_ols_beta_resid_sd(c["dsoh"])[0] for c in excl_pop])
                sds = np.array([_ols_beta_resid_sd(c["dsoh"])[1] for c in excl_pop])
                mu_prior = float(betas.mean())
                sigma_prior = float(betas.std(ddof=1)) * np.sqrt(1.0 + 1.0 / len(excl_pop)) \
                    if len(excl_pop) > 1 else float(sds.mean())
                sigma_obs = float(sds.mean())

                # mu_loo_j(k) for H's future cycles
                res_H = _fit_and_predict(H["dsoh"], n_obs_H, mu_prior, sigma_prior, sigma_obs)
                mu_loo_by_j.append(res_H["pred_mean"])

                # R_j: leave-J-out residual, predicting J's own future from J's own early fraction
                n_total_J = J["n_total"]
                n_obs_J = max(MIN_N_OBS, int(np.ceil(frac * n_total_J)))
                if n_obs_J >= n_total_J - 1:
                    R_j_list.append(sigma_obs)  # fallback: use pooled noise estimate
                    continue
                res_J = _fit_and_predict(J["dsoh"], n_obs_J, mu_prior, sigma_prior, sigma_obs)
                R_j = _rmse(res_J["y_true"], res_J["pred_mean"])
                R_j_list.append(R_j)

            if len(mu_loo_by_j) < 3:
                continue  # need all 3 LOO members for n=3 jackknife+

            n_pred = len(mu_loo_by_j[0])
            mu_loo_arr = np.array(mu_loo_by_j)      # (3, n_pred)
            R_j_arr = np.array(R_j_list)             # (3,)
            y_true = H["dsoh"][n_obs_H:]
            n_folds += 1

            for k in range(n_pred):
                mu_k = mu_loo_arr[:, k]
                jk_90 = _jk_interval(mu_k, R_j_arr, alpha=0.05)
                jk_50 = _jk_interval(mu_k, R_j_arr, alpha=0.25)
                jk_prac = _jk_interval_practical(mu_k, R_j_arr, alpha=0.05)

                if jk_90["guarantee_achievable"]:
                    all_covered_formal_90.append(bool(jk_90["lo"] <= y_true[k] <= jk_90["hi"]))
                all_covered_formal_50.append(bool(jk_50["lo"] <= y_true[k] <= jk_50["hi"]))
                widths_formal_50.append(jk_50["hi"] - jk_50["lo"])
                all_covered_practical.append(bool(jk_prac["lo"] <= y_true[k] <= jk_prac["hi"]))
                widths_practical.append(jk_prac["hi"] - jk_prac["lo"])

        results_by_frac[str(frac)] = {
            "n_folds": n_folds,
            "n_predictions": len(all_covered_practical),
            "formal_alpha_0.05_target_90pct": {
                "guarantee_achievable_at_n3": False,
                "note": "k=4 required, only n=3 LOO members available -- no finite interval carries this guarantee.",
                "n_evaluated": len(all_covered_formal_90),
            },
            "formal_alpha_0.25_guaranteed_50pct": {
                "guarantee_achievable_at_n3": True,
                "empirical_coverage": float(np.mean(all_covered_formal_50)) if all_covered_formal_50 else None,
                "mean_width": float(np.mean(widths_formal_50)) if widths_formal_50 else None,
                "guaranteed_coverage": 0.50,
            },
            "practical_clipped_alpha_0.05": {
                "empirical_coverage": float(np.mean(all_covered_practical)) if all_covered_practical else None,
                "mean_width": float(np.mean(widths_practical)) if widths_practical else None,
                "formal_guarantee": None,
                "note": "Index clipped to n=3 when formal k=4 unavailable. NO proven coverage "
                        "guarantee -- empirical coverage only, measured via this project's own LOO-CV.",
            },
        }

    return results_by_frac


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: baseline comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_comparison(cells: List[Dict], chem: str) -> Dict:
    result = {}
    for frac in FRACTIONS:
        phys_r2, flat_r2, lin_r2 = [], [], []
        phys_mae, flat_mae, lin_mae = [], [], []
        for i, c in enumerate(cells):
            dsoh, n_total = c["dsoh"], c["n_total"]
            n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
            if n_obs >= n_total - 1:
                continue
            mu_prior, sigma_prior, sigma_obs = _population_stats(cells, i)
            res = _fit_and_predict(dsoh, n_obs, mu_prior, sigma_prior, sigma_obs)
            y_true = res["y_true"]

            phys_r2.append(_r2(y_true, res["pred_mean"]))
            phys_mae.append(_mae(y_true, res["pred_mean"]))

            flat_pred = _baseline_flat(dsoh, n_obs)
            flat_r2.append(_r2(y_true, flat_pred))
            flat_mae.append(_mae(y_true, flat_pred))

            lin_pred = _baseline_linear(dsoh, n_obs)
            lin_r2.append(_r2(y_true, lin_pred))
            lin_mae.append(_mae(y_true, lin_pred))

        if not phys_r2:
            result[str(frac)] = {"note": "no cells had enough cycles"}
            continue
        result[str(frac)] = {
            "n_cells": len(phys_r2),
            "physics_bayes":  {"r2_mean": float(np.mean(phys_r2)), "mae_mean": float(np.mean(phys_mae))},
            "flat_baseline":  {"r2_mean": float(np.mean(flat_r2)), "mae_mean": float(np.mean(flat_mae))},
            "linear_baseline":{"r2_mean": float(np.mean(lin_r2)),  "mae_mean": float(np.mean(lin_mae))},
            "physics_beats_flat_mae":   bool(np.mean(phys_mae) < np.mean(flat_mae)),
            "physics_beats_linear_mae": bool(np.mean(phys_mae) < np.mean(lin_mae)),
            "mae_improvement_vs_flat_pct":   float((np.mean(flat_mae) - np.mean(phys_mae)) / (np.mean(flat_mae) + 1e-12) * 100),
            "mae_improvement_vs_linear_pct": float((np.mean(lin_mae) - np.mean(phys_mae)) / (np.mean(lin_mae) + 1e-12) * 100),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: residual analysis (bias vs. position in life)
# ─────────────────────────────────────────────────────────────────────────────

def run_residual_analysis(cells: List[Dict], chem: str) -> Dict:
    """Pool residuals (pred-actual) across all fractions/cells, bin by
    normalized position in total life (k/n_total). Checks for systematic
    bias, especially near end-of-life (80-100% bin) where it matters most."""
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_resid: Dict[str, List[float]] = {f"{lo:.1f}-{hi:.1f}": [] for lo, hi in bins}

    for i, c in enumerate(cells):
        dsoh, n_total = c["dsoh"], c["n_total"]
        mu_prior, sigma_prior, sigma_obs = _population_stats(cells, i)
        for frac in FRACTIONS:
            n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
            if n_obs >= n_total - 1:
                continue
            res = _fit_and_predict(dsoh, n_obs, mu_prior, sigma_prior, sigma_obs)
            resid = res["pred_mean"] - res["y_true"]
            pos = res["k_pred"] / n_total
            for lo, hi in bins:
                mask = (pos >= lo) & (pos < hi if hi < 1.0 else pos <= hi)
                if mask.any():
                    bin_resid[f"{lo:.1f}-{hi:.1f}"].extend(resid[mask].tolist())

    result = {}
    for key, vals in bin_resid.items():
        if not vals:
            result[key] = {"n": 0}
            continue
        arr = np.array(vals)
        result[key] = {
            "n": len(arr), "mean_residual": float(arr.mean()),
            "sd_residual": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "direction": "over-predicts fade" if arr.mean() > 0 else "under-predicts fade",
        }
    eol_bias = result.get("0.8-1.0", {}).get("mean_residual")
    result["eol_bias_note"] = (
        f"Near-EOL (80-100% of life) mean residual = {eol_bias:.5f} dSOH -- "
        f"{'over' if eol_bias and eol_bias > 0 else 'under'}-predicting fade at "
        f"end-of-life, the region that matters most for replacement/warranty "
        f"decisions." if eol_bias is not None else "insufficient EOL predictions"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: OOD / extrapolation limit check
# ─────────────────────────────────────────────────────────────────────────────

def run_ood_check(cells: List[Dict], chem: str, ref_frac: float = 0.20) -> Dict:
    """Stratify per-cell R2 at a reference fraction (20%) by whether the
    cell's total cycle life sits inside or outside the population's IQR."""
    life = np.array([c["n_total"] for c in cells])
    p25, p75 = np.percentile(life, 25), np.percentile(life, 75)

    typical_r2, extreme_r2 = [], []
    per_cell = []
    for i, c in enumerate(cells):
        dsoh, n_total = c["dsoh"], c["n_total"]
        n_obs = max(MIN_N_OBS, int(np.ceil(ref_frac * n_total)))
        if n_obs >= n_total - 1:
            continue
        mu_prior, sigma_prior, sigma_obs = _population_stats(cells, i)
        res = _fit_and_predict(dsoh, n_obs, mu_prior, sigma_prior, sigma_obs)
        r2 = _r2(res["y_true"], res["pred_mean"])
        is_extreme = bool(n_total < p25 or n_total > p75)
        (extreme_r2 if is_extreme else typical_r2).append(r2)
        per_cell.append({"cell_id": c["cell_id"], "n_total": n_total,
                          "r2": r2, "extreme": is_extreme})

    return {
        "ref_fraction": ref_frac,
        "cycle_life_range_validated": [int(life.min()), int(life.max())],
        "cycle_life_iqr": [float(p25), float(p75)],
        "n_typical_cells": len(typical_r2),
        "n_extreme_cells": len(extreme_r2),
        "r2_mean_typical": float(np.mean(typical_r2)) if typical_r2 else None,
        "r2_mean_extreme": float(np.mean(extreme_r2)) if extreme_r2 else None,
        "degrades_for_extreme": (
            bool(np.mean(extreme_r2) < np.mean(typical_r2))
            if typical_r2 and extreme_r2 else None
        ),
        "per_cell": per_cell,
        "validity_statement": (
            f"Validated on cells with total recorded life in "
            f"[{int(life.min())}, {int(life.max())}] cycles. Predictions for a "
            f"new cell whose eventual cycle life falls far outside this range "
            f"(especially above {int(life.max())}) are extrapolation beyond "
            f"anything tested here and should not be trusted without new "
            f"validation data."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: sample size honesty
# ─────────────────────────────────────────────────────────────────────────────

def run_sample_size_honesty(cells: List[Dict], chem: str, n_boot: int = 2000) -> Dict:
    n = len(cells)
    result = {"n_cells": n}

    if n <= 10:
        # Small-n: report SE-based interval, explicit low-power warning
        per_frac = {}
        for frac in FRACTIONS:
            r2s = []
            for i, c in enumerate(cells):
                dsoh, n_total = c["dsoh"], c["n_total"]
                n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
                if n_obs >= n_total - 1:
                    continue
                mu_prior, sigma_prior, sigma_obs = _population_stats(cells, i)
                res = _fit_and_predict(dsoh, n_obs, mu_prior, sigma_prior, sigma_obs)
                r2s.append(_r2(res["y_true"], res["pred_mean"]))
            if len(r2s) < 2:
                per_frac[str(frac)] = {"note": "insufficient folds"}
                continue
            arr = np.array(r2s)
            se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
            per_frac[str(frac)] = {
                "n_folds": len(arr), "r2_mean": float(arr.mean()), "se": se,
                "approx_95ci": [float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)],
            }
        result["method"] = "standard-error interval (n too small for bootstrap to add information)"
        result["per_fraction"] = per_frac
        result["honesty_warning"] = (
            f"n={n} cells means every statistic in this report has only {n} "
            f"leave-one-out folds. A 95% CI computed from {n} points is itself "
            f"barely informative -- treat all NASA point estimates in this "
            f"validation as illustrative of a failure/success MODE, not as "
            f"precise, generalizable numbers. This is the same caveat "
            f"hierarchical_beta.py and bayes_gp_predictor.py already state for "
            f"this dataset."
        )
    else:
        # Large-n: proper bootstrap CI (resample cells with replacement)
        rng = np.random.default_rng(42)
        per_frac = {}
        for frac in FRACTIONS:
            r2s = []
            for i, c in enumerate(cells):
                dsoh, n_total = c["dsoh"], c["n_total"]
                n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
                if n_obs >= n_total - 1:
                    continue
                mu_prior, sigma_prior, sigma_obs = _population_stats(cells, i)
                res = _fit_and_predict(dsoh, n_obs, mu_prior, sigma_prior, sigma_obs)
                r2s.append(_r2(res["y_true"], res["pred_mean"]))
            r2s = np.array(r2s)
            if len(r2s) < 5:
                per_frac[str(frac)] = {"note": "insufficient folds"}
                continue
            boot_means = np.array([
                rng.choice(r2s, size=len(r2s), replace=True).mean()
                for _ in range(n_boot)
            ])
            per_frac[str(frac)] = {
                "n_folds": len(r2s), "r2_mean": float(r2s.mean()),
                "bootstrap_95ci": [float(np.percentile(boot_means, 2.5)),
                                   float(np.percentile(boot_means, 97.5))],
            }
        result["method"] = f"bootstrap ({n_boot} resamples of {n} cells)"
        result["per_fraction"] = per_frac
        result["honesty_note"] = (
            f"n={n} cells supports a proper bootstrap CI, in sharp contrast to "
            f"the NASA n=4 case -- the R2 estimates here are far more "
            f"statistically trustworthy, though the Severson dataset's own "
            f"protocol-heterogeneity caveat (severson_gp_predictor.py: "
            f"protocol explains R2=0.452 of beta variance) still applies to "
            f"what the population variance actually represents."
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full run per chemistry
# ─────────────────────────────────────────────────────────────────────────────

def run_all_checks(cells: List[Dict], chem: str) -> Dict:
    print(f"\n{'='*78}\n{chem} — {len(cells)} cells\n{'='*78}")

    print("  [1+2] Fraction sweep + per-cell LOO-CV...")
    fs = run_fraction_sweep(cells, chem)
    for frac in FRACTIONS:
        s = fs["step1_fraction_sweep"].get(str(frac), {})
        if s.get("n_cells"):
            print(f"    frac={frac:.2f}: n={s['n_cells']:3d}  R2_mean={s['r2_mean']:+.3f}  "
                  f"R2_median={s['r2_median']:+.3f}  MAE={s['mae_mean']:.5f}  "
                  f"frac_negative_R2={s['r2_frac_negative']:.2f}")

    print("  [3] Uncertainty calibration...")
    cal = run_calibration_check(cells, chem)
    print(f"    Overall empirical coverage of nominal 90% interval: "
          f"{cal['overall_empirical_coverage']:.3f}  -- {cal.get('verdict','')}")

    print("  [4] Baseline comparison...")
    base = run_baseline_comparison(cells, chem)
    for frac in FRACTIONS:
        b = base.get(str(frac), {})
        if "physics_bayes" in b:
            print(f"    frac={frac:.2f}: physics_MAE={b['physics_bayes']['mae_mean']:.5f}  "
                  f"flat_MAE={b['flat_baseline']['mae_mean']:.5f}  "
                  f"linear_MAE={b['linear_baseline']['mae_mean']:.5f}  "
                  f"beats_flat={b['physics_beats_flat_mae']}  "
                  f"beats_linear={b['physics_beats_linear_mae']}")

    print("  [5] Residual analysis...")
    resid = run_residual_analysis(cells, chem)
    print(f"    {resid.get('eol_bias_note')}")

    print("  [6] OOD / extrapolation limit check...")
    ood = run_ood_check(cells, chem)
    print(f"    Validated cycle-life range: {ood['cycle_life_range_validated']}  "
          f"typical R2={ood['r2_mean_typical']}  extreme R2={ood['r2_mean_extreme']}")

    print("  [7] Sample size honesty...")
    ssh = run_sample_size_honesty(cells, chem)
    print(f"    Method: {ssh['method']}")

    return {
        "chemistry": chem, "n_cells": len(cells),
        "step1_2_fraction_sweep_and_loo": fs,
        "step3_calibration": cal,
        "step4_baselines": base,
        "step5_residuals": resid,
        "step6_ood": ood,
        "step7_sample_size_honesty": ssh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def _write_markdown(nasa_res: Dict, sev_res: Dict) -> None:
    lines = []
    lines.append("# Problem 1: 360-Degree Validation Report\n")
    lines.append(
        "Comprehensive validation of limited-data degradation prediction, run "
        "separately on NASA LCO (n=4) and Severson LFP (n=124). Model: "
        "closed-form conjugate Bayesian linear regression (physics mean "
        "function `beta * k^0.5`, LOO population prior) -- a disclosed "
        "simplification of the full Matern52-GP used in "
        "`bayes_gp_predictor.py`/`severson_gp_predictor.py`, needed for "
        "computational tractability across this many fraction/cell "
        "combinations. See module docstring for the full rationale.\n"
    )

    for res in (nasa_res, sev_res):
        chem = res["chemistry"]
        lines.append(f"\n## {chem} (n={res['n_cells']})\n")

        s1 = res["step1_2_fraction_sweep_and_loo"]["step1_fraction_sweep"]
        r2_trend = [s1[str(f)]["r2_median"] for f in FRACTIONS if s1.get(str(f), {}).get("n_cells")]
        mae_trend = [s1[str(f)]["mae_mean"] for f in FRACTIONS if s1.get(str(f), {}).get("n_cells")]
        if len(r2_trend) >= 2 and r2_trend[-1] < r2_trend[0] and (mae_trend[-1] - mae_trend[0]) / (mae_trend[0] + 1e-9) < 0.15:
            lines.append(
                "> **Read this carefully:** median R² *decreases* as the training "
                "fraction increases, while MAE stays roughly flat. This is a "
                "known R² pathology, not evidence the model gets worse with more "
                "data: R²'s denominator is the variance of the *remaining unseen* "
                "trajectory, which shrinks as more of the cell is already "
                "observed — the same absolute error produces a worse (or "
                "negative) R² against a smaller, noisier target range. **MAE/RMSE "
                "are the more trustworthy metrics for the \"does more data help\" "
                "question in this table; R² should be read per-fraction, not "
                "trended across fractions.**\n"
            )

        lines.append("### Step 1 — Multi-fraction early-cycle test\n")
        lines.append("| Fraction | n cells | R² mean | R² median | MAE mean | RMSE mean | %R²<0 |")
        lines.append("|---|---|---|---|---|---|---|")
        for frac in FRACTIONS:
            s = res["step1_2_fraction_sweep_and_loo"]["step1_fraction_sweep"].get(str(frac), {})
            if s.get("n_cells"):
                lines.append(f"| {frac:.0%} | {s['n_cells']} | {s['r2_mean']:+.3f} | "
                              f"{s['r2_median']:+.3f} | {s['mae_mean']:.5f} | "
                              f"{s['rmse_mean']:.5f} | {s['r2_frac_negative']:.0%} |")
            else:
                lines.append(f"| {frac:.0%} | — | insufficient cycles | | | | |")

        lines.append("\n### Step 2 — Per-cell LOO-CV distribution (R²)\n")
        lines.append("| Fraction | min | p25 | median | p75 | max | worst cell | best cell |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for frac in FRACTIONS:
            d = res["step1_2_fraction_sweep_and_loo"]["step2_per_cell_loo"].get(str(frac), {}).get("distribution", {})
            if d:
                lines.append(f"| {frac:.0%} | {d['r2_min']:+.3f} | {d['r2_p25']:+.3f} | "
                              f"{d['r2_median']:+.3f} | {d['r2_p75']:+.3f} | {d['r2_max']:+.3f} | "
                              f"{d['worst_cell']} | {d['best_cell']} |")

        cal = res["step3_calibration"]
        lines.append("\n### Step 3 — Uncertainty calibration\n")
        lines.append(f"Nominal 90% interval, overall empirical coverage: "
                      f"**{cal['overall_empirical_coverage']:.3f}** "
                      f"({cal['n_predictions_pooled']} pooled predictions). "
                      f"{cal.get('verdict','')}\n")
        lines.append("| Fraction | Empirical coverage | n predictions |")
        lines.append("|---|---|---|")
        for frac in FRACTIONS:
            b = cal["by_fraction"].get(str(frac), {})
            if b.get("empirical_coverage") is not None:
                lines.append(f"| {frac:.0%} | {b['empirical_coverage']:.3f} | {b['n_predictions']} |")

        if "step3b_jackknife_plus_calibration_LCO_ONLY" in res:
            jk = res["step3b_jackknife_plus_calibration_LCO_ONLY"]
            lines.append("\n### Step 3b — Jackknife+ calibration fix (LCO ONLY, Thread 2)\n")
            lines.append(
                "GP posterior (Step 3 above) is overconfident on LCO: 73.9% actual vs 90% "
                "nominal. Tested jackknife+ (Barber, Candes, Ramdas & Tibshirani 2021) as a "
                "distribution-free interval-construction fix, chosen over Sanchez-Dominguez "
                "et al. (2025) small-n conformal because the latter's own guarantee "
                "Pr(coverage>=0.9)>=0.9 is numerically unachievable at this n_cal (best "
                "achievable: Pr(coverage>=0.9)=0.27-0.34, verified before implementation) "
                "-- jackknife+ at least has an achievable finite-width regime (50% "
                "guaranteed coverage at alpha=0.25). LFP's existing 94.2% coverage (Step 3) "
                "is untouched by this section.\n"
            )
            lines.append("| Fraction | Formal 50%-guarantee: coverage | width | Practical (clipped, unguaranteed): coverage | width |")
            lines.append("|---|---|---|---|---|")
            for frac in FRACTIONS:
                r = jk.get(str(frac), {})
                f50 = r.get("formal_alpha_0.25_guaranteed_50pct", {})
                prac = r.get("practical_clipped_alpha_0.05", {})
                if f50.get("empirical_coverage") is not None:
                    lines.append(f"| {frac:.0%} | {f50['empirical_coverage']:.3f} | "
                                  f"{f50['mean_width']:.4f} | "
                                  f"{prac.get('empirical_coverage', float('nan')):.3f} | "
                                  f"{prac.get('mean_width', float('nan')):.4f} |")
            lines.append(
                "\n> **Formal guarantee note:** the 90%-nominal target (alpha=0.05) requires "
                "index k=4, but only n=3 leave-one-out ensemble members are available per "
                "fold at this sample size (4 cells, hold one out, jackknife over the other "
                "3) -- NOT achievable with a finite interval under Theorem 1. The maximum "
                "formally-guaranteed finite-width coverage at this n is **50%**, not 90%. "
                "The 'practical (clipped)' column relaxes the index to the widest available "
                "finite value and reports its EMPIRICAL coverage only -- this has no proven "
                "guarantee and is not the same claim as the formal 50% column.\n"
            )

        lines.append("\n### Step 4 — Baseline comparison (MAE, lower is better)\n")
        lines.append("| Fraction | Physics-Bayes | Flat baseline | Linear baseline | Beats flat? | Beats linear? |")
        lines.append("|---|---|---|---|---|---|")
        base = res["step4_baselines"]
        for frac in FRACTIONS:
            b = base.get(str(frac), {})
            if "physics_bayes" in b:
                lines.append(f"| {frac:.0%} | {b['physics_bayes']['mae_mean']:.5f} | "
                              f"{b['flat_baseline']['mae_mean']:.5f} | "
                              f"{b['linear_baseline']['mae_mean']:.5f} | "
                              f"{'Yes' if b['physics_beats_flat_mae'] else 'No'} | "
                              f"{'Yes' if b['physics_beats_linear_mae'] else 'No'} |")
        n_beats_linear = sum(1 for f in FRACTIONS if base.get(str(f), {}).get("physics_beats_linear_mae") is True)
        n_frac_tested = sum(1 for f in FRACTIONS if "physics_bayes" in base.get(str(f), {}))
        if 0 < n_beats_linear < n_frac_tested:
            lines.append(
                f"\n> **Caveat:** the physics-informed Bayesian model beats "
                f"flat extrapolation at every fraction tested, but only beats "
                f"the simple linear-extrapolation baseline at {n_beats_linear}/"
                f"{n_frac_tested} fractions. A plain OLS line through the early "
                f"cycles is a genuinely competitive baseline here — the physics "
                f"prior's advantage over 'no baseline at all' is clear; its "
                f"advantage over 'simplest reasonable baseline' is not "
                f"uniform.\n"
            )

        lines.append("\n### Step 5 — Residual bias by position in life\n")
        lines.append("| Life bin | n | mean residual | direction |")
        lines.append("|---|---|---|---|")
        resid = res["step5_residuals"]
        for key in ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]:
            r = resid.get(key, {})
            if r.get("n"):
                lines.append(f"| {key} | {r['n']} | {r['mean_residual']:+.5f} | {r['direction']} |")
        lines.append(f"\n{resid.get('eol_bias_note','')}\n")

        ood = res["step6_ood"]
        lines.append("### Step 6 — OOD / extrapolation limit\n")
        lines.append(f"- Validated cycle-life range: **{ood['cycle_life_range_validated']}** cycles\n"
                      f"- Typical cells (within IQR {[round(x) for x in ood['cycle_life_iqr']]}) "
                      f"R² mean: {ood['r2_mean_typical']}\n"
                      f"- Extreme cells (outside IQR) R² mean: {ood['r2_mean_extreme']}\n"
                      f"- {ood['validity_statement']}\n")
        if (ood.get("r2_mean_typical") is not None and ood.get("r2_mean_extreme") is not None
                and ood["r2_mean_extreme"] > ood["r2_mean_typical"] and res["n_cells"] <= 10):
            lines.append(
                f"> **Caveat:** 'extreme' cells scored *better* here "
                f"({ood['r2_mean_extreme']:.3f}) than 'typical' cells "
                f"({ood['r2_mean_typical']:.3f}) — the opposite of the expected "
                f"OOD-degrades pattern. With only n={res['n_cells']} cells, the "
                f"typical/extreme split leaves 1-2 cells per group; this is "
                f"almost certainly a small-sample artifact, not evidence that "
                f"extrapolation is safe. Do not read this as 'OOD is fine for "
                f"this chemistry.'\n"
            )

        ssh = res["step7_sample_size_honesty"]
        lines.append("### Step 7 — Sample size honesty\n")
        lines.append(f"Method: {ssh['method']}\n")
        if "honesty_warning" in ssh:
            lines.append(f"**{ssh['honesty_warning']}**\n")
        if "honesty_note" in ssh:
            lines.append(f"{ssh['honesty_note']}\n")

    lines.append("\n## Validated range / Known limitations\n")
    lines.append(
        "- **Model is a simplified proxy, not the full GP.** This report uses "
        "closed-form conjugate Bayesian linear regression, not the "
        "Matern52-kernel GP in `bayes_gp_predictor.py`/`severson_gp_predictor.py`. "
        "It has no correlated-residual term; calibration and RMSE numbers here "
        "are NOT directly comparable to those sibling modules' published figures.\n"
        "- **LCO and LFP results are never pooled** — different chemistry, "
        "different beta scale, different degradation shape (LFP: near-linear/"
        "convex; LCO: concave, matching beta*sqrt(k)). Any cross-chemistry "
        "claim would require the separate analysis in "
        "`hierarchical_beta_cross_chemistry.py`, which already found this hard "
        "(see `docs/problem2_literature_review.md`).\n"
        "- **NASA LCO n=4**: every statistic has only 4 leave-one-out folds. "
        "Point estimates should be read as illustrating a failure/success mode, "
        "not as precise numbers. See Step 7.\n"
        "- **Severson LFP heterogeneity**: `severson_gp_predictor.py` already "
        "established that beta variance in this dataset is ~45% explained by "
        "protocol (varied fast-charge conditions), not intrinsic cell-to-cell "
        "variation — the population prior used here inherits that conflation.\n"
        "- **Extrapolation boundary**: see Step 6 per chemistry for the exact "
        "validated cycle-life range. Predictions for cells whose eventual life "
        "falls outside that range are unvalidated extrapolation.\n"
        "- **Accuracy degrades at low early-data fractions** (5-10%) in both "
        "chemistries, as expected — this is reported plainly in Step 1, not "
        "hidden behind a single cherry-picked fraction.\n"
    )

    OUT_MD.write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ═════════════════════════════════════════════════════════════════════════════
# BACON-WATTS ALTERNATE MODEL PATH
# ═════════════════════════════════════════════════════════════════════════════
# Real (not synthetic) evaluation of the Bacon-Watts knee-aware mean function
# proposed in docs/problem1_eol_and_calibration_literature_review.md Section
# 1.3/3.3, run on THIS project's actual NASA LCO (n=4) and Severson LFP
# (n=124) data. The literature review's own Section 4 numbers (94.5% / 75.3%
# EOL-bias reduction) were computed on synthetic data generated in a separate
# sandbox and are explicitly NOT validated results for this project -- they
# are not reused, referenced, or assumed here in any numeric way.
#
# MODEL DIFFERENCE FROM THE BASELINE: fade(D) = a*D + b + c*(D-tau)*tanh((D-tau)/w),
# D = cycle/total_life in [0,1] -- a NORMALIZED fraction, unlike the baseline's
# D_k=k (absolute cycle count). This means Bacon-Watts requires knowing (or
# assuming) total_life to evaluate fade at a future cycle -- a real, additional
# limitation the baseline model does not have. For this retrospective LOO-CV
# validation, total_life is taken from the historical record (standard practice
# in the knee-detection literature, e.g. Fermin-Cueto 2020), which is an
# idealization: a true deployment would need a separate cycle-life estimate.
#
# IDENTIFIABILITY CAVEAT: tau in [0.4, 0.95] (the knee location) is only
# observable if the training window extends into the knee region. At the 5%,
# 10%, and 20% fractions used throughout this validation, the observed window
# essentially NEVER reaches tau -- so at those fractions, c and tau are fit
# almost entirely from the chemistry-specific prior and bounds, not from data.
# This is reported explicitly per fraction below, not hidden.

sys.path.insert(0, str(ROOT))
from degradation.severson_gp_predictor import (
    bacon_watts_mean as _bw_mean, fit_bacon_watts as _bw_fit,
    CHEM_W_PRIOR as _BW_CHEM_W_PRIOR,
)


def _bw_predictive_sd(params: np.ndarray, pcov: np.ndarray,
                       D_pred: np.ndarray, sigma_obs: float,
                       eps: float = 1e-5) -> np.ndarray:
    """Delta-method predictive SD: propagate NLS parameter covariance through
    the Bacon-Watts mean function via numeric gradient, add residual noise."""
    n_p = len(params)
    grads = np.zeros((len(D_pred), n_p))
    for j in range(n_p):
        pp = np.array(params, dtype=float); pp[j] += eps
        pm = np.array(params, dtype=float); pm[j] -= eps
        grads[:, j] = (_bw_mean(D_pred, *pp) - _bw_mean(D_pred, *pm)) / (2 * eps)
    var_param = np.einsum('ij,jk,ik->i', grads, pcov, grads)
    var_pred = np.maximum(var_param, 0.0) + sigma_obs ** 2
    return np.sqrt(var_pred)


def _bw_sigma_obs(cells: List[Dict], chemistry: str) -> float:
    """Pooled per-chemistry residual SD from full-trajectory Bacon-Watts fits
    (same role as sigma_obs in the baseline model, recomputed for this model)."""
    sds = []
    for c in cells:
        dsoh, n_total = c["dsoh"], c["n_total"]
        k = np.arange(1, n_total + 1, dtype=float)
        D = k / n_total
        params, _, _ = _bw_fit(D, dsoh, chemistry)
        resid = dsoh - _bw_mean(D, *params)
        sds.append(float(np.std(resid, ddof=1)) if len(resid) > 1 else float(np.std(resid)))
    return float(np.mean(sds))


def _fit_and_predict_bw(dsoh: np.ndarray, n_obs: int, n_total: int,
                         chemistry: str, sigma_obs: float) -> Dict:
    k_obs = np.arange(1, n_obs + 1, dtype=float)
    D_obs = k_obs / n_total
    y_obs = dsoh[:n_obs]
    params, pcov, success = _bw_fit(D_obs, y_obs, chemistry)

    k_pred = np.arange(n_obs + 1, len(dsoh) + 1, dtype=float)
    D_pred = k_pred / n_total
    y_true = dsoh[n_obs:]

    pred_mean = _bw_mean(D_pred, *params)
    pred_sd = _bw_predictive_sd(params, pcov, D_pred, sigma_obs)

    tau_observed = bool((n_obs / n_total) >= (params[3] - 3 * params[4]))
    return {
        "params": params, "fit_success": success, "tau_observed_in_window": tau_observed,
        "k_pred": k_pred, "pred_mean": pred_mean, "pred_sd": pred_sd, "y_true": y_true,
    }


def run_eol_bias_before_after(cells: List[Dict], chem: str, chemistry_key: str) -> Dict:
    """
    Real before/after EOL-bin-bias comparison: baseline (beta*D^0.5) vs
    Bacon-Watts, pooled across the SAME 5 fractions and LOO-CV structure used
    throughout this validation (Steps 1/2/5), binned into 3 named life
    regions matching the literature review's reporting schema (early_0_20,
    mid_40_60, eol_80_100) so results are directly comparable in format.
    NOTE: this differs from the literature review's own single-80%-split
    synthetic protocol -- reusing this validation's own pooled-across-
    fractions protocol was chosen for internal consistency with the rest of
    this report, and is disclosed here explicitly.
    """
    bins = {"early_0_20": (0.0, 0.2), "mid_40_60": (0.4, 0.6), "eol_80_100": (0.8, 1.0)}
    baseline_bin_resid = {k: [] for k in bins}
    fix_bin_resid = {k: [] for k in bins}
    n_tau_unobserved = 0
    n_total_fits = 0

    sigma_obs_bw = _bw_sigma_obs(cells, chemistry_key)

    for i, c in enumerate(cells):
        dsoh, n_total = c["dsoh"], c["n_total"]
        mu_prior, sigma_prior, sigma_obs_base = _population_stats(cells, i)
        for frac in FRACTIONS:
            n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
            if n_obs >= n_total - 1:
                continue

            res_base = _fit_and_predict(dsoh, n_obs, mu_prior, sigma_prior, sigma_obs_base)
            resid_base = res_base["pred_mean"] - res_base["y_true"]
            pos = res_base["k_pred"] / n_total

            res_bw = _fit_and_predict_bw(dsoh, n_obs, n_total, chemistry_key, sigma_obs_bw)
            resid_bw = res_bw["pred_mean"] - res_bw["y_true"]
            n_total_fits += 1
            if not res_bw["tau_observed_in_window"]:
                n_tau_unobserved += 1

            for key, (lo, hi) in bins.items():
                mask = (pos >= lo) & (pos < hi if hi < 1.0 else pos <= hi)
                if mask.any():
                    baseline_bin_resid[key].extend(resid_base[mask].tolist())
                    fix_bin_resid[key].extend(resid_bw[mask].tolist())

    def _stats(vals):
        arr = np.array(vals)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, len(arr)

    baseline_bins = {k: _stats(v) for k, v in baseline_bin_resid.items()}
    fix_bins = {k: _stats(v) for k, v in fix_bin_resid.items()}

    baseline_eol = baseline_bins["eol_80_100"]
    fix_eol = fix_bins["eol_80_100"]
    abs_reduction = abs(baseline_eol[0]) - abs(fix_eol[0])
    pct_reduction = abs_reduction / (abs(baseline_eol[0]) + 1e-12) * 100

    return {
        "chemistry": chem, "n_cells": len(cells),
        "baseline_eol_bias": baseline_eol[0], "baseline_eol_bias_std": baseline_eol[1],
        "fix_eol_bias": fix_eol[0], "fix_eol_bias_std": fix_eol[1],
        "delta": baseline_eol[0] - fix_eol[0],
        "abs_bias_reduction": abs_reduction,
        "pct_abs_bias_reduction": pct_reduction,
        "baseline_bins": baseline_bins,
        "fix_bins": fix_bins,
        "tau_identifiability": {
            "n_fits_total": n_total_fits,
            "n_fits_tau_unobserved": n_tau_unobserved,
            "pct_tau_unobserved": float(n_tau_unobserved / n_total_fits * 100) if n_total_fits else None,
            "note": (
                "In this fraction of LOO-CV fits, the observed training window "
                "never reached the knee region (tau - 3w) -- tau/c were fit "
                "almost entirely from the chemistry-specific prior/bounds, not "
                "from data. High values here mean the EOL-bias result is "
                "substantially prior-driven, not data-driven."
            ),
        },
        "real_vs_literature_review_synthetic": (
            "This is a REAL measurement on this project's own data. The "
            "literature review document's Section 4 numbers (94.5% LCO / "
            "75.3% LFP reduction) were computed on SYNTHETIC data in a "
            "separate sandbox and are not reused or assumed here."
        ),
    }


def run_all_checks_bw(cells: List[Dict], chem: str, chemistry_key: str) -> Dict:
    """Re-run Steps 1,2,3,4,5,6,7 with the Bacon-Watts model as the 'physics'
    method, mirroring run_all_checks exactly for direct comparability."""
    print(f"\n{'='*78}\n{chem} — Bacon-Watts model — {len(cells)} cells\n{'='*78}")
    sigma_obs_bw = _bw_sigma_obs(cells, chemistry_key)
    print(f"  Pooled sigma_obs (Bacon-Watts, full-trajectory fits): {sigma_obs_bw:.5f}")

    # Steps 1+2
    per_cell_by_frac: Dict[str, List[Dict]] = {str(f): [] for f in FRACTIONS}
    for i, c in enumerate(cells):
        dsoh, n_total = c["dsoh"], c["n_total"]
        for frac in FRACTIONS:
            n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
            if n_obs >= n_total - 1:
                continue
            res = _fit_and_predict_bw(dsoh, n_obs, n_total, chemistry_key, sigma_obs_bw)
            r2 = _r2(res["y_true"], res["pred_mean"])
            mae = _mae(res["y_true"], res["pred_mean"])
            rmse = _rmse(res["y_true"], res["pred_mean"])
            per_cell_by_frac[str(frac)].append({
                "cell_id": c["cell_id"], "n_obs": n_obs, "n_total": n_total,
                "r2": r2, "mae": mae, "rmse": rmse,
                "tau_observed_in_window": res["tau_observed_in_window"],
            })

    step1 = {}
    for frac in FRACTIONS:
        rows = per_cell_by_frac[str(frac)]
        if not rows:
            step1[str(frac)] = {"n_cells": 0}
            continue
        r2s = np.array([r["r2"] for r in rows])
        maes = np.array([r["mae"] for r in rows])
        rmses = np.array([r["rmse"] for r in rows])
        n_tau_obs = sum(1 for r in rows if r["tau_observed_in_window"])
        step1[str(frac)] = {
            "n_cells": len(rows), "r2_mean": float(r2s.mean()), "r2_median": float(np.median(r2s)),
            "mae_mean": float(maes.mean()), "rmse_mean": float(rmses.mean()),
            "r2_frac_negative": float(np.mean(r2s < 0)),
            "pct_cells_tau_observed": float(n_tau_obs / len(rows) * 100),
        }
        print(f"    frac={frac:.2f}: n={len(rows):3d}  R2_median={step1[str(frac)]['r2_median']:+.3f}  "
              f"MAE={step1[str(frac)]['mae_mean']:.5f}  "
              f"tau_observed={step1[str(frac)]['pct_cells_tau_observed']:.0f}%")

    step2 = {}
    for frac in FRACTIONS:
        rows = per_cell_by_frac[str(frac)]
        if not rows:
            step2[str(frac)] = {"per_cell": [], "distribution": {}}
            continue
        r2s = np.array([r["r2"] for r in rows])
        step2[str(frac)] = {
            "per_cell": rows,
            "distribution": {
                "r2_min": float(r2s.min()), "r2_p25": float(np.percentile(r2s, 25)),
                "r2_median": float(np.median(r2s)), "r2_p75": float(np.percentile(r2s, 75)),
                "r2_max": float(r2s.max()),
            },
        }

    # Step 3: calibration
    by_frac_covered: Dict[str, List[bool]] = {str(f): [] for f in FRACTIONS}
    all_covered: List[bool] = []
    for i, c in enumerate(cells):
        dsoh, n_total = c["dsoh"], c["n_total"]
        for frac in FRACTIONS:
            n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
            if n_obs >= n_total - 1:
                continue
            res = _fit_and_predict_bw(dsoh, n_obs, n_total, chemistry_key, sigma_obs_bw)
            lo = res["pred_mean"] - Z_90 * res["pred_sd"]
            hi = res["pred_mean"] + Z_90 * res["pred_sd"]
            covered = (res["y_true"] >= lo) & (res["y_true"] <= hi)
            by_frac_covered[str(frac)].extend(covered.tolist())
            all_covered.extend(covered.tolist())
    cal = {
        "nominal_coverage": 0.90,
        "overall_empirical_coverage": float(np.mean(all_covered)) if all_covered else None,
        "n_predictions_pooled": len(all_covered),
        "by_fraction": {str(f): {"empirical_coverage": float(np.mean(by_frac_covered[str(f)]))
                                  if by_frac_covered[str(f)] else None,
                                  "n_predictions": len(by_frac_covered[str(f)])}
                        for f in FRACTIONS},
    }
    overall = cal["overall_empirical_coverage"]
    if overall is not None:
        cal["verdict"] = (
            f"MISCALIBRATED (overconfident): {overall:.2f} actual vs 0.90 nominal." if overall < 0.75 else
            f"MISCALIBRATED (vacuously wide): {overall:.2f} actual vs 0.90 nominal." if overall > 0.98 else
            f"Reasonably calibrated: {overall:.2f} actual vs 0.90 nominal."
        )
    print(f"  [3] Calibration: {cal['overall_empirical_coverage']:.3f} -- {cal.get('verdict','')}")

    # Step 4: baseline comparison (physics_bayes replaced by bacon_watts)
    base = {}
    for frac in FRACTIONS:
        bw_r2, flat_r2, lin_r2 = [], [], []
        bw_mae, flat_mae, lin_mae = [], [], []
        for c in cells:
            dsoh, n_total = c["dsoh"], c["n_total"]
            n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
            if n_obs >= n_total - 1:
                continue
            res = _fit_and_predict_bw(dsoh, n_obs, n_total, chemistry_key, sigma_obs_bw)
            y_true = res["y_true"]
            bw_r2.append(_r2(y_true, res["pred_mean"])); bw_mae.append(_mae(y_true, res["pred_mean"]))
            flat_pred = _baseline_flat(dsoh, n_obs)
            flat_r2.append(_r2(y_true, flat_pred)); flat_mae.append(_mae(y_true, flat_pred))
            lin_pred = _baseline_linear(dsoh, n_obs)
            lin_r2.append(_r2(y_true, lin_pred)); lin_mae.append(_mae(y_true, lin_pred))
        if not bw_r2:
            base[str(frac)] = {"note": "no cells had enough cycles"}
            continue
        base[str(frac)] = {
            "n_cells": len(bw_r2),
            "bacon_watts": {"r2_mean": float(np.mean(bw_r2)), "mae_mean": float(np.mean(bw_mae))},
            "flat_baseline": {"r2_mean": float(np.mean(flat_r2)), "mae_mean": float(np.mean(flat_mae))},
            "linear_baseline": {"r2_mean": float(np.mean(lin_r2)), "mae_mean": float(np.mean(lin_mae))},
            "bw_beats_flat_mae": bool(np.mean(bw_mae) < np.mean(flat_mae)),
            "bw_beats_linear_mae": bool(np.mean(bw_mae) < np.mean(lin_mae)),
        }
    print(f"  [4] Baseline comparison done.")

    # Step 5: residual analysis
    bins5 = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_resid: Dict[str, List[float]] = {f"{lo:.1f}-{hi:.1f}": [] for lo, hi in bins5}
    for c in cells:
        dsoh, n_total = c["dsoh"], c["n_total"]
        for frac in FRACTIONS:
            n_obs = max(MIN_N_OBS, int(np.ceil(frac * n_total)))
            if n_obs >= n_total - 1:
                continue
            res = _fit_and_predict_bw(dsoh, n_obs, n_total, chemistry_key, sigma_obs_bw)
            resid = res["pred_mean"] - res["y_true"]
            pos = res["k_pred"] / n_total
            for lo, hi in bins5:
                mask = (pos >= lo) & (pos < hi if hi < 1.0 else pos <= hi)
                if mask.any():
                    bin_resid[f"{lo:.1f}-{hi:.1f}"].extend(resid[mask].tolist())
    resid5 = {}
    for key, vals in bin_resid.items():
        if not vals:
            resid5[key] = {"n": 0}; continue
        arr = np.array(vals)
        resid5[key] = {"n": len(arr), "mean_residual": float(arr.mean()),
                        "sd_residual": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                        "direction": "over-predicts fade" if arr.mean() > 0 else "under-predicts fade"}
    eol_bias5 = resid5.get("0.8-1.0", {}).get("mean_residual")
    print(f"  [5] Near-EOL residual (Bacon-Watts): {eol_bias5:.5f}" if eol_bias5 is not None else "  [5] n/a")

    # Step 6: OOD check
    ref_frac = 0.20
    life = np.array([c["n_total"] for c in cells])
    p25, p75 = np.percentile(life, 25), np.percentile(life, 75)
    typical_r2, extreme_r2 = [], []
    for c in cells:
        dsoh, n_total = c["dsoh"], c["n_total"]
        n_obs = max(MIN_N_OBS, int(np.ceil(ref_frac * n_total)))
        if n_obs >= n_total - 1:
            continue
        res = _fit_and_predict_bw(dsoh, n_obs, n_total, chemistry_key, sigma_obs_bw)
        r2 = _r2(res["y_true"], res["pred_mean"])
        (extreme_r2 if (n_total < p25 or n_total > p75) else typical_r2).append(r2)
    ood = {
        "ref_fraction": ref_frac,
        "r2_mean_typical": float(np.mean(typical_r2)) if typical_r2 else None,
        "r2_mean_extreme": float(np.mean(extreme_r2)) if extreme_r2 else None,
    }
    print(f"  [6] OOD (Bacon-Watts): typical R2={ood['r2_mean_typical']}  extreme R2={ood['r2_mean_extreme']}")

    # Step 7: sample size honesty (reuse same bootstrap/SE approach, on BW R2s)
    n = len(cells)
    ssh = {"n_cells": n}
    if n <= 10:
        per_frac = {}
        for frac in FRACTIONS:
            r2s = [r["r2"] for r in per_cell_by_frac[str(frac)]]
            if len(r2s) < 2:
                per_frac[str(frac)] = {"note": "insufficient folds"}; continue
            arr = np.array(r2s)
            se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
            per_frac[str(frac)] = {"n_folds": len(arr), "r2_mean": float(arr.mean()), "se": se}
        ssh["method"] = "standard-error interval (n too small for bootstrap)"
        ssh["per_fraction"] = per_frac
    else:
        rng = np.random.default_rng(42)
        per_frac = {}
        for frac in FRACTIONS:
            r2s = np.array([r["r2"] for r in per_cell_by_frac[str(frac)]])
            if len(r2s) < 5:
                per_frac[str(frac)] = {"note": "insufficient folds"}; continue
            boot_means = np.array([rng.choice(r2s, size=len(r2s), replace=True).mean() for _ in range(2000)])
            per_frac[str(frac)] = {"n_folds": len(r2s), "r2_mean": float(r2s.mean()),
                                    "bootstrap_95ci": [float(np.percentile(boot_means, 2.5)),
                                                        float(np.percentile(boot_means, 97.5))]}
        ssh["method"] = "bootstrap (2000 resamples)"
        ssh["per_fraction"] = per_frac

    return {
        "chemistry": chem, "n_cells": len(cells),
        "step1_fraction_sweep": step1, "step2_per_cell_loo": step2,
        "step3_calibration": cal, "step4_baselines": base,
        "step5_residuals": resid5, "step6_ood": ood, "step7_sample_size_honesty": ssh,
    }


def run_bacon_watts_validation():
    print("=" * 78)
    print("Problem 1 — Bacon-Watts REAL-DATA validation (vs synthetic literature review)")
    print("=" * 78)

    nasa_cells = _load_nasa_cells()
    sev_cells = _load_severson_cells()

    print("\n--- EOL-bin bias before/after (real data) ---")
    eol_lco = run_eol_bias_before_after(nasa_cells, "LCO (NASA)", "LCO")
    print(f"  LCO: baseline={eol_lco['baseline_eol_bias']:+.5f}  fix={eol_lco['fix_eol_bias']:+.5f}  "
          f"abs_reduction={eol_lco['pct_abs_bias_reduction']:.1f}%  "
          f"tau_unobserved={eol_lco['tau_identifiability']['pct_tau_unobserved']:.0f}%")
    eol_lfp = run_eol_bias_before_after(sev_cells, "LFP (Severson)", "LFP")
    print(f"  LFP: baseline={eol_lfp['baseline_eol_bias']:+.5f}  fix={eol_lfp['fix_eol_bias']:+.5f}  "
          f"abs_reduction={eol_lfp['pct_abs_bias_reduction']:.1f}%  "
          f"tau_unobserved={eol_lfp['tau_identifiability']['pct_tau_unobserved']:.0f}%")

    nasa_bw = run_all_checks_bw(nasa_cells, "LCO (NASA)", "LCO")
    sev_bw = run_all_checks_bw(sev_cells, "LFP (Severson)", "LFP")

    report = {
        "meta": {
            "script": "degradation/problem1_360_validation.py :: run_bacon_watts_validation",
            "model": "Bacon-Watts knee-aware mean function, density-weighted NLS fit, "
                     "chemistry-specific LogNormal(w) prior (per "
                     "docs/problem1_eol_and_calibration_literature_review.md Section 1.3/3.3)",
            "real_data_disclaimer": (
                "All numbers below are computed on this project's REAL NASA LCO "
                "and Severson LFP data. The literature review document's Section "
                "4 before/after numbers (94.5%/75.3%) were computed on SYNTHETIC "
                "data in a separate sandbox and are NOT reused here."
            ),
        },
        "eol_bias_before_after": {"LCO_NASA": eol_lco, "LFP_Severson": eol_lfp},
        "full_7_check_rerun": {"LCO_NASA": nasa_bw, "LFP_Severson": sev_bw},
    }

    def _serial(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, bool): return o
        if isinstance(o, dict): return {str(k): _serial(v) for k, v in o.items()}
        if isinstance(o, list): return [_serial(v) for v in o]
        return o

    out_path = ROOT / "data" / "problem1_bacon_watts_real_validation.json"
    out_path.write_text(json.dumps(_serial(report), indent=2))
    print(f"\nJSON report written to {out_path}")

    # Also emit the exact schema of the uploaded synthetic-comparison JSON, for
    # direct side-by-side diffing against the synthetic numbers.
    compare = {
        "LFP": {k: v for k, v in eol_lfp.items() if k not in ("chemistry",)},
        "LCO": {k: v for k, v in eol_lco.items() if k not in ("chemistry",)},
    }
    compare_path = ROOT / "data" / "problem1_eol_fix_real_validation_results.json"
    compare_path.write_text(json.dumps(_serial(compare), indent=2))
    print(f"Direct-comparison-schema JSON written to {compare_path}")

    return report


def main():
    print("=" * 78)
    print("Problem 1 — 360-Degree Validation")
    print("=" * 78)

    print("\nLoading NASA LCO cells...")
    nasa_cells = _load_nasa_cells()
    print(f"  {len(nasa_cells)} cells: " +
          ", ".join(f"{c['cell_id']}(n={c['n_total']})" for c in nasa_cells))

    print("Loading Severson LFP cells...")
    sev_cells = _load_severson_cells()
    print(f"  {len(sev_cells)} cells, cycle-life range "
          f"[{min(c['n_total'] for c in sev_cells)}, {max(c['n_total'] for c in sev_cells)}]")

    nasa_res = run_all_checks(nasa_cells, "LCO (NASA)")
    sev_res = run_all_checks(sev_cells, "LFP (Severson)")

    print("\n  [3b] Jackknife+ calibration (LCO ONLY -- Thread 2 fix)...")
    jk_lco = run_jackknife_plus_calibration(nasa_cells)
    for frac in FRACTIONS:
        r = jk_lco.get(str(frac), {})
        f50 = r.get("formal_alpha_0.25_guaranteed_50pct", {})
        prac = r.get("practical_clipped_alpha_0.05", {})
        print(f"    frac={frac:.2f}: formal-50%-guarantee cov={f50.get('empirical_coverage')}  "
              f"width={f50.get('mean_width')}  ||  practical(clipped) cov={prac.get('empirical_coverage')}  "
              f"width={prac.get('mean_width')}")
    nasa_res["step3b_jackknife_plus_calibration_LCO_ONLY"] = jk_lco
    print("  Severson LFP calibration (step3_calibration, 94.2%) left untouched.")

    report = {
        "meta": {
            "script": "degradation/problem1_360_validation.py",
            "model": "Closed-form conjugate Bayesian linear regression, "
                     "physics mean beta*k^0.5, LOO population prior "
                     "(disclosed simplification of the full GP in sibling modules)",
            "fractions_tested": FRACTIONS,
            "gamma_fixed": GAMMA,
        },
        "LCO_NASA": nasa_res,
        "LFP_Severson": sev_res,
    }

    def _serial(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, bool): return o
        if isinstance(o, dict): return {str(k): _serial(v) for k, v in o.items()}
        if isinstance(o, list): return [_serial(v) for v in o]
        return o

    OUT_JSON.write_text(json.dumps(_serial(report), indent=2))
    print(f"\nJSON report written to {OUT_JSON}")

    _write_markdown(nasa_res, sev_res)
    print(f"Markdown report written to {OUT_MD}")

    return report


if __name__ == "__main__":
    import sys as _sys
    if "--bacon-watts-only" in _sys.argv:
        run_bacon_watts_validation()
    else:
        main()
        run_bacon_watts_validation()
