#!/usr/bin/env python3
"""
degradation/severson_gp_predictor.py
=====================================
Physics-informed Bayesian GP — early-cycle degradation prediction on the
Severson et al. (2019) LFP dataset.  Parallel to bayes_gp_predictor.py
(NASA LCO, n=4), but powered by n=124 cells.

SCOPE
-----
This module demonstrates the same physics-informed GP methodology as
bayes_gp_predictor.py on a dataset with adequate statistical power:
n=124 LFP cells (A123 APR18650M1A, 1.1 Ah) from Severson et al. (2019)
Nature Energy 4:383–391, single facility (MIT), varied fast-charge protocols.

The analysis is SEPARATE from the NASA LCO results and is NOT directly
comparable to them.  Different chemistry (LFP vs LCO), different V-grid
([2.0–3.5 V] vs [2.75–4.15 V]), different beta scale (~0.0025 vs ~0.022),
different facility and protocol.  No merging or cross-chemistry comparison
is performed.

PROTOCOL HETEROGENEITY (pre-flight check, Step 0)
--------------------------------------------------
OLS regression of beta_i on three numeric protocol features (fast_c, soc_pct,
slow_c) gives R²=0.452.  Combined with batch dummies: R²=0.620.  Policy
fixed effects (upper bound, 68 groups): R²=0.984.  Interpretation: the
"cell heterogeneity" in this dataset is overwhelmingly protocol-driven.  Any
hierarchical sigma_beta estimate conflates within-protocol cell variance with
between-protocol variance by design (varied charge C-rates are the
independent variable in Severson's experiment).  This is stated alongside
all heterogeneity estimates — it is not presented as intrinsic manufacturing
variation.

KEY DIFFERENCES FROM NASA VERSION
----------------------------------
  - n=124 vs n=4: LOO now has real statistical power; sigma_beta and coverage
    statistics are trustworthy rather than illustrative.
  - Kernel calibration: Matern52 hyperparameters are fitted ONCE on a held-aside
    12-cell stratified calibration split (4 cells per batch), NOT per LOO fold.
    Fitting the kernel on 123 training cells per fold at full cycle counts would
    require ~17 hours per run (0.038 × 10 × 20.11 × 8000 s under empirical
    α=2.2 scaling — see benchmark note below).  The calibration-once approach
    is a named assumption: kernel covariance structure is assumed to be shared
    across cells (conditional on cell type and protocol regime).
  - Every-4th-cycle subsampling for calibration cells: reduces Cholesky matrix
    from ~783×783 to ~196×196 per cell while preserving full cycle-range
    temporal extent (cycle indices 1 to ~1934), which is critical for fitting
    the length-scale on a dataset spanning 148–1935 cycle lives.  This is NOT
    truncation — subsampled indices span the full range.
  - N values as fractions of median cycle_life (736): N ∈ {36, 110, 220}
    cycles (≈5%, 15%, 30%).  Raw absolute cycle counts are meaningless across
    a 13× range of cycle lives.
  - LOO beta prior: per-fold predictive-SD correction sqrt(1+1/123) ≈ 1.004
    (vs sqrt(1+1/3) ≈ 1.155 in the NASA 4-cell case).  At n=123, the
    population SD estimate is stable and the inflation is negligible.

COMPUTE BENCHMARK (transparency note)
---------------------------------------
Benchmark: 2 calibration cells × 200 cycles, 1 chain, 200T+200D.
  Cold start (PyTensor compile not cached): 15.2s / 400 iters = 0.038 s/iter
  Warm cache (second run, same config):     20.0s / 800 iters = 0.025 s/iter
  Actual NUTS sampling speed (progress bar, both runs): ~50 draws/s = 0.020 s/iter
The 34% discrepancy between cold and warm timing is explained by fixed PyTensor
graph-compilation overhead (~7s cold, ~4s warm-chain-1) amortized over
different iteration counts.  For 8000-iteration calibration runs, this overhead
is <0.1% of runtime.

Projection for calibration run (12 cells, n_eff=196/cell, 4 chains, 2000 iters):
  t(n_cal, n_eff) = T_BASE × (n_cal/2) × (n_eff/200)^2.2
  Pessimistic (T_BASE=0.038): ~29 min
  Central     (T_BASE=0.025): ~19 min
  Optimistic  (sampling-only): ~15 min
All three within the 30–60 min budget.  T_BASE=0.038 used as conservative bound.

NAMED LIMITATIONS (same as bayes_gp_predictor.py — inherited and confirmed)
-----------------------------------------------------------------------------
  1. Kernel amplitude, length-scale, and sigma_obs are estimated from calibration
     cells' within-cell residual structure only.  They do not capture between-cell
     systematic deviation on held-out cells.  Coverage will degrade at higher N
     as within-cell likelihood dominates — convergence of the kernel posterior is
     not the same as calibration of the uncertainty model.
  2. The calibration kernel is fixed across all LOO folds.  This is an assumption:
     cells with anomalous protocol assignments (e.g. Batch 2 slow-charge outliers)
     may have different covariance structures than the calibration-set average.
     Results for cells whose policy deviates strongly from calibration-set policies
     should be interpreted with additional caution.
  3. n=124 cells, single facility, single chemistry, varied protocols at MIT only.
     Generalisation to other LFP cell types, other facilities, or other chemistries
     is not supported by this analysis.
  4. NAMED LIMITATION (LFP-specific): the physics mean function m(k) = beta*sqrt(k)
     assumes concave (fast-early, decelerating-late) degradation, which holds for
     the NASA LCO cells (see bayes_gp_predictor.py) but does NOT hold for this
     Severson LFP population.  Residual analysis shows the opposite curvature
     (mean residual -0.019 in the first 10% of life, +0.086 in the last 10%),
     consistent with LFP's known slow-linear-then-knee degradation shape.
     Physics-mean GP underperforms a zero-mean baseline at every N tested
     (RMSE ratio 1.49-2.21x worse; coverage 75-80% vs 90% nominal) as a direct
     consequence.  This is a genuine finding about model-chemistry mismatch, not
     a code defect: a physics mean function must match the target chemistry's
     degradation shape to provide any benefit over a data-only GP.

PRE-REGISTERED EXPECTATIONS (stated before any run)
-----------------------------------------------------
  (a) Hierarchical mu_beta / sigma_beta: expect ESS_min ≥ 3000, R-hat ≤ 1.001.
      sigma_beta HDI will be tighter relative to its mean than the NASA result
      because n=124 overwhelms the prior.
  (b) Coverage at N=36 (5%): physics prior dominates likelihood at 36 obs;
      expect 80–93% coverage (nominal 90%).
  (c) Coverage at N=110 (15%) and N=220 (30%): expect degradation to 55–75%,
      reproducing the NASA N≥50 failure mode (same named limitation applies).
  (d) Batch-stratified coverage: Batch 2 cells (highest mean beta, widest
      protocol spread) expected to show lower coverage — analogous to B0006
      stress test in NASA version but now population-level, not one cell.
  (e) Honest scope: this demonstrates that the methodology scales to n=124
      and produces trustworthy population estimates.  It does NOT resolve
      cross-chemistry or cross-facility generalisation.

Dataset:   Severson et al. (2019) Nature Energy 4:383–391
Citation:  Severson KA et al., "Data-driven prediction of battery cycle life
           before capacity degradation", Nature Energy 4, 383–391 (2019).
           https://doi.org/10.1038/s41560-019-0356-8
Source:    https://data.matr.io/1/ (registration required)
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from scipy.linalg import cho_factor, cho_solve

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent

# ── Constants ─────────────────────────────────────────────────────────────────

CELLS_PER_BATCH   = 4          # calibration cells per batch (4 × 3 batches = 12 total)
CALIB_SUBSAMPLE_K = 4          # every-Kth-cycle subsampling for calibration
N_FRACS           = [0.05, 0.15, 0.30]   # early-cycle fractions of median cycle_life
MEDIAN_CYCLE_LIFE = 736        # median cycle_life across 124 cells (from loader)
N_VALUES          = [max(5, int(MEDIAN_CYCLE_LIFE * f)) for f in N_FRACS]  # [36, 110, 220]
COVERAGE_NOMINAL  = 0.90

# Kernel calibration priors — verified against empirical residual SDs (Item 4 pre-flight).
# Calibration-cell OLS residual SDs: mean=0.033, max=0.049.
# HalfNormal(0.005) 95th pct ≈ 0.010 — does NOT cover observed range. CORRECTED below.
# HalfNormal(0.05) 95th pct ≈ 0.098 — comfortably covers max=0.049 for both terms.
# Amplitude and sigma_obs each explain roughly half the residual variance (split unknown);
# priors are kept equal and weakly informative at this scale.
AMPLITUDE_PRIOR_SIGMA = 0.05   # HalfNormal; 95th pct=0.098 covers empirical max resid_sd=0.049
SIGMA_OBS_PRIOR       = 0.05   # HalfNormal; same reasoning
LS_PRIOR_ALPHA        = 2      # Gamma shape
# LS_PRIOR_SCALE is NOT the operative value at runtime — _calibrate_kernel() receives
# ls_scale computed from ACF decay lags on non-calibration cells (no leakage, done in main()).
# This constant documents the data-derived value for reference only; it is not used
# unless explicitly passed to _calibrate_kernel() as ls_prior_scale.
LS_PRIOR_SCALE_REF    = 98     # reference: median ACF 1/e decay lag across 112 non-calib cells

N_CHAINS     = 4
N_TUNE       = 1000
N_DRAWS      = 1000
TARGET_AC    = 0.99
N_PRED_SAMPS = 2000

CALIB_SEED   = 42   # for stratified cell selection reproducibility
SAMPLE_SEED  = 42


# ── Loader import ─────────────────────────────────────────────────────────────

import sys as _sys
_sys.path.insert(0, str(ROOT))
from data.loaders.severson_loader import load_severson  # noqa: E402


# ── Physics mean function (same as bayes_gp_predictor.py) ─────────────────────

class _PhysicsMeanFn(pm.gp.mean.Mean):
    """m(k) = beta * sqrt(k)  (unit-cycle damage, no intercept)."""
    def __init__(self, beta):
        pm.gp.mean.Mean.__init__(self)
        self.beta = beta

    def __call__(self, X):
        return self.beta * pt.sqrt(X[:, 0])


# ── Knee-aware Bacon-Watts mean function (standalone, not yet wired into the ──
# ── PyMC marginal_likelihood pipeline above -- see problem1_360_validation.py ──
# ── for the applied fit/predict/evaluate loop that actually uses this)      ──
#
# Source: Fermin-Cueto et al. (2020) Energy and AI 1:100006 (Bacon-Watts
# change-point form); Attia et al. (2022) J. Electrochem. Soc. 169:060517
# ("LFP knees sharper than NMC due to flat voltage plateau" -- motivates the
# LFP-specific tight w prior below); Greenbank & Howey (2022) Mech. Syst.
# Signal Process. 184:109612 (piecewise form reduces tail/EOL error vs GPR).
# Added per docs/problem1_eol_and_calibration_literature_review.md Section 1.3
# and 3.3. That document's own before/after numbers were computed on
# SYNTHETIC data in a separate sandbox -- they are NOT validated results for
# this project. See data/problem1_360_validation_report_bacon_watts.json for
# the real measurement on this project's own NASA/Severson data.

CHEM_W_PRIOR = {"LFP": 0.03, "LCO": 0.08}   # LogNormal mean, sigma=0.5 in log space
BW_PRIOR_SIGMA_LOG = 0.5
BW_BOUNDS_LO = np.array([-2.0, -0.5, 0.0, 0.4, 0.005])
BW_BOUNDS_HI = np.array([5.0, 0.5, 3.0, 0.95, 0.5])


def bacon_watts_mean(D: np.ndarray, a: float, b: float, c: float,
                      tau: float, w: float) -> np.ndarray:
    """fade(D) = a*D + b + c*(D-tau)*tanh((D-tau)/w), D = cycle/total_life in [0,1].
    c >= 0 (curve can only accelerate, not decelerate, past the knee)."""
    D = np.asarray(D, dtype=float)
    return a * D + b + c * (D - tau) * np.tanh((D - tau) / w)


def density_weights(D: np.ndarray, h: float = 0.1) -> np.ndarray:
    """Inverse local-density weight in cycle-life-fraction space, bandwidth h.
    Sparse (typically late-life) regions get upweighted relative to dense
    (typically early-life) regions -- see literature review Section 1.3."""
    D = np.asarray(D, dtype=float)
    counts = np.array([max(1, int(np.sum(np.abs(D - d) < h))) for d in D])
    return 1.0 / counts


def fit_bacon_watts(D_obs: np.ndarray, y_obs: np.ndarray, chemistry: str):
    """
    Density-weighted, chemistry-prior-regularized NLS fit of the Bacon-Watts
    mean function. Returns (params, pcov, success) where params =
    [a, b, c, tau, w] and pcov is the (5,5) parameter covariance from the
    Jacobian at the optimum (standard NLS covariance estimate).

    The chemistry-specific w prior is enforced as an extra soft residual term
    (log(w) pulled toward log(w_prior_mean) with sigma=0.5 in log space) --
    NOT a hard constraint. At small observed fractions the knee region
    (tau in [0.4,0.95]) is often outside the observed window, so tau/c are
    weakly identified from data alone and the fit leans on this prior and on
    the initial value; this is a real identifiability limitation, not a bug,
    and is reported explicitly by problem1_360_validation.py.
    """
    from scipy.optimize import least_squares

    w_prior_mean = CHEM_W_PRIOR.get(chemistry, 0.05)
    dens_w = density_weights(D_obs, h=0.1)

    # Initial guess: simple linear OLS for a,b; conservative defaults for c,tau,w
    A = np.vstack([D_obs, np.ones_like(D_obs)]).T
    a0, b0 = np.linalg.lstsq(A, y_obs, rcond=None)[0]
    tau0 = float(np.clip(0.7, 0.4, 0.95))
    x0 = np.array([a0, b0, 0.05, tau0, w_prior_mean])
    x0 = np.clip(x0, BW_BOUNDS_LO, BW_BOUNDS_HI)

    def _resid(params):
        a, b, c, tau, w = params
        pred = bacon_watts_mean(D_obs, a, b, c, tau, w)
        data_resid = np.sqrt(dens_w) * (pred - y_obs)
        prior_resid = np.array([
            (np.log(max(w, 1e-6)) - np.log(w_prior_mean)) / BW_PRIOR_SIGMA_LOG
        ])
        return np.concatenate([data_resid, prior_resid])

    try:
        res = least_squares(_resid, x0, bounds=(BW_BOUNDS_LO, BW_BOUNDS_HI))
        J = res.jac
        resid_var = float(np.sum(res.fun ** 2)) / max(1, len(res.fun) - len(x0))
        try:
            pcov = np.linalg.inv(J.T @ J) * resid_var
        except np.linalg.LinAlgError:
            pcov = np.eye(len(x0)) * resid_var
        return res.x, pcov, bool(res.success)
    except Exception:
        return x0, np.eye(len(x0)) * 0.01, False


# ── OLS helpers ───────────────────────────────────────────────────────────────

def _ols_beta(soh: np.ndarray) -> float:
    k = np.arange(1, len(soh) + 1, dtype=float)
    x = np.sqrt(k)
    return float(np.dot(x, 1.0 - soh) / np.dot(x, x))


def _dsoh(soh: np.ndarray) -> np.ndarray:
    return 1.0 - soh


# ── Stratified calibration split ──────────────────────────────────────────────

def _select_calibration_cells(
    cells: List[Dict],
    n_per_batch: int = CELLS_PER_BATCH,
    seed: int = CALIB_SEED,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified split: n_per_batch cells randomly selected per batch.
    Returns (calibration_cells, loo_cells).
    """
    rng = np.random.default_rng(seed)
    by_batch: Dict[int, List[Dict]] = {1: [], 2: [], 3: []}
    for c in cells:
        by_batch[c["batch"]].append(c)

    calib_ids: set = set()
    for b in [1, 2, 3]:
        pool = by_batch[b]
        chosen_idx = rng.choice(len(pool), size=n_per_batch, replace=False)
        for idx in chosen_idx:
            calib_ids.add(pool[idx]["cell_id"])

    calib = [c for c in cells if c["cell_id"] in calib_ids]
    loo   = [c for c in cells if c["cell_id"] not in calib_ids]
    return calib, loo


# ── ACF length-scale calibration ──────────────────────────────────────────────

def _acf_decay_lag(soh: np.ndarray, beta: float) -> Optional[int]:
    """1/e ACF decay lag on OLS residuals."""
    k = np.arange(1, len(soh) + 1, dtype=float)
    resid = _dsoh(soh) - beta * np.sqrt(k)
    n = len(resid)
    var = np.var(resid)
    if var < 1e-12:
        return None
    target = var / np.e
    mu = resid.mean()
    for lag in range(1, n):
        c_lag = float(np.mean((resid[:n-lag] - mu) * (resid[lag:] - mu)))
        if c_lag <= target:
            return lag
    return n


# ── Matern 5/2 kernel (numpy, for Stage 2 analytical) ────────────────────────

def _matern52(x1: np.ndarray, x2: np.ndarray, ls: float) -> np.ndarray:
    d = np.abs(x1[:, None] - x2[None, :]) / ls
    return (1.0 + np.sqrt(5.0) * d + 5.0 / 3.0 * d**2) * np.exp(-np.sqrt(5.0) * d)


# ── Stage 1: calibrate kernel on calibration cells ────────────────────────────

def _calibrate_kernel(
    calib_cells: List[Dict],
    ls_prior_scale: float = LS_PRIOR_SCALE_REF,
    verbose: bool = True,
) -> Tuple[Dict, bool]:
    """
    NUTS on calibration cells (every-4th-cycle subsampled) to get
    posterior over (amplitude, ls, sigma_obs).  Chains run sequentially
    (cores=1) to avoid multiprocessing EOFError on macOS/Python 3.14.

    Returns (kernel_samples dict, nuts_converged bool).
    """
    ols_betas = {c["cell_id"]: _ols_beta(c["soh"]) for c in calib_cells}

    # Every-4th-cycle subsampled data
    resids: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for c in calib_cells:
        soh_full = c["soh"]
        idx = np.arange(0, len(soh_full), CALIB_SUBSAMPLE_K)
        soh_sub = soh_full[idx]
        k_sub   = (idx + 1).astype(float)          # 1-indexed cycle counts
        b       = ols_betas[c["cell_id"]]
        r       = _dsoh(soh_sub) - b * np.sqrt(k_sub)
        resids[c["cell_id"]] = (k_sub, r)

    if verbose:
        n_eff = int(np.mean([len(v[0]) for v in resids.values()]))
        print(f"  Calibration: {len(calib_cells)} cells, n_eff≈{n_eff} pts/cell "
              f"(every-{CALIB_SUBSAMPLE_K}th cycle)")

    t0 = time.time()
    try:
        with pm.Model() as calib_model:
            amplitude = pm.HalfNormal("amplitude", sigma=AMPLITUDE_PRIOR_SIGMA)
            ls        = pm.Gamma("ls", alpha=LS_PRIOR_ALPHA,
                                 beta=1.0 / ls_prior_scale)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=SIGMA_OBS_PRIOR)
            cov = amplitude**2 * pm.gp.cov.Matern52(1, ls=ls)
            for c in calib_cells:
                k_vec, r_vec = resids[c["cell_id"]]
                gp = pm.gp.Marginal(cov_func=cov)
                gp.marginal_likelihood(
                    f"ml_{c['cell_id']}", X=k_vec[:, None], y=r_vec, sigma=sigma_obs
                )
            idata = pm.sample(
                draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, cores=1,
                target_accept=TARGET_AC, return_inferencedata=True,
                progressbar=verbose, random_seed=SAMPLE_SEED,
            )

        elapsed = time.time() - t0
        post = idata.posterior

        amp_draws = post["amplitude"].values.flatten()
        ls_draws  = post["ls"].values.flatten()
        so_draws  = post["sigma_obs"].values.flatten()

        # Convergence check
        import pandas as pd
        summ = az.summary(idata, var_names=["amplitude", "ls", "sigma_obs"])
        r_hats = pd.to_numeric(summ["r_hat"], errors="coerce")
        ess    = pd.to_numeric(summ["ess_bulk"], errors="coerce")
        max_rhat = float(r_hats.max())
        min_ess  = float(ess.min())
        n_div    = int(idata.sample_stats.diverging.values.sum())
        converged = (max_rhat < 1.01) and (min_ess > 400) and (n_div == 0)

        if verbose:
            print(f"  Calibration elapsed: {elapsed:.0f}s  "
                  f"R-hat_max={max_rhat:.4f}  ESS_min={min_ess:.0f}  "
                  f"divergences={n_div}  "
                  f"{'CONVERGED' if converged else 'WARNING: not converged'}")

        return {
            "amplitude": amp_draws,
            "ls":        ls_draws,
            "sigma_obs": so_draws,
            "idata":     idata,
            "elapsed_s": elapsed,
            "max_rhat":  max_rhat,
            "min_ess":   min_ess,
            "divergences": n_div,
        }, converged

    except Exception as exc:
        elapsed = time.time() - t0
        if verbose:
            print(f"  NUTS FAILED after {elapsed:.0f}s: {exc}")
            print(f"  Falling back to MAP point estimate.")

        with pm.Model():
            amplitude = pm.HalfNormal("amplitude", sigma=AMPLITUDE_PRIOR_SIGMA)
            ls        = pm.Gamma("ls", alpha=LS_PRIOR_ALPHA,
                                 beta=1.0 / ls_prior_scale)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=SIGMA_OBS_PRIOR)
            cov = amplitude**2 * pm.gp.cov.Matern52(1, ls=ls)
            for c in calib_cells:
                k_vec, r_vec = resids[c["cell_id"]]
                gp = pm.gp.Marginal(cov_func=cov)
                gp.marginal_likelihood(
                    f"ml_{c['cell_id']}", X=k_vec[:, None], y=r_vec, sigma=sigma_obs
                )
            map_est = pm.find_MAP()

        n_samp = N_DRAWS * N_CHAINS
        return {
            "amplitude": np.full(n_samp, float(map_est["amplitude"])),
            "ls":        np.full(n_samp, float(map_est["ls"])),
            "sigma_obs": np.full(n_samp, float(map_est["sigma_obs"])),
            "idata":     None,
            "elapsed_s": elapsed,
            "max_rhat":  float("nan"),
            "min_ess":   float("nan"),
            "divergences": -1,
        }, False


# ── Stage 2: analytical LOO prediction ───────────────────────────────────────

def _loo_predict(
    held_out: Dict,
    N_obs: int,
    kernel_samples: Dict,
    training_betas: np.ndarray,
    physics_mean: bool = True,
) -> Dict:
    """
    Analytical conjugate prediction for one LOO fold at one N_obs value.

    Beta prior: Normal(mu_prior, sd_prior) where
      mu_prior = mean of 123 training OLS betas
      sd_prior = std(123 training betas, ddof=1) × sqrt(1 + 1/123)
    (predictive-SD correction per DeGroot & Schervish 2012 §8.6)

    GP conditional: same einsum-corrected formulation as bayes_gp_predictor.py.
    """
    soh   = held_out["soh"]
    n_tot = len(soh)
    if N_obs >= n_tot:
        return None   # not enough cycles to predict anything

    # Observed and prediction grids
    k_obs  = np.arange(1, N_obs + 1, dtype=float)
    k_pred = np.arange(N_obs + 1, n_tot + 1, dtype=float)
    n_pred = len(k_pred)

    y_obs   = _dsoh(soh[:N_obs])
    y_true  = _dsoh(soh[N_obs:])
    x_obs   = np.sqrt(k_obs)
    x_pred  = np.sqrt(k_pred)

    # LOO-consistent prior
    mu_prior   = float(training_betas.mean())
    sd_prior   = float(training_betas.std(ddof=1)) * np.sqrt(1.0 + 1.0 / len(training_betas))
    prior_prec = 1.0 / sd_prior**2

    amps = kernel_samples["amplitude"]
    lss  = kernel_samples["ls"]
    soos = kernel_samples["sigma_obs"]
    rng  = np.random.default_rng(SAMPLE_SEED)

    n_s = min(N_PRED_SAMPS, len(amps))
    idx_s = rng.choice(len(amps), size=n_s, replace=False)
    pred_draws = np.empty((n_s, n_pred))

    for i, ii in enumerate(idx_s):
        amp, ls, soo = float(amps[ii]), float(lss[ii]), float(soos[ii])
        K_oo     = amp**2 * _matern52(k_obs, k_obs, ls) + soo**2 * np.eye(N_obs)
        K_po     = amp**2 * _matern52(k_pred, k_obs, ls)
        # Stationary kernel diagonal: k(x,x)=1 for all x (d=0 → Matern52=1),
        # so diag(K_pp) = amp^2 * ones(n_pred). Avoids O(n_pred^2) matrix build.
        K_pp_diag = amp**2 * np.ones(n_pred)

        try:
            L_fac = cho_factor(K_oo, lower=True)
        except Exception:
            pred_draws[i] = np.nan
            continue

        K_oo_inv_y = cho_solve(L_fac, y_obs)

        if physics_mean:
            K_oo_inv_x = cho_solve(L_fac, x_obs)
            data_prec  = float(x_obs @ K_oo_inv_x)
            post_prec  = prior_prec + data_prec
            post_mu    = (prior_prec * mu_prior + float(x_obs @ K_oo_inv_y)) / post_prec
            beta_s     = rng.normal(post_mu, 1.0 / np.sqrt(post_prec))
            r_obs      = y_obs - beta_s * x_obs
            mu_gp_star = K_po @ cho_solve(L_fac, r_obs)
            mu_pred    = beta_s * x_pred + mu_gp_star
        else:
            mu_pred = K_po @ K_oo_inv_y

        # Corrected variance: diag(K_pp - K_po K_oo^{-1} K_po^T)
        v = cho_solve(L_fac, K_po.T)          # (N_obs, n_pred)
        var_gp = K_pp_diag - np.einsum("ij,ji->i", K_po, v)
        var_pred = np.maximum(var_gp, 0.0) + soo**2

        pred_draws[i] = rng.normal(mu_pred, np.sqrt(var_pred))

    # Drop NaN rows (Cholesky failures)
    valid = ~np.any(np.isnan(pred_draws), axis=1)
    pred_draws = pred_draws[valid]

    lo = np.quantile(pred_draws, (1 - COVERAGE_NOMINAL) / 2, axis=0)
    hi = np.quantile(pred_draws, 1 - (1 - COVERAGE_NOMINAL) / 2, axis=0)
    mu = pred_draws.mean(axis=0)

    covered    = np.mean((y_true >= lo) & (y_true <= hi))
    rmse       = float(np.sqrt(np.mean((mu - y_true)**2)))
    mae        = float(np.mean(np.abs(mu - y_true)))

    return {
        "covered":    float(covered),
        "rmse":       rmse,
        "mae":        mae,
        "n_valid_draws": int(valid.sum()),
    }


# ── Hierarchical beta model (partial pooling over 124 cells) ─────────────────

def _fit_hierarchical_beta(
    cells: List[Dict],
    verbose: bool = True,
) -> Dict:
    """
    Partial-pooling hierarchical model for beta across all 124 cells.
    mu_beta ~ Normal(0.0025, 0.005)
    sigma_beta ~ HalfNormal(0.002)
    beta_i ~ Normal(mu_beta, sigma_beta)
    sigma_obs_h ~ HalfNormal(0.05)   # empirical OLS resid SD: mean=0.037, p95=0.050, max=0.072
    dsoh_ik ~ Normal(beta_i * sqrt(k), sigma_obs_h)
    """
    import pandas as pd

    ols_betas = np.array([_ols_beta(c["soh"]) for c in cells])
    if verbose:
        print(f"  Hierarchical beta: n={len(cells)} cells, "
              f"OLS mean={ols_betas.mean():.5f}, sd={ols_betas.std(ddof=1):.5f}")

    with pm.Model() as hier_model:
        mu_beta    = pm.Normal("mu_beta", mu=0.0025, sigma=0.005)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.002)
        sigma_obs_h = pm.HalfNormal("sigma_obs_h", sigma=0.005)
        beta_raw   = pm.Normal("beta_raw", mu=0.0, sigma=1.0, shape=len(cells))
        beta_i     = pm.Deterministic("beta_i", mu_beta + sigma_beta * beta_raw)

        for j, c in enumerate(cells):
            k  = np.arange(1, len(c["soh"]) + 1, dtype=float)
            mu = beta_i[j] * pt.sqrt(pt.as_tensor_variable(k))
            pm.Normal(f"obs_{c['cell_id']}", mu=mu,
                      sigma=sigma_obs_h, observed=_dsoh(c["soh"]))

        idata_h = pm.sample(
            draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, cores=1,
            target_accept=TARGET_AC, return_inferencedata=True,
            progressbar=verbose, random_seed=SAMPLE_SEED,
        )

    post = idata_h.posterior
    mu_draws = post["mu_beta"].values.flatten()
    sb_draws = post["sigma_beta"].values.flatten()
    so_draws = post["sigma_obs_h"].values.flatten()

    summ = az.summary(idata_h, var_names=["mu_beta", "sigma_beta", "sigma_obs_h"])
    r_hats = pd.to_numeric(summ["r_hat"], errors="coerce")
    ess    = pd.to_numeric(summ["ess_bulk"], errors="coerce")
    hdi_mu = az.hdi(mu_draws, prob=0.94)
    hdi_sb = az.hdi(sb_draws, prob=0.94)
    hdi_so = az.hdi(so_draws, prob=0.94)

    return {
        "mu_beta":    float(mu_draws.mean()),
        "mu_beta_sd": float(mu_draws.std()),
        "mu_beta_hdi94": [float(hdi_mu[0]), float(hdi_mu[1])],
        "sigma_beta":    float(sb_draws.mean()),
        "sigma_beta_sd": float(sb_draws.std()),
        "sigma_beta_hdi94": [float(hdi_sb[0]), float(hdi_sb[1])],
        "sigma_obs_h":   float(so_draws.mean()),
        "sigma_obs_h_hdi94": [float(hdi_so[0]), float(hdi_so[1])],
        "max_rhat":  float(r_hats.max()),
        "min_ess":   float(ess.min()),
        "divergences": int(idata_h.sample_stats.diverging.values.sum()),
        "idata": idata_h,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import pandas as pd

    print("=" * 70)
    print("Severson GP Predictor — LFP early-cycle degradation prediction")
    print("=" * 70)

    # Load data
    print("\nLoading Severson dataset...")
    t_load = time.time()
    cells = load_severson(verbose=False)
    print(f"  {len(cells)} cells loaded in {time.time()-t_load:.1f}s")

    # SOH clip [0.0, 1.05] applied once at the data boundary, before any downstream use.
    # Severson dataset contains sensor/test artifacts: b1c0 cycle 11 (soh=1.4374),
    # b1c18 soh=2.705, and a few Batch 2 cells from reference performance tests (RPTs)
    # with anomalous capacity readings.  Upper bound 1.05 permits small measurement
    # noise above nominal but rejects gross test artifacts; lower bound 0.0 rules out
    # sign-flipped sensor readings.  Applied here so _ols_beta, _fit_hierarchical_beta,
    # and _loo_predict all receive consistent data without separate per-function patches.
    soh_clip_count = 0
    soh_clip_cells = 0
    for c in cells:
        raw = np.array(c["soh"], dtype=float)
        clipped = np.clip(raw, 0.0, 1.05)
        n_clipped = int((raw != clipped).sum())
        if n_clipped:
            soh_clip_cells += 1
            soh_clip_count += n_clipped
        c["soh"] = clipped.tolist()
    print(f"  SOH clip [0.0, 1.05]: {soh_clip_count} measurements clipped in {soh_clip_cells} cells")

    # ── Stratified calibration split ─────────────────────────────────────────
    calib_cells, loo_cells = _select_calibration_cells(cells, CELLS_PER_BATCH, CALIB_SEED)
    calib_ids = {c["cell_id"] for c in calib_cells}
    print(f"\nCalibration cells ({len(calib_cells)}, 4 per batch):")
    for b in [1, 2, 3]:
        batch_calib = [c["cell_id"] for c in calib_cells if c["batch"] == b]
        print(f"  Batch {b}: {batch_calib}")
    print(f"LOO cells: {len(loo_cells)}")
    print(f"N_VALUES (5/15/30% of median cycle_life={MEDIAN_CYCLE_LIFE}): {N_VALUES}")

    # ── ACF length-scale (computed on non-calibration cells, no leakage) ─────
    print("\nComputing ACF decay lags on non-calibration cells...")
    lags = []
    for c in loo_cells:
        b = _ols_beta(c["soh"])
        lag = _acf_decay_lag(c["soh"], b)
        if lag is not None:
            lags.append(lag)
    ls_scale = float(np.median(lags))
    print(f"  {len(lags)} cells, median lag={ls_scale:.1f}  "
          f"→ Gamma(2, 1/{ls_scale:.0f})")

    # ── Stage 1: calibrate kernel (cache to disk so crashes don't require rerun) ──
    calib_cache = ROOT / "data" / "severson_calib_idata.nc"
    print("\n" + "─" * 70)
    print("Stage 1: Kernel calibration (NUTS, cores=1 sequential chains)")
    print("─" * 70)
    if calib_cache.exists():
        print(f"  Loading cached kernel idata from {calib_cache.name} (skipping NUTS)")
        import arviz as _az2
        cached_idata = _az2.from_netcdf(str(calib_cache))
        post = cached_idata.posterior
        amp_draws = post["amplitude"].values.flatten()
        ls_draws  = post["ls"].values.flatten()
        so_draws  = post["sigma_obs"].values.flatten()
        import pandas as _pd
        summ = az.summary(cached_idata, var_names=["amplitude", "ls", "sigma_obs"])
        r_hats = _pd.to_numeric(summ["r_hat"], errors="coerce")
        ess    = _pd.to_numeric(summ["ess_bulk"], errors="coerce")
        kernel_samples = {
            "amplitude": amp_draws, "ls": ls_draws, "sigma_obs": so_draws,
            "idata": cached_idata,
            "elapsed_s": float("nan"),
            "max_rhat": float(r_hats.max()), "min_ess": float(ess.min()),
            "divergences": int(cached_idata.sample_stats.diverging.values.sum()),
        }
        nuts_ok = True
        print(f"  Cached: R-hat_max={kernel_samples['max_rhat']:.4f}  "
              f"ESS_min={kernel_samples['min_ess']:.0f}  "
              f"divergences={kernel_samples['divergences']}")
    else:
        kernel_samples, nuts_ok = _calibrate_kernel(calib_cells, ls_prior_scale=ls_scale)
        print(f"  NUTS converged: {nuts_ok}  (used_full_nuts={nuts_ok})")
        if nuts_ok and kernel_samples.get("idata") is not None:
            kernel_samples["idata"].to_netcdf(str(calib_cache))
            print(f"  Kernel idata cached → {calib_cache.name}")

    # ── Hierarchical beta model ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Hierarchical beta model (partial pooling, all 124 cells)")
    print("─" * 70)
    hier = _fit_hierarchical_beta(cells)
    print(f"\n  mu_beta:    {hier['mu_beta']:.6f} ± {hier['mu_beta_sd']:.6f}  "
          f"HDI 94% [{hier['mu_beta_hdi94'][0]:.6f}, {hier['mu_beta_hdi94'][1]:.6f}]")
    print(f"  sigma_beta: {hier['sigma_beta']:.6f} ± {hier['sigma_beta_sd']:.6f}  "
          f"HDI 94% [{hier['sigma_beta_hdi94'][0]:.6f}, {hier['sigma_beta_hdi94'][1]:.6f}]")
    print(f"  sigma_obs_h:{hier['sigma_obs_h']:.6f}  "
          f"HDI 94% [{hier['sigma_obs_h_hdi94'][0]:.6f}, {hier['sigma_obs_h_hdi94'][1]:.6f}]")
    print(f"  R-hat_max={hier['max_rhat']:.4f}  ESS_min={hier['min_ess']:.0f}  "
          f"divergences={hier['divergences']}")
    print(f"\n  Protocol heterogeneity note: sigma_beta={hier['sigma_beta']:.6f} conflates")
    print(f"  within-protocol cell variance with between-protocol design variance.")
    print(f"  Step 0 regression: protocol features explain R²=0.452 of OLS beta variance.")

    # ── Stage 2: LOO GP predictions ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"Stage 2: LOO GP predictions ({len(loo_cells)} cells × {len(N_VALUES)} N values)")
    print("─" * 70)

    # Precompute OLS betas for all 124 cells once — avoids 112×3 redundant refits
    all_ols_betas_dict = {c["cell_id"]: _ols_beta(c["soh"]) for c in cells}
    all_ols_betas_arr  = np.array(list(all_ols_betas_dict.values()))

    results: Dict = {
        "per_cell": {},
        "aggregate_by_N": {},
        "aggregate_by_N_by_batch": {},
        "zero_mean_by_N": {},
    }

    for n_val in N_VALUES:
        physics_covered, physics_rmse, physics_mae = [], [], []
        zero_covered, zero_rmse = [], []
        by_batch: Dict[int, Dict[str, List]] = {1:{}, 2:{}, 3:{}}
        for bk in by_batch:
            by_batch[bk] = {"covered": [], "rmse": []}

        t_n = time.time()
        for c in loo_cells:
            # LOO-consistent training betas: exclude held-out cell from precomputed dict
            other_betas = np.array(
                [b for cid, b in all_ols_betas_dict.items() if cid != c["cell_id"]]
            )

            res_p = _loo_predict(c, n_val, kernel_samples, other_betas, physics_mean=True)
            res_z = _loo_predict(c, n_val, kernel_samples, other_betas, physics_mean=False)

            if res_p is None:
                continue

            physics_covered.append(res_p["covered"])
            physics_rmse.append(res_p["rmse"])
            physics_mae.append(res_p["mae"])
            by_batch[c["batch"]]["covered"].append(res_p["covered"])
            by_batch[c["batch"]]["rmse"].append(res_p["rmse"])

            if res_z is not None:
                zero_covered.append(res_z["covered"])
                zero_rmse.append(res_z["rmse"])

            if c["cell_id"] not in results["per_cell"]:
                results["per_cell"][c["cell_id"]] = {}
            results["per_cell"][c["cell_id"]][n_val] = {
                "physics": res_p,
                "zero_mean": res_z,
                "batch": c["batch"],
            }

        elapsed_n = time.time() - t_n
        n_cells_done = len(physics_covered)
        phys_cov = float(np.mean(physics_covered)) if physics_covered else float("nan")
        phys_rmse_m = float(np.mean(physics_rmse)) if physics_rmse else float("nan")
        zero_cov = float(np.mean(zero_covered)) if zero_covered else float("nan")
        zero_rmse_m = float(np.mean(zero_rmse)) if zero_rmse else float("nan")

        print(f"\n  N={n_val:3d}:  {n_cells_done} cells  "
              f"physics_cov={phys_cov:.3f}  physics_rmse={phys_rmse_m:.5f}  "
              f"zero_cov={zero_cov:.3f}  zero_rmse={zero_rmse_m:.5f}  "
              f"[{elapsed_n:.0f}s]")
        print(f"         RMSE ratio (physics/zero): "
              f"{phys_rmse_m/zero_rmse_m:.3f}x"
              if zero_rmse_m > 0 else "         RMSE ratio: N/A")

        # Per-batch breakdown
        for b in [1, 2, 3]:
            bc = by_batch[b]["covered"]
            br = by_batch[b]["rmse"]
            if bc:
                print(f"         Batch {b}: cov={np.mean(bc):.3f}  rmse={np.mean(br):.5f}  "
                      f"(n={len(bc)})")

        results["aggregate_by_N"][n_val] = {
            "n_cells": n_cells_done,
            "physics_coverage": phys_cov,
            "physics_rmse_mean": phys_rmse_m,
            "zero_mean_coverage": zero_cov,
            "zero_mean_rmse_mean": zero_rmse_m,
            "rmse_ratio_physics_over_zero": (
                phys_rmse_m / zero_rmse_m if zero_rmse_m > 0 else float("nan")
            ),
        }
        results["aggregate_by_N_by_batch"][n_val] = {
            b: {
                "n_cells": len(by_batch[b]["covered"]),
                "coverage": float(np.mean(by_batch[b]["covered"])) if by_batch[b]["covered"] else float("nan"),
                "rmse_mean": float(np.mean(by_batch[b]["rmse"])) if by_batch[b]["rmse"] else float("nan"),
            }
            for b in [1, 2, 3]
        }

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "meta": {
            "module": "severson_gp_predictor.py",
            "dataset": "Severson et al. (2019) Nature Energy 4:383-391 — LFP/graphite, A123 APR18650M1A",
            "n_cells_total": len(cells),
            "n_calib_cells": len(calib_cells),
            "n_loo_cells": len(loo_cells),
            "calib_cell_ids": sorted(calib_ids),
            "calib_subsample_k": CALIB_SUBSAMPLE_K,
            "n_values": N_VALUES,
            "n_fracs": N_FRACS,
            "median_cycle_life": MEDIAN_CYCLE_LIFE,
            "coverage_nominal": COVERAGE_NOMINAL,
            "nuts_ok": nuts_ok,
            "ls_scale_from_data": ls_scale,
            "protocol_heterogeneity_r2_numeric": 0.452,
            "protocol_heterogeneity_r2_upper_bound": 0.984,
            "scope": (
                "Demonstrates physics-informed GP methodology at n=124 (adequate "
                "statistical power). Separate from NASA LCO analysis — different "
                "chemistry, V-grid, scale, facility. NOT a cross-chemistry result."
            ),
            "named_limitation": (
                "Kernel amplitude/ls/sigma_obs estimated from calibration-cell "
                "within-cell structure only; cannot capture between-cell systematic "
                "deviation. Coverage degrades at higher N. Convergence of the "
                "kernel posterior != correctness of the uncertainty model."
            ),
            "separate_from_nasa": True,
        },
        "calibration_kernel": {
            "used_full_nuts": nuts_ok,
            "max_rhat": kernel_samples["max_rhat"],
            "min_ess": kernel_samples["min_ess"],
            "divergences": kernel_samples["divergences"],
            "elapsed_s": kernel_samples["elapsed_s"],
            "amplitude_mean": float(kernel_samples["amplitude"].mean()),
            "ls_mean": float(kernel_samples["ls"].mean()),
            "sigma_obs_mean": float(kernel_samples["sigma_obs"].mean()),
        },
        "hierarchical_beta": {
            "mu_beta": hier["mu_beta"],
            "mu_beta_sd": hier["mu_beta_sd"],
            "mu_beta_hdi94": hier["mu_beta_hdi94"],
            "sigma_beta": hier["sigma_beta"],
            "sigma_beta_sd": hier["sigma_beta_sd"],
            "sigma_beta_hdi94": hier["sigma_beta_hdi94"],
            "sigma_obs_h": hier["sigma_obs_h"],
            "sigma_obs_h_hdi94": hier["sigma_obs_h_hdi94"],
            "max_rhat": hier["max_rhat"],
            "min_ess": hier["min_ess"],
            "divergences": hier["divergences"],
            "protocol_note": (
                "sigma_beta conflates within-protocol cell variance with "
                "between-protocol design variance (Step 0: protocol explains "
                "R²=0.452 of OLS beta variance, upper bound R²=0.984)."
            ),
        },
        "loo_results": {
            "aggregate_by_N": {str(k): v for k, v in results["aggregate_by_N"].items()},
            "aggregate_by_N_by_batch": {
                str(k): {str(b): vv for b, vv in v.items()}
                for k, v in results["aggregate_by_N_by_batch"].items()
            },
        },
    }

    out_path = ROOT / "data" / "severson_gp_report.json"

    def _serial(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): _serial(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_serial(v) for v in obj]
        return obj

    out_path.write_text(json.dumps(_serial(report), indent=2))
    print(f"\nReport written → {out_path}")

    # ── Final console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Hierarchical beta (n=124 LFP cells):")
    print(f"    mu_beta    = {hier['mu_beta']:.6f} ± {hier['mu_beta_sd']:.6f}  "
          f"HDI 94% [{hier['mu_beta_hdi94'][0]:.6f}, {hier['mu_beta_hdi94'][1]:.6f}]")
    print(f"    sigma_beta = {hier['sigma_beta']:.6f} ± {hier['sigma_beta_sd']:.6f}  "
          f"HDI 94% [{hier['sigma_beta_hdi94'][0]:.6f}, {hier['sigma_beta_hdi94'][1]:.6f}]")
    print(f"    R-hat_max={hier['max_rhat']:.4f}  ESS_min={hier['min_ess']:.0f}  "
          f"divergences={hier['divergences']}")

    print(f"\n  GP LOO predictions (physics mean vs zero-mean baseline):")
    print(f"  {'N':>4}  {'Physics cov':>12}  {'Zero cov':>10}  {'Physics RMSE':>13}  {'Ratio':>7}")
    for n_val in N_VALUES:
        agg = results["aggregate_by_N"].get(n_val, {})
        print(f"  {n_val:>4}  {agg.get('physics_coverage', float('nan')):>12.3f}  "
              f"{agg.get('zero_mean_coverage', float('nan')):>10.3f}  "
              f"{agg.get('physics_rmse_mean', float('nan')):>13.5f}  "
              f"{agg.get('rmse_ratio_physics_over_zero', float('nan')):>7.3f}x")

    print(f"\n  Pre-registered coverage check (nominal=0.90):")
    n36_cov = results["aggregate_by_N"].get(N_VALUES[0], {}).get("physics_coverage", float("nan"))
    result_str = (
        "WITHIN pre-registered 80-93% range" if 0.80 <= n36_cov <= 0.93
        else f"OUTSIDE pre-registered range — report as honest failure"
    )
    print(f"    N={N_VALUES[0]}: {n36_cov:.3f}  → {result_str}")


if __name__ == "__main__":
    main()
