#!/usr/bin/env python3
"""
degradation/cross_cell_predictor.py  —  Module 4: Cross-Cell Degradation Prediction

Problem
-------
Within-cell R²=0.9725 (Module 2, β fitted per cell).
Cross-cell R²=−0.68 (fixed population-mean β applied to a held-out cell).
B0006 degrades ~45% faster than B0005 at nearly identical cumulative damage.
The open question: given a new cell and only its first ~30 cycles, can we arrive
at a useful β before the slow fade signal accumulates?

Approach
--------
Hierarchical empirical-Bayes + online windowed β-adaptation. NOT deep/meta-learning
(need 100+ cells for that; we have 4). Three explicit honesty checks:

  CHECK A  ΔQ(V) feature utility:
    Does feature-mapped β₀ beat plain-mean β₀?  (3-point fit may add nothing)
    Report honestly: if not, conclude feature adds no value on this 4-cell dataset.

  CHECK B  Dead-band utility:
    Run online update WITH and WITHOUT dead-band.
    If dead-band makes no measurable difference, state that the bone/mechanostat
    framing is purely cosmetic here and report it accordingly.

  CHECK C  Few-shot framing:
    This is ~30-cycle adaptation, NOT zero-cycle early prediction.
    State explicitly that β is unidentifiable from 0 cycles of a new cell.

Honest limits (stated in report):
  - 4 NASA cells only (ideal ≥10 for stable hierarchical Bayes)
  - Same manufacturer / same lab protocol — NOT cross-manufacturer
  - First ~10 cycles: too little cumulative D signal to identify β reliably
  - Realistic target: R²≥0.6 after 30 cycles, ≥0.8 after 50 cycles

References
----------
  Severson et al. (2019) Nature Energy 4:383–391       ΔQ(V) early features
  Cripps & Pecht (2017) Reliab. Eng. Syst. Saf.        Bayesian random-effects
  Plett (2004) J. Power Sources 134:252–261             dual EKF β co-state
  Frost (1987) Anat. Rec. 219:1–9                      mechanostat dead-band
                                                        DESIGN INSPIRATION ONLY
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit

ROOT     = Path(__file__).resolve().parent.parent
NASA_DIR = ROOT / "data" / "nasa"
OUT_JSON = ROOT / "data" / "cross_cell_report.json"
OUT_JSON_SEV  = ROOT / "data" / "cross_cell_severson_report.json"
OUT_JSON_CL   = ROOT / "data" / "cross_cell_severson_cyclelife_report.json"
SEVERSON_DIR  = ROOT / "data" / "severson"
SEVERSON_BATCH_FILES = [
    (1, "2017-05-12_batchdata_updated_struct_errorcorrect.mat"),
    (2, "2017-06-30_batchdata_updated_struct_errorcorrect.mat"),
    (3, "2018-04-12_batchdata_updated_struct_errorcorrect.mat"),
]
SEVERSON_EXCLUDE: Dict[int, set] = {
    1: {"b1c8",  "b1c10", "b1c12", "b1c13", "b1c22"},
    2: {"b2c7",  "b2c8",  "b2c9",  "b2c15", "b2c16"},
    3: {"b3c2",  "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"},
}

# ── model constants ───────────────────────────────────────────────────────────
CELLS   = ["B0005", "B0006", "B0007", "B0018"]
GAMMA   = 0.5          # fixed population exponent (Module 2 finding: SEI ∝ √k)

# Windowed RLS
WINDOW_W      = 10     # cycles in the β-adaptation window
FREEZE_CYCLES = 10     # hold β=β₀ for first N cycles (too little D signal)
DEAD_BAND_TAU = 0.005  # SOH units; |error|<τ → skip update (mechanostat dead-band)

# ΔQ(V) feature (Severson-style)
N_FEATURE_CYCLES = 20         # use cycles 2..N for ΔQ(V) variance
V_GRID           = np.linspace(2.75, 4.15, 200)   # 200-point voltage grid

# Evaluation snapshots: "new-cell cycles seen" before predicting remainder
SNAPSHOTS = [0, 10, 20, 30, 50, 100]

# ── data loading ──────────────────────────────────────────────────────────────

def _load_cell(cell_id: str) -> List[Dict]:
    """Return list of discharge cycle dicts {V, I, t, Q_total, cycle_n}."""
    path = NASA_DIR / f"{cell_id}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    mat  = loadmat(str(path))
    key  = [k for k in mat if not k.startswith("_")][0]
    cycs = mat[key]["cycle"][0, 0]
    n    = cycs.shape[1]

    discharges: List[Dict] = []
    d_num = 0
    for i in range(n):
        c = cycs[0, i]
        if "discharge" not in str(c["type"][0]).strip().lower():
            continue
        d_num += 1
        data = c["data"][0, 0]
        V  = data["Voltage_measured"][0].astype(np.float64)
        I  = data["Current_measured"][0].astype(np.float64)
        t  = data["Time"][0].astype(np.float64)
        dt = np.diff(t, prepend=t[0])
        Q  = np.cumsum(np.abs(I) * dt) / 3600.0
        discharges.append({"cycle_n": d_num, "V": V, "I": I, "t": t,
                           "Q": Q, "Q_total": float(Q[-1])})
    return discharges

# ── SOH + damage ──────────────────────────────────────────────────────────────

def _soh_damage(cycles: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    SOH_k = Q_total_k / Q_total_1  (normalised fade, dimensionless).
    D_k   = k  (unit damage per full 1C cycle; DOD≈1.0 and T constant for NASA).

    Using D_k=k rather than DOD-weighted Miner avoids circularity
    (SOH appearing on both sides) and is valid when C-rate and depth
    are nearly constant across cycles, which holds for NASA lab data.
    Model: ΔSOH_k = β · k^γ  →  SOH_k = 1 − β · k^0.5
    """
    q1 = cycles[0]["Q_total"]
    soh = np.array([c["Q_total"] / q1 for c in cycles])
    D   = np.arange(1, len(cycles) + 1, dtype=float)
    return soh, D

# ── ΔQ(V) feature (Severson 2019 style) ──────────────────────────────────────

def _qv_curve(cycle: Dict) -> Optional[np.ndarray]:
    """
    Interpolate Q onto V_GRID for one discharge cycle.
    Discharge: V decreases monotonically (approximately) as Q increases.
    We flip arrays so V is ascending for np.interp.
    Returns Q(V_GRID) or None if cycle is too short / voltage range too narrow.
    """
    V = cycle["V"]
    Q = cycle["Q"]
    if len(V) < 10:
        return None
    # Keep only the strictly-falling portion (drop any initial plateau noise)
    diff_v = np.diff(V)
    # Find first index where V starts falling
    start = next((i for i, d in enumerate(diff_v) if d < 0), 0)
    V = V[start:]
    Q = Q[start:]
    if V[-1] >= V[0] or (V[0] - V[-1]) < 0.3:
        return None
    # Sort ascending for interp (V is descending, Q ascending → flip both)
    sort_idx = np.argsort(V)
    V_asc = V[sort_idx]
    Q_asc = Q[sort_idx]
    v_min, v_max = V_asc[0], V_asc[-1]
    in_range = (V_GRID >= v_min) & (V_GRID <= v_max)
    if in_range.sum() < 20:
        return None
    q_interp = np.full(len(V_GRID), np.nan)
    q_interp[in_range] = np.interp(V_GRID[in_range], V_asc, Q_asc)
    return q_interp

def _dqv_feature(cycles: List[Dict], n_early: int = N_FEATURE_CYCLES) -> float:
    """
    Severson-style ΔQ(V) feature: variance of Q_k(V) − Q_2(V) across k=3..n_early,
    averaged over the voltage grid. Returns log10(variance + ε).

    A high variance means Q(V) is changing rapidly in early cycles → high β.
    A low variance means the cell is stable early on → low β.

    With only 4 cells the feature→β mapping is a 3-point fit (in LOO-CV).
    We prove below (CHECK A) whether this adds value over the plain prior mean.
    """
    if len(cycles) < 3:
        return 0.0
    n_use = min(n_early, len(cycles))
    curves = []
    for c in cycles[1:n_use]:   # cycles index 1 = cycle 2 (0-indexed)
        q = _qv_curve(c)
        if q is not None:
            curves.append(q)
    if len(curves) < 2:
        return 0.0
    ref = _qv_curve(cycles[1])  # cycle 2 as reference
    if ref is None:
        return 0.0
    deltas = []
    for q in curves:
        dq = q - ref
        mask = ~np.isnan(dq)
        if mask.sum() > 20:
            deltas.append(dq[mask])
    if not deltas:
        return 0.0
    # Pad all deltas to same length with NaN, then compute variance per V-point
    max_len = max(len(d) for d in deltas)
    mat = np.full((len(deltas), max_len), np.nan)
    for i, d in enumerate(deltas):
        mat[i, :len(d)] = d
    var_per_v = np.nanvar(mat, axis=0)
    mean_var  = float(np.nanmean(var_per_v))
    return float(np.log10(mean_var + 1e-9))

# ── per-cell β fit ────────────────────────────────────────────────────────────

def _fit_beta(soh: np.ndarray, D: np.ndarray, gamma: float = GAMMA,
              mu_prior: float = 0.0, lam0: float = 0.0) -> float:
    """
    Fit β via regularised least squares:
      minimize Σ_k (SOH_k − (1 − β·D_k^γ))² + λ₀·(β − μ_prior)²
    Closed form: β = (Σ x_k² + λ₀)⁻¹ · (Σ x_k·y_k + λ₀·μ_prior)
    where x_k = D_k^γ, y_k = 1 − SOH_k  (= ΔSOH_k).
    """
    x = np.power(D, gamma)
    y = 1.0 - soh
    denom = float(np.dot(x, x)) + lam0
    numer = float(np.dot(x, y)) + lam0 * mu_prior
    beta  = max(numer / (denom + 1e-12), 0.0)
    return float(beta)

def _predict_soh(D: np.ndarray, beta: float, gamma: float = GAMMA) -> np.ndarray:
    return np.clip(1.0 - beta * np.power(D, gamma), 0.0, 1.0)

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)

# ── windowed regularised RLS β-update ─────────────────────────────────────────

def _rls_update(window_D: np.ndarray, window_soh: np.ndarray,
                gamma: float, mu_prior: float, lam0: float) -> float:
    """Closed-form windowed β from the last W cycles + population prior."""
    return _fit_beta(window_soh, window_D, gamma, mu_prior, lam0)

# ── online β-adaptation for a new cell ───────────────────────────────────────

def _online_adapt(soh_new: np.ndarray, D_new: np.ndarray,
                  beta0: float, mu_prior: float, lam0: float,
                  gamma: float = GAMMA,
                  use_deadband: bool = True,
                  tau: float = DEAD_BAND_TAU,
                  freeze: int = FREEZE_CYCLES,
                  window: int = WINDOW_W) -> Dict:
    """
    Online windowed β-adaptation for a held-out cell.

    Parameters
    ----------
    soh_new  : measured SOH for every cycle of the new cell
    D_new    : damage (cycle index k) for every cycle
    beta0    : initial β guess (prior mean OR feature-mapped)
    mu_prior : population prior mean β
    lam0     : prior precision (1/σ²_β)
    use_deadband : if True, skip update when |innovation| < tau
    freeze   : hold β = beta0 for first N cycles unconditionally

    Returns
    -------
    dict with beta_trace, soh_pred_trace, innovation_trace, n_updates
    """
    n           = len(soh_new)
    beta_trace  = np.full(n, np.nan)
    beta        = beta0
    n_updates   = 0
    innovations = []

    for k in range(n):
        # Predict using current β
        soh_pred = float(_predict_soh(np.array([D_new[k]]), beta, gamma)[0])
        innov    = float(soh_new[k] - soh_pred)
        innovations.append(innov)

        # Update decision
        if k < freeze:
            pass   # frozen — hold β=β₀
        elif use_deadband and abs(innov) < tau:
            pass   # within noise dead-band — no update
        else:
            # Windowed RLS
            w_start = max(0, k + 1 - window)
            w_D     = D_new[w_start: k + 1]
            w_soh   = soh_new[w_start: k + 1]
            beta    = _rls_update(w_D, w_soh, gamma, mu_prior, lam0)
            n_updates += 1

        beta_trace[k] = beta

    return {
        "beta_trace":  beta_trace,
        "n_updates":   n_updates,
        "innovations": np.array(innovations),
    }

# ── R²(n_cycles_seen) for one cell and one β strategy ────────────────────────

def _r2_at_snapshots(soh_true: np.ndarray, D: np.ndarray,
                     beta_trace: np.ndarray,
                     gamma: float = GAMMA,
                     snapshots: List[int] = SNAPSHOTS) -> Dict[int, float]:
    """
    At each snapshot n (cycles seen), use β(n) to predict SOH for cycles n+1..end.
    Returns {n: R²} dict.
    """
    results: Dict[int, float] = {}
    n_total = len(soh_true)
    for n in snapshots:
        if n >= n_total - 2:
            results[n] = float("nan")
            continue
        beta_n    = float(beta_trace[min(n, n_total - 1)])
        future_D  = D[n:]
        soh_pred  = _predict_soh(future_D, beta_n, gamma)
        soh_true_ = soh_true[n:]
        results[n] = _r2(soh_true_, soh_pred)
    return results

def _r2_full(soh_true: np.ndarray, D: np.ndarray,
             beta: float, gamma: float = GAMMA) -> float:
    """R² on all cycles with a fixed β (for baselines)."""
    return _r2(soh_true, _predict_soh(D, beta, gamma))

# ── LOO-CV ────────────────────────────────────────────────────────────────────

def run_loo_cv(cell_data: Dict[str, Dict], gamma: float = GAMMA) -> Dict:
    """
    Leave-one-cell-out cross-validation.

    For each held-out cell:
      1. Fit β_i per training cell → population prior (μ_β, σ²_β)
      2. Fit ΔQ(V) feature → β linear map on training cells
      3. Compute F_new for held-out cell
      4. Run 4 methods and record R²(n_cycles_seen):
           A  Fixed mean   : β = μ_β, no adaptation
           B  Feature-mapped: β = a·F + b, no adaptation
           C  Online no-db  : windowed RLS, dead-band OFF
           D  Online + db   : windowed RLS, dead-band ON

    Returns nested dict with per-cell and aggregated results.
    """
    results: Dict = {}

    for held_out in CELLS:
        train_ids = [c for c in CELLS if c != held_out]
        print(f"\n  LOO fold: held-out={held_out}, train={train_ids}")

        # ── 1. Per-training-cell β fit ──────────────────────────────────────
        train_betas: List[float] = []
        train_features: List[float] = []
        for cid in train_ids:
            soh_t, D_t = cell_data[cid]["soh"], cell_data[cid]["D"]
            b = _fit_beta(soh_t, D_t, gamma)
            f = cell_data[cid]["dqv_feature"]
            train_betas.append(b)
            train_features.append(f)
            print(f"    {cid}: β={b:.4f}, ΔQ(V) feature={f:.4f}")

        # ── 2. Population prior ──────────────────────────────────────────────
        mu_beta  = float(np.mean(train_betas))
        var_beta = float(np.var(train_betas, ddof=1)) if len(train_betas) > 1 else 1e-4
        lam0     = 1.0 / (var_beta + 1e-8)   # prior precision
        print(f"    Prior: μ_β={mu_beta:.4f}, σ_β={var_beta**0.5:.4f}, λ₀={lam0:.2f}")

        # ── 3. Feature→β map (linear, 3 training points) ───────────────────
        F_train = np.array(train_features)
        B_train = np.array(train_betas)
        # Simple linear regression: β = a·F + b
        F_mean, B_mean = float(np.mean(F_train)), float(np.mean(B_train))
        ss_F = float(np.sum((F_train - F_mean) ** 2))
        a_feat = float(np.sum((F_train - F_mean) * (B_train - B_mean))) / (ss_F + 1e-12)
        b_feat = B_mean - a_feat * F_mean
        F_new  = cell_data[held_out]["dqv_feature"]
        beta0_feature = float(np.clip(a_feat * F_new + b_feat, 1e-6, None))
        beta0_mean    = mu_beta
        print(f"    Feature map: a={a_feat:.4f}, b={b_feat:.4f}")
        print(f"    F_new={F_new:.4f}  →  β₀_feature={beta0_feature:.4f}  "
              f"β₀_mean={beta0_mean:.4f}")

        # ── 4. Held-out cell data ────────────────────────────────────────────
        soh_new = cell_data[held_out]["soh"]
        D_new   = cell_data[held_out]["D"]
        n_cyc   = len(soh_new)

        # True β for this cell (within-cell fit — the oracle target)
        beta_true = _fit_beta(soh_new, D_new, gamma)

        # ── β_predicted vs β_true (CHECK A: 3-point fit quality) ───────────
        beta_err_pct = (beta0_feature - beta_true) / (beta_true + 1e-9) * 100
        is_extrapolation = (F_new < min(train_features) or F_new > max(train_features))
        print(f"    β_true={beta_true:.4f}  β₀_feature={beta0_feature:.4f}  "
              f"err={beta_err_pct:+.1f}%"
              f"{'  [EXTRAPOLATION]' if is_extrapolation else '  [interpolation]'}")
        print(f"    β₀_mean={beta0_mean:.4f}  "
              f"mean_err={(beta0_mean - beta_true)/(beta_true+1e-9)*100:+.1f}%")

        # Method A: fixed population mean β for all cycles
        beta_trace_A = np.full(n_cyc, beta0_mean)

        # Method B: CORRECTED labelling — feature requires N_FEATURE_CYCLES cycles.
        # Before that: use prior mean (same as A). Feature-mapped β available only at
        # cycle N_FEATURE_CYCLES (n=20), not at n=0. This fixes the labelling contradiction.
        beta_trace_B = np.full(n_cyc, beta0_mean)        # prior-only for cycles 0..19
        beta_trace_B[N_FEATURE_CYCLES:] = beta0_feature  # feature-mapped from cycle 20 on

        # Method C: online windowed RLS, dead-band OFF
        res_C = _online_adapt(soh_new, D_new, beta0=beta0_mean,
                              mu_prior=mu_beta, lam0=lam0, gamma=gamma,
                              use_deadband=False)
        beta_trace_C = res_C["beta_trace"]

        # Method D: online windowed RLS, dead-band ON
        res_D = _online_adapt(soh_new, D_new, beta0=beta0_mean,
                              mu_prior=mu_beta, lam0=lam0, gamma=gamma,
                              use_deadband=True)
        beta_trace_D = res_D["beta_trace"]

        print(f"    β_trace_C final: {beta_trace_C[-1]:.4f}  "
              f"β_trace_D final: {beta_trace_D[-1]:.4f}  "
              f"updates_C: {res_C['n_updates']}  updates_D: {res_D['n_updates']}")

        # ── 5. R²(n_cycles_seen) ─────────────────────────────────────────────
        r2_A = _r2_at_snapshots(soh_new, D_new, beta_trace_A, gamma)
        r2_B = _r2_at_snapshots(soh_new, D_new, beta_trace_B, gamma)
        r2_C = _r2_at_snapshots(soh_new, D_new, beta_trace_C, gamma)
        r2_D = _r2_at_snapshots(soh_new, D_new, beta_trace_D, gamma)

        results[held_out] = {
            "held_out":            held_out,
            "train_ids":           train_ids,
            "mu_beta":             mu_beta,
            "sigma_beta":          var_beta ** 0.5,
            "lam0":                lam0,
            "beta0_mean":          beta0_mean,
            "beta0_feature":       beta0_feature,
            "beta_true":           beta_true,
            "beta_feature_err_pct": beta_err_pct,
            "beta_mean_err_pct":   float((beta0_mean - beta_true) / (beta_true + 1e-9) * 100),
            "feature_extrapolation": is_extrapolation,
            "F_new":               F_new,
            "feature_map":         {"a": a_feat, "b": b_feat},
            "n_updates_no_db":     res_C["n_updates"],
            "n_updates_db":        res_D["n_updates"],
            "r2_snapshots": {
                "A_fixed_mean":     r2_A,
                "B_feature_mapped": r2_B,
                "C_online_no_db":   r2_C,
                "D_online_db":      r2_D,
            },
        }

    return results

# ── aggregated summary + 3 checks ────────────────────────────────────────────

def _aggregate(loo_results: Dict) -> Dict:
    """Average R²(n) across all 4 LOO folds per method."""
    methods = ["A_fixed_mean", "B_feature_mapped", "C_online_no_db", "D_online_db"]
    agg: Dict = {m: {} for m in methods}
    for n in SNAPSHOTS:
        for m in methods:
            vals = [
                loo_results[cid]["r2_snapshots"][m].get(n, float("nan"))
                for cid in CELLS
            ]
            finite = [v for v in vals if not np.isnan(v)]
            agg[m][n] = float(np.mean(finite)) if finite else float("nan")
    return agg

def _check_feature_utility(agg: Dict, loo_results: Dict) -> Tuple[bool, str]:
    """
    CHECK A: does feature-mapped β₀ beat plain mean at n=20?
    n=20 is the FIRST snapshot where Method B differs from Method A —
    before cycle 20 both methods use β=μ_β (feature not yet computable).
    """
    # At n=20, Method B has switched to feature-mapped β; Method A still uses mean.
    r2_A_20 = agg["A_fixed_mean"][20]
    r2_B_20 = agg["B_feature_mapped"][20]
    margin   = r2_B_20 - r2_A_20

    # Also inspect raw β_predicted vs β_true per fold
    errs_feat = [abs(loo_results[c]["beta_feature_err_pct"]) for c in CELLS]
    errs_mean = [abs(loo_results[c]["beta_mean_err_pct"])    for c in CELLS]
    n_extrap  = sum(1 for c in CELLS if loo_results[c]["feature_extrapolation"])

    err_str = (f"β_feature errors: "
               + ", ".join(f"{loo_results[c]['held_out']}={loo_results[c]['beta_feature_err_pct']:+.1f}%"
                           for c in CELLS)
               + f"; β_mean errors: "
               + ", ".join(f"{loo_results[c]['held_out']}={loo_results[c]['beta_mean_err_pct']:+.1f}%"
                           for c in CELLS))

    caveat = (f"3-point linear fit, {n_extrap}/4 folds require extrapolation beyond "
              f"training F range. ")

    if margin > 0.02:
        verdict = (True,
                   f"Feature-mapped β₀ BEATS plain mean at n=20 — the first "
                   f"snapshot where B diverges from A (ΔR²={margin:+.3f}). "
                   f"{caveat}{err_str}")
    elif margin > -0.02:
        verdict = (False,
                   f"Feature-mapped β₀ is INDISTINGUISHABLE from plain mean "
                   f"at n=20 (ΔR²={margin:+.3f}). {caveat}{err_str} "
                   f"ΔQ(V) feature adds no reliable value on this 4-cell dataset.")
    else:
        verdict = (False,
                   f"Feature-mapped β₀ is WORSE than plain mean at n=20 "
                   f"(ΔR²={margin:+.3f}). {caveat}{err_str} "
                   f"Feature discarded; use plain prior mean as β₀.")
    return verdict

def _check_deadband_utility(agg: Dict) -> Tuple[bool, str]:
    """CHECK B: does dead-band ON beat dead-band OFF after freeze period?"""
    # Compare at n=20 (post-freeze, some cycles seen)
    r2_C_20 = agg["C_online_no_db"][20]
    r2_D_20 = agg["D_online_db"][20]
    # Also compare at n=10 (near freeze boundary — where early-cycle overfitting hurts)
    r2_C_10 = agg["C_online_no_db"][10]
    r2_D_10 = agg["D_online_db"][10]
    margin_20 = r2_D_20 - r2_C_20
    margin_10 = r2_D_10 - r2_C_10
    if margin_20 > 0.02 or margin_10 > 0.02:
        verdict = (True,
                   f"Dead-band HELPS: R²(n=10) Δ={margin_10:+.3f}, "
                   f"R²(n=20) Δ={margin_20:+.3f}. The dead-band reduces "
                   f"early-cycle β-overfitting. The mechanostat analogy is "
                   f"load-bearing as a design rationale.")
    else:
        verdict = (False,
                   f"Dead-band makes NO MEASURABLE DIFFERENCE: "
                   f"R²(n=10) Δ={margin_10:+.3f}, R²(n=20) Δ={margin_20:+.3f}. "
                   f"The bone/mechanostat framing is COSMETIC on this dataset — "
                   f"the noise floor is low enough that all innovations exceed τ. "
                   f"Report this honestly; do not claim the dead-band as a contribution.")
    return verdict

# ── printing ──────────────────────────────────────────────────────────────────

def print_r2_table(agg: Dict, loo_results: Dict) -> None:
    cols = ["A_fixed_mean", "B_feature_mapped", "C_online_no_db", "D_online_db"]
    labels = {
        "A_fixed_mean":     "A  Fixed mean (baseline)",
        "B_feature_mapped": "B  Feature-mapped β₀   ",
        "C_online_no_db":   "C  Online (no dead-band)",
        "D_online_db":      "D  Online + dead-band  ",
    }
    print("\n" + "="*72)
    print("  R²(n_cycles_seen) — averaged over 4 LOO folds")
    print("  Prediction target: cycles n+1 → end of cell")
    print(f"  NOTE: Method B uses prior-mean β for n<{N_FEATURE_CYCLES} (feature not yet")
    print(f"        computable). B diverges from A only at n≥{N_FEATURE_CYCLES}.")
    print("="*72)
    header = f"  {'Method':<27}" + "".join(f"  n={n:<5}" for n in SNAPSHOTS)
    print(header)
    print("  " + "-"*68)
    for m in cols:
        row = f"  {labels[m]}"
        for n in SNAPSHOTS:
            v = agg[m].get(n, float("nan"))
            if np.isnan(v):
                row += "    nan"
            elif m == "B_feature_mapped" and n < N_FEATURE_CYCLES:
                row += f"  {v:+.3f}*"   # * = same as A, feature not available yet
            else:
                row += f"  {v:+.3f} "
        print(row)
    print(f"  * Method B columns n<{N_FEATURE_CYCLES}: identical to A "
          f"(feature requires {N_FEATURE_CYCLES} cycles of the new cell)")
    print()

    # β_predicted vs β_true comparison (CHECK A: 3-point fit honesty)
    print("  β_predicted vs β_true — feature map quality (CHECK A):")
    print(f"  {'Cell':<8}  {'β_true':>8}  {'β_feat':>8}  {'feat_err%':>10}  "
          f"{'β_mean':>8}  {'mean_err%':>10}  {'extrap?':>8}")
    print("  " + "-"*70)
    for cid in CELLS:
        r = loo_results[cid]
        extrap = "YES" if r["feature_extrapolation"] else "no"
        print(f"  {cid:<8}  {r['beta_true']:>8.4f}  {r['beta0_feature']:>8.4f}  "
              f"{r['beta_feature_err_pct']:>+9.1f}%  "
              f"{r['beta0_mean']:>8.4f}  {r['beta_mean_err_pct']:>+9.1f}%  "
              f"{extrap:>8}")
    print()

    # Per-cell R² at n=20 (first point B diverges from A)
    print(f"  Per-cell R² at n=20 (first snapshot where B uses feature-mapped β):")
    print(f"  {'Cell':<8}  {'A mean':>8}  {'B feat':>8}  {'C online':>9}")
    for cid in CELLS:
        r = loo_results[cid]
        rA = r["r2_snapshots"]["A_fixed_mean"].get(20, float("nan"))
        rB = r["r2_snapshots"]["B_feature_mapped"].get(20, float("nan"))
        rC = r["r2_snapshots"]["C_online_no_db"].get(20, float("nan"))
        print(f"  {cid:<8}  {rA:>+8.3f}  {rB:>+8.3f}  {rC:>+9.3f}")
    print()

def print_checks(loo_results: Dict, agg: Dict) -> None:
    feat_ok, feat_msg   = _check_feature_utility(agg, loo_results)
    db_ok,   db_msg     = _check_deadband_utility(agg)

    print("="*72)
    print("  3 HONESTY CHECKS")
    print("="*72)
    print(f"\n  CHECK A — ΔQ(V) feature utility:")
    print(f"    {'✓' if feat_ok else '✗'}  {feat_msg}")

    print(f"\n  CHECK B — Dead-band utility:")
    print(f"    {'✓' if db_ok else '✗'}  {db_msg}")

    print(f"\n  CHECK C — Few-shot framing (structural, not data-driven):")
    print(f"    ✗  This is NOT Severson-style cycle-0 prediction.")
    print(f"       R²(n=0) reflects only the population prior — it will be")
    print(f"       negative or near-zero for a cell outside the prior's range.")
    print(f"       Adaptation requires seeing ~30 cycles of the new cell.")
    print(f"       State this explicitly in any viva presentation.")
    print()

def print_honest_validation(loo_results: Dict, agg: Dict) -> None:
    print("="*72)
    print("  VALIDATION SCOPE — WHAT THIS ANALYSIS CAN AND CANNOT CLAIM")
    print("="*72)

    r2_A_0  = agg["A_fixed_mean"][0]
    r2_D_30 = agg["D_online_db"][30]
    r2_D_50 = agg["D_online_db"][50]

    print(f"""
  CAN  claim:
    · Cross-cell R² improves over fixed-β baseline ({r2_A_0:+.3f})
      as new-cell cycles accumulate:
        n=30 cycles seen → R²≈{r2_D_30:+.3f}  (best online method)
        n=50 cycles seen → R²≈{r2_D_50:+.3f}
    · Whether ΔQ(V) feature warm-start beats plain prior (CHECK A above)
    · Whether dead-band reduces early β-overfitting (CHECK B above)
    · Honest R²(n_cycles) curve showing adaptation rate

  CANNOT claim:
    · True cross-manufacturer generalisation — all 4 cells are Sanyo
      18650, same lab, same 1C protocol, same temperature. "Cross-cell"
      here means cross-individual within one manufacturer batch only.
    · Statistical power from 4-cell LOO-CV: each fold trains the prior
      on 3 cells. Cripps & Pecht (2017) recommend ≥10 for stable
      hierarchical Bayes. Our σ_β estimate has high uncertainty.
    · Zero-cycle early prediction — β is unidentifiable from 0 new-cell
      cycles. The prior mean is the only estimate at n=0, and it can be
      arbitrarily wrong for an outlier cell (B0006 is a clear outlier).
    · Real BMS deployment: online adaptation needs per-cycle Ah-integrated
      SOH as a measurement, which may not be available in all field BMSs.
    · Exact %LLI / %LAM for the degradation mechanism — Module 3 result:
      mode unresolved at 1C.

  WITHIN-CELL R² DISCREPANCY vs MODULE 2:
    This module: within-cell R²≈0.83 (B0005=0.755, B0006=0.891, B0007=0.787,
                 B0018=0.870). Model: ΔSOH = β·k^0.5, D_k=k (unit-cycle).
    Module 2:    within-cell R²=0.9725. Model: ΔSOH = β·D^γ with full
                 rainflow-Miner D (DOD × C-rate × Arrhenius) and per-cell
                 γ optimised by curve_fit (not fixed at 0.5).
    Both are honest within-cell numbers for their respective models.
    The 0.9725 is not applicable here because this module fixes γ=0.5
    and uses simplified D_k=k damage (valid for constant-protocol lab data
    but lower R² than the full model). Quote 0.83 for this module.

  WHAT'S GENUINELY NOVEL (honest framing):
    The unpublished combination is:
      rainflow-Miner fatigue damage (simplified to √k for lab-constant protocol)
      + per-cell β population prior (empirical-Bayes from 3 training cells)
      + online windowed gated β-update
      + explicit few-cell (4-cell) cross-cell validation with 3 honest baselines
    The bone/mechanostat framing names the dead-band DESIGN CHOICE only.
    Whether the dead-band is load-bearing is proved empirically (CHECK B).

  REFERENCES:
    Severson et al. (2019) Nature Energy 4:383–391
    Cripps & Pecht (2017) Reliab. Eng. Syst. Saf.
    Plett (2004) J. Power Sources 134:252–261
    Frost (1987) Anat. Rec. 219:1–9 (design inspiration only)
""")

# ── JSON serialisation ────────────────────────────────────────────────────────

def _serialise(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Not serialisable: {type(obj)}")

def save_json(loo_results: Dict, agg: Dict) -> None:
    feat_ok, feat_msg = _check_feature_utility(agg, loo_results)
    db_ok,   db_msg   = _check_deadband_utility(agg)

    report = {
        "module":  "Module 4 — Cross-Cell Degradation Prediction",
        "model":   f"ΔSOH = β · k^{GAMMA}  (γ={GAMMA} fixed, D_k=k unit-cycle damage)",
        "honest_scope": {
            "cells":            CELLS,
            "n_cells":          len(CELLS),
            "manufacturer":     "Sanyo 18650 (same manufacturer, same lab protocol — NOT cross-manufacturer)",
            "ideal_n_cells":    "≥10 for stable hierarchical Bayes (Cripps & Pecht 2017)",
            "adaptation_note":  "NOT zero-cycle prediction. β unidentifiable at n=0. Requires ~30 new-cell cycles.",
        },
        "check_A_feature_utility": {
            "passed":  feat_ok,
            "verdict": feat_msg,
            "note":    (f"Method B uses prior-mean β for n<{N_FEATURE_CYCLES} cycles "
                        f"(feature not computable yet). Comparison is at n={N_FEATURE_CYCLES}, "
                        f"the FIRST snapshot where B diverges from A."),
            "beta_comparison": [
                {
                    "held_out":          c,
                    "beta_true":         loo_results[c]["beta_true"],
                    "beta_feature_pred": loo_results[c]["beta0_feature"],
                    "beta_feature_err_pct": loo_results[c]["beta_feature_err_pct"],
                    "beta_mean":         loo_results[c]["beta0_mean"],
                    "beta_mean_err_pct": loo_results[c]["beta_mean_err_pct"],
                    "extrapolation":     loo_results[c]["feature_extrapolation"],
                }
                for c in CELLS
            ],
        },
        "check_B_deadband_utility":  {"passed": db_ok,   "verdict": db_msg},
        "check_C_fewshot_framing":   {
            "passed": False,
            "verdict": (
                "Structural check — always fails. This is few-shot adaptation (~30 cycles), "
                "NOT Severson-style zero-cycle prediction. R²(n=0) reflects prior only."
            ),
        },
        "r2_snapshots_avg_over_4_folds": {
            m: {str(n): v for n, v in vals.items()}
            for m, vals in agg.items()
        },
        "per_cell_results": {
            cid: {
                k: v for k, v in res.items()
                if k not in ("r2_snapshots",)   # keep it flat; snapshots in agg
            }
            for cid, res in loo_results.items()
        },
        "per_cell_r2_snapshots": {
            cid: {
                m: {str(n): v for n, v in snaps.items()}
                for m, snaps in res["r2_snapshots"].items()
            }
            for cid, res in loo_results.items()
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_JSON), "w") as f:
        json.dump(report, f, indent=2, default=_serialise)
    print(f"\n[JSON] Saved → {OUT_JSON}")

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("OpenCATHODE — Module 4: Cross-Cell Degradation Prediction")
    print(f"Model: ΔSOH = β · k^{GAMMA}  (γ={GAMMA} fixed, D_k=k unit-cycle)")
    print(f"Cells: {CELLS}")
    print(f"Evaluation snapshots (cycles seen): {SNAPSHOTS}")
    print()

    # Load and pre-compute per-cell data
    print("Loading cells and computing SOH, Miner damage, ΔQ(V) features...")
    cell_data: Dict[str, Dict] = {}
    for cid in CELLS:
        mat_path = NASA_DIR / f"{cid}.mat"
        if not mat_path.exists():
            print(f"  [SKIP] {cid}.mat not found at {mat_path}")
            continue
        cycles       = _load_cell(cid)
        soh, D       = _soh_damage(cycles)
        dqv_feature  = _dqv_feature(cycles, N_FEATURE_CYCLES)
        beta_within  = _fit_beta(soh, D, GAMMA)
        soh_pred_within = _predict_soh(D, beta_within, GAMMA)
        r2_within    = _r2(soh, soh_pred_within)
        print(f"  {cid}: {len(cycles)} cycles, β_within={beta_within:.4f}, "
              f"R²_within={r2_within:.4f}, ΔQ(V)_feat={dqv_feature:.4f}")
        cell_data[cid] = {
            "cycles":       cycles,
            "soh":          soh,
            "D":            D,
            "dqv_feature":  dqv_feature,
            "beta_within":  beta_within,
            "r2_within":    r2_within,
        }

    if len(cell_data) < 3:
        print("\n[ERROR] Need at least 3 cells for LOO-CV. "
              "Place B0005/6/7/18.mat in data/nasa/")
        sys.exit(1)

    # Within-cell summary (the 0.97 baseline)
    r2_within_mean = float(np.mean([cell_data[c]["r2_within"] for c in cell_data]))
    print(f"\n  Within-cell R² (fit β on own data): mean={r2_within_mean:.4f}  "
          f"— this is the 0.97 baseline from Module 2\n")

    # LOO-CV
    print("Running leave-one-cell-out cross-validation...")
    loo_results = run_loo_cv(cell_data, GAMMA)

    # Aggregate
    agg = _aggregate(loo_results)

    # Print results
    print_r2_table(agg, loo_results)
    print_checks(loo_results, agg)
    print_honest_validation(loo_results, agg)

    # Save
    save_json(loo_results, agg)

    # Final summary line
    r2_A0  = agg["A_fixed_mean"][0]
    r2_D30 = agg["D_online_db"][30]
    r2_D50 = agg["D_online_db"][50]
    print(f"\n  Summary: fixed-β baseline R²(n=0)={r2_A0:+.3f}  →  "
          f"online+db R²(n=30)={r2_D30:+.3f}  →  R²(n=50)={r2_D50:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Module 4 — Severson LFP extension  (run_severson)
#
#  124 A123 APR18650M1A LFP cells, 3 batches.  Leave-One-Batch-Out (LOBO) CV.
#  Same model: ΔSOH = β · k^γ, γ=0.5, D_k=k.
#  Feature: log10(var_k(Qdlin_k − Qdlin_ref)) on pre-computed 1000-pt LFP grid.
#
#  Key upgrade over 4-cell NASA run:
#   · feature→β map is now a proper linear regression on ~83 training cells
#   · R²_feat→β reported on TRAINING and on HELD-OUT cells separately
#   · β_predicted vs β_true for all 124 cells (not just 4)
#   · prior domination ratio λ₀/Σx² per fold
#   · within-cell R² distribution (γ=0.5 adequacy check)
#   · dead-band innovation magnitude check (LFP fades slowly — τ may gate updates)
#
#  Honest caveats wired in:
#   · LFP feature range wide due to flat 3.3 V plateau; may degrade R²_feat→β
#   · β variance partly protocol-driven (varied charge rates by design)
#   · same manufacturer (A123), same test lab (MIT) — not cross-manufacturer
#   · CHECK A reports training vs held-out R² separately; no tuning to rescue it
# ══════════════════════════════════════════════════════════════════════════════

def _load_severson_cells() -> List[Dict]:
    """Load 124 Severson cells via the severson_loader. Raises if files missing."""
    sys.path.insert(0, str(ROOT / "data" / "loaders"))
    import severson_loader
    return severson_loader.load_severson(verbose=False)


def _beta_regression(F_train: np.ndarray,
                     B_train: np.ndarray) -> Tuple[float, float, float]:
    """
    OLS linear regression: β = a·F + b.
    Returns (a, b, r2_training).
    No regularisation, no outlier removal — report what the data gives.
    """
    F_m = float(np.mean(F_train))
    B_m = float(np.mean(B_train))
    ss_F = float(np.sum((F_train - F_m) ** 2))
    a    = float(np.sum((F_train - F_m) * (B_train - B_m))) / (ss_F + 1e-15)
    b    = B_m - a * F_m
    B_pred = a * F_train + b
    r2   = _r2(B_train, B_pred)
    return a, b, r2


def _run_lobo_cv(cells: List[Dict], gamma: float = GAMMA) -> Dict:
    """
    Leave-One-Batch-Out cross-validation on 124 Severson cells.

    3 folds: held-out batch = 1, 2, or 3.
    Within each fold:
      1. Fit β_i per training cell → (μ_β, σ_β, λ₀)
      2. Fit feature→β OLS on training cells → (a, b, R²_train)
      3. For each held-out cell:
           - compute β₀_feature = clip(a·F_new + b, 1e-6, ∞)
           - run 4 methods (A/B/C/D)
           - record R²(n_cycles_seen) and β prediction quality

    Returns nested dict with per-fold and per-cell results.
    """
    results_by_cell: Dict = {}
    fold_summaries:  List[Dict] = []

    for held_batch in [1, 2, 3]:
        train_cells = [c for c in cells if c["batch"] != held_batch]
        test_cells  = [c for c in cells if c["batch"] == held_batch]
        print(f"\n{'='*68}")
        print(f"  LOBO fold: held-out Batch {held_batch}  "
              f"({len(test_cells)} test, {len(train_cells)} train)")
        print(f"{'='*68}")

        # ── 1. Per-training-cell β fit ──────────────────────────────────
        train_betas    = np.array([_fit_beta(c["soh"], c["D"], gamma)
                                   for c in train_cells])
        train_features = np.array([c["dqv_feature"] for c in train_cells])

        # ── 2. Population prior ─────────────────────────────────────────
        mu_beta  = float(np.mean(train_betas))
        var_beta = float(np.var(train_betas, ddof=1))
        sig_beta = float(var_beta ** 0.5)
        lam0     = 1.0 / (var_beta + 1e-12)

        # Prior domination ratio: compare λ₀ to expected Σx² in the
        # adaptation window [freeze .. freeze+W] at cycle n≈30.
        # x_k = k^γ, Σx² = Σ_{k=freeze}^{freeze+W} k  (γ=0.5, D_k=k)
        window_sum_x2 = float(sum(
            (k ** gamma) ** 2
            for k in range(FREEZE_CYCLES, FREEZE_CYCLES + WINDOW_W + 1)
        ))
        prior_dom_ratio = lam0 / (window_sum_x2 + 1e-12)

        print(f"  Prior: μ_β={mu_beta:.5f}  σ_β={sig_beta:.5f}  "
              f"λ₀={lam0:.1f}")
        print(f"  Window Σx²≈{window_sum_x2:.1f}  "
              f"→ λ₀/Σx²={prior_dom_ratio:.1f}:1  "
              f"{'[PRIOR DOMINATES]' if prior_dom_ratio > 10 else '[BALANCED]'}")

        # ── 3. Feature→β regression on training cells ───────────────────
        # Remove NaN features before fitting (shouldn't happen but guard)
        valid_mask = np.isfinite(train_features) & np.isfinite(train_betas)
        a_feat, b_feat, r2_feat_train = _beta_regression(
            train_features[valid_mask], train_betas[valid_mask]
        )
        print(f"  Feature→β regression:  a={a_feat:.5f}  b={b_feat:.5f}  "
              f"R²_train={r2_feat_train:.3f}")
        feat_range_train = (float(train_features[valid_mask].min()),
                            float(train_features[valid_mask].max()))

        # ── 4. Evaluate each test cell ──────────────────────────────────
        for cell in test_cells:
            cid   = cell["cell_id"]
            soh_n = cell["soh"]
            D_n   = cell["D"]
            F_new = cell["dqv_feature"]
            n_cyc = len(soh_n)

            # Oracle β (within-cell fit; used only for error reporting)
            beta_true = _fit_beta(soh_n, D_n, gamma)

            # Feature-mapped β₀
            if np.isfinite(F_new):
                beta0_feature = float(np.clip(a_feat * F_new + b_feat, 1e-6, None))
                is_extrap = (F_new < feat_range_train[0] or
                             F_new > feat_range_train[1])
                beta_feat_err_pct = (beta0_feature - beta_true) / (beta_true + 1e-9) * 100
            else:
                beta0_feature    = mu_beta
                is_extrap        = False
                beta_feat_err_pct = float("nan")

            beta_mean_err_pct = (mu_beta - beta_true) / (beta_true + 1e-9) * 100

            # ── Method A: fixed population mean ─────────────────────────
            beta_trace_A = np.full(n_cyc, mu_beta)

            # ── Method B: fixed feature-mapped β₀ (no online update)
            # Prior-mean for cycles 0..N_FEATURE_CYCLES-1 (feature not yet
            # computable), feature-mapped β₀ from cycle N_FEATURE_CYCLES on.
            beta_trace_B = np.full(n_cyc, mu_beta)
            if n_cyc > N_FEATURE_CYCLES:
                beta_trace_B[N_FEATURE_CYCLES:] = beta0_feature

            # ── Method C: online RLS, dead-band OFF ─────────────────────
            res_C = _online_adapt(soh_n, D_n, beta0=mu_beta,
                                  mu_prior=mu_beta, lam0=lam0, gamma=gamma,
                                  use_deadband=False)

            # ── Method D: online RLS, dead-band ON ──────────────────────
            res_D = _online_adapt(soh_n, D_n, beta0=mu_beta,
                                  mu_prior=mu_beta, lam0=lam0, gamma=gamma,
                                  use_deadband=True)

            # Innovation magnitude check (for CHECK B)
            innov_abs = np.abs(res_C["innovations"])
            n_below_tau = int(np.sum(innov_abs < DEAD_BAND_TAU))
            max_innov   = float(np.max(innov_abs)) if len(innov_abs) > 0 else 0.0

            # ── R²(n_cycles_seen) ────────────────────────────────────────
            r2_A = _r2_at_snapshots(soh_n, D_n, beta_trace_A, gamma)
            r2_B = _r2_at_snapshots(soh_n, D_n, beta_trace_B, gamma)
            r2_C = _r2_at_snapshots(soh_n, D_n, res_C["beta_trace"], gamma)
            r2_D = _r2_at_snapshots(soh_n, D_n, res_D["beta_trace"], gamma)

            results_by_cell[cid] = {
                "cell_id":              cid,
                "batch":                cell["batch"],
                "held_out_batch":       held_batch,
                "n_cycles":             n_cyc,
                "cycle_life":           cell["cycle_life"],
                # Population prior for this fold
                "mu_beta":              mu_beta,
                "sigma_beta":           sig_beta,
                "lam0":                 lam0,
                "prior_dom_ratio":      prior_dom_ratio,
                # Feature regression for this fold
                "feat_map_a":           a_feat,
                "feat_map_b":           b_feat,
                "r2_feat_train":        r2_feat_train,
                "feat_range_train":     feat_range_train,
                # Per-cell predictions
                "dqv_feature":          F_new,
                "beta_true":            beta_true,
                "beta0_feature":        beta0_feature,
                "beta0_mean":           mu_beta,
                "beta_feat_err_pct":    beta_feat_err_pct,
                "beta_mean_err_pct":    beta_mean_err_pct,
                "feature_extrapolation": is_extrap,
                # Adaptation diagnostics
                "n_cycles_below_tau":   n_below_tau,
                "max_innovation":       max_innov,
                "n_updates_no_db":      res_C["n_updates"],
                "n_updates_db":         res_D["n_updates"],
                # R² snapshots
                "r2_snapshots": {
                    "A_fixed_mean":     r2_A,
                    "B_feature_mapped": r2_B,
                    "C_online_no_db":   r2_C,
                    "D_online_db":      r2_D,
                },
            }

        fold_summaries.append({
            "held_batch":       held_batch,
            "n_train":          len(train_cells),
            "n_test":           len(test_cells),
            "mu_beta":          mu_beta,
            "sigma_beta":       sig_beta,
            "lam0":             lam0,
            "prior_dom_ratio":  prior_dom_ratio,
            "r2_feat_train":    r2_feat_train,
            "feat_map":         {"a": a_feat, "b": b_feat},
            "feat_range_train": feat_range_train,
        })

    return {"cells": results_by_cell, "fold_summaries": fold_summaries}


def _aggregate_lobo(lobo_results: Dict) -> Dict:
    """
    Average R²(n) across ALL 124 held-out cells per method.
    Equal weight per cell (not per fold — folds have unequal sizes).
    """
    methods = ["A_fixed_mean", "B_feature_mapped", "C_online_no_db", "D_online_db"]
    agg: Dict = {m: {} for m in methods}
    cells_dict = lobo_results["cells"]
    for n in SNAPSHOTS:
        for m in methods:
            vals = [
                cells_dict[cid]["r2_snapshots"][m].get(n, float("nan"))
                for cid in cells_dict
            ]
            finite = [v for v in vals if not np.isnan(v)]
            agg[m][n] = float(np.mean(finite)) if finite else float("nan")
    return agg


def _within_cell_r2_distribution(cells: List[Dict],
                                  gamma: float = GAMMA) -> Dict:
    """
    For each cell fit β on its own data and compute within-cell R².
    This checks whether γ=0.5 is adequate for LFP.
    """
    r2s = []
    betas = []
    for c in cells:
        b = _fit_beta(c["soh"], c["D"], gamma)
        r2 = _r2_full(c["soh"], c["D"], b, gamma)
        r2s.append(r2)
        betas.append(b)
    r2s   = np.array(r2s)
    betas = np.array(betas)
    n_poor = int(np.sum(r2s < 0.7))
    n_ok   = int(np.sum(r2s >= 0.7))
    return {
        "r2_min":    float(r2s.min()),
        "r2_p10":    float(np.percentile(r2s, 10)),
        "r2_median": float(np.median(r2s)),
        "r2_p90":    float(np.percentile(r2s, 90)),
        "r2_max":    float(r2s.max()),
        "n_poor_fit":  n_poor,   # R² < 0.7
        "n_good_fit":  n_ok,
        "beta_min":  float(betas.min()),
        "beta_median": float(np.median(betas)),
        "beta_max":  float(betas.max()),
        "per_cell":  [{"cell_id": c["cell_id"],
                       "r2_within": float(r2s[i]),
                       "beta_within": float(betas[i])}
                      for i, c in enumerate(cells)],
    }


def _check_feature_utility_severson(agg: Dict,
                                     lobo_results: Dict) -> Tuple[bool, str]:
    """
    CHECK A (Severson): does feature-mapped β₀ beat plain mean at n=20?

    Reports:
    - ΔR² at n=20 (first snapshot where B diverges from A)
    - R²_feat→β on training cells (per fold)
    - R²_feat→β on ALL held-out cells (the honest test: training vs held-out)
    - β prediction errors across all 124 held-out cells
    """
    r2_A_20 = agg["A_fixed_mean"][20]
    r2_B_20 = agg["B_feature_mapped"][20]
    margin   = r2_B_20 - r2_A_20

    cells_dict = lobo_results["cells"]

    # Held-out R²_feat→β: R² of (β₀_feature vs β_true) across all 124 cells
    feat_pred  = np.array([cells_dict[c]["beta0_feature"]  for c in cells_dict
                           if np.isfinite(cells_dict[c]["beta_feat_err_pct"])])
    beta_true_ = np.array([cells_dict[c]["beta_true"]      for c in cells_dict
                           if np.isfinite(cells_dict[c]["beta_feat_err_pct"])])
    r2_feat_heldout = _r2(beta_true_, feat_pred) if len(feat_pred) > 1 else float("nan")

    # Training R²_feat→β (average across 3 folds)
    r2_feat_train_avg = float(np.mean([f["r2_feat_train"]
                                       for f in lobo_results["fold_summaries"]]))

    # β error statistics
    errs_feat = np.array([abs(cells_dict[c]["beta_feat_err_pct"]) for c in cells_dict
                          if np.isfinite(cells_dict[c]["beta_feat_err_pct"])])
    errs_mean = np.array([abs(cells_dict[c]["beta_mean_err_pct"]) for c in cells_dict])
    n_extrap  = sum(1 for c in cells_dict if cells_dict[c]["feature_extrapolation"])

    err_summary = (
        f"R²_feat→β: train_avg={r2_feat_train_avg:.3f} / held_out={r2_feat_heldout:.3f}  "
        f"({n_extrap}/{len(cells_dict)} cells extrapolate outside training F-range).  "
        f"β_feature MAE={np.mean(errs_feat):.1f}% vs β_mean MAE={np.mean(errs_mean):.1f}%."
    )

    if r2_feat_train_avg > 0.3 and r2_feat_heldout < 0.1:
        training_vs_heldout = (
            f"OVERFITTING DETECTED: train R²={r2_feat_train_avg:.3f} >> "
            f"held-out R²={r2_feat_heldout:.3f}. "
            f"The ΔQ(V) feature does not generalise cross-batch for LFP. "
            f"LFP plateau noise likely corrupts the feature signal. "
        )
    elif r2_feat_heldout < 0.1:
        training_vs_heldout = (
            f"Feature→β regression is WEAK on held-out cells "
            f"(R²_held_out={r2_feat_heldout:.3f}). "
            f"The ΔQ(V) feature does not reliably predict β for LFP cells. "
            f"LFP plateau noise (flat 3.3 V) likely inflates variance non-informatively. "
        )
    elif r2_feat_heldout > 0.3:
        training_vs_heldout = (
            f"Feature→β regression generalises cross-batch "
            f"(train R²={r2_feat_train_avg:.3f}, held-out R²={r2_feat_heldout:.3f}). "
        )
    else:
        training_vs_heldout = (
            f"Feature→β regression is MODEST on held-out cells "
            f"(train R²={r2_feat_train_avg:.3f}, held-out R²={r2_feat_heldout:.3f}). "
        )

    if margin > 0.02:
        verdict = (True,
                   f"Feature-mapped β₀ BEATS plain mean at n=20 (ΔR²={margin:+.3f}). "
                   f"{training_vs_heldout}{err_summary}")
    elif margin > -0.02:
        verdict = (False,
                   f"Feature-mapped β₀ INDISTINGUISHABLE from plain mean at n=20 "
                   f"(ΔR²={margin:+.3f}). {training_vs_heldout}{err_summary}")
    else:
        verdict = (False,
                   f"Feature-mapped β₀ WORSE than plain mean at n=20 "
                   f"(ΔR²={margin:+.3f}). {training_vs_heldout}{err_summary}")
    return verdict


def _check_deadband_severson(agg: Dict, lobo_results: Dict) -> Tuple[bool, str]:
    """
    CHECK B (Severson): does dead-band ON beat dead-band OFF?

    Additional check: what fraction of innovations fall BELOW τ?
    LFP cells fade slowly → innovations may be small → dead-band may
    actually gate updates here (unlike NASA where all innovations > τ).
    """
    r2_C_10  = agg["C_online_no_db"][10]
    r2_D_10  = agg["D_online_db"][10]
    r2_C_20  = agg["C_online_no_db"][20]
    r2_D_20  = agg["D_online_db"][20]
    margin_10 = r2_D_10 - r2_C_10
    margin_20 = r2_D_20 - r2_C_20

    cells_dict = lobo_results["cells"]
    total_cycles = sum(c["n_cycles"] for c in cells_dict.values())
    total_below  = sum(c["n_cycles_below_tau"] for c in cells_dict.values())
    frac_gated   = total_below / max(total_cycles, 1)
    max_innov_median = float(np.median([c["max_innovation"] for c in cells_dict.values()]))

    db_activity = (
        f"Dead-band τ={DEAD_BAND_TAU} SOH: {total_below}/{total_cycles} "
        f"cycle-innovations below τ ({100*frac_gated:.1f}% gated). "
        f"Median max|innovation| per cell = {max_innov_median:.4f} SOH."
    )

    if margin_20 > 0.02 or margin_10 > 0.02:
        verdict = (True,
                   f"Dead-band HELPS: R²(n=10) Δ={margin_10:+.3f}, "
                   f"R²(n=20) Δ={margin_20:+.3f}. {db_activity}")
    elif frac_gated > 0.05:
        verdict = (False,
                   f"Dead-band gates {100*frac_gated:.1f}% of updates but makes "
                   f"NO MEASURABLE R² DIFFERENCE: n=10 Δ={margin_10:+.3f}, "
                   f"n=20 Δ={margin_20:+.3f}. The gated updates were not improving "
                   f"predictions anyway — λ₀ dominates. {db_activity}")
    else:
        verdict = (False,
                   f"Dead-band makes NO MEASURABLE DIFFERENCE: "
                   f"n=10 Δ={margin_10:+.3f}, n=20 Δ={margin_20:+.3f}. "
                   f"Nearly all innovations exceed τ — no updates gated. {db_activity}")
    return verdict


def _print_severson_r2_table(agg: Dict, lobo_results: Dict,
                              within_r2: Dict) -> None:
    cells_dict = lobo_results["cells"]
    methods = ["A_fixed_mean", "B_feature_mapped", "C_online_no_db", "D_online_db"]
    labels  = {
        "A_fixed_mean":     "A  Fixed mean (baseline)",
        "B_feature_mapped": "B  Feature-mapped β₀   ",
        "C_online_no_db":   "C  Online (no dead-band)",
        "D_online_db":      "D  Online + dead-band  ",
    }
    print("\n" + "="*72)
    print("  Severson LFP — R²(n_cycles_seen) averaged over 124 held-out cells")
    print("  Evaluation: LOBO (Leave-One-Batch-Out), 3 folds")
    print(f"  NOTE: Method B uses prior-mean β for n<{N_FEATURE_CYCLES}; "
          f"diverges from A at n≥{N_FEATURE_CYCLES}.")
    print("="*72)
    header = f"  {'Method':<27}" + "".join(f"  n={n:<5}" for n in SNAPSHOTS)
    print(header)
    print("  " + "-"*68)
    for m in methods:
        row = f"  {labels[m]}"
        for n in SNAPSHOTS:
            v = agg[m].get(n, float("nan"))
            if np.isnan(v):
                row += "    nan"
            elif m == "B_feature_mapped" and n < N_FEATURE_CYCLES:
                row += f"  {v:+.3f}*"
            else:
                row += f"  {v:+.3f} "
        print(row)
    print(f"  * Method B n<{N_FEATURE_CYCLES}: identical to A (feature requires "
          f"{N_FEATURE_CYCLES} cycles)")
    print()

    # Within-cell R² distribution (γ=0.5 adequacy)
    w = within_r2
    print("  Within-cell R² distribution (γ=0.5 fixed, D_k=k — adequacy check):")
    print(f"    min={w['r2_min']:.3f}  p10={w['r2_p10']:.3f}  "
          f"median={w['r2_median']:.3f}  p90={w['r2_p90']:.3f}  "
          f"max={w['r2_max']:.3f}")
    print(f"    Poor fits (R²<0.70): {w['n_poor_fit']}/{w['n_poor_fit']+w['n_good_fit']} cells")
    if w["n_poor_fit"] > 10:
        print(f"    NOTE: γ=0.5 gives poor within-cell fits for "
              f"{w['n_poor_fit']} cells — the power-law model may not hold "
              f"for all LFP degradation trajectories in this dataset.")
    print()

    # β distribution
    print(f"  β distribution across 124 cells: "
          f"min={w['beta_min']:.5f}  "
          f"median={w['beta_median']:.5f}  "
          f"max={w['beta_max']:.5f}")
    print()

    # LOBO fold summaries
    print("  LOBO fold diagnostics:")
    print(f"  {'Held batch':>11}  {'n_train':>7}  {'n_test':>6}  "
          f"{'μ_β':>8}  {'σ_β':>8}  {'λ₀':>10}  {'λ₀/Σx²':>8}  "
          f"{'R²feat(train)':>13}")
    for fs in lobo_results["fold_summaries"]:
        print(f"  {'Batch '+str(fs['held_batch']):>11}  "
              f"{fs['n_train']:>7}  {fs['n_test']:>6}  "
              f"{fs['mu_beta']:>8.5f}  {fs['sigma_beta']:>8.5f}  "
              f"{fs['lam0']:>10.1f}  {fs['prior_dom_ratio']:>8.1f}  "
              f"{fs['r2_feat_train']:>13.3f}")
    print()

    # β_predicted vs β_true — per-batch summary
    print("  β_feature vs β_true by batch (all held-out cells):")
    print(f"  {'Batch':>6}  {'n':>4}  {'β_true range':>14}  "
          f"{'feat MAE%':>10}  {'mean MAE%':>10}  {'R²_feat→β':>10}  "
          f"{'n_extrap':>8}")
    for held_b in [1, 2, 3]:
        batch_cells = [cells_dict[cid] for cid in cells_dict
                       if cells_dict[cid]["batch"] == held_b]
        bt   = np.array([c["beta_true"]        for c in batch_cells])
        bf   = np.array([c["beta0_feature"]     for c in batch_cells])
        bm   = np.array([c["beta0_mean"]        for c in batch_cells])
        ferr = np.array([abs(c["beta_feat_err_pct"]) for c in batch_cells
                         if np.isfinite(c["beta_feat_err_pct"])])
        merr = np.array([abs(c["beta_mean_err_pct"]) for c in batch_cells])
        r2fb = _r2(bt, bf) if len(bt) > 1 else float("nan")
        n_ex = sum(1 for c in batch_cells if c["feature_extrapolation"])
        print(f"  {held_b:>6}  {len(batch_cells):>4}  "
              f"[{bt.min():.4f}, {bt.max():.4f}]  "
              f"{np.mean(ferr):>9.1f}%  {np.mean(merr):>9.1f}%  "
              f"{r2fb:>10.3f}  {n_ex:>8}")
    print()


def _print_severson_checks(lobo_results: Dict, agg: Dict,
                            within_r2: Dict) -> None:
    feat_ok, feat_msg = _check_feature_utility_severson(agg, lobo_results)
    db_ok,   db_msg   = _check_deadband_severson(agg, lobo_results)

    gamma_note = (
        f"γ=0.5 ADEQUATE for {within_r2['n_good_fit']} cells (R²≥0.70), "
        f"POOR for {within_r2['n_poor_fit']} cells (R²<0.70, "
        f"median within-cell R²={within_r2['r2_median']:.3f})."
    )
    if within_r2["n_poor_fit"] > 20:
        gamma_note += (
            " The power-law ΔSOH=β·k^0.5 likely does not capture "
            "all LFP degradation trajectories — some cells may have "
            "a different functional form (concave / two-phase fade). "
            "γ should be treated as approximate for LFP."
        )

    print("="*72)
    print("  3 HONESTY CHECKS  (Severson LFP, 124-cell LOBO)")
    print("="*72)

    print(f"\n  CHECK A — ΔQ(V) feature utility (training vs held-out R²):")
    print(f"    {'✓' if feat_ok else '✗'}  {feat_msg}")

    print(f"\n  CHECK B — Dead-band utility + innovation magnitude:")
    print(f"    {'✓' if db_ok else '✗'}  {db_msg}")

    print(f"\n  CHECK C — Few-shot framing (structural):")
    print(f"    ✗  NOT zero-cycle prediction. β unidentifiable at n=0. "
          f"R²(n=0) reflects only the population prior. "
          f"With 83-84 training cells the prior mean is better estimated "
          f"than NASA's 3-cell prior, but a new cell can still fall outside "
          f"the prior range (especially at manufacturing extremes or novel "
          f"fast-charge protocols). Adaptation requires ~20–50 cycles.")

    print(f"\n  CHECK γ — Power-law adequacy for LFP (additional):")
    print(f"    {gamma_note}")
    print()


def _print_severson_scope(agg: Dict, within_r2: Dict,
                          lobo_results: Dict) -> None:
    r2_A0  = agg["A_fixed_mean"][0]
    r2_D30 = agg["D_online_db"][30]
    r2_D50 = agg["D_online_db"][50]
    dom_ratios = [f["prior_dom_ratio"] for f in lobo_results["fold_summaries"]]

    print("="*72)
    print("  VALIDATION SCOPE — SEVERSON LFP")
    print("="*72)
    print(f"""
  CAN claim:
    · Cross-cell R² on 124 LFP cells (A123 APR18650M1A) using LOBO-CV
      — the largest within-chemistry cross-cell validation in this stack.
    · Prior mean R²(n=0) = {r2_A0:+.3f}. Online adaptation to
      R²(n=30)={r2_D30:+.3f}, R²(n=50)={r2_D50:+.3f}.
    · Whether ΔQ(V) feature warm-start generalises cross-batch (CHECK A).
    · Whether dead-band gates LFP updates (LFP fades slowly — CHECK B).
    · Within-cell R² distribution for γ=0.5 adequacy (CHECK γ).

  CANNOT claim:
    · Cross-manufacturer or cross-chemistry generalisation —
      all 124 cells are A123 APR18650M1A, MIT test facility.
    · Cross-chemistry comparison with NASA (NMC vs LFP, different feature
      space, different degradation mechanism prominence).
    · Zero-cycle early prediction (CHECK C — structural).

  HONEST CAVEATS:
    · β variance is partly PROTOCOL-driven: Severson varied charge C-rates
      by design. Higher C-rate charge → higher β. The ΔQ(V) feature is
      correlated with β FOR THIS REASON (Severson 2019 finding). In a
      real deployment the charge protocol may be unknown or uncontrolled.
    · LFP feature noise: dqv_feature range spans ~8 log-decades due to
      voltage sensitivity on the flat 3.3 V LFP plateau. This may degrade
      the feature→β regression relative to NMC.
    · Prior domination ratios per fold: {[f'{r:.1f}:1' for r in dom_ratios]}.
      {('Prior STILL dominates adaptation — λ₀/Σx² >> 1 even with 83-84 '
        'training cells.') if min(dom_ratios) > 10 else
       ('Prior no longer dominates — adaptation now makes a real contribution.')}
    · γ=0.5 assumed (SEI ∝ √k). Median within-cell R²={within_r2['r2_median']:.3f}.
      If LFP degradation is concave or two-phase for some cells,
      the model is misspecified and R² will be lower than Module 2's 0.9725.

  WHAT'S GENUINELY NOVEL vs NASA:
    · 4 cells → 124 cells: feature→β map is now a proper regression,
      not a 3-point interpolation.
    · LOBO-CV is a cross-batch test — stronger than within-batch LOO-CV.
    · Explicit training vs held-out R²_feat→β reveals whether the feature
      signal survives the LFP plateau noise (CHECK A above).
""")


def _save_severson_json(lobo_results: Dict, agg: Dict,
                        within_r2: Dict) -> None:
    feat_ok, feat_msg = _check_feature_utility_severson(agg, lobo_results)
    db_ok,   db_msg   = _check_deadband_severson(agg, lobo_results)

    cells_dict = lobo_results["cells"]

    # held-out R²_feat→β overall
    feat_pred  = np.array([cells_dict[c]["beta0_feature"] for c in cells_dict
                           if np.isfinite(cells_dict[c]["beta_feat_err_pct"])])
    beta_true_ = np.array([cells_dict[c]["beta_true"]     for c in cells_dict
                           if np.isfinite(cells_dict[c]["beta_feat_err_pct"])])
    r2_feat_heldout = _r2(beta_true_, feat_pred) if len(feat_pred) > 1 else None

    # β_predicted vs β_true table (all 124 cells, sorted by cell_id)
    beta_table = sorted([
        {
            "cell_id":           cid,
            "batch":             cells_dict[cid]["batch"],
            "beta_true":         cells_dict[cid]["beta_true"],
            "beta_feature_pred": cells_dict[cid]["beta0_feature"],
            "beta_mean_pred":    cells_dict[cid]["beta0_mean"],
            "beta_feat_err_pct": cells_dict[cid]["beta_feat_err_pct"],
            "beta_mean_err_pct": cells_dict[cid]["beta_mean_err_pct"],
            "dqv_feature":       cells_dict[cid]["dqv_feature"],
            "extrapolation":     cells_dict[cid]["feature_extrapolation"],
        }
        for cid in cells_dict
    ], key=lambda x: (x["batch"], x["cell_id"]))

    report = {
        "module": "Module 4 — Cross-Cell Degradation Prediction (Severson LFP)",
        "model":  f"ΔSOH = β · k^{GAMMA}  (γ={GAMMA} fixed, D_k=k unit-cycle damage)",
        "dataset": {
            "name":         "Severson et al. (2019) Nature Energy 4:383–391",
            "chemistry":    "LFP/graphite — A123 APR18650M1A, 1.1 Ah nominal",
            "manufacturer": "A123, single test facility (MIT) — NOT cross-manufacturer",
            "n_cells":      len(cells_dict),
            "evaluation":   "Leave-One-Batch-Out (LOBO), 3 folds",
            "honest_note":  (
                "β variance is partly protocol-driven (varied charge C-rates). "
                "LFP feature range wide due to flat 3.3 V plateau. "
                "Cross-manufacturer generalisation NOT tested."
            ),
        },
        "check_A_feature_utility": {
            "passed":  feat_ok,
            "verdict": feat_msg,
            "r2_feat_heldout_overall": r2_feat_heldout,
            "r2_feat_train_per_fold":  [f["r2_feat_train"]
                                        for f in lobo_results["fold_summaries"]],
            "note": (f"Method B uses prior-mean for n<{N_FEATURE_CYCLES} cycles. "
                     f"Training and held-out R² reported separately to detect overfitting."),
            "beta_table_all_124": beta_table,
        },
        "check_B_deadband_utility": {"passed": db_ok, "verdict": db_msg},
        "check_C_fewshot_framing":  {
            "passed": False,
            "verdict": ("Structural — always fails. NOT zero-cycle prediction. "
                        "R²(n=0) reflects population prior only."),
        },
        "check_gamma_adequacy": {
            "gamma_fixed": GAMMA,
            "model":       "ΔSOH = β · k^0.5",
            "r2_within_cell_distribution": {
                k: v for k, v in within_r2.items() if k != "per_cell"
            },
            "verdict": (
                f"γ=0.5 gives adequate fits (R²≥0.70) for "
                f"{within_r2['n_good_fit']}/{within_r2['n_good_fit']+within_r2['n_poor_fit']} "
                f"cells (median R²={within_r2['r2_median']:.3f}). "
                + ("Poor fits for many cells suggest the power-law model "
                   "may be misspecified for some LFP degradation trajectories."
                   if within_r2["n_poor_fit"] > 20 else
                   "γ=0.5 is a reasonable approximation for this dataset.")
            ),
        },
        "r2_snapshots_avg_124_cells": {
            m: {str(n): v for n, v in vals.items()}
            for m, vals in agg.items()
        },
        "lobo_fold_summaries": lobo_results["fold_summaries"],
        "per_cell_r2_snapshots": {
            cid: {
                m: {str(n): v for n, v in snaps.items()}
                for m, snaps in cells_dict[cid]["r2_snapshots"].items()
            }
            for cid in cells_dict
        },
    }

    OUT_JSON_SEV.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_JSON_SEV), "w") as fh:
        json.dump(report, fh, indent=2, default=_serialise)
    print(f"\n[JSON] Saved → {OUT_JSON_SEV}")


def run_severson() -> None:
    """Module 4 on Severson 124-cell LFP dataset. LOBO-CV."""
    print("OpenCATHODE — Module 4 (Severson LFP Extension)")
    print(f"Model: ΔSOH = β · k^{GAMMA}  (γ={GAMMA} fixed, D_k=k unit-cycle)")
    print(f"Evaluation: Leave-One-Batch-Out (LOBO), 3 folds")
    print(f"Snapshots (cycles seen): {SNAPSHOTS}")
    print()

    # ── Load cells ──────────────────────────────────────────────────────────
    print("Loading Severson cells via severson_loader (may take ~30 s)...")
    try:
        cells = _load_severson_cells()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    print(f"  Loaded {len(cells)} cells  "
          f"(Batch 1: {sum(1 for c in cells if c['batch']==1)}, "
          f"Batch 2: {sum(1 for c in cells if c['batch']==2)}, "
          f"Batch 3: {sum(1 for c in cells if c['batch']==3)})")

    # ── Within-cell R² distribution (γ=0.5 adequacy) ───────────────────────
    print("\nComputing within-cell β fits and R² distribution (γ=0.5 check)...")
    within_r2 = _within_cell_r2_distribution(cells, GAMMA)
    print(f"  Within-cell R²: median={within_r2['r2_median']:.3f}  "
          f"p10={within_r2['r2_p10']:.3f}  p90={within_r2['r2_p90']:.3f}")
    print(f"  Poor fits (R²<0.70): {within_r2['n_poor_fit']} cells")
    print(f"  β range: [{within_r2['beta_min']:.5f}, {within_r2['beta_max']:.5f}]  "
          f"median={within_r2['beta_median']:.5f}")

    # ── LOBO-CV ─────────────────────────────────────────────────────────────
    print("\nRunning LOBO cross-validation (3 folds × up to 43 cells)...")
    lobo_results = _run_lobo_cv(cells, GAMMA)

    # ── Aggregate R²(n) over all 124 held-out cells ─────────────────────────
    agg = _aggregate_lobo(lobo_results)

    # ── Print results ────────────────────────────────────────────────────────
    _print_severson_r2_table(agg, lobo_results, within_r2)
    _print_severson_checks(lobo_results, agg, within_r2)
    _print_severson_scope(agg, within_r2, lobo_results)

    # ── Save JSON ────────────────────────────────────────────────────────────
    _save_severson_json(lobo_results, agg, within_r2)

    r2_A0  = agg["A_fixed_mean"][0]
    r2_D30 = agg["D_online_db"][30]
    r2_D50 = agg["D_online_db"][50]
    print(f"\n  Summary: prior-only R²(n=0)={r2_A0:+.3f}  →  "
          f"online+db R²(n=30)={r2_D30:+.3f}  →  R²(n=50)={r2_D50:+.3f}")


# ── Severson cycle-life predictor (Option A, separate from β-model) ───────────
#
# Predicts log10(cycle_life) — cycles to 80% nominal capacity — from a single
# early-cycle ΔQ(V) feature.  This is "Ather's Problem 1": limited-data lifetime
# prediction before significant degradation occurs.
#
# What this is NOT:
#   · Does NOT give a cycle-by-cycle SOH trajectory.
#   · Does NOT track within-life degradation rate (that is the β-model above).
#
# Feature: log( var( Qdlin[99] − Qdlin[9] ) )   [natural log, population var]
#   Validated in validation/severson_reproduce.py:
#   ρ=−0.89 (paper −0.93), mean % error 15.2% on B1+B2-odd / B2-even split.
#
# Primary evaluation: Leave-One-Batch-Out (LOBO) cross-validation, 3 folds.
# Reference-only:     B1+B2-odd-indexed / B2-even-indexed split (labelled clearly).


def _load_severson_cl_features() -> Tuple[List[Dict], List[Dict]]:
    """
    Load ΔQ(V) feature and cycle_life for all 124 Severson cells directly from
    the raw HDF5 .mat files.  Does NOT use severson_loader's dqv_feature field,
    which uses a different formula (early-cycle variance relative to cycle 2).

    Returns
    -------
    (included, skipped)
      included : list of dicts — cell_id, batch, cycle_life, cell_idx, feature
      skipped  : list of dicts — cell_id, batch, cycle_life, cell_idx, reason
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py")

    included: List[Dict] = []
    skipped:  List[Dict] = []

    for batch_num, fname in SEVERSON_BATCH_FILES:
        path = SEVERSON_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Severson file: {path}\nDownload from https://data.matr.io/1/"
            )
        prefix = f"b{batch_num}c"
        excl   = SEVERSON_EXCLUDE[batch_num]

        with h5py.File(path, "r") as f:
            batch  = f["batch"]
            n_raw  = batch["summary"].shape[0]

            for i in range(n_raw):
                cid = f"{prefix}{i}"
                if cid in excl:
                    continue

                cl = int(f[batch["cycle_life"][i, 0]][()].flat[0])
                cyc = f[batch["cycles"][i, 0]]
                n_stored = cyc["Qdlin"].shape[0]

                if n_stored < 100:
                    skipped.append({
                        "cell_id":    cid,
                        "batch":      batch_num,
                        "cycle_life": cl,
                        "cell_idx":   i,
                        "reason":     f"n_stored={n_stored} < 100 (cannot compute Qdlin[99])",
                    })
                    continue

                q9_arr  = f[cyc["Qdlin"][9,  0]][()].flatten()
                q99_arr = f[cyc["Qdlin"][99, 0]][()].flatten()

                if len(q9_arr) != 1000 or len(q99_arr) != 1000:
                    skipped.append({
                        "cell_id":    cid,
                        "batch":      batch_num,
                        "cycle_life": cl,
                        "cell_idx":   i,
                        "reason":     f"Qdlin not 1000 pts (len9={len(q9_arr)}, len99={len(q99_arr)})",
                    })
                    continue

                dq      = q99_arr.astype(np.float64) - q9_arr.astype(np.float64)
                feature = float(np.log(np.var(dq) + 1e-20))

                included.append({
                    "cell_id":    cid,
                    "batch":      batch_num,
                    "cycle_life": cl,
                    "cell_idx":   i,
                    "feature":    feature,
                })

    return included, skipped


def _ols_cl(F: np.ndarray, L: np.ndarray) -> Tuple[float, float, float]:
    """Plain OLS: L = w0 + w1·F.  Returns (w0, w1, R²_training)."""
    Fm, Lm = float(F.mean()), float(L.mean())
    denom   = float(np.dot(F - Fm, F - Fm)) + 1e-15
    w1      = float(np.dot(F - Fm, L - Lm)) / denom
    w0      = Lm - w1 * Fm
    Lp      = w0 + w1 * F
    r2      = 1.0 - float(np.sum((L - Lp) ** 2)) / (float(np.sum((L - Lm) ** 2)) + 1e-15)
    return w0, w1, float(r2)


def _eval_cl(w0: float, w1: float,
             F_te: np.ndarray, cl_te: np.ndarray,
             cell_ids: List[str]) -> Dict:
    """
    Evaluate OLS predictions on a test set.

    mean_pct_err = mean over test cells of
                   |pred_cycles − true_cycles| / true_cycles × 100
    (computed in CYCLE space, not log space)
    """
    pred_log    = w0 + w1 * F_te
    pred_cycles = np.power(10.0, pred_log)
    true_cycles = cl_te.astype(float)

    abs_pct     = np.abs(pred_cycles - true_cycles) / (true_cycles + 1e-6) * 100.0
    rmse        = float(np.sqrt(np.mean((pred_cycles - true_cycles) ** 2)))
    mean_pct    = float(np.mean(abs_pct))

    L_true = np.log10(true_cycles + 1e-6)
    Lm     = float(L_true.mean())
    r2_log = 1.0 - float(np.sum((L_true - pred_log) ** 2)) / (
                 float(np.sum((L_true - Lm) ** 2)) + 1e-15)

    per_cell = []
    for cid, f_val, cl_true, cl_pred, pct_err in zip(
            cell_ids, F_te, true_cycles, pred_cycles, abs_pct):
        per_cell.append({
            "cell_id":         cid,
            "true_cycles":     int(cl_true),
            "pred_log10":      round(float(np.log10(cl_pred + 1e-6)), 4),
            "pred_cycles":     round(float(cl_pred), 1),
            "abs_pct_err":     round(float(pct_err), 2),
        })

    return {
        "n_test":       len(F_te),
        "rmse_cycles":  round(rmse, 1),
        "mean_pct_err": round(mean_pct, 2),
        "r2_log":       round(r2_log, 4),
        "per_cell":     per_cell,
    }


def _lobo_cv_cl(cells: List[Dict]) -> Dict:
    """
    3-fold Leave-One-Batch-Out cross-validation for cycle-life prediction.

    Folds:
      fold_test_B1 : train=B2+B3 (83), test=B1 (41)
      fold_test_B2 : train=B1+B3 (81), test=B2 (43)
      fold_test_B3 : train=B1+B2 (84), test=B3 (40)
    """
    def _split(test_batch: int):
        tr = [c for c in cells if c["batch"] != test_batch]
        te = [c for c in cells if c["batch"] == test_batch]
        return tr, te

    results = {}
    for held_batch in (1, 2, 3):
        tr, te = _split(held_batch)
        F_tr = np.array([c["feature"] for c in tr])
        L_tr = np.log10(np.array([c["cycle_life"] for c in tr], dtype=float))
        F_te = np.array([c["feature"] for c in te])
        cl_te = np.array([c["cycle_life"] for c in te], dtype=float)
        ids_te = [c["cell_id"] for c in te]

        w0, w1, r2_tr = _ols_cl(F_tr, L_tr)
        fold_eval = _eval_cl(w0, w1, F_te, cl_te, ids_te)
        fold_eval["n_train"]   = len(tr)
        fold_eval["w0"]        = round(w0, 6)
        fold_eval["w1"]        = round(w1, 6)
        fold_eval["r2_train"]  = round(r2_tr, 4)
        results[f"fold_test_B{held_batch}"] = fold_eval

    return results


def _paper_split_cl(cells: List[Dict]) -> Dict:
    """
    Reference split: train = B1 + B2-odd-indexed, test = B2-even-indexed.
    'Odd-indexed' means cells whose cell_idx is odd (b2c1, b2c3, ...).
    Clearly labelled as reference-only — not the primary LOBO metric.
    """
    b2_train = [c for c in cells if c["batch"] == 2 and c["cell_idx"] % 2 == 1]
    b2_test  = [c for c in cells if c["batch"] == 2 and c["cell_idx"] % 2 == 0]
    b1       = [c for c in cells if c["batch"] == 1]
    tr       = b1 + b2_train

    F_tr  = np.array([c["feature"] for c in tr])
    L_tr  = np.log10(np.array([c["cycle_life"] for c in tr], dtype=float))
    F_te  = np.array([c["feature"] for c in b2_test])
    cl_te = np.array([c["cycle_life"] for c in b2_test], dtype=float)
    ids_te = [c["cell_id"] for c in b2_test]

    w0, w1, r2_tr = _ols_cl(F_tr, L_tr)
    result = _eval_cl(w0, w1, F_te, cl_te, ids_te)
    result["n_train"]  = len(tr)
    result["w0"]       = round(w0, 6)
    result["w1"]       = round(w1, 6)
    result["r2_train"] = round(r2_tr, 4)
    result["label"]    = ("B1+B2-odd-indexed / B2-even-indexed "
                          "(reference only — NOT the primary LOBO metric)")
    return result


def run_severson_cycle_life() -> None:
    """
    Predict log10(cycle_life) from the validated ΔQ(V) variance feature.

    This is a separate code path from run_severson() (β-model).
    The β/√k model is wrong for LFP two-phase degradation; this function
    replaces that evaluation with the correct target (cycle-life, not β).

    Invoke via:  python degradation/cross_cell_predictor.py --cycle-life
    """
    print("OpenCATHODE — Severson Cycle-Life Predictor (Option A)")
    print("Target : log10(cycle_life)  [cycles to 80% capacity]")
    print("Feature: log( var( Qdlin[99] − Qdlin[9] ) )  [validated: ρ=−0.89]")
    print("Eval   : Leave-One-Batch-Out (LOBO), 3 folds  [primary metric]")
    print()

    # ── Load features ────────────────────────────────────────────────────────
    print("Loading features from Severson .mat files...")
    try:
        included, skipped = _load_severson_cl_features()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    by_batch = {1: [], 2: [], 3: []}
    for c in included:
        by_batch[c["batch"]].append(c)

    print(f"  Included: {len(included)} cells  "
          f"(B1={len(by_batch[1])}, B2={len(by_batch[2])}, B3={len(by_batch[3])})")
    if skipped:
        print(f"  Skipped:  {len(skipped)} cells — "
              + ", ".join(f"{s['cell_id']} ({s['reason']})" for s in skipped))
    print()

    # ── Global Pearson ρ ─────────────────────────────────────────────────────
    all_feats = np.array([c["feature"]    for c in included])
    all_cl    = np.array([c["cycle_life"] for c in included], dtype=float)
    rho       = float(np.corrcoef(all_feats, np.log10(all_cl))[0, 1])
    print(f"Global Pearson ρ(feature, log10(cycle_life)): {rho:.4f}")
    print()

    # ── LOBO-CV (primary) ────────────────────────────────────────────────────
    print("Running LOBO-CV...")
    lobo = _lobo_cv_cl(included)

    print(f"\n{'Fold':<20}  {'N_train':>7}  {'N_test':>6}  "
          f"{'RMSE (cyc)':>11}  {'Mean%err':>9}  {'R²(log)':>8}")
    print("─" * 70)
    for fold_key, res in lobo.items():
        batch_label = fold_key.replace("fold_test_", "test=Batch")
        print(f"  {batch_label:<18}  {res['n_train']:>7}  {res['n_test']:>6}  "
              f"{res['rmse_cycles']:>11.1f}  {res['mean_pct_err']:>8.1f}%  "
              f"{res['r2_log']:>8.4f}")

    lobo_pct = [res["mean_pct_err"] for res in lobo.values()]
    lobo_rmse = [res["rmse_cycles"] for res in lobo.values()]
    print("─" * 70)
    print(f"  {'Macro mean':<18}  {'':>7}  {'':>6}  "
          f"{float(np.mean(lobo_rmse)):>11.1f}  "
          f"{float(np.mean(lobo_pct)):>8.1f}%")
    print()

    # ── Paper-style split (reference) ────────────────────────────────────────
    ref = _paper_split_cl(included)
    print(f"Reference split (B1+B2-odd / B2-even, n_train={ref['n_train']}, "
          f"n_test={ref['n_test']}):")
    print(f"  RMSE={ref['rmse_cycles']:.1f} cycles   "
          f"Mean%err={ref['mean_pct_err']:.1f}%   R²={ref['r2_log']:.4f}")
    print(f"  [reference only — not the primary metric; validated in severson_reproduce.py]")
    print()

    # ── Honest caveats ────────────────────────────────────────────────────────
    caveats = [
        ("scalar_only",
         "Predicts cycle_life (cycles to 80% nominal capacity) — a single scalar per cell. "
         "Does NOT give the cycle-by-cycle SOH trajectory."),
        ("single_manufacturer",
         "Trained on A123 APR18650M1A (1.1 Ah LFP/graphite) only. "
         "Cross-manufacturer accuracy is untested."),
        ("protocol_confounded",
         "Severson varied fast-charge rates by design; the variance feature partly captures "
         "protocol aggressiveness alongside cell health. OLS slope is calibrated to "
         "Severson's specific protocol range."),
        ("lobo_conservative",
         "LOBO tests worst-case batch-level generalization (batches differ in protocol "
         "distribution). Within-population accuracy would likely be better."),
        ("lfp_specific",
         "Feature exploits the LFP plateau's sensitivity to capacity fade. "
         "Not directly applicable to NMC (different voltage profile, different mechanism)."),
    ]

    print("Honest caveats:")
    for key, txt in caveats:
        print(f"  [{key}] {txt}")
    print()

    # ── Assemble JSON ─────────────────────────────────────────────────────────
    per_cell_lobo = []
    for fold_key, res in lobo.items():
        fold_num = int(fold_key[-1])
        for pc in res["per_cell"]:
            per_cell_lobo.append({**pc, "lobo_held_batch": fold_num})
    per_cell_lobo.sort(key=lambda x: (x["lobo_held_batch"], x["cell_id"]))

    report = {
        "meta": {
            "script":        "degradation/cross_cell_predictor.py --cycle-life",
            "dataset":       "Severson et al. 2019, 124 LFP cells (A123 APR18650M1A)",
            "target":        "log10(cycle_life)",
            "feature":       "log( var( Qdlin[99] − Qdlin[9] ) )  [natural log, population var]",
            "feature_note":  ("Qdlin[k] is the pre-computed 1000-point discharge capacity "
                              "on the fixed Vdlin grid [3.5→2.0 V].  Indices 9 and 99 "
                              "correspond to MATLAB cycles 10 and 100 (0-indexed)."),
            "mean_pct_err_formula": (
                "mean over test cells of "
                "|pred_cycles − true_cycles| / true_cycles × 100 "
                "(computed in CYCLE space, not log space)"),
            "what_this_predicts": (
                "Cycles to 80% nominal capacity (cycle_life scalar). "
                "This is 'limited-data lifetime prediction from early cycles' "
                "(Ather's Problem 1). It does NOT give the cycle-by-cycle SOH curve."),
            "primary_eval":       "LOBO-CV (3 folds: train on 2 batches, test on 1)",
            "reference_eval":     "B1+B2-odd-indexed / B2-even-indexed split",
            "n_cells_included":   len(included),
            "n_cells_skipped":    len(skipped),
            "skipped_cells":      skipped,
            "global_pearson_rho": round(rho, 4),
            "ols_model":          "log10(cycle_life) = w0 + w1 * feature  (plain OLS)",
        },
        "lobo_cv": {
            **{k: {kk: vv for kk, vv in v.items() if kk != "per_cell"}
               for k, v in lobo.items()},
            "macro_mean_pct_err":  round(float(np.mean(lobo_pct)), 2),
            "macro_mean_rmse":     round(float(np.mean(lobo_rmse)), 1),
        },
        "paper_split_reference": {
            kk: vv for kk, vv in ref.items() if kk != "per_cell"
        },
        "honest_caveats": {k: v for k, v in caveats},
        "per_cell_lobo_predictions": per_cell_lobo,
    }

    OUT_JSON_CL.parent.mkdir(parents=True, exist_ok=True)
    with open(str(OUT_JSON_CL), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[JSON] Saved → {OUT_JSON_CL}")


if __name__ == "__main__":
    if "--cycle-life" in sys.argv:
        run_severson_cycle_life()
    elif "--severson" in sys.argv:
        run_severson()
    else:
        main()
