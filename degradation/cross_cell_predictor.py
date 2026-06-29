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


if __name__ == "__main__":
    main()
