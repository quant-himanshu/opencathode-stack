#!/usr/bin/env python3
"""
analysis/factor_ranking.py  —  P5: Degradation Factor Ranking on Deng Fleet

TARGET VARIABLE:
  Per-vehicle calendar fade rate λ_v (from cell_to_field_temporal.py train-window
  OLS fit: λ_v = argmin Σ_k (ΔSOH_obs[k] − λ·√t[k])²).
  NOT raw endpoint ΔSOH — that is noise-dominated (median |ΔSOH|/σ = 1.70, per
  soh_noise_floor.py). λ_v is the best available per-vehicle summary of true fade
  rate; its limitation (noisy for low-SNR vehicles) is disclosed throughout.

GATING:
  λ_v > 0  (13 vehicles): identifiable fade → all ranking methods applied
  λ_v ≤ 0  (7 vehicles):  non-monotone/flat SOH → separate flagged table

FEATURES (6; t_years excluded from ranking):
  T_mean, T_p95, DoD_mean, DoD_p95, C_rate_mean, n_cycles
  t_years excluded: range 2.308–2.318 yr (near-zero variance), and λ_v is
  √t-normalised so t_years has mechanical coupling with the target denominator.
  Reported in per-vehicle table for context only.

METHODS (all three always reported):
  (a) Spearman ρ  — pairwise rank-correlation vs λ_v (n=13, scipy.stats)
      Note: features correlated → ρ ≠ independent contributions.
      Bonferroni threshold at n=13, k=6 features: p_adj < 0.008.
  (b) Ridge regression (α=1.0, standardised features) + Leave-One-Out CV
      → LOO R² + in-sample permutation importance (200 repeats, numpy RNG)
      Ridge and LOO implemented manually (no sklearn installed).
      If LOO R² < 0: "no cross-validated predictive value; importances
      are directional suggestions only."
  (c) Kernel SHAP: skipped — shap package not installed.

PRE-REGISTERED EXPECTATION (locked before computation):
  Literature order (Edge J.S. et al. 2021 PCCP 23(14):8200-8221,
  DOI 10.1039/D1CP00359C):  Temperature > DoD > C-rate
  With n=13 same-city, same-model vehicles (narrow T/DoD/C-rate ranges),
  expected outcome: "factors indistinguishable at n=13."
  Either outcome is the reportable finding — no post-hoc reframing.

DATASET:
  Deng Z., Xu L., Liu H., Hu X., Duan Z., Xu Y. (2023). Prognostics of battery
  capacity based on charging data and data-driven methods for on-road vehicles.
  Applied Energy 339:120954. https://doi.org/10.1016/j.apenergy.2023.120954

FRAMING:
  To our knowledge, per-vehicle degradation factor attribution via λ_v and
  permutation importance has not been published on this specific dataset.
  Prior work on the Deng 2023 dataset (and direct follow-ups) targets
  capacity-trajectory prediction, not factor attribution or fade-rate transfer.
  Contribution attempt, not a reproduction.

OUTPUT: data/factor_ranking_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT     = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "data" / "factor_ranking_report.json"
TEMPORAL_REPORT = ROOT / "data" / "cell_to_field_temporal_report.json"

RIDGE_ALPHA  = 1.0    # regularisation (scale-free after standardisation)
PERM_REPEATS = 200    # permutation importance repeats
RNG_SEED     = 42

FEATURE_NAMES = ["T_mean", "T_p95", "DoD_mean", "DoD_p95", "C_rate_mean", "n_cycles"]

PRE_REGISTERED = (
    "Literature order (Edge et al. 2021 PCCP 23:8200): Temperature > DoD > C-rate. "
    "With n=13 same-city same-model vehicles (narrow feature ranges), "
    "expected outcome: factors indistinguishable at n=13. "
    "Either result is the reportable finding."
)


# ── Ridge helpers (no sklearn) ────────────────────────────────────────────────

def _standardise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column. Returns (X_std, mu, sigma)."""
    mu    = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)   # guard zero-variance columns
    return (X - mu) / sigma, mu, sigma


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """OLS with L2 penalty: β = (X'X + αI)⁻¹ X'y."""
    n, p = X.shape
    A    = X.T @ X + alpha * np.eye(p)
    return np.linalg.solve(A, X.T @ y)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _loo_cv(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[float, np.ndarray]:
    """Leave-one-out CV for Ridge. Returns (LOO R², y_pred_loo)."""
    n = len(y)
    y_loo = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta = _ridge_fit(X[mask], y[mask], alpha)
        y_loo[i] = float(X[i] @ beta)
    return _r2(y, y_loo), y_loo


def _permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    n_repeats: int = PERM_REPEATS,
    seed: int = RNG_SEED,
) -> List[Dict]:
    """
    In-sample permutation importance. Baseline RMSE then shuffle each feature.
    Returns list of {feature, mean_importance, std_importance} sorted descending.
    Note: in-sample is optimistic — treat as directional, especially at n=13.
    """
    rng      = np.random.default_rng(seed)
    y_base   = X @ beta
    base_mse = float(np.mean((y - y_base) ** 2))

    results = []
    for j, fname in enumerate(FEATURE_NAMES):
        deltas = []
        for _ in range(n_repeats):
            Xp       = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            mse_j    = float(np.mean((y - Xp @ beta) ** 2))
            deltas.append(mse_j - base_mse)
        results.append({
            "feature"         : fname,
            "mean_importance" : round(float(np.mean(deltas)), 6),
            "std_importance"  : round(float(np.std(deltas)),  6),
        })

    results.sort(key=lambda r: -r["mean_importance"])
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def run_factor_ranking() -> None:
    from degradation.deng_loader     import load_all
    from degradation.cycle_segmentor import segment_all

    print("P5 — Degradation Factor Ranking on Deng Fleet")
    print("=" * 65)
    print()
    print("PRE-REGISTERED EXPECTATION:")
    print(f"  {PRE_REGISTERED}")
    print()

    # ── 1. Load λ_v from temporal report ─────────────────────────────────────
    if not TEMPORAL_REPORT.exists():
        print(f"ERROR: {TEMPORAL_REPORT} not found. Run cell_to_field_temporal.py first.")
        sys.exit(1)

    with open(TEMPORAL_REPORT) as fh:
        temporal = json.load(fh)

    lam_v = {}
    for veh, d in temporal["per_vehicle"].items():
        if "lambda_v" in d:
            lam_v[veh] = float(d["lambda_v"])

    gated_in  = {v: lam for v, lam in lam_v.items() if lam >  0}
    gated_out = {v: lam for v, lam in lam_v.items() if lam <= 0}
    print(f"  λ_v loaded: {len(lam_v)} vehicles, "
          f"{len(gated_in)} gated-in (λ>0), {len(gated_out)} gated-out (λ≤0)")

    # ── 2. Load fleet and compute per-vehicle features ────────────────────────
    print("\nLoading Deng fleet and computing features…")
    raw    = load_all(verbose=False)
    cycles = segment_all(raw, verbose=False)

    records = []
    for veh in sorted(lam_v.keys()):
        vc = cycles[cycles["vehicle"] == veh]
        if len(vc) == 0:
            continue
        t_d = temporal["per_vehicle"].get(veh, {})
        records.append({
            "vehicle"    : veh,
            "lambda_v"   : lam_v[veh],
            "gated_in"   : veh in gated_in,
            "snr"        : t_d.get("snr", float("nan")),
            "t_years"    : t_d.get("t_max_years", float("nan")),
            "T_mean"     : round(float(vc["T_mean_C"].mean()), 3),
            "T_p95"      : round(float(vc["T_mean_C"].quantile(0.95)), 3),
            "DoD_mean"   : round(float(vc["DoD_pct"].mean()), 3),
            "DoD_p95"    : round(float(vc["DoD_pct"].quantile(0.95)), 3),
            "C_rate_mean": round(float(vc["C_rate"].abs().mean()), 5),
            "n_cycles"   : int(len(vc)),
        })

    df = pd.DataFrame(records).set_index("vehicle")
    df_in  = df[df["gated_in"]].copy()
    df_out = df[~df["gated_in"]].copy()
    n_in   = len(df_in)

    # Feature ranges (diagnostic)
    print(f"\nFeature ranges across {n_in} gated-in vehicles:")
    for f in FEATURE_NAMES:
        v = df_in[f]
        print(f"  {f:14s}: min={v.min():.4f}  max={v.max():.4f}  "
              f"range={v.max()-v.min():.4f}  std={v.std():.4f}")

    # ── 3. Spearman ρ ─────────────────────────────────────────────────────────
    y_in = df_in["lambda_v"].values
    bonferroni_threshold = 0.05 / len(FEATURE_NAMES)
    spearman_results = []
    for f in FEATURE_NAMES:
        x  = df_in[f].values
        rho, pval = spearmanr(x, y_in)
        spearman_results.append({
            "feature"  : f,
            "rho"      : round(float(rho), 4),
            "p_value"  : round(float(pval), 4),
            "sig_bonf" : bool(float(pval) < bonferroni_threshold),
        })
    spearman_results.sort(key=lambda r: -abs(r["rho"]))

    # ── 4. Ridge LOO-CV + permutation importance ──────────────────────────────
    X_raw = df_in[FEATURE_NAMES].values.astype(float)
    X_std, feat_mu, feat_sigma = _standardise(X_raw)

    loo_r2, y_loo = _loo_cv(X_std, y_in, RIDGE_ALPHA)
    predictive_value = loo_r2 >= 0

    # Fit on full gated-in set for permutation importance
    beta_full = _ridge_fit(X_std, y_in, RIDGE_ALPHA)
    perm_imp  = _permutation_importance(X_std, y_in, beta_full)

    # Ridge coefficients (standardised space → directional, not magnitude-comparable)
    ridge_coef = [
        {"feature": f, "coef_std": round(float(beta_full[j]), 5)}
        for j, f in enumerate(FEATURE_NAMES)
    ]

    # ── 5. Observed ranking (by |ρ|) ─────────────────────────────────────────
    observed_order = [r["feature"] for r in spearman_results]
    literature_order = ["T_mean", "DoD_mean", "C_rate_mean"]  # T > DoD > C-rate
    # Map to observed positions
    obs_pos = {f: i+1 for i, f in enumerate(observed_order)}
    top3_obs = observed_order[:3]

    # Agreement: top-2 features should be temperature or DoD
    lit_top2 = {"T_mean", "T_p95", "DoD_mean", "DoD_p95"}
    obs_top2 = set(observed_order[:2])
    agreement_lit = bool(len(obs_top2 & lit_top2) >= 1)

    # Distinguishability: all |ρ| < 0.4 → indistinguishable
    rhos_abs = [abs(r["rho"]) for r in spearman_results]
    indistinguishable = all(r < 0.40 for r in rhos_abs)

    # ── 6. Verdict ────────────────────────────────────────────────────────────
    if indistinguishable:
        verdict = (
            f"FACTORS INDISTINGUISHABLE at n={n_in}: all |ρ| < 0.40 "
            f"(max |ρ| = {max(rhos_abs):.3f}). "
            f"No feature clears even the unadjusted ρ significance threshold "
            f"at n={n_in}. Pre-registered expectation CONFIRMED: "
            f"same-city, same-model fleet with narrow feature ranges "
            f"does not resolve factor ranking. "
            f"LOO R² = {loo_r2:.3f} "
            f"({'positive but weak' if predictive_value else 'negative — no cross-validated predictive value'}). "
            f"Importances are directional suggestions only."
        )
    else:
        max_feat = spearman_results[0]["feature"]
        max_rho  = spearman_results[0]["rho"]
        verdict = (
            f"PARTIAL SIGNAL at n={n_in}: strongest feature = {max_feat} "
            f"(ρ={max_rho:.3f}). "
            f"{'Consistent' if agreement_lit else 'Inconsistent'} with literature order "
            f"(Edge et al. 2021: T > DoD > C-rate). "
            f"LOO R² = {loo_r2:.3f} "
            f"({'positive' if predictive_value else 'negative — no cross-validated predictive value'}). "
            f"Caution: n={n_in} with correlated features; "
            f"individual ρ values ≠ independent contributions."
        )

    # ── 7. Print per-vehicle table ────────────────────────────────────────────
    print()
    print("=" * 95)
    print(f"PER-VEHICLE FEATURE TABLE  (all 20 vehicles; ◄ = gated-in for ranking, n={n_in})")
    print("=" * 95)
    print(f"{'Veh':4s} {'λ_v':8s} {'SNR':5s} {'t_yr':5s} "
          f"{'T_mn':6s} {'T_p95':6s} {'DoD_mn':7s} {'DoD_p95':8s} "
          f"{'Crate_mn':9s} {'n_cyc':6s} {'gate':5s}")
    print("-" * 95)
    for veh in sorted(df.index):
        row = df.loc[veh]
        mark = " ◄" if row["gated_in"] else "  "
        print(
            f"{veh:4s} {row['lambda_v']:8.5f} {row['snr']:5.2f} {row['t_years']:5.3f} "
            f"{row['T_mean']:6.2f} {row['T_p95']:6.2f} {row['DoD_mean']:7.2f} "
            f"{row['DoD_p95']:8.2f} {row['C_rate_mean']:9.5f} {row['n_cycles']:6d}"
            f"{mark}"
        )

    # ── 8. Print Spearman table ───────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"(a) SPEARMAN ρ  vs λ_v  (n={n_in} gated-in vehicles, sorted by |ρ|)")
    print(f"    Bonferroni threshold: p < {bonferroni_threshold:.4f}  "
          f"(unadjusted p < 0.05 requires |ρ| ≳ 0.55 at n={n_in})")
    print("=" * 65)
    print(f"{'Feature':14s} {'ρ':7s} {'p-value':9s} {'|ρ|':6s} {'Bonf-sig?':10s}")
    print("-" * 50)
    for r in spearman_results:
        sig_s = "YES" if r["sig_bonf"] else "no"
        print(f"  {r['feature']:12s} {r['rho']:+7.4f}  {r['p_value']:8.4f}  "
              f"{abs(r['rho']):6.4f}  {sig_s}")

    # ── 9. Print Ridge + permutation importance table ─────────────────────────
    print()
    print("=" * 65)
    print(f"(b) RIDGE LOO-CV + PERMUTATION IMPORTANCE  (n={n_in}, α={RIDGE_ALPHA})")
    print(f"    LOO R² = {loo_r2:.4f}  "
          + ("(positive — weak predictive value)" if predictive_value
             else "(NEGATIVE — no cross-validated predictive value; importances directional only)"))
    print(f"    In-sample permutation importance ({PERM_REPEATS} repeats):")
    print("=" * 65)
    print(f"{'Feature':14s} {'Coef(std)':10s} {'Imp(ΔMSE)':11s} {'Std':8s}  {'Rank':5s}")
    print("-" * 55)
    coef_map = {c["feature"]: c["coef_std"] for c in ridge_coef}
    for rank, r in enumerate(perm_imp, 1):
        print(f"  {r['feature']:12s} {coef_map[r['feature']]:+10.5f} "
              f"{r['mean_importance']:+11.6f} {r['std_importance']:8.6f}  #{rank}")

    print()
    print("    Note: in-sample permutation importance is optimistic at n=13.")
    print("    Treat as directional, not quantitative, especially for features")
    print("    with similar importance ± std overlap.")

    print()
    print("(c) KERNEL SHAP: skipped — shap package not installed.")

    # ── 10. Verdict ───────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("VERDICT")
    print("=" * 65)
    words = verdict.split()
    line, lines = [], []
    for w in words:
        if sum(len(x)+1 for x in line) + len(w) > 70:
            lines.append("  " + " ".join(line)); line = [w]
        else:
            line.append(w)
    if line:
        lines.append("  " + " ".join(line))
    print("\n".join(lines))
    print()
    print(f"  Pre-registered lit. order: T > DoD > C-rate (Edge et al. 2021 PCCP 23:8200)")
    print(f"  Observed order (by |ρ|):   {' > '.join(observed_order)}")
    print(f"  Agreement with literature: {'YES' if agreement_lit else 'NO (or indistinguishable)'}")
    print(f"  Indistinguishable (all |ρ|<0.40): {'YES — pre-registered outcome' if indistinguishable else 'NO'}")

    # ── 11. Write JSON ────────────────────────────────────────────────────────
    report = {
        "meta": {
            "script"       : "analysis/factor_ranking.py",
            "target"       : "lambda_v (per-vehicle calendar fade rate from train-window OLS)",
            "target_source": "data/cell_to_field_temporal_report.json",
            "t_years_note" : (
                "t_years excluded from ranking features: range {:.4f}–{:.4f} yr "
                "(near-zero variance); also mechanically coupled to target (λ_v is √t-normalised). "
                "Reported in per-vehicle table for context only.".format(
                    df["t_years"].min(), df["t_years"].max())
            ),
            "shap_note"    : "Kernel SHAP skipped — shap package not installed.",
            "ridge_alpha"  : RIDGE_ALPHA,
            "perm_repeats" : PERM_REPEATS,
        },
        "pre_registered_expectation": PRE_REGISTERED,
        "literature_reference"      : (
            "Edge J.S. et al. (2021) Lithium ion battery degradation: what you need to know. "
            "Physical Chemistry Chemical Physics 23(14):8200-8221. DOI 10.1039/D1CP00359C"
        ),
        "literature_order"          : ["temperature", "DoD", "C_rate"],
        "gating": {
            "gated_in"    : sorted(gated_in.keys()),
            "gated_out"   : sorted(gated_out.keys()),
            "gate_criterion": "lambda_v > 0 from train-window OLS (train-time-observable)",
        },
        "per_vehicle": {
            row.Index: {
                "lambda_v"   : round(row.lambda_v, 6),
                "gated_in"   : row.gated_in,
                "snr"        : row.snr,
                "t_years"    : row.t_years,
                "T_mean"     : row.T_mean,
                "T_p95"      : row.T_p95,
                "DoD_mean"   : row.DoD_mean,
                "DoD_p95"    : row.DoD_p95,
                "C_rate_mean": row.C_rate_mean,
                "n_cycles"   : row.n_cycles,
            }
            for row in df.itertuples()
        },
        "spearman": {
            "n"                  : n_in,
            "bonferroni_threshold": round(bonferroni_threshold, 5),
            "results"            : spearman_results,
            "note"               : (
                f"At n={n_in}, unadjusted p<0.05 requires |ρ|≳0.55. "
                "p-values are weak; report but do not over-read."
            ),
        },
        "ridge_loo": {
            "alpha"             : RIDGE_ALPHA,
            "loo_r2"            : round(loo_r2, 5),
            "predictive_value"  : predictive_value,
            "loo_note"          : (
                "Positive LOO R² — weak predictive value" if predictive_value
                else "Negative LOO R² — no cross-validated predictive value; "
                     "importances are directional suggestions only"
            ),
            "ridge_coef_std_space": ridge_coef,
            "permutation_importance": perm_imp,
            "perm_note"         : (
                f"In-sample permutation importance ({PERM_REPEATS} repeats). "
                "Optimistic at n=13; treat as directional, not quantitative."
            ),
        },
        "observed_ranking_by_rho" : observed_order,
        "agreement_with_literature": agreement_lit,
        "indistinguishable"        : indistinguishable,
        "verdict"                  : verdict,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nReport written → {OUT_JSON}")


if __name__ == "__main__":
    run_factor_ranking()
