"""
sobol_factor_ranking.py — Physics-based global sensitivity analysis (P5 follow-up).

WHY THIS EXISTS:
  factor_ranking.py (P5) proved statistical factor ranking is impossible on the Deng
  fleet at n=13: all |ρ|<0.40, LOO R²=−12.83. Root cause: T range <1°C, DoD range
  <2.5%, C_rate range <0.012 — no feature variance → no attribution power.

  This module substitutes a physics-model sensitivity analysis using Sobol indices
  and Morris elementary effects over a virtual Design-of-Experiments sweep. It ranks
  which factor (T, DoD, C_rate) drives ΔSOH most in the EXISTING DFN-SPM +
  Palmgren-Miner physics stack — not from data.

IMPORTANT CAVEATS (repeated in output JSON):
  1. This is a MODEL sensitivity analysis, not a data-driven finding. Results are
     only as good as stress_model.py's functional form (empirical proxy, not
     diffusion-induced stress from first principles).
  2. λ_sei (calendar term) is temperature-invariant in this model. Temperature only
     enters via Arrhenius in the CYCLING stress term. Real SEI kinetics are T-dependent
     (Arrhenius with Ea≈0.5 eV); this understates T sensitivity for calendar-dominated
     regimes and long-duration operation.
  3. Does NOT resolve P5's original ask. This substitutes a physics-based ranking
     because the data-based ranking is provably underpowered at n=13 (P5 result).
  4. Analytical single-amplitude model (no rainflow). Assumes one full cycle per
     session at amplitude=DoD_pct. Valid for sweep; may differ from real mixed
     amplitude distributions in fleet.

FROZEN PARAMETERS (do not modify — sourced from committed JSONs):
  β_NASA   = 0.021545  (cell_to_field_report.json → frozen_params.beta_NASA)
                        NASA LOO-CV mean: B0005/B0006/B0007/B0018
  γ_NASA   = 0.5       (fixed throughout Module 4; cell_to_field_report.json)
  λ_sei    = 0.02639332 (deng_degradation_report.json → degradation_driver)
  A (S-N)  = 1e6       (nasa_degradation_report.json → sn_A; fatigue.py default)
  m (S-N)  = 2.5       (nasa_degradation_report.json → sn_m; fatigue.py default)

NOTE ON SCALE:
  eval_B_cross_cell.beta = 2730.959 in nasa_degradation_report.json uses the raw
  fatigue.py D scale (D≈1e-3 at NASA EOL). cell_to_field_bridge.py uses a different
  D normalisation → β=0.021545. These are NOT interchangeable. The β=0.021545 is
  the consistent value across Module 3/4 and gives ΔSOH_cyc ∈ [0, 1] at all corners
  of the parameter box. eval_B beta was verified to produce ΔSOH_cyc≈484 at
  max-stress — physically impossible; it must not be used here.

PRE-REGISTERED EXPECTATION (locked before running, not adjusted post-hoc):
  T > DoD > C_rate
  Source: Edge J.S. et al. (2021) Phys. Chem. Chem. Phys. 23(14):8200.
          DOI 10.1039/D1CP00359C

PARAMETER RANGES (pre-registered):
  T_C:     [0, 45] °C
  DoD_pct: [10, 100] %
  C_rate:  [0.1, 3.0] C

REFERENCE DURATION (fixed across all sweep points):
  N_cycles = 800  (approx. annual sessions for Deng fleet)
  t_years  = 1.0

METHOD:
  Step A — Morris elementary effects screening (~80 evaluations, cheap).
            Identifies negligible factors before Sobol budget allocation.
  Step B — Sobol first-order (S1) and total-order (ST) indices via Saltelli
            sampling, N=1024, calc_second_order=False → 5120 evaluations.
"""

from __future__ import annotations

import json
import sys
import importlib.metadata
from pathlib import Path

import numpy as np
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sampler
from SALib.analyze import morris as morris_analyze
from SALib.analyze import sobol as sobol_analyze

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from degradation.stress_model import compute_stress

# ── Frozen parameters ─────────────────────────────────────────────────────────
BETA_NASA   = 0.021545    # cell_to_field_report.json → frozen_params.beta_NASA
GAMMA_NASA  = 0.5         # fixed throughout Module 4
LAMBDA_SEI  = 0.02639332  # deng_degradation_report.json → degradation_driver
SN_A        = 1.0e6       # Basquin coefficient  (nasa_degradation_report.json)
SN_M        = 2.5         # Basquin exponent     (nasa_degradation_report.json)

# ── Reference duration ────────────────────────────────────────────────────────
N_CYCLES    = 800         # sessions per reference year (Deng fleet approx.)
T_YEARS     = 1.0

# ── SALib problem definition (pre-registered) ─────────────────────────────────
PROBLEM = {
    "num_vars": 3,
    "names":   ["T_C", "DoD_pct", "C_rate"],
    "bounds":  [[0.0, 45.0], [10.0, 100.0], [0.1, 3.0]],
}

PRE_REGISTERED_EXPECTATION = (
    "T > DoD > C-rate  "
    "(Edge J.S. et al. 2021, Phys. Chem. Chem. Phys. 23(14):8200, "
    "DOI 10.1039/D1CP00359C)"
)

CAVEATS = [
    "PHYSICS-MODEL sensitivity only — reflects the current model structure "
    "(stress_model.py Arrhenius + Basquin + SEI √t), NOT empirical fleet data. "
    "Results are only as defensible as the model's functional form.",

    "λ_sei (calendar term) is temperature-invariant in this model: λ·√t does not "
    "depend on T, DoD, or C_rate. Temperature enters ONLY via Arrhenius in the "
    "cycling stress. Real SEI kinetics are T-dependent (Arrhenius, Ea≈0.5 eV); "
    "this analysis UNDERSTATES T sensitivity in calendar-dominated, long-duration "
    "regimes.",

    "Does NOT resolve P5's original ask (rank factors from real Deng fleet data). "
    "Substituted because fleet is statistically underpowered at n=13: all |ρ|<0.40, "
    "LOO R²=−12.83 (see factor_ranking_report.json). The present result shows what "
    "the PHYSICS MODEL believes, not what the data confirms.",

    "Analytical single-amplitude model: one full cycle per session at "
    "amplitude=DoD_pct, mean temperature T_C, rate C_rate. No rainflow on SOC "
    "time-series. Valid for virtual DoE sweep; may differ from real fleet's "
    "mixed-amplitude distributions.",

    "β=0.021545, γ=0.5 sourced from cell_to_field_bridge.py (Module 3/4 D scale). "
    "eval_B_cross_cell.beta=2730.959 in nasa_degradation_report.json uses the raw "
    "fatigue.py D scale and is NOT interchangeable — produces ΔSOH≈484 at max stress.",

    "Calendar term ΔSOH_cal = λ·√t = 0.02639 is constant across all sweep points. "
    "It shifts the mean output but does not contribute to output variance. Sobol "
    "indices therefore reflect sensitivity of the CYCLING term only. S1+ST sum ≈ 1 "
    "for the three factors (constant term adds no variance).",

    "N_cycles=800 is held constant across all C_rate values in this sweep. "
    "Physically, higher C_rate typically enables more charge sessions per unit time, "
    "which would compound the cycling effect. This sweep isolates per-cycle stress "
    "sensitivity only and likely UNDERSTATES C_rate's true real-world impact on "
    "annual ΔSOH.",
]


# ── Model function ─────────────────────────────────────────────────────────────

def _delta_soh(T_C: float, DoD_pct: float, C_rate: float) -> float:
    """
    ΔSOH after N_CYCLES at (T_C, DoD_pct, C_rate) over T_YEARS.

    Cycling term:
      stress     = compute_stress(DoD_pct, C_rate, T_C)          [stress_model.py]
      d_per_cyc  = stress^m / A                                   [Basquin, 1 full cycle]
      D_total    = N_CYCLES * d_per_cyc                           [Palmgren-Miner]
      ΔSOH_cyc   = β * D_total^γ

    Calendar term:
      ΔSOH_cal   = λ_sei * √t_years                              [Pinson-Bazant SEI]

    Returns ΔSOH_total = ΔSOH_cyc + ΔSOH_cal  (clipped to [0, 1]).
    """
    stress      = float(compute_stress(dod_pct=DoD_pct, c_rate=C_rate, T_mean_C=T_C))
    d_per_cycle = (stress ** SN_M) / SN_A
    D_total     = N_CYCLES * d_per_cycle
    dsoh_cyc    = BETA_NASA * (D_total ** GAMMA_NASA)
    dsoh_cal    = LAMBDA_SEI * np.sqrt(T_YEARS)
    return float(np.clip(dsoh_cyc + dsoh_cal, 0.0, 1.0))


def _model_batch(X: np.ndarray) -> np.ndarray:
    """Vectorised wrapper: X shape (N, 3) → Y shape (N,)."""
    return np.array([_delta_soh(row[0], row[1], row[2]) for row in X])


# ── Corner sanity check ────────────────────────────────────────────────────────

def _run_corner_check() -> dict:
    corners = {
        "max_stress (T=45, DoD=100%, C=3.0)":  (45.0, 100.0, 3.0),
        "min_stress (T=0,  DoD=10%,  C=0.1)":  (0.0,   10.0, 0.1),
        "deng_mean  (T=22, DoD=57%,  C=0.41)": (22.0,  57.0, 0.41),
    }
    results = {}
    ok = True
    for label, (T, DoD, C) in corners.items():
        val = _delta_soh(T, DoD, C)
        stress = float(compute_stress(DoD, C, T))
        d_pc   = (stress ** SN_M) / SN_A
        D_tot  = N_CYCLES * d_pc
        cyc    = BETA_NASA * (D_tot ** GAMMA_NASA)
        cal    = LAMBDA_SEI * np.sqrt(T_YEARS)
        in_range = 0.0 <= val <= 1.0
        if not in_range:
            ok = False
        results[label] = {
            "T_C": T, "DoD_pct": DoD, "C_rate": C,
            "stress": round(stress, 6),
            "D_total": float(f"{D_tot:.4e}"),
            "ΔSOH_cyc": round(cyc, 6),
            "ΔSOH_cal": round(cal, 6),
            "ΔSOH_total": round(val, 6),
            "in_range_0_1": in_range,
        }
    return {"all_pass": ok, "corners": results}


# ── Absolute contribution table ───────────────────────────────────────────────

def _absolute_contribution() -> dict:
    """
    Compute cycling vs calendar share of total ΔSOH at two reference points.
    These numbers give the reader context for interpreting Sobol indices:
    the indices rank sensitivity within the cycling sub-model only.
    """
    points = {
        "deng_fleet_mean": (22.0, 57.0, 0.41),
        "max_stress_corner": (45.0, 100.0, 3.0),
    }
    result = {}
    for label, (T, DoD, C) in points.items():
        stress  = float(compute_stress(dod_pct=DoD, c_rate=C, T_mean_C=T))
        d_pc    = (stress ** SN_M) / SN_A
        D_tot   = N_CYCLES * d_pc
        cyc     = BETA_NASA * (D_tot ** GAMMA_NASA)
        cal     = LAMBDA_SEI * np.sqrt(T_YEARS)
        total   = cyc + cal
        result[label] = {
            "T_C": T, "DoD_pct": DoD, "C_rate": C,
            "ΔSOH_cyc":   round(cyc,   6),
            "ΔSOH_cal":   round(cal,   6),
            "ΔSOH_total": round(total, 6),
            "cyc_pct_of_total": round(100.0 * cyc / total, 2),
            "cal_pct_of_total": round(100.0 * cal / total, 2),
        }
    return result


# ── Step A: Morris screening ───────────────────────────────────────────────────

def _run_morris(n_trajectories: int = 20, num_levels: int = 4) -> dict:
    X = morris_sample.sample(PROBLEM, N=n_trajectories, num_levels=num_levels)
    Y = _model_batch(X)
    Si = morris_analyze.analyze(
        PROBLEM, X, Y,
        num_levels=num_levels,
        print_to_console=False,
    )
    n_evals = X.shape[0]
    return {
        "n_evaluations": n_evals,
        "n_trajectories": n_trajectories,
        "num_levels": num_levels,
        "T_C":    {"mu_star": round(float(Si["mu_star"][0]), 6),
                   "sigma":   round(float(Si["sigma"][0]),   6)},
        "DoD_pct":{"mu_star": round(float(Si["mu_star"][1]), 6),
                   "sigma":   round(float(Si["sigma"][1]),   6)},
        "C_rate": {"mu_star": round(float(Si["mu_star"][2]), 6),
                   "sigma":   round(float(Si["sigma"][2]),   6)},
    }


# ── Step B: Sobol analysis ─────────────────────────────────────────────────────

def _run_sobol(N: int = 1024) -> dict:
    X = sobol_sampler.sample(PROBLEM, N=N, calc_second_order=False)
    Y = _model_batch(X)
    Si = sobol_analyze.analyze(
        PROBLEM, Y,
        calc_second_order=False,
        print_to_console=False,
    )
    n_evals = X.shape[0]
    return {
        "n_evaluations": n_evals,
        "N_base_samples": N,
        "calc_second_order": False,
        "T_C":    {"S1": round(float(Si["S1"][0]), 6),
                   "ST": round(float(Si["ST"][0]), 6),
                   "S1_conf": round(float(Si["S1_conf"][0]), 6),
                   "ST_conf": round(float(Si["ST_conf"][0]), 6)},
        "DoD_pct":{"S1": round(float(Si["S1"][1]), 6),
                   "ST": round(float(Si["ST"][1]), 6),
                   "S1_conf": round(float(Si["S1_conf"][1]), 6),
                   "ST_conf": round(float(Si["ST_conf"][1]), 6)},
        "C_rate": {"S1": round(float(Si["S1"][2]), 6),
                   "ST": round(float(Si["ST"][2]), 6),
                   "S1_conf": round(float(Si["S1_conf"][2]), 6),
                   "ST_conf": round(float(Si["ST_conf"][2]), 6)},
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Sobol/Morris Factor Sensitivity Analysis")
    print("Pre-registered expectation:", PRE_REGISTERED_EXPECTATION)
    print("=" * 60)

    # Corner sanity check
    print("\n[0] Corner sanity check ...")
    corners = _run_corner_check()
    if not corners["all_pass"]:
        print("FATAL: ΔSOH out of [0,1] at corner — aborting.")
        for label, r in corners["corners"].items():
            if not r["in_range_0_1"]:
                print(f"  FAIL: {label} → ΔSOH={r['ΔSOH_total']}")
        sys.exit(1)
    for label, r in corners["corners"].items():
        print(f"  {label}: ΔSOH_cyc={r['ΔSOH_cyc']:.6f}  "
              f"ΔSOH_cal={r['ΔSOH_cal']:.6f}  total={r['ΔSOH_total']:.6f}  ✓")

    # Absolute contribution table
    print("\n[0b] Absolute contribution — cycling vs calendar share of ΔSOH")
    abs_contrib = _absolute_contribution()
    print(f"  {'Point':<30} {'ΔSOH_cyc':>10} {'ΔSOH_cal':>10} "
          f"{'total':>8} {'cyc%':>6} {'cal%':>6}")
    print(f"  {'-'*74}")
    for label, r in abs_contrib.items():
        print(f"  {label:<30} {r['ΔSOH_cyc']:>10.6f} {r['ΔSOH_cal']:>10.6f} "
              f"{r['ΔSOH_total']:>8.6f} {r['cyc_pct_of_total']:>5.1f}% {r['cal_pct_of_total']:>5.1f}%")
    print()
    print("  INTERPRETATION: Sobol indices rank sensitivity of the CYCLING term only.")
    print("  Calendar aging (λ·√t) is 69–99% of total ΔSOH in realistic-to-worst-case")
    print("  conditions and does NOT respond to T/DoD/C-rate in this model. Changing")
    print("  T/DoD/C-rate shifts total ΔSOH by at most ~1 percentage point.")

    # Step A — Morris
    print("\n[A] Morris elementary effects screening ...")
    morris_res = _run_morris(n_trajectories=20, num_levels=4)
    print(f"    Evaluations: {morris_res['n_evaluations']}")
    for f in ["T_C", "DoD_pct", "C_rate"]:
        r = morris_res[f]
        print(f"    {f:<10}  μ*={r['mu_star']:.6f}  σ={r['sigma']:.6f}")
    morris_order = sorted(["T_C", "DoD_pct", "C_rate"],
                          key=lambda f: morris_res[f]["mu_star"], reverse=True)
    print(f"    Morris order (by μ*): {' > '.join(morris_order)}")

    # Step B — Sobol
    print("\n[B] Sobol indices (N=1024, S1+ST only) ...")
    sobol_res = _run_sobol(N=1024)
    print(f"    Evaluations: {sobol_res['n_evaluations']}")
    for f in ["T_C", "DoD_pct", "C_rate"]:
        r = sobol_res[f]
        print(f"    {f:<10}  S1={r['S1']:.4f} ± {r['S1_conf']:.4f}  "
              f"ST={r['ST']:.4f} ± {r['ST_conf']:.4f}")
    sobol_order = sorted(["T_C", "DoD_pct", "C_rate"],
                         key=lambda f: sobol_res[f]["ST"], reverse=True)
    print(f"    Sobol order (by ST): {' > '.join(sobol_order)}")

    # Observed order and agreement
    observed_order_st = sobol_order
    expected_order    = ["T_C", "DoD_pct", "C_rate"]
    agreement         = (observed_order_st == expected_order)
    print(f"\n    Expected:  {' > '.join(expected_order)}")
    print(f"    Observed:  {' > '.join(observed_order_st)}")
    print(f"    Agreement: {'YES' if agreement else 'NO — report as-is, do not adjust model'}")

    # Verdict
    top    = observed_order_st[0]
    second = observed_order_st[1]
    third  = observed_order_st[2]
    st_top = sobol_res[top]["ST"]
    st_2nd = sobol_res[second]["ST"]
    st_3rd = sobol_res[third]["ST"]
    fm  = abs_contrib["deng_fleet_mean"]
    mx  = abs_contrib["max_stress_corner"]
    if agreement:
        verdict = (
            f"T > DoD > C-rate confirmed by ST ({st_top:.3f} > {st_2nd:.3f} > {st_3rd:.3f}). "
            f"Consistent with Edge et al. (2021). "
            f"NOTE: T sensitivity is through Arrhenius cycling stress only — "
            f"calendar λ is T-invariant in this model. "
            f"This Sobol ranking describes sensitivity WITHIN the cycling sub-model "
            f"only. It does NOT contradict the M2 finding that field degradation is "
            f"calendar-dominated (deng_degradation_report.json: stress_frac=0%) — "
            f"rather, it shows that even the (small) cycling contribution is primarily "
            f"driven by temperature. In the fleet's actual operating regime, changing "
            f"T/DoD/C-rate would shift total ΔSOH by at most ~1 percentage point "
            f"(cycling is {fm['cyc_pct_of_total']:.1f}% of total ΔSOH at fleet mean, "
            f"{mx['cyc_pct_of_total']:.1f}% at max-stress corner — see "
            f"absolute_contribution table), because calendar aging "
            f"({fm['cal_pct_of_total']:.1f}–{mx['cal_pct_of_total']:.1f}% of fade) "
            f"is unaffected by these factors in the current model."
        )
    else:
        verdict = (
            f"Observed order {' > '.join(observed_order_st)} "
            f"(ST: {st_top:.3f} > {st_2nd:.3f} > {st_3rd:.3f}) "
            f"does NOT match Edge et al. (2021) expectation T > DoD > C-rate. "
            f"This flags a potential mis-calibration of the Arrhenius or Basquin "
            f"term in stress_model.py relative to literature. "
            f"Report as-is; do not adjust model to force agreement. "
            f"Note: cycling is {fm['cyc_pct_of_total']:.1f}% of total ΔSOH at fleet "
            f"mean and {mx['cyc_pct_of_total']:.1f}% at max-stress corner — Sobol "
            f"indices reflect sensitivity within this sub-model only; the M2 "
            f"calendar-dominance finding is unaffected."
        )
    print(f"\n    Verdict: {verdict}")

    # Build report
    salib_version = importlib.metadata.version("SALib")
    report = {
        "meta": {
            "script": "analysis/sobol_factor_ranking.py",
            "salib_version": salib_version,
            "n_morris_samples": morris_res["n_evaluations"],
            "n_sobol_samples":  sobol_res["n_evaluations"],
            "reference_n_cycles": N_CYCLES,
            "reference_t_years":  T_YEARS,
            "frozen_params": {
                "beta_NASA":  BETA_NASA,
                "gamma_NASA": GAMMA_NASA,
                "lambda_sei": LAMBDA_SEI,
                "sn_A": SN_A,
                "sn_m": SN_M,
                "beta_source": "cell_to_field_report.json frozen_params.beta_NASA "
                               "(NASA LOO-CV mean: B0005/B0006/B0007/B0018)",
                "gamma_source": "fixed γ=0.5 throughout Module 4",
                "lambda_source": "deng_degradation_report.json degradation_driver",
                "scale_note": (
                    "eval_B_cross_cell.beta=2730.959 in nasa_degradation_report.json "
                    "uses raw fatigue.py D scale (D≈1e-3 at NASA EOL) and is NOT "
                    "interchangeable with beta=0.021545 — produces ΔSOH≈484 at "
                    "max-stress corner. Physically impossible. Do not use for Sobol."
                ),
            },
        },
        "parameter_ranges": {
            "T_C":     [0.0, 45.0],
            "DoD_pct": [10.0, 100.0],
            "C_rate":  [0.1, 3.0],
        },
        "pre_registered_expectation": PRE_REGISTERED_EXPECTATION,
        "corner_sanity_check": corners,
        "morris_screening": {
            k: v for k, v in morris_res.items()
            if k in ("n_evaluations", "n_trajectories", "num_levels",
                     "T_C", "DoD_pct", "C_rate")
        },
        "morris_order_by_mu_star": morris_order,
        "sobol_indices": {
            k: v for k, v in sobol_res.items()
            if k in ("n_evaluations", "N_base_samples",
                     "T_C", "DoD_pct", "C_rate")
        },
        "absolute_contribution": abs_contrib,
        "observed_order_by_ST": observed_order_st,
        "agreement_with_literature": agreement,
        "caveats": CAVEATS,
        "verdict": verdict,
    }

    out = Path(__file__).resolve().parents[1] / "data" / "sobol_ranking_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[✓] Report written → {out}")


if __name__ == "__main__":
    main()
