# Problem 1: 360-Degree Validation Report

Comprehensive validation of limited-data degradation prediction, run separately on NASA LCO (n=4) and Severson LFP (n=124). Model: closed-form conjugate Bayesian linear regression (physics mean function `beta * k^0.5`, LOO population prior) -- a disclosed simplification of the full Matern52-GP used in `bayes_gp_predictor.py`/`severson_gp_predictor.py`, needed for computational tractability across this many fraction/cell combinations. See module docstring for the full rationale.


## LCO (NASA) (n=4)

> **Read this carefully:** median R² *decreases* as the training fraction increases, while MAE stays roughly flat. This is a known R² pathology, not evidence the model gets worse with more data: R²'s denominator is the variance of the *remaining unseen* trajectory, which shrinks as more of the cell is already observed — the same absolute error produces a worse (or negative) R² against a smaller, noisier target range. **MAE/RMSE are the more trustworthy metrics for the "does more data help" question in this table; R² should be read per-fraction, not trended across fractions.**

### Step 1 — Multi-fraction early-cycle test

| Fraction | n cells | R² mean | R² median | MAE mean | RMSE mean | %R²<0 |
|---|---|---|---|---|---|---|
| 5% | 4 | +0.456 | +0.427 | 0.05729 | 0.06588 | 0% |
| 10% | 4 | +0.395 | +0.346 | 0.05774 | 0.06616 | 0% |
| 20% | 4 | +0.193 | +0.291 | 0.05627 | 0.06382 | 50% |
| 30% | 4 | -0.192 | +0.253 | 0.05515 | 0.06031 | 50% |
| 50% | 4 | -1.547 | -0.349 | 0.05311 | 0.05586 | 75% |

### Step 2 — Per-cell LOO-CV distribution (R²)

| Fraction | min | p25 | median | p75 | max | worst cell | best cell |
|---|---|---|---|---|---|---|---|
| 5% | +0.121 | +0.180 | +0.427 | +0.703 | +0.849 | B0007 | B0018 |
| 10% | +0.053 | +0.054 | +0.346 | +0.687 | +0.834 | B0006 | B0018 |
| 20% | -0.604 | -0.207 | +0.291 | +0.691 | +0.794 | B0006 | B0018 |
| 30% | -2.004 | -0.657 | +0.253 | +0.718 | +0.730 | B0006 | B0018 |
| 50% | -6.120 | -1.997 | -0.349 | +0.100 | +0.631 | B0006 | B0005 |

### Step 3 — Uncertainty calibration

Nominal 90% interval, overall empirical coverage: **0.739** (2441 pooled predictions). MISCALIBRATED (overconfident): 0.74 actual vs 0.90 nominal.

| Fraction | Empirical coverage | n predictions |
|---|---|---|
| 5% | 0.731 | 602 |
| 10% | 0.720 | 571 |
| 20% | 0.744 | 507 |
| 30% | 0.765 | 443 |
| 50% | 0.748 | 318 |

### Step 3b — Jackknife+ calibration fix (LCO ONLY, Thread 2)

GP posterior (Step 3 above) is overconfident on LCO: 73.9% actual vs 90% nominal. Tested jackknife+ (Barber, Candes, Ramdas & Tibshirani 2021) as a distribution-free interval-construction fix, chosen over Sanchez-Dominguez et al. (2025) small-n conformal because the latter's own guarantee Pr(coverage>=0.9)>=0.9 is numerically unachievable at this n_cal (best achievable: Pr(coverage>=0.9)=0.27-0.34, verified before implementation) -- jackknife+ at least has an achievable finite-width regime (50% guaranteed coverage at alpha=0.25). LFP's existing 94.2% coverage (Step 3) is untouched by this section.

| Fraction | Formal 50%-guarantee: coverage | width | Practical (clipped, unguaranteed): coverage | width |
|---|---|---|---|---|
| 5% | 0.811 | 0.2035 | 0.811 | 0.2035 |
| 10% | 0.806 | 0.2061 | 0.806 | 0.2061 |
| 20% | 0.787 | 0.2052 | 0.787 | 0.2052 |
| 30% | 0.756 | 0.2051 | 0.756 | 0.2051 |
| 50% | 0.739 | 0.2164 | 0.739 | 0.2164 |

> **Formal guarantee note:** the 90%-nominal target (alpha=0.05) requires index k=4, but only n=3 leave-one-out ensemble members are available per fold at this sample size (4 cells, hold one out, jackknife over the other 3) -- NOT achievable with a finite interval under Theorem 1. The maximum formally-guaranteed finite-width coverage at this n is **50%**, not 90%. The 'practical (clipped)' column relaxes the index to the widest available finite value and reports its EMPIRICAL coverage only -- this has no proven guarantee and is not the same claim as the formal 50% column.


### Step 4 — Baseline comparison (MAE, lower is better)

| Fraction | Physics-Bayes | Flat baseline | Linear baseline | Beats flat? | Beats linear? |
|---|---|---|---|---|---|
| 5% | 0.05729 | 0.15816 | 0.04391 | Yes | No |
| 10% | 0.05774 | 0.14709 | 0.06838 | Yes | Yes |
| 20% | 0.05627 | 0.15001 | 0.08280 | Yes | Yes |
| 30% | 0.05515 | 0.13699 | 0.06662 | Yes | Yes |
| 50% | 0.05311 | 0.06493 | 0.03484 | Yes | No |

> **Caveat:** the physics-informed Bayesian model beats flat extrapolation at every fraction tested, but only beats the simple linear-extrapolation baseline at 3/5 fractions. A plain OLS line through the early cycles is a genuinely competitive baseline here — the physics prior's advantage over 'no baseline at all' is clear; its advantage over 'simplest reasonable baseline' is not uniform.


### Step 5 — Residual bias by position in life

| Life bin | n | mean residual | direction |
|---|---|---|---|
| 0.0-0.2 | 151 | +0.06126 | over-predicts fade |
| 0.2-0.4 | 440 | +0.04787 | over-predicts fade |
| 0.4-0.6 | 565 | +0.00417 | over-predicts fade |
| 0.6-0.8 | 640 | -0.02294 | under-predicts fade |
| 0.8-1.0 | 645 | -0.03804 | under-predicts fade |

Near-EOL (80-100% of life) mean residual = -0.03804 dSOH -- under-predicting fade at end-of-life, the region that matters most for replacement/warranty decisions.

### Step 6 — OOD / extrapolation limit

- Validated cycle-life range: **[132, 168]** cycles
- Typical cells (within IQR [159, 168]) R² mean: -0.007451932418683842
- Extreme cells (outside IQR) R² mean: 0.7936454624303476
- Validated on cells with total recorded life in [132, 168] cycles. Predictions for a new cell whose eventual cycle life falls far outside this range (especially above 168) are extrapolation beyond anything tested here and should not be trusted without new validation data.

> **Caveat:** 'extreme' cells scored *better* here (0.794) than 'typical' cells (-0.007) — the opposite of the expected OOD-degrades pattern. With only n=4 cells, the typical/extreme split leaves 1-2 cells per group; this is almost certainly a small-sample artifact, not evidence that extrapolation is safe. Do not read this as 'OOD is fine for this chemistry.'

### Step 7 — Sample size honesty

Method: standard-error interval (n too small for bootstrap to add information)

**n=4 cells means every statistic in this report has only 4 leave-one-out folds. A 95% CI computed from 4 points is itself barely informative -- treat all NASA point estimates in this validation as illustrative of a failure/success MODE, not as precise, generalizable numbers. This is the same caveat hierarchical_beta.py and bayes_gp_predictor.py already state for this dataset.**


## LFP (Severson) (n=124)

> **Read this carefully:** median R² *decreases* as the training fraction increases, while MAE stays roughly flat. This is a known R² pathology, not evidence the model gets worse with more data: R²'s denominator is the variance of the *remaining unseen* trajectory, which shrinks as more of the cell is already observed — the same absolute error produces a worse (or negative) R² against a smaller, noisier target range. **MAE/RMSE are the more trustworthy metrics for the "does more data help" question in this table; R² should be read per-fraction, not trended across fractions.**

### Step 1 — Multi-fraction early-cycle test

| Fraction | n cells | R² mean | R² median | MAE mean | RMSE mean | %R²<0 |
|---|---|---|---|---|---|---|
| 5% | 124 | -0.237 | +0.244 | 0.03671 | 0.04379 | 17% |
| 10% | 124 | -0.289 | +0.233 | 0.03726 | 0.04449 | 16% |
| 20% | 124 | -0.372 | +0.201 | 0.03773 | 0.04556 | 16% |
| 30% | 124 | -0.451 | +0.171 | 0.03784 | 0.04655 | 18% |
| 50% | 124 | -0.779 | +0.128 | 0.03904 | 0.05029 | 35% |

### Step 2 — Per-cell LOO-CV distribution (R²)

| Fraction | min | p25 | median | p75 | max | worst cell | best cell |
|---|---|---|---|---|---|---|---|
| 5% | -17.029 | +0.119 | +0.244 | +0.279 | +0.442 | b1c2 | b1c18 |
| 10% | -18.348 | +0.070 | +0.233 | +0.262 | +0.417 | b1c2 | b1c18 |
| 20% | -20.461 | +0.091 | +0.201 | +0.243 | +0.365 | b1c2 | b1c18 |
| 30% | -22.078 | +0.083 | +0.171 | +0.249 | +0.302 | b1c2 | b1c18 |
| 50% | -27.259 | -0.106 | +0.128 | +0.196 | +0.247 | b1c1 | b3c16 |

### Step 3 — Uncertainty calibration

Nominal 90% interval, overall empirical coverage: **0.942** (373460 pooled predictions). Reasonably calibrated: 0.94 actual vs 0.90 nominal.

| Fraction | Empirical coverage | n predictions |
|---|---|---|
| 5% | 0.957 | 92157 |
| 10% | 0.954 | 87307 |
| 20% | 0.947 | 77608 |
| 30% | 0.936 | 67885 |
| 50% | 0.895 | 48503 |

### Step 4 — Baseline comparison (MAE, lower is better)

| Fraction | Physics-Bayes | Flat baseline | Linear baseline | Beats flat? | Beats linear? |
|---|---|---|---|---|---|
| 5% | 0.03671 | 0.04293 | 0.06753 | Yes | Yes |
| 10% | 0.03726 | 0.04423 | 0.04699 | Yes | Yes |
| 20% | 0.03773 | 0.04572 | 0.03817 | Yes | Yes |
| 30% | 0.03784 | 0.04705 | 0.03683 | Yes | No |
| 50% | 0.03904 | 0.05033 | 0.04109 | Yes | Yes |

> **Caveat:** the physics-informed Bayesian model beats flat extrapolation at every fraction tested, but only beats the simple linear-extrapolation baseline at 4/5 fractions. A plain OLS line through the early cycles is a genuinely competitive baseline here — the physics prior's advantage over 'no baseline at all' is clear; its advantage over 'simplest reasonable baseline' is not uniform.


### Step 5 — Residual bias by position in life

| Life bin | n | mean residual | direction |
|---|---|---|---|
| 0.0-0.2 | 24000 | +0.02926 | over-predicts fade |
| 0.2-0.4 | 67653 | +0.03491 | over-predicts fade |
| 0.4-0.6 | 87317 | +0.03310 | over-predicts fade |
| 0.6-0.8 | 97030 | +0.02027 | over-predicts fade |
| 0.8-1.0 | 97460 | -0.04283 | under-predicts fade |

Near-EOL (80-100% of life) mean residual = -0.04283 dSOH -- under-predicting fade at end-of-life, the region that matters most for replacement/warranty decisions.

### Step 6 — OOD / extrapolation limit

- Validated cycle-life range: **[170, 1934]** cycles
- Typical cells (within IQR [526, 945]) R² mean: 0.17853222613632724
- Extreme cells (outside IQR) R² mean: -0.9221974584174348
- Validated on cells with total recorded life in [170, 1934] cycles. Predictions for a new cell whose eventual cycle life falls far outside this range (especially above 1934) are extrapolation beyond anything tested here and should not be trusted without new validation data.

### Step 7 — Sample size honesty

Method: bootstrap (2000 resamples of 124 cells)

n=124 cells supports a proper bootstrap CI, in sharp contrast to the NASA n=4 case -- the R2 estimates here are far more statistically trustworthy, though the Severson dataset's own protocol-heterogeneity caveat (severson_gp_predictor.py: protocol explains R2=0.452 of beta variance) still applies to what the population variance actually represents.


## Validated range / Known limitations

- **Model is a simplified proxy, not the full GP.** This report uses closed-form conjugate Bayesian linear regression, not the Matern52-kernel GP in `bayes_gp_predictor.py`/`severson_gp_predictor.py`. It has no correlated-residual term; calibration and RMSE numbers here are NOT directly comparable to those sibling modules' published figures.
- **LCO and LFP results are never pooled** — different chemistry, different beta scale, different degradation shape (LFP: near-linear/convex; LCO: concave, matching beta*sqrt(k)). Any cross-chemistry claim would require the separate analysis in `hierarchical_beta_cross_chemistry.py`, which already found this hard (see `docs/problem2_literature_review.md`).
- **NASA LCO n=4**: every statistic has only 4 leave-one-out folds. Point estimates should be read as illustrating a failure/success mode, not as precise numbers. See Step 7.
- **Severson LFP heterogeneity**: `severson_gp_predictor.py` already established that beta variance in this dataset is ~45% explained by protocol (varied fast-charge conditions), not intrinsic cell-to-cell variation — the population prior used here inherits that conflation.
- **Extrapolation boundary**: see Step 6 per chemistry for the exact validated cycle-life range. Predictions for cells whose eventual life falls outside that range are unvalidated extrapolation.
- **Accuracy degrades at low early-data fractions** (5-10%) in both chemistries, as expected — this is reported plainly in Step 1, not hidden behind a single cherry-picked fraction.
