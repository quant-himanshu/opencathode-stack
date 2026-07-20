# METRICS — standardized definitions (Phase 1)

Single source of truth for every metric in the paper. Implementation:
`validation/metrics.py` (unit tests: `tests/test_metrics.py`, 20 synthetic
known-answer cases). Regeneration: `venv/bin/python data/run_main_table.py`
→ `results/main_table.csv` / `.md` + per-segment dump.

Units: SOC internally a fraction in [0, 1]; **all reported errors are in
percentage points** (pp). "Trip" = one held-out evaluation segment (a
BMW/VED trip, a Deng charging session, a CALCE 30-min or UMich 15-min lab
window).

---

## 1. What the OLD headline numbers were, precisely

The pre-Phase-1 numbers of the form "BMW 17.69 / 29.42 / 37.75" were:

> **arithmetic MEAN over held-out segments of the per-segment full-trip SOC
> RMSE**, where per-segment RMSE = `sqrt(mean((SOC_est − SOC_true)²)) × 100`
> computed over **every sample of the segment including the initial
> +20-pp-wrong-init transient**
> (`data/soc_baseline_benchmark.py:133/139/154`, aggregated at `:254-269`).

Properties to be aware of (why Phase 1 standardizes):

- **Mean across segments** — a single diverged segment inflates it
  arbitrarily. On CALCE the EKF's mean is 35.18 pp but the median is
  25.29 pp: the gap is entirely a divergence tail.
- **Full-trip window** — deliberately includes the recovery transient from
  the adversarial wrong init; short trips are therefore dominated by it.
  This is a *stress-test* number, not a steady-state accuracy number.
- The old per-fleet **convergence time** (`validate_generic.py:221-231`,
  reported in `reports/real_fleet_validation.md`) required 30 CONSECUTIVE
  SAMPLES below 5 pp — a **sample-count** window, hence dt-dependent
  (≈30 s on 1 s BMW data, 240 s on 8 s Deng data, 10 min on 20 s-resampled
  data), and it did **not** require staying converged afterwards.

The old headline numbers remain reproducible byte-identically (Phase 0) and
are retained in the new table as the `rmse_full_mean` column — the mean and
the old headline are THE SAME NUMBER by construction, verified per-segment
against the committed reports at tolerance 1e-6 pp on every run of
`data/run_main_table.py` (abort-on-mismatch).

## 2. Standardized per-trip metrics (NEW)

For each trip and each method (EKF / coulomb / OCV-lookup), with
`err(t) = SOC_est(t) − SOC_true(t)`:

| Metric | Definition |
|---|---|
| `rmse_full_pct` | `sqrt(mean(err²)) × 100` over the full trip (identical sample set to the old numbers) |
| `mae_full_pct` | `mean(abs(err)) × 100`, full trip |
| `maxerr_full_pct` | `max(abs(err)) × 100`, full trip |
| `rmse_postconv_pct` (+ MAE, max) | same, restricted to `t ≥ t_c(strict)`; `None` when the trip never strictly converges |

## 3. Convergence definitions (threshold = 5 pp for all three)

| Name | Definition | Role |
|---|---|---|
| **STRICT** | first `t_c` with `abs(err) < 5 pp` at `t_c` **and at every later sample of the trip** | **primary** (paper headline) |
| **HOLD-600** | first `t_c` whose continuous below-threshold run lasts ≥ 600 s, or reaches the trip end when < 600 s remain ("whichever the trip length supports") | sensitivity: forgives re-divergence after 10 min of holding |
| **LEGACY** | verbatim port of the pre-Phase-1 rule: first sample whose next 30 consecutive samples are below threshold | comparability: shows exactly how much the old definition flattered/penalized each dataset |

Reported per dataset × method: **convergence rate** (fraction of trips
converged) and **median time-to-converge among converged trips**, for each
of the three definitions side by side (per 2026-07-19 review decision (b)).

Known LEGACY quirks, preserved deliberately in the port
(`validation/metrics.py:convergence_time_legacy`): trips with ≤ 30 samples
can never converge; a below-threshold window cannot *start* within the last
30 samples; dt-dependence as above.

Boundary conventions (all three definitions): comparison is strict
(`< 0.05`); an error of exactly 5.000 pp counts as NOT converged. A trip
already below threshold at its first sample has `t_c = 0`.

## 4. Cross-trip aggregation

Primary: **median and IQR (q25–q75)** across held-out trips — robust to the
divergence tail. **Mean is also reported** for every metric for
comparability with the old headline numbers (§1). Time-to-converge is
aggregated as the **median among converged trips only**, always next to the
corresponding convergence *rate* (never one without the other — a fast
median over 8% of trips is not a good number).

## 5. Standing table footnotes

Every generated results table (CSV: `#` comment lines; markdown: `>`
blockquote; LaTeX later: `\tablenote`) carries the standing footnotes from
`validation/metrics.py:TABLE_FOOTNOTES` (per 2026-07-19 review decision (c)):

1. Calibration split is 10% per vehicle for the fleet datasets (BMW i3,
   Deng, VED) but **40% per cell for CALCE** and **30% per module for
   UMich/Ford** — data-size-driven exceptions, decided before evaluation
   (documented in `data/soc_baseline_benchmark_calce.py` / `_module.py`).
2. All estimators, including the coulomb-counting baseline, start from the
   same deliberately wrong initial SOC (+20 pp headline protocol).

## 6. Ground truth per dataset (unchanged from Phase 0)

| Dataset | SOC ground truth |
|---|---|
| BMW i3, Deng, VED | vehicle BMS-reported SOC (imperfect; disclosed) |
| CALCE A123 | Arbin BT2000 cumulative charge/discharge capacity |
| UMich/Ford module | Maccor S4000 cumulative capacity |

Lab ground truths are themselves coulomb-counting-derived, giving the pure
coulomb baseline a disclosed home-field advantage on CALCE/UMich.
