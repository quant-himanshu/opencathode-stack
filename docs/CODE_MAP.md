# CODE MAP — SOC/SOH Dual-EKF Field-Study Pipeline

> Phase 0 repository audit (2026-07-19). Documents where every piece of the
> SOC/SOH estimation pipeline lives, with file paths and line references, as
> the basis for the paper-preparation work. No code was changed in this phase.
>
> NOTE: line numbers are accurate as of commit `9b749c0`. They will drift as
> files are edited; section/function names are the stable reference.

---

## 1. Entry points

| What | Command | Notes |
|---|---|---|
| Fleet validation harness (Mode A / Mode B) | `python data/validate_generic.py --all` | Writes `reports/real_fleet_validation.md`. CLI: `validate_generic.py:1147-1238` |
| **Headline benchmark, BMW + Deng + VED** | `python data/soc_baseline_benchmark.py` | Writes `data/soc_baseline_benchmark_report.json` + `docs/soc_baseline_benchmark.md` |
| **Headline benchmark, CALCE A123** | `python data/soc_baseline_benchmark_calce.py` | Writes `data/soc_baseline_benchmark_calce_report.json` |
| **Headline benchmark, UMich/Ford module** | `python data/soc_baseline_benchmark_module.py` | Writes `data/soc_baseline_benchmark_module_report.json` |
| Non-headline: iontech/BattGP portable LFP | `python data/soc_baseline_benchmark_device.py` | Excluded from headline table (dataset self-flagged as field-returned faulty units; see report meta `known_bias`) |
| Non-headline: Quartz WLTP | `python data/soc_baseline_benchmark_quartz.py` | |
| Legacy full-stack simulation demo | `python main.py` | Not part of the paper pipeline |

All benchmark scripts import and reuse `validate_generic.py`'s loaders,
calibration split, and `run_mode_b_ekf()` unmodified, so the EKF being
benchmarked is exactly the one used in fleet validation.

## 2. The Dual EKF — `diagnosis/dual_ekf_lfp.py`

Class `DualEKF_LFP` (`dual_ekf_lfp.py:77`).

| Piece | Location |
|---|---|
| Module docstring: design history (Round 2 coupled-Jacobian failure, Round 3 decoupling, Round 4 slow loops + safety guard) | `dual_ekf_lfp.py:1-64` |
| Fast-loop state `x1 = [SOC, V_polarization]`, `P1`, `Q_base = diag(1e-6, 1e-5)` | `dual_ekf_lfp.py:118-121` |
| Slow-loop state `x2 = [SOH, R_int]`, `P2` | `dual_ekf_lfp.py:123-124` |
| **Predict step** (coulomb integration + 1st-order RC with τ = 50 s, fixed) | `dual_ekf_lfp.py:277-281` |
| **Measurement model** `V_pred = OCV(SOC) − I·R + V_pol + δV(SOC) + δR0·I` | `dual_ekf_lfp.py:284-286` |
| **Decoupled Jacobian** `H = [∂OCV/∂SOC, 1]` — ∂δV/∂SOC deliberately EXCLUDED (core design decision; Round 2 coupling bug documented in docstring at lines 6-10 and 258-263) | `dual_ekf_lfp.py:288-290` |
| Kalman update (standard form, not Joseph) | `dual_ekf_lfp.py:292-297` |
| **Adaptive Q**: `Q_base × min(1/max(|∂OCV/∂SOC|, 0.02), 50) × γ` | `_adaptive_Q`, `dual_ekf_lfp.py:179-182` |
| ∂OCV/∂SOC by central difference (h = 0.005) | `_docv_dsoc`, `dual_ekf_lfp.py:169-171` |
| **R_int slow loop**: scalar Kalman update on IR-drop term, gated to \|I\| > 1 A (`R_INT_MIN_CURRENT_A`, line 68), clipped to [0.1, 5]× init | `_update_r_int`, `dual_ekf_lfp.py:184-218` |
| **SOH slow loop**: accumulate Ah + ΔSOC; fire when \|ΔSOC\| ≥ 0.05 (`SOH_MIN_DSOC`, line 74); `SOH_obs = Ah / (ΔSOC·Q_nom)`; confidence-weighted blend; clip [0.5, 1.05] | `_update_soh`, `dual_ekf_lfp.py:220-254` |
| **Calibration sanity gate**: R_int loop disabled entirely when \|cal_dR0\| ≥ 50 mΩ (`CAL_DR0_SANITY_THRESHOLD_OHM`, line 69) — fires on CALCE (−260 mΩ) and UMich module (−233 mΩ) | `dual_ekf_lfp.py:127-140` |
| Built-in Prada-2012 LFP OCV table (fallback when no `ocv_fn` given) | `dual_ekf_lfp.py:156-162` |
| SOC init (`set_soc`) | `dual_ekf_lfp.py:311-315` |
| Built-in self-test | `validate()`, `dual_ekf_lfp.py:318-411` |

## 3. Mode B runner + offset injection — `data/validate_generic.py`

| Piece | Location |
|---|---|
| `ValidationConfig` (per-fleet: n_series, n_parallel, Q_cell, chemistry, dt, **`ekf_soc_offset = 0.20`**) | `validate_generic.py:92-107` |
| **+20 %-point wrong-init injection**: `soc_init_offset = clip(SOC_true[0] + ekf_soc_offset, 0.02, 0.98)` | `run_mode_b_ekf`, `validate_generic.py:291` |
| Initial covariance matched to offset: `P0_soc = ekf_soc_offset²` | `validate_generic.py:296-297` |
| Free-running EKF loop (per-sample `ekf.update(V, −I, dt, T)`; no state carryover between segments — a fresh `DualEKF_LFP` per segment) | `run_mode_b_ekf`, `validate_generic.py:268-334` |
| Mode A (forced-SOC voltage prediction, used only to fit calibration residuals) | `run_mode_a_forced`, `validate_generic.py:238-261` |
| **Convergence criterion (current)**: first t where 30 *consecutive samples* have \|SOC_est − SOC_true\| < 0.05 | `_ekf_convergence_time`, `validate_generic.py:221-231` |
| Per-segment dispatcher + short-segment filter/resampling | `validate_segment`, `validate_generic.py:514-606` |
| Markdown fleet report writer | `write_report`, `validate_generic.py:768-890` |

⚠️ Phase-1 relevant: the 30-consecutive-samples window is **dt-dependent**
(30×20 s = 10 min on 20 s-resampled fleets, 30×5 s = 2.5 min on 5 s data) and
does not require holding to end-of-trip. Phase 1 replaces this with the
standardized definition.

## 4. Offline calibration (δV(SOC), δR0, OCV, γ, R_meas)

All in `data/validate_generic.py` unless noted:

| Piece | Location |
|---|---|
| **Calibration split**: first 10 % of segments per vehicle (`cal_frac = 0.10`) | `_split_by_vehicle`, `validate_generic.py:452-466` |
| Exception: CALCE uses 40 % (data-size-driven, disclosed in comment) | `soc_baseline_benchmark_calce.py:174-184` |
| Exception: UMich module split | `soc_baseline_benchmark_module.py` (mirrors CALCE pattern) |
| **δR0 fit**: OLS of voltage residual vs I (slope), joint with constant | `fit_soc_calibration`, `validate_generic.py:365-435` (OLS at 389-395) |
| **δV(SOC) fit**: median residual in 12 uniform SOC bins (≥5 pts/bin) → PCHIP knots | `validate_generic.py:397-435`; spline built in `FleetCalibration.soc_cal_fn`, `validate_generic.py:138-143` |
| Residual source: Mode A zero-cal predictions on calibration segments | `_collect_cal_quad`, `validate_generic.py:438-449` |
| **Empirical OCV curve** from near-rest fleet points (PCHIP; generic-table fallback if <4 bins) | `build_fleet_ocv`, `diagnosis/nmc_ocv.py:134-186` |
| **γ sweep** {0.5, 1, 2, 4} on calibration segments only | `_tune_gamma`, `validate_generic.py:469-507` |
| **Fleet R_meas table** (sensor-quantization-based) | `_FLEET_R_MEAS`, `validate_generic.py:905-910` |
| Orchestration of all of the above per fleet | `_build_calibration_for_fleet`, `validate_generic.py:913-965` |

Inside the EKF, δV(SOC) enters via `cal_soc_fn` → `_cal_offset`
(`dual_ekf_lfp.py:173-177`) and δR0 via `cal_dR0` (`dual_ekf_lfp.py:285`).

## 5. Dataset loaders

| Dataset | Loader | Data dir | Notes |
|---|---|---|---|
| BMW i3 (TUM RDC, 63 trips) | `BMWI3Loader`, `data/loaders/bmw_i3_loader.py:205` (`iter_segments`:234) | `data/bmw_i3/` (TripA*.csv, TripB*.csv) | NMC pack, cartridge `BMW_I3_60AH` (`pack_cartridge.py:99`) |
| Deng BAIC EU500 (charging sessions) | `DengChargingLoader`, `data/loaders/deng_charging_loader.py:354` (`iter_segments`:392; SOH trajectories: `SOHTrajectory`, :286) | `data/deng20/` | Cartridge `BAIC_EU500_90S` (`pack_cartridge.py:278`); >12 h sessions dropped as merge artifacts; eval = seeded 2000-session sample (seed 42, `soc_baseline_benchmark.py:187-203`) |
| VED (Michigan, mixed fleet) | `VEDLoader`, `data/loaders/ved_loader.py:431` (`iter_segments`:466) | `data/ved/` | Per-vehicle cartridge via `lookup_ved_cartridge` (`pack_cartridge.py:377`); <120 s segments skipped; 5 s resample for 120–600 s trips |
| CALCE A123 (2 cells, DST/US06/FUDS 25 °C) | inline `load_calce_file`, `soc_baseline_benchmark_calce.py:104-140` | `data/calce/DST-US06-FUDS-25/` | Ground truth from Arbin cumulative capacity columns; 30-min windows; Q_eff measured from file |
| UMich/Ford 3-parallel module (4 of 78 modules) | inline `load_module_folder`, `soc_baseline_benchmark_module.py:94` | `data/parallel_module_dataset/` | Mendeley DOI 10.17632/ssrgfmb8vw.2; 15-min windows; module-level CSV |

Shared segment schema/utilities: `data/loaders/common_schema.py`
(`SegmentMeta`:48, `split_segments`:122, `resample_to_uniform_dt`:179,
`make_schema_df`:199).

## 6. Baselines — `data/soc_baseline_benchmark.py`

| Baseline | Location | Key fairness detail |
|---|---|---|
| Pure coulomb counting | `coulomb_counting_soc`, `soc_baseline_benchmark.py:62-86` | Starts from the **same +20 %-offset wrong init** as the EKF (line 80) — documented fix for an earlier unfair version |
| Naive OCV lookup | `ocv_lookup_soc` + `_invert_ocv`, `soc_baseline_benchmark.py:93-118` | Inverts the same fleet-fitted OCV curve against raw terminal voltage; no IR compensation |
| Per-segment eval (all three methods on identical data) | `evaluate_segment`, `soc_baseline_benchmark.py:125-159` | |
| Aggregation (**mean** of per-segment RMSE across segments) | `_aggregate`, `soc_baseline_benchmark.py:254-269` | Phase 1 adds median/IQR |

## 7. Metric definition behind the headline numbers

The headline "17.69 %"-style numbers are, precisely:

> **mean over held-out segments of ( per-segment RMSE(SOC_est − SOC_true) × 100 )**,
> full-trip window (including the pre-convergence transient from the +20 %
> wrong init), computed at `soc_baseline_benchmark.py:133/139/154`,
> averaged at `soc_baseline_benchmark.py:254-269`.

Ground truth per dataset: BMS-reported SOC (BMW/Deng/VED), Arbin cumulative
capacity (CALCE), Maccor cumulative capacity (UMich module).

## 8. Results files (current, all git-tracked)

| File | Producer |
|---|---|
| `data/soc_baseline_benchmark_report.json` + `docs/soc_baseline_benchmark.md` | `data/soc_baseline_benchmark.py` |
| `data/soc_baseline_benchmark_calce_report.json` | `..._calce.py` |
| `data/soc_baseline_benchmark_module_report.json` | `..._module.py` |
| `data/soc_baseline_benchmark_device_report.json` | `..._device.py` (non-headline) |
| `reports/real_fleet_validation.md` | `data/validate_generic.py` |
| `data/fleet_validation_report*.json` | earlier validation rounds |

## 9. Phase 0 reproduction status (2026-07-19)

Environment: Python 3.14.2, macOS 26.5.1 (arm64), numpy 2.4.6, pandas 3.0.3,
scipy 1.17.1. Full freeze: `requirements-lock.txt` (repo root).

| Dataset | EKF | Coulomb | OCV lookup | Match vs committed report |
|---|---|---|---|---|
| BMW i3 (n=63) | 17.69 | 29.42 | 37.75 | byte-identical JSON + md |
| Deng BAIC EU500 (n=2000) | 9.69 | 40.12 | 6.05 | byte-identical JSON + md |
| VED (n=408) | 25.71 | 16.05 | 14.83 | byte-identical JSON + md |
| CALCE A123 (n=26) | 35.18 | 15.66 | 36.81 | byte-identical JSON |
| UMich/Ford module (n=216) | 15.49 | 15.33 | 21.07 | byte-identical JSON |

All 15 headline numbers match the committed reports exactly (0.00 deviation);
`git diff` on every regenerated report file is empty, i.e. the pipeline is
fully deterministic on this machine (Deng eval sample seeded, seed=42).

## 10. Known sharp edges (candidate paper/Phase-1+ items — no code changed)

1. **Convergence definition is dt-dependent** (§3) — Phase 1 standardizes it.
2. **Aggregation is mean, not median** (§6) — one diverged segment can dominate.
3. `DualEKF_LFP` uses the **standard covariance update, not Joseph form**
   (`dual_ekf_lfp.py:297`); the RBC-DEKF baseline (Phase 2) specifies Joseph form.
4. CALCE/UMich δR0 calibration fits are wildly non-physical (−260/−233 mΩ) and
   trip the EKF's sanity gate — already flagged in the Round 4 docstring; this
   is Phase 4 failure-analysis material.
5. `run_mode_b_ekf` passes `−I` to the EKF (sign convention flip at
   `validate_generic.py:322/327`): loaders emit discharge-positive? — actually
   loaders normalize to the project convention and the flip makes `I_A`
   discharge-positive inside the EKF; verify per-loader in Phase 4 diagnostics.
6. The DFN cell stepped alongside the EKF in `run_mode_b_ekf`
   (`validate_generic.py:299-329`) does not feed the EKF's SOC estimate — its
   `V_pred` is only used for Mode-B voltage-accuracy metrics. The EKF is
   self-contained (RC Thevenin + OCV table).
