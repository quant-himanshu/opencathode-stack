# VED failure-mode breakdown — Dual EKF, +20 pp protocol

Generated 20260719T193500Z by `data/run_offset_sweep.py`. n = 408 held-out VED trips; strict convergence threshold 5 pp; 'short' = duration < censoring threshold (139 s = VED EKF median strict t_conv); 'recovered' = trip-end error ≤ 10 pp without strict convergence.

| Tier | n | share | short trips | recovered-at-end | median duration (s) | median min\|err\| (pp) | median end-err (pp) |
|---|---|---|---|---|---|---|---|
| converged (strict) | 66 | 16.2% | 4 | 0 | 423 | 0.0 | 1.9 |
| re-diverging (entered 5 pp band, did not hold) | 133 | 32.6% | 2 | 30 | 633 | 0.4 | 36.1 |
| never-approaching (never within 5 pp) | 209 | 51.2% | 9 | 14 | 394 | 19.0 | 27.0 |

> Calibration split: 10% of segments per vehicle for the fleet datasets (BMW i3, Deng, VED); 40% per cell for CALCE and 30% per module for UMich/Ford — data-size-driven exceptions, decided before evaluation (see data/soc_baseline_benchmark_calce.py / _module.py).
> All SOC errors in percentage points; every estimator (including coulomb counting) starts from the same deliberately wrong initial SOC.

## Hypothesis-test evidence (2026-07-19, `analysis/ved_hypothesis_test.py`, run 20260719T195454Z)

### H-TOPO — topology / chemistry window

| VehId | static name | cartridge used | topology | chem assumed | per-cell p1–p99 (V) | NMC-window OK | LFP-window OK | coulomb-vs-BMS drift (pp/h, median) |
|---|---|---|---|---|---|---|---|---|
| VehId_0010 | EV_Car | Generic EV Pack (fallback) | 96s2p, Q=40.0Ah | NMC | 3.745–4.109 | True | False | +9.91 |
| VehId_0455 | EV_Car | Generic EV Pack (fallback) | 96s2p, Q=40.0Ah | NMC | 3.719–4.104 | True | False | +12.37 |
| VehId_0541 | EV_Car | Generic EV Pack (fallback) | 96s2p, Q=40.0Ah | NMC | 3.740–4.021 | True | False | +19.97 |

### H-OCV — table residuals (near-rest, coulomb SOC)

| VehId | table | bias (mV) | structured RMS (mV) |
|---|---|---|---|
| VehId_0010 | generic_NMC | -1.1 | 56.2 |
| VehId_0010 | generic_LMO-NMC | +34.7 | 52.8 |
| VehId_0010 | LFP_Prada2012 | +524.6 | 54.3 |
| VehId_0010 | NMC811_DFN_rest | -40.7 | 116.7 |
| VehId_0010 | fleet_fitted_empirical **← best** | +2.1 | 5.1 |
| VehId_0455 | generic_NMC | +46.0 | 47.5 |
| VehId_0455 | generic_LMO-NMC | +78.9 | 44.4 |
| VehId_0455 | LFP_Prada2012 | +556.4 | 49.2 |
| VehId_0455 | NMC811_DFN_rest | +38.6 | 124.9 |
| VehId_0455 | fleet_fitted_empirical **← best** | -4.3 | 29.7 |
| VehId_0541 | generic_NMC | +90.5 | 9.8 |
| VehId_0541 | generic_LMO-NMC | +120.8 | 10.0 |
| VehId_0541 | LFP_Prada2012 | +598.5 | 49.2 |
| VehId_0541 | NMC811_DFN_rest | +93.6 | 73.4 |
| VehId_0541 | fleet_fitted_empirical **← best** | -3.8 | 6.8 |

### H-CAL — fleet-level vs per-vehicle calibration (+20 pp, DIAGNOSTIC variant — headline tables stay fleet-level)

| VehId | δR0 (mΩ) | physical? | γ | cal | n | conv | re-div | never | median RMSE (pp) |
|---|---|---|---|---|---|---|---|---|---|
| VehId_0010 | (fleet fit) |  |  | fleet | 165 | 38 | 58 | 69 | 19.8 |
| VehId_0010 | +2.12 | True | 0.5 | per-veh | 165 | 69 | 21 | 75 | 7.7 |
| VehId_0455 | (fleet fit) |  |  | fleet | 236 | 27 | 72 | 137 | 26.1 |
| VehId_0455 | +1.18 | True | 1.0 | per-veh | 236 | 21 | 86 | 129 | 25.9 |
| VehId_0541 | (fleet fit) |  |  | fleet | 7 | 1 | 3 | 3 | 23.9 |
| VehId_0541 | +0.59 | True | 0.5 | per-veh | 7 | 0 | 1 | 6 | 24.0 |

## Per-vehicle outcomes and verdicts (post-sign-fix, 2026-07-20)

Loader is BEV-only (`EngineType == 'EV'`: VehIds 10, 455, 541, all 'EV_Car';
no PHEV/HEV/ICE segments — type contamination excluded). EKF rows were
sign-correct throughout on VED; coulomb columns changed with the fix.

| VehId | n eval | converged | re-diverging | never-approach | median RMSE (pp) |
|---|---|---|---|---|---|
| 0010 | 165 | 38 (23%) | 58 (35%) | 69 (42%) | 19.8 |
| 0455 | 236 | 27 (11%) | 72 (31%) | 137 (58%) | 26.1 |
| 0541 | 7 | 1 | 3 | 3 | 23.9 |

**(a) Chemistry-class mismatch — REJECTED.** Per-cell windows (3.72–4.11 V
at 96s) are NMC-consistent, LFP-inconsistent, for all three vehicles; the
fleet-fitted empirical OCV beats every generic table (structured RMS
5.1 / 29.7 / 6.8 mV for 0010/0455/0541, corrected coulomb-SOC axis).
0455's residual structure is ~6× worse than its siblings' — real
within-fleet heterogeneity, but not a wrong chemistry class.

**(b) Per-vehicle heterogeneity — SUPPORTED for 0010, INSUFFICIENT for
0455.** Per-vehicle δV/δR0/γ (diagnostic variant; same splits, same
held-out trips): 0010 median RMSE 19.8 → **7.7 pp**, converged 38 → 69 of
165; 0455 unchanged (26.1 → 25.9); 0541 unusable (1 cal segment). All
per-vehicle δR0 physical (+0.6 to +2.1 mΩ).

**(c′) Cartridge capacity error — CONFIRMED with corrected sign.** All
three fall back to GENERIC_EV_PACK (96s2p × 40 Ah = 80 Ah); implied
|∫I dt / ΔSOC_BMS| ≈ 40–43 Ah — ~1.9× overstatement, numerically
consistent with a 96s1p ≈ 40 Ah pack (evidence-based inference only).
Corrected coulomb-vs-BMS drift: +9.9 / +12.4 / +20.0 pp/h — a standing
process-model bias on every trip; leading fleet-wide suspect for the
re-diverging tier and for the online bias state staying active on VED.

**(c) Residual unknown — VehId_0455.** ~26 pp floor survives per-vehicle
calibration and the online-bias variant; contributes 66% of never-approach.
Phase-4 candidate once the capacity question is settled.
