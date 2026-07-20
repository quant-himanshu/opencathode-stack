# VED failure-mode breakdown — Dual EKF, +20 pp protocol

Generated 20260719T174532Z by `data/run_offset_sweep.py`. n = 408 held-out VED trips; strict convergence threshold 5 pp; 'short' = duration < censoring threshold (139 s = VED EKF median strict t_conv); 'recovered' = trip-end error ≤ 10 pp without strict convergence.

| Tier | n | share | short trips | recovered-at-end | median duration (s) | median min\|err\| (pp) | median end-err (pp) |
|---|---|---|---|---|---|---|---|
| converged (strict) | 66 | 16.2% | 4 | 0 | 423 | 0.0 | 1.9 |
| re-diverging (entered 5 pp band, did not hold) | 133 | 32.6% | 2 | 30 | 633 | 0.4 | 36.1 |
| never-approaching (never within 5 pp) | 209 | 51.2% | 9 | 14 | 394 | 19.0 | 27.0 |

> Calibration split: 10% of segments per vehicle for the fleet datasets (BMW i3, Deng, VED); 40% per cell for CALCE and 30% per module for UMich/Ford — data-size-driven exceptions, decided before evaluation (see data/soc_baseline_benchmark_calce.py / _module.py).
> All SOC errors in percentage points; every estimator (including coulomb counting) starts from the same deliberately wrong initial SOC.

## Per-vehicle breakdown (added 2026-07-19, from the same +20 pp per-trip data)

The VED loader is **BEV-only**: it filters `VED_Static_Data_PHEV&EV.xlsx` to
`EngineType == 'EV'`, which matches exactly three vehicles (VehIds 10, 455,
541, all listed as 'Car ELECTRIC'; specific models not identified in the
static file — the loader assigns a generic EV pack with inferred n_series).
**No PHEV/HEV/ICE segments are present**, so the never-approach group is
0% non-BEV — vehicle-type contamination is ruled out as an explanation.

| VehId | n eval trips | converged | re-diverging | never-approach | median RMSE (pp) | median duration (s) |
|---|---|---|---|---|---|---|
| VehId_0010 | 165 | 38 (23%) | 58 (35%) | 69 (42%) | 19.8 | 522 |
| VehId_0455 | 236 | 27 (11%) | 72 (31%) | 137 (58%) | 26.1 | 459 |
| VehId_0541 | 7 | 1 (14%) | 3 (43%) | 3 (43%) | 23.9 | 328 |

All three vehicles fail the same way (never-approach 42–58%, re-diverge
31–43%): the failure mode is fleet-wide, not one bad vehicle, though
VehId_0455 (58% never-approach, median RMSE 26.1 pp) is worst and
VehId_0010 (23% converged, 19.8 pp) least bad. VehId_0541 has only 7
held-out trips — too few for stable rates on its own.

## Hypothesis-test evidence (2026-07-19, `analysis/ved_hypothesis_test.py`, run 20260719T182755Z)

### H-TOPO — topology / chemistry window

| VehId | static name | cartridge used | topology | chem assumed | per-cell p1–p99 (V) | NMC-window OK | LFP-window OK | coulomb-vs-BMS drift (pp/h, median) |
|---|---|---|---|---|---|---|---|---|
| VehId_0010 | EV_Car | Generic EV Pack (fallback) | 96s2p, Q=40.0Ah | NMC | 3.745–4.109 | True | False | +43.15 |
| VehId_0455 | EV_Car | Generic EV Pack (fallback) | 96s2p, Q=40.0Ah | NMC | 3.719–4.104 | True | False | +40.03 |
| VehId_0541 | EV_Car | Generic EV Pack (fallback) | 96s2p, Q=40.0Ah | NMC | 3.740–4.021 | True | False | +53.40 |

### H-OCV — table residuals (near-rest, coulomb SOC)

| VehId | table | bias (mV) | structured RMS (mV) |
|---|---|---|---|
| VehId_0010 | generic_NMC | -28.9 | 61.4 |
| VehId_0010 | generic_LMO-NMC | +7.9 | 57.7 |
| VehId_0010 | LFP_Prada2012 | +488.7 | 61.5 |
| VehId_0010 | NMC811_DFN_rest | -62.4 | 124.1 |
| VehId_0010 | fleet_fitted_empirical **← best** | -5.5 | 6.3 |
| VehId_0455 | generic_NMC | +20.6 | 52.2 |
| VehId_0455 | generic_LMO-NMC | +54.2 | 48.7 |
| VehId_0455 | LFP_Prada2012 | +532.7 | 55.8 |
| VehId_0455 | NMC811_DFN_rest | +6.5 | 123.5 |
| VehId_0455 | fleet_fitted_empirical **← best** | -14.9 | 26.5 |
| VehId_0541 | generic_NMC | +78.1 | 21.2 |
| VehId_0541 | generic_LMO-NMC | +109.0 | 20.6 |
| VehId_0541 | LFP_Prada2012 | +593.6 | 48.9 |
| VehId_0541 | NMC811_DFN_rest | +73.0 | 78.4 |
| VehId_0541 | fleet_fitted_empirical **← best** | -9.9 | 7.3 |

### H-CAL — fleet-level vs per-vehicle calibration (+20 pp, DIAGNOSTIC variant — headline tables stay fleet-level)

| VehId | δR0 (mΩ) | physical? | γ | cal | n | conv | re-div | never | median RMSE (pp) |
|---|---|---|---|---|---|---|---|---|---|
| VehId_0010 | (fleet fit) |  |  | fleet | 165 | 38 | 58 | 69 | 19.8 |
| VehId_0010 | +2.12 | True | 0.5 | per-veh | 165 | 69 | 21 | 75 | 7.7 |
| VehId_0455 | (fleet fit) |  |  | fleet | 236 | 27 | 72 | 137 | 26.1 |
| VehId_0455 | +1.18 | True | 1.0 | per-veh | 236 | 21 | 86 | 129 | 25.9 |
| VehId_0541 | (fleet fit) |  |  | fleet | 7 | 1 | 3 | 3 | 23.9 |
| VehId_0541 | +0.59 | True | 0.5 | per-veh | 7 | 0 | 1 | 6 | 24.0 |

## Verdicts (2026-07-20, evidence from `analysis/ved_hypothesis_test.py` + sign audit)

**(a) Generic-OCV / chemistry-class mismatch — REJECTED as the primary cause.**
All three vehicles' per-cell windows (p1–p99 ≈ 3.72–4.11 V at n_series=96) are
NMC/NCA-consistent and LFP-inconsistent, and the fleet-fitted empirical OCV
beats every generic table in the repo for all three vehicles (structured RMS
6.3 / 26.5 / 7.3 mV for 0010 / 0455 / 0541) — the benchmark was already using
the best available OCV. Caveat: the residual-vs-SOC axis used coulomb-derived
SOC, which the sign/capacity finding below corrupts; treat residual *shapes*
qualitatively. VehId_0455's structured residual is ~4× worse than its
siblings' — within-fleet OCV heterogeneity is real but secondary.

**(b) Per-vehicle heterogeneity that fleet-level calibration can't capture —
SUPPORTED for VehId_0010, INSUFFICIENT for 0455.** Per-vehicle δV(SOC)/δR0/γ
(same split discipline, same held-out trips; diagnostic variant only):
0010 median RMSE 19.8 → **7.7 pp**, converged 38 → 69 of 165. 0455:
26.1 → 25.9 pp (no help). 0541: 7 trips, 1 cal segment — unusable. All
per-vehicle δR0 fits are physical (+0.6 to +2.1 mΩ).

**(c′) NEW — assumed cartridge capacity/topology is wrong, and the schema
current sign inverts the coulomb baseline.** All three vehicles fall back to
GENERIC_EV_PACK (96s2p × 40 Ah = 80 Ah); the data imply |∫I dt / ΔSOC_BMS| ≈
40–43 Ah pack-equivalent — a ~1.9× capacity overstatement, numerically
consistent with a 96s**1p** ≈ 40 Ah pack (evidence-based inference only; no
vehicle identity is hard-coded). The negative sign of the implied capacity
(all 154 qualifying trips) confirms VED's schema is discharge-negative, so
the coulomb BASELINE columns for VED are sign-inverted (the EKF chain is
sign-correct here; see PHASE2_FINDINGS §5). The capacity error biases the
EKF's process model on every VED trip and is the leading fleet-wide suspect
for the re-diverging tier and for θ staying active on VED.

**(c) Residual unknown — VehId_0455's ~26 pp floor.** Neither per-vehicle
calibration nor the online bias variant moves it; it contributes 66% of the
never-approach tier and has the worst OCV residual structure. Its identity/
pack differs in some way the current model family does not capture —
candidate for Phase 4 with the capacity fix in place.

> Numbers from runs `ved_hypothesis_*.json` and the cross-checked sweep dump;
> no estimator or headline-pipeline code was changed for this analysis.
