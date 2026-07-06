# OpenCATHODE Stack — Final Project Report

**Methodology contract (Section 0):** Every numerical claim in this report traces to
a commit hash or a specific field in a recorded JSON file. Nulls are reported as nulls
with root causes. No claim is smoothed into a partial success. Every deviation from a
pre-registered plan, every retraction, and every figure correction is listed explicitly
in the Appendix, with commit hashes. This contract takes precedence over any desire for
a cleaner narrative.

---

## 1. Core BMS Stack — Validated

### 1.1 Architecture

Six-layer stack: DFN-SPM electrochemical model → EIS impedance calibration → Dual EKF
SOC/SOH estimator → GraphSAGE GNN (prototype) → NSA anomaly detector (prototype) →
ACO+Kuramoto action control (prototype). The first three layers are validated on real
data. The last three are prototype — implemented and architecture-verified, not
end-to-end validated. None of the prototype layer metrics appear in any headline figure.

### 1.2 Primary accuracy figures

**Two validation contexts. Do not mix.**

| Dataset | Context | Metric | Value | Source |
|---|---|---|---|---|
| Quartz WLTP 36-cell NMC811 pack | Lab protocol, sensor-update rows only | MAE | **18.6 mV** | `stack_validation_report.json`, `mae_mv` |
| Quartz WLTP 36-cell NMC811 pack | Lab protocol, sensor-update rows only | R² | **0.9217** | `stack_validation_report.json`, `r2_sensor_update_rows` |
| Quartz WLTP 36-cell NMC811 pack | Lab protocol, all rows (83% repeated BMS readings) | R² | 0.9810 | `stack_validation_report.json`, `r2_all_rows` |
| BMW i3 RDC, 63 held-out trips | Fleet, scale-calibrated | MAE | 35.6 mV | `stack_validation_report.json`, `bmw_i3_ekf.mae_scale_cal_mv` |
| BMW i3 RDC, 63 held-out trips | Fleet | SOC RMSE | 20.8 % | `stack_validation_report.json`, `bmw_i3_ekf.soc_rmse_pct` |
| Deng BAIC EU500, 2,000 sessions | Fleet, scale-calibrated | MAE | 15.5 mV | `stack_validation_report.json`, `deng_charging_ekf.mae_scale_cal_mv` |
| Deng BAIC EU500, 2,000 sessions | Fleet | SOC RMSE | 11.9 % | `stack_validation_report.json`, `deng_charging_ekf.soc_rmse_pct` |
| VED (30 vehicles, 38 segments) | Fleet, scale-calibrated | MAE | 40.7 mV | `stack_validation_report.json`, `ved_ekf.mae_scale_cal_mv` |
| VED (30 vehicles, 38 segments) | Fleet | SOC RMSE | 20.2 % | `stack_validation_report.json`, `ved_ekf.soc_rmse_pct` |

**On the R² values:** The 0.9810 figure (all rows) is inflated because 83% of Quartz
timestamps are step-constant repeated BMS readings at the 6-minute sensor update
interval — trivial to predict. The 0.9217 figure covers only the ~12,690 rows
(~17% of timestamps) where a genuinely new sensor measurement arrives. R²=0.9217 is
the honest headline; 0.9810 is noted for completeness only. Source: commit `cde6a74`
(README correction) and `8409f8d` (architecture doc). Original headline of 0.9810
appeared in commit `b45ce3f` and is documented in Appendix Entry 7.

**On the 5–8% SOC RMSE figure:** A "5–8% SOC RMSE" figure appeared in comparison
materials before commit `8409f8d` marked it as unsourced and replaced it with the
recorded fleet figures above (11.9–25.5%). See Appendix Entry 7.

### 1.3 Jacobian-decoupling design choice

The Dual EKF retains the δV(SOC) PCHIP calibration curve in the *innovation* term —
so the filter sees calibration-corrected residuals — but deliberately sets the
calibration slope to zero in the measurement Jacobian:

```
H = [∂OCV/∂SOC,   1.0]        (∂δV/∂SOC := 0)
```

**Evidence for why the full Jacobian fails.** Round 2 (commit `25c71c7`) included
dcal/dSOC in H. The PCHIP spline slope introduced large, erratic Kalman gain swings.
Result: BMW i3 — 0/63 trips converged (SOC RMSE = N/A); VED — SOC RMSE = 45.6%.
Round 3 fix (commit `6585528`): zeroed ∂δV/∂SOC in H. Post-fix figures:
BMW 20.8%, Deng 11.9%, VED 25.5% (commit `6585528`) / 20.2% (JSON, `ved_ekf.soc_rmse_pct` —
later run; the two figures bracket the same regime). The decoupling is the binding
architectural fix that makes fleet convergence possible.

**Honest cost of the decoupling.** On flat-OCV chemistry regions (LFP plateau,
SOC 30–75%), ∂OCV/∂SOC → 0 → H ≈ 0 → Kalman gain K ≈ 0. The filter becomes
voltage-blind and falls back to open-loop coulomb counting. Adaptive process noise
Q ∝ 1/|∂OCV/∂SOC| (source: Mikhak 2024, PMC12936157) partially mitigates this by
widening P, but does not restore observability. This is a known architectural
limitation — not a future-work item, a structural property.

### 1.4 Scope

Validation is on NMC811 (Quartz WLTP) and mixed NMC fleet data (BMW i3, Deng BAIC,
VED). The NASA B0018 result (MAE=102 mV, R²=0.784) reflects an OCV approximation
error from a dynamic-discharge fit (not GITT) and a chemistry mismatch (NMC811 DFN
vs Sanyo LCO cell). It is not a headline accuracy number.

---

## 2. Module 2 — Stress-Fatigue Degradation

### 2.1 Lab result

β (stress-fatigue rate coefficient) extracted per cell from NASA PCoE B0005/B0007/B0018
(B0006 excluded — see Section 3). Within-cell R²=0.9725, MAE=0.0126 SOH units.
Source: `stack_validation_report.json`, `nasa_degradation_module2`.

**Interpretation:** D_per_cycle ≈ constant for identical CC discharge protocol. β
variation across cells therefore captures manufacturing susceptibility to mechanical
damage accumulation, not stress amplitude. β is not comparable across cells on
different protocols without normalising by D.

### 2.2 Field null — honestly documented

Cross-cell R²=−0.683 (source: `stack_validation_report.json`,
`nasa_degradation_module2.eval_B_cross_cell_r2`). On the Deng BAIC EU500 fleet
(20 vehicles, 2-year operation), SEI/calendar aging dominates:
λ=0.026 SOH/√yr contributes ~4% fade at 2.3 years; the stress term contributes
≈3.5×10⁻⁹ ΔSOH — negligible. Model B (stress + SEI) achieves MAE≈3.7% ΔSOH,
indistinguishable from Model A (stress only) because the signal is below the
per-session BMS capacity noise (~2.8% SOH std). SNR<1 for 8/20 vehicles.

The field result is a null. The lab result (β extraction) is valid in its regime.
The two are not contradictory — they characterise different degradation regimes
(stress-dominated lab cycling vs calendar-dominated urban fleet operation).

---

## 3. Module 3 — LLI/LAM/CL ICA Decomposition

### 3.1 Genuine finding

Incremental Capacity Analysis on NASA PCoE cells. Dominant-peak tracking is stable
(0 jumps >50 mV for B0005/B0007/B0018). slope_m Δμ tracks LLI across these three
cells; quantitative results in `data/nasa_ica_report.json`.

**K-instability clarification:** Total K (peak count) varies 3–12 across cycles.
This reflects noise peaks >150 mV from the dominant peak, not tracking failures.
Dominant-peak tracking is the meaningful metric; total-K variation is noise.
Source: commit `c224fe2`.

### 3.2 B0006 exclusion

B0006 is qualitative only: K unstable (2–11), dominant peak identity ambiguous
across lifetime. This exclusion was established in commit `0dbbe37`, which
predates all cross-module synthesis. It is not a post-hoc exclusion.

### 3.3 B0007 structural collinearity

LLI and LAM are structurally collinear for B0007 (Spearman r=−0.927). This is
because both vary monotonically with cycle number on the identical protocol —
co-monotonicity, not identity swap. The original wording implied identity
swap; corrected in `c224fe2`. See Appendix Entry 1.

---

## 4. Cross-Module Synthesis — Honest Underpowered Null

**Question:** Does β (Module 2 stress-fatigue rate) correlate with slope_m (LAM
proxy) or ΔRe+ΔRct (CL resistance rise) from Module 3?

**Check A** (slope_m vs β, n=3): Spearman rs=−1.000, p=0.333.
Pre-registered category: PERFECT DISAGREEMENT — but slope_m range across 3 cells
is 0.049 (~2.6% of baseline). The "perfect disagreement" is a rank ordering of
near-identical values. Not a meaningful correlation in either direction.

**Check B** (ΔRe+ΔRct vs β, n=4): Spearman rs=−0.400, p=0.600.
Pre-registered category: WEAK/MIXED.

**Power:** Minimum achievable p at n=3 is 1/6≈0.167; result cannot be statistically
significant. This is a plausibility check, not a test.

**Conclusion:** No coupling found. Pre-registered null categories applied verbatim.
Source: `data/nasa_ica_report.json`, `cross_module_synthesis`. Commit `c72914b`.

---

## 5. Prototype Layers

### 5a. GNN (GraphSAGE) — Architecture verified; accuracy not testable

**Architecture checks:** 7/7 PASS — output shape (20,4), [0,1] bounds (Sigmoid head),
KCL residual finite/≥0, adjacency partitioning, p99 latency 309 µs < 500 µs target,
NumPy interface. Source: `data/prototype_layer_validation.json`, `gnn_graphsage`.

**Near-zero inter-node std on real Quartz inputs:** Mean std=0.000819 across 20 output
nodes. Originally framed as "BatchNorm collapse (architectural concern)." Retracted
and re-attributed: BN-disabled control produces std=0.000819 — identical. Layer-1
BN probe: pre-BN std=0.003359, post-BN std=0.003359 — BatchNorm acts as identity at
initialization in eval mode (running_mean=0, running_var=1). The near-zero std is a
random-weight artifact on near-uniform inputs (V spread 37 mV = 0.025 normalized
units): ||W·(x_i−x_j)|| ≤ ||W||·||x_i−x_j|| is tiny when ||x_i−x_j|| is tiny. See
Appendix Entry 3. Whether trained weights would amplify small inter-cell differences
cannot be assessed without training data.

**Accuracy validation status:** Not possible. No per-cell SOC/SOH ground truth exists
in the Quartz dataset. N_NODES=20 hardcoded; Quartz topology has 36 cells —
architecture change required before deployment. This is a data limitation, not a
model limitation. Source: `data/prototype_layer_validation.json`, `gnn_graphsage.data_limitation`.

### 5b. NSA Weakest-Cell Detector — Null Type A

**Original failure:** 6D hand-crafted fault centroids. Minimum distance from any
real Quartz observation to any centroid = 0.70; matching radius r=0.25. Gap factor
2.8×. All 36 cells scored 0.0000 in both Test A (full dataset) and Test B (SOC<5%,
15,292 rows). Source: commit `60f443b`.

**Root cause:** 4/6 features (SOC, SOH, SEI_norm, plating_risk) are imputed
constants identical across all cells at any timestep. The 6D self-cloud is a 2D
slice embedded at [SOC(t), 1.0, *, 0.05, *, 0.01]. Hand-crafted centroids placed
in the remaining 6D space (SOH=0.60–0.85, SEI_norm=2.5–3.0) are structurally
unreachable by any real observation from any cell, healthy or not.

**Recalibration:** Rewrote `diagnosis/weakest_cell.py` as genuine 2D NSA operating
on V_norm × T_norm only. Genuine negative-selection training: data-derived self-set
(100,940 per-timestep observations, every 220th row × 35 cells); r=0.4×p95=0.124
(single pre-chosen multiplier, not tuned); 45,386/50,000 candidate detectors survive
self-deletion. All 20 `validate()` tests PASS. See `diagnosis/weakest_cell.py` module
docstring for 6D expansion path.

**Recalibration outcome — Null Type A confirmed:**
- PRIMARY (mean activation fraction): exactly 0.000000 for all 35 cells.
- SECONDARY (max activation fraction): std=0.000071 < 0.001 threshold; trace
  non-zero for three thermal-fringe cells (P3S4=0.000242, P3S9=0.000110,
  P1S1=0.000022), single-timestep activations at the T_norm boundary.
- Root cause: 97.7% of surviving detectors lie outside the Quartz WLTP operating
  envelope entirely (T_norm>0.466 i.e.>48°C; Quartz max 40°C). The remaining 2.3%
  sit at exactly r beyond the envelope boundary — unreachable by construction. The
  cycling amplitude (V_norm std=0.188) dominates inter-cell V differences (~0.004)
  by 47×.
- LOCO diagnostic (Null Type C test): at r=0.124 no single cell defines any part of
  the self-boundary. Removing 2,884 readings per cell (2.9%) produces zero change in
  detector count or scores. Null Type C ruled out by structural entailment, not
  empirical power.

**Permanent 2D scope limitation:** The recalibrated NSA detects anomalies in V_norm
× T_norm only. No SOH, SEI, or plating sensitivity is possible without real per-cell
sensors. This limitation is a hardware/data constraint — it does not disappear with
further algorithmic refinement.

**Pre-registration deviation documented:** Commit `1eb8ff2` used 35 per-cell means
as self-set instead of per-timestep readings. Corrected in `3ddc594`. Final null
verdict unchanged. See Appendix Entry 4.

**Recommended replacement for this use case:** Pack-centroid relative outlier score
(Mahalanobis distance in V_norm × T_norm). See Section 6.1 for the validation
constraint that must be stated before building it.

Source: `data/prototype_layer_validation.json`, `nsa_weakest_cell`.

### 5c. ACO + Kuramoto — Functional, Two Calibration Concerns

**Functional checks:** 4/4 PASS. KCL conservation: max violation 4.44×10⁻¹⁶ A
(machine epsilon). Monotone preference: 10/10 scenarios. Kuramoto order parameter:
R=0.739→0.995 on real-parameter SOC imbalance. TR risk: 0.0 for all Quartz
temperatures (max 42°C, margin 38°C to 80°C onset); correct boundary behaviour at
80°C, 85°C, 150°C. Source: `data/prototype_layer_validation.json`, `aco_kuramoto`.

**Calibration concern 1 — ACO 0A starvation (8/10 scenarios):** The heuristic
η = SOH/(risk+ε) differential between the best and worst cells (19 vs 1.875) is so
large that all 20 ants route to the best cell, starving the worst cell to 0A. This is
full current concentration, not modulation. Real packs require a minimum current floor
before deployment. Source: `aco_kuramoto.check_2_monotone.calibration_concern`.

**Calibration concern 2 — Kuramoto delta_soc unrealizable (±13–17% per call):**
Mean absolute delta_soc=9.9% per call. Physically realizable passive or active cell
balancers achieve <1% SOC correction per cycle. The Kuramoto output must be treated
as a synchronization target direction, not a directly actionable setpoint.
Source: `aco_kuramoto.check_3_kuramoto_convergence.calibration_concern`.

---

## 6. Future Work

### 6.1 NSA replacement: pack-centroid relative outlier score

Mahalanobis distance from the pack mean in the (V_norm, T_norm) plane is the
structurally appropriate tool for relative health ranking within a healthy homogeneous
pack — it directly measures how far each cell's operating trajectory deviates from
the pack mean, which is what NSA was trying to approximate via detectors.

**Validation constraint stated now, before building:** Must NOT validate against
Criterion A (mean voltage deficit from pack mean). The two quantities are
near-tautologically equivalent: the cell with the highest mean voltage deficit is by
construction the cell furthest below the pack mean in V_norm, which is exactly what
a V_norm Mahalanobis score measures. rs≈1 between them would be circular, not a
finding. Honest validation targets:

- **Criterion B** (minimum instantaneous voltage at SOC<5%): rs=−0.01 vs Criterion A
  (Spearman, n=35, p=0.952) — statistically independent, measures capacity exhaustion
  not chronic resistance bias. A genuine test.
- **A held-out property** not derived from mean voltage (e.g. capacity fade estimated
  from a separate reference performance test, or a time-to-low-voltage metric).

### 6.2 GNN accuracy validation

Requires either (a) a dataset with per-cell SOC/SOH ground truth (reference performance
tests or high-precision lab equipment per cell), or (b) synthetic training data
generated from the DFN cell model with per-cell parameter variation. The N_NODES=20
hardcode must also be resolved for the Quartz 36-cell topology.

### 6.3 ACO minimum current floor + Kuramoto setpoint scaling

Before any real deployment: ACO needs a parameter specifying minimum current fraction
per string to prevent 0A starvation; Kuramoto delta_soc needs a saturation cap
(≤1% per call) matching physical balancer rates. Both are one-parameter fixes, not
architectural changes.

---

## Appendix — Deviations, Retractions, Figure Corrections

All items in chronological order of correction commit. The Section 0 contract
requires every entry to appear here regardless of whether the final conclusion changes.

| # | Type | Original claim | Correction | Commit |
|---|---|---|---|---|
| 1 | Retraction | "LLI and LAM are collinear for B0007 — identity swap possible" | Structural collinearity from co-monotonicity with cycle number on identical protocol; not an identity swap | `c224fe2` |
| 2 | Retraction | "K=5 stable" for all four NASA cells | Two-part characterization: dominant-peak tracking stable (0 jumps >50 mV); total K varies 3–12 from noise peaks >150 mV from dominant | `c224fe2` |
| 3 | Retraction | GNN near-zero inter-node std framed as "BatchNorm collapse (architectural concern)" | Random-weight artifact confirmed by BN-disabled control: std=0.000819 with and without BN; BatchNorm acts as identity at init in eval mode | `3ced657` |
| 4 | Deviation + correction | NSA self-set in commit `1eb8ff2` used 35 per-cell means, not per-timestep readings as pre-registered | Pre-registration specified "real healthy readings sampled uniformly across all cells AND TIMESTEPS"; rerun with 100,940-point timestep self-set in `3ddc594`; null verdict unchanged (Null Type A) | `1eb8ff2` → `3ddc594` |
| 5 | Retraction | "Per-cell-means LOCO ruled out Null Type C definitively" | That LOCO removed 1/35 mean points — no statistical power. Replaced by structural entailment argument: at r=0.124, no single cell defines any part of the self-boundary; LOCO re-run with 2,884 points confirms but is not the operative argument | `b8b44e5` |
| 6 | Framing fix | NSA LOCO zero-delta presented as empirical evidence against Type C | Replaced with structural argument: geometry prevents any individual cell from influencing the self-boundary at r=0.124 | `b8b44e5` |
| 7 | Figure correction | R²=0.9810 headlined as the core BMS accuracy figure (commit `b45ce3f`) | 0.9810 computed on all rows including 83% step-constant repeated BMS readings; 0.9217 on sensor-update rows (genuinely new measurements, n=12,690) is the honest headline. Both figures are in `data/stack_validation_report.json`. Badge and architecture doc corrected in `cde6a74` and `8409f8d`. "5–8% SOC RMSE" figure also replaced in `8409f8d` as unsourced; actual recorded fleet figures are BMW 20.8%, Deng 11.9%, VED 20.2–25.5% SOC RMSE. |
| 8 | NSA "working" check | — | Check performed: no commit or recorded document described NSA as "working" or "validated" prior to the degenerate-null discovery in `60f443b`. README marked NSA as `🔬 prototype — not validated` from the first commit that included it. Check came back clean; no entry required. |

---

*All source files: `data/stack_validation_report.json`, `data/nasa_ica_report.json`,
`data/prototype_layer_validation.json`. All commit hashes reference this repository's
`main` branch.*
