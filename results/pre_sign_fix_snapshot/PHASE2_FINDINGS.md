# PHASE 2 FINDINGS — offline structured corrections vs online scalar bias

Status: complete (wide-grid θ re-tune of 2026-07-19/20). Every number from
actual runs; my-EKF/coulomb/OCV rows cross-checked byte-identical against
the committed benchmark reports. Written to drop into the paper as a
subsection: **prediction first, fleet evidence second, implications third.**

⚠️ **Scope caveat discovered during this phase (§5): a current-sign
convention inconsistency between the fleet loaders (discharge-negative
schema) and the CALCE/UMich benchmark loaders (discharge-positive).** As
the estimator chains consume signed current, this means: on BMW/Deng/VED
the EKF-family results are valid but the coulomb baseline is sign-inverted;
on CALCE/UMich the coulomb baseline is valid but every EKF-family result
(mine and the scalar-bias variants, plus the δV/δR0 calibrations) ran with
an inverted process model. Findings below are stated per-dataset with this
validity map applied. No code has been changed pending review sign-off.

---

## 1. Prediction (stated before the fleet evidence was in)

A single online scalar voltage-bias state θ (RBC-DEKF, Guo et al.,
arXiv:2510.22813) is identifiable only when SOC and θ offer *separable*
explanations of the voltage residual. On a flat OCV plateau (LFP), SOC
cannot move the predicted voltage, so the residual belongs to θ — the
RBC-DEKF operating regime. On a steep OCV (NMC region), a constant bias
and a SOC offset are near-indistinguishable; whichever state carries more
covariance wins the residual, so an online bias filter must either be
tuned toward OFF or it will trade accuracy with the SOC state. First
observed while building the known-answer unit tests
(`diagnosis/scalar_bias_dekf.py:validate()`), i.e. before any fleet run.

## 2. Fleet evidence

### 2.1 The tuner's own choices are the first result

Narrow grid (Q_θ ∈ [1e-10, 1e-6] V²/s, R_θ ∈ [1e-6, 1e-4] V²) put 4 of 5
fleets on grid edges. Per the 2026-07-19 review the grid was widened
adaptively (start Q_θ 1e-14…1e-2 × R_θ 1e-10…1e-2; every selected edge
extended 2 decades repeatedly; hard bounds [1e-18, 1]). Selections
(calibration splits only, +20 pp, logged in
`results/theta_tuning_wide_*.json`):

| Fleet | narrow-grid pick | wide-grid pick | cal RMSE narrow→wide | reading |
|---|---|---|---|---|
| BMW i3 | 1e-10, 1e-4 | **1e-15, 1e0** | 16.35 → 16.35 | θ **OFF** (flat plateau of equivalent inert configs) |
| Deng | 1e-6, 1e-6 | **1e-18, 1e0** | 14.16 → **9.33** | θ **OFF** — narrow grid had genuinely mis-tuned it |
| VED | 1e-6, 1e-4 | unchanged (interior) | 16.24 | the only genuinely *adaptive* θ |
| CALCE | 1e-6, 1e-4 | **1e0, 1e-3** | 20.72 → 19.00 | θ **instant-absorb** extreme |
| UMich | 1e-6, 1e-6 | **1e-18, 1e0** | 14.15 → 13.56 | θ OFF |

Grid-edge caveat resolution: the wide-grid selections still sit on bounds,
but these are *semantic endpoints* — Q_θ=1e-18/R_θ=1 makes K_θ ≈ 0 (bias
filter disabled) and Q_θ=1 makes K_θ ≈ 1 (absorb the whole residual every
step); further decades cannot change the filter's behavior, so
"improvement stops" is satisfied. The BMW frozen-bias observation from the
narrow grid **survives and sharpens**: given full freedom, the tuner turns
the bias mechanism off entirely on all three discharge-negative-schema
fleets except VED.

### 2.2 Identifiability check (`results/identifiability_check.csv`)

θ_end vs −slope·err_end correlation (the "θ absorbs SOC error" signature),
rbc_dekf @ +20 pp, final tuned params:

| Fleet | mean \|∂OCV/∂SOC\| (V) | Q_θ picked | median \|θ_end\| (mV) | corr(θ, −slope·err) | validity |
|---|---|---|---|---|---|
| BMW i3 | 0.24 | 1e-15 (off) | 0.0008 | 0.18 (θ degenerate) | valid |
| Deng | 0.65 | 1e-18 (off) | 0.40 | 0.01 | valid |
| VED | 0.29 | 1e-6 (active) | 10.0 | −0.25 | valid |
| CALCE | 0.86* | 1e0 (absorb-all) | 124 | 0.47 | ⚠ inverted-process run |
| UMich | 1.62* | 1e-18 (off) | 0.18 | 0.36 | ⚠ inverted-process run |

*fleet-fitted OCV slope, which on CALCE/UMich was itself fitted under the
sign inversion — treat as unreliable.

On the three valid fleets the prediction holds in the actionable sense:
**wherever the OCV channel is informative, the honest-best tuning of the
online bias is OFF** — Deng, the steepest valid fleet (0.65 V/SOC), is the
clearest case. VED keeps an active θ not because its OCV is flat but
because its process model carries a large genuine bias for θ to absorb
(assumed pack capacity ≈ 2× the implied one — see VED_BREAKDOWN.md), i.e.
θ works there as a *model-error* compensator, not a sensor-bias estimator.
A clean flat-plateau (LFP) test point is NOT currently available: CALCE's
run is invalidated by the sign inversion (§5).

### 2.3 Where scalar bias wins/loses vs my offline corrections (+20 pp, valid fleets)

Median full-trip RMSE (pp), `results/baseline_comparison.csv`:

| Fleet | my EKF (offline δV/δR0) | rbc_dekf (online θ) | verdict |
|---|---|---|---|
| BMW i3 | **17.61** | 19.39 | offline wins (+1.8) |
| Deng | 9.28 | **7.66** | online wins (+1.6) — n.b. with θ≈OFF, so this is really "no corrections beat my offline corrections on Deng" |
| VED | 23.98 | **21.00** | online wins (+3.0) — θ compensating the capacity-driven model bias |

The honest reading: on Deng and VED the **offline fleet-level corrections
actively hurt**, and the rbc variant's advantage comes less from online
bias estimation than from *not carrying* those corrections (Deng: θ off;
VED: θ absorbing model error the offline recipe baked in wrongly).
BMW is the one valid fleet where the offline structured corrections beat
every alternative tried.

### 2.4 Coupled vs decoupled (the paper's central architectural claim)

+20 pp, `results/baseline_comparison.csv`:

| Fleet | decoupled rbc_dekf | coupled rbc_coupled | effect |
|---|---|---|---|
| BMW i3 | 19.39 | 16.89 | neutral (θ inert in both; differences within trip-to-trip spread) |
| Deng | **7.66** | 16.03 | **coupling degrades 2.1×** — strict convergence collapses 80.6% → 32.5% |
| VED | 21.00 | 20.35 | neutral |
| CALCE / UMich | — | — | invalid pending sign fix |

Deng is the decisive case: with identical (inert) bias parameters, the
DECOUPLED architecture is harmless, while AUGMENTING the same state into
one joint filter lets the initial bias covariance (P_θ0=(50 mV)²) couple
into SOC through the shared gain and halves the convergence rate. This is
the empirical, fleet-scale counterpart of the project's Round-2
∂δV/∂SOC failure (0/63 BMW convergence) and of Guo et al.'s theoretical
decoupling argument.

## 3. Implications for positioning against Guo et al. (RBC-DEKF)

1. **Their decoupling argument is confirmed at fleet scale** (Deng, 2000
   sessions: 2.1× degradation when coupled) — evidence they did not have
   (one lab cell).
2. **Their bias-state mechanism does not transfer to steep-OCV fleet
   data**: honestly tuned, it switches itself off (BMW, Deng, UMich) or
   turns into a model-error sponge (VED). RBC-DEKF should be described in
   the paper as an LFP-plateau technique whose operating assumption —
   separability of bias from SOC in the voltage channel — fails on NMC
   fleets; we could not obtain a valid LFP fleet test (CALCE sign issue)
   and must say so.
3. **Offline vs online is not a winner-take-all**: offline structured
   corrections won only on the fleet whose calibration split matches its
   evaluation conditions well (BMW). Where the fleet-level calibration is
   itself misfit (VED) or unnecessary (Deng), corrections should be
   omitted or estimated per-vehicle (VED per-vehicle probe:
   VehId_0010 median RMSE 19.8 → 7.7 pp; see VED_BREAKDOWN.md).

## 4. Tuning-hygiene notes

- Grids, per-combo cal RMSE, extension rounds, and bound hits are logged in
  `results/theta_tuning_wide_*.json`; selection used calibration splits
  only, tie-breaks prefer smaller Q_θ then smaller R_θ.
- The +20 pp protocol was used for tuning (matches the headline protocol);
  P0_soc = max(offset², (2 pp)²), initial SOC clipped to [0.02, 0.98].

## 5. The current-sign discovery (must be resolved before Phase 4+)

Empirical check (`sign(∫I dt)` vs `sign(ΔSOC_true)` per segment,
2026-07-20): BMW 0/65, Deng 0/8368, VED 0/154 segments consistent with
discharge-positive → those schemas are **discharge-negative** (as
`common_schema.enforce_discharge_negative` intends); CALCE 40/40 and
UMich 221/221 → **discharge-positive** (their standalone benchmark loaders
flipped raw current the wrong way relative to the project schema, while
documenting the flip as deliberate).

Consequences, mechanically traced:

- `coulomb_counting_soc` integrates assuming discharge-positive → its
  BMW/Deng/VED columns are **sign-inverted** (Deng's are the smoking gun:
  at offset 0 a correct integrator cannot produce 55.7 pp median RMSE over
  a charging session; CALCE's 0.10 pp at offset 0 shows the same code is
  correct where the schema is discharge-positive).
- `run_mode_b_ekf`/`run_lean_traj` negate schema current before the EKF →
  EKF-family results are **valid on BMW/Deng/VED, inverted on
  CALCE/UMich**. This also explains the long-standing "non-physical"
  δR0 fits that trip the EKF's sanity gate (CALCE −260 mΩ, UMich −233 mΩ):
  Mode A fed `−I` to the DFN, so the fitted current-proportional residual
  slope has the wrong sign and roughly twice the magnitude it should.
- BMW magnitude sanity: implied |∫I dt/ΔSOC| = 61.7 Ah vs the 60 Ah
  cartridge — magnitudes are fine everywhere; only signs and the VED
  capacity (§VED_BREAKDOWN) are wrong.

**No fix has been applied** (project rule: describe, show evidence, ask).
The proposed fix — align CALCE/UMich loaders to the schema convention and
correct `coulomb_counting_soc`'s integral sign, then regenerate
everything including the Phase 0 reproduction — changes headline numbers
in both directions and therefore needs explicit sign-off.
