# PHASE 2 FINDINGS — offline structured corrections vs online scalar bias

Status: FINAL on sign-corrected data (2026-07-20; the pre-fix version of
this file is preserved at `results/pre_sign_fix_snapshot/PHASE2_FINDINGS.md`
and superseded — see `docs/SIGN_BUG_POSTMORTEM.md`). Every number from
actual runs; my-EKF/coulomb/OCV cells cross-checked byte-identical against
the regenerated committed reports (max |Δ| = 0.0 on all five datasets).
Structured for the paper: **prediction first, fleet evidence second,
implications third.**

Standing table notes: calibration splits 10% (fleets) / 40% (CALCE) /
30% (UMich); +20 pp wrong-init protocol unless stated; initial-SOC clipping
to [2%, 98%] materially flatters coulomb AGGREGATES in 23/45 sweep cells
(medians far less affected) — `results/coulomb_clipping_diagnostic.csv`.

---

## 1. Prediction (stated before the fleet evidence was in)

A single online scalar voltage-bias state θ (RBC-DEKF, Guo et al.,
arXiv:2510.22813) is identifiable only when SOC and θ offer *separable*
explanations of the voltage residual. On a flat OCV plateau (LFP), SOC
cannot move predicted voltage, so the residual belongs to θ — RBC-DEKF's
operating regime. On a steep OCV (NMC/NCA), a constant bias and a SOC
offset are near-indistinguishable; an honestly tuned online bias filter
must either switch itself off or trade accuracy with the SOC state. First
observed while building the known-answer unit tests
(`diagnosis/scalar_bias_dekf.py:validate()`), before any fleet run.

Corollary (sharpened during analysis): the *harm of coupling* the bias
into one joint filter should also scale with OCV slope — the shared-gain
leak from θ into SOC is proportional to ∂OCV/∂SOC.

## 2. Fleet evidence (sign-corrected data)

### 2.1 The tuner's own choices are the first result

Wide ADAPTIVE grid (Q_θ from 1e-14…1e-2 V²/s, R_θ 1e-10…1e-2 V², every
selected edge extended 2 decades repeatedly, hard bounds [1e-18, 1];
calibration splits only, +20 pp; log:
`results/theta_tuning_wide_20260719T193500Z.json`):

| Fleet | chemistry / regime | picked (Q_θ, R_θ) | reading |
|---|---|---|---|
| BMW i3 | NMC fleet | 1e-15, 1e0 (bound) | θ **OFF** (K_θ→0 endpoint; flat plateau of equivalent inert configs) |
| Deng | NMC fleet, steepest valid OCV (0.65 V/SOC) | 1e-18, 1e0 (bound) | θ **OFF** — turning it off improved cal RMSE 14.16→9.33 |
| VED | NMC fleet, known ~1.9× capacity model error | 1e-6, 1e-4 (interior) | θ active as a **model-error sponge** |
| CALCE | **LFP lab** | 1e-3, 1e-7 (**interior**) | θ **most aggressive of all fleets** — genuine bias compensation |
| UMich | NCA lab, steep (1.62 V/SOC) | 1e-10, 1e-9 (interior) | θ mildly active |

Grid-edge caveat resolution: the only remaining bound hits (BMW, Deng) are
semantic OFF endpoints — further decades cannot change K_θ ≈ 0, so
"improvement stops" is satisfied. The BMW frozen-bias observation survives
every widening. Post-sign-fix, both lab datasets moved to interior optima.

### 2.2 Identifiability signatures (`results/identifiability_check.csv`)

θ_end vs −slope·err_end correlation (the "θ absorbs SOC error" signature),
rbc_dekf @ +20 pp, final tuned params, all five datasets now valid:

| Fleet | mean \|∂OCV/∂SOC\| (V) | median \|θ_end\| (mV) | corr(θ, −slope·err) | reading |
|---|---|---|---|---|
| BMW i3 | 0.24 | 0.001 | 0.18 (θ degenerate) | θ off — nothing to absorb with |
| Deng | 0.65 | 0.40 | 0.01 | θ off |
| VED | 0.29 | 10.0 | −0.25 | θ tracking model error, not SOC error |
| CALCE | 0.86¹ | **164** | 0.31 | large genuine bias compensation |
| UMich | 1.62 | 44.5 | **0.69** | textbook ambiguity: θ absorbing SOC error on the steepest OCV |

¹CALCE's fleet-fitted OCV mean slope is inflated by the steep LFP tails;
the cell spends most operating time in the flat 30–75% plateau.

**Does the chemistry-dependence prediction get its clean CALCE test now?
Yes, qualified.** On the corrected LFP dataset the tuner drives θ to its
most aggressive setting, θ carries a real ~160 mV correction, and the
scalar-bias variant is the best EKF-family method there (§2.3) — exactly
RBC-DEKF's design regime. On the steepest chemistry (UMich NCA) the active
θ instead correlates 0.69 with −slope·err — it is eating SOC error, the
predicted failure mode. The qualification: CALCE evidence rests on n=26
windows from 2 cells, and my EKF's CALCE run remains handicapped by a
still-non-physical δR0 (+122 mΩ, §4).

### 2.3 Offline corrections vs online bias, +20 pp (median RMSE, pp)

| Fleet | my EKF (offline δV/δR0) | rbc_dekf (online θ) | verdict |
|---|---|---|---|
| BMW i3 | **17.61** | 19.39 | offline wins (+1.8) |
| Deng | 9.28 | **7.66** | "no corrections" wins (θ off) |
| VED | 23.98 | **21.00** | online wins (θ compensating capacity error) |
| CALCE | 31.70 | **26.80** | online wins on LFP (its design regime) |
| UMich | 16.89 | **16.57** | statistical tie |

Honest reading: offline structured corrections win **only on BMW** — the
fleet whose calibration split matches evaluation conditions well. On Deng
they *cost* 1.6 pp vs carrying no correction at all; on VED they bake in a
capacity-driven bias that the online θ partially undoes; on CALCE the LFP
plateau plus the broken δR0 fit make the online bias clearly better.

### 2.4 Coupled vs decoupled — harm scales with OCV slope

| Fleet | slope (V/SOC) | decoupled | coupled | effect |
|---|---|---|---|---|
| Deng | 0.65 | **7.66** | 16.03 | **2.1× degradation; strict conv 80.6→32.5%** |
| UMich | 1.62 | 16.57 | 17.30 | mild degradation |
| BMW | 0.24 | 19.39 | 16.89 | neutral (θ inert; means 18.58 vs 18.49) |
| VED | 0.29 | 21.00 | 20.35 | neutral |
| CALCE | plateau | 26.80 | **21.94** | coupling *helps* on the flat plateau |

This is the corollary of §1 confirmed: the joint filter's θ→SOC leak is
gain-weighted by ∂OCV/∂SOC. Where the OCV is steep and informative (Deng),
coupling is catastrophic at fleet scale — the empirical, 2000-session
counterpart of the project's Round-2 ∂δV/∂SOC failure (0/63 BMW trips) and
of Guo et al.'s theoretical argument. Where the OCV is flat (CALCE
plateau), H ≈ [0, 1, 1] and the coupled filter degenerates gracefully
toward the decoupled one (and here even benefits, since the bias update
uses the jointly-updated covariance).

## 3. Implications for positioning against Guo et al. (RBC-DEKF)

1. **Their decoupling argument is confirmed at fleet scale** on steep-OCV
   data (Deng: 2.1× RMSE degradation, convergence rate halved, n=2000) —
   evidence unavailable from their single lab cell. Refinement we can
   state: the coupling penalty is slope-dependent, vanishing on the LFP
   plateau where their method was validated.
2. **Their bias mechanism is chemistry-scoped, and now we can show both
   sides**: on LFP lab data it is the best EKF-family variant (26.80 vs
   31.70); on NMC/NCA data an honest tuner either disables it (BMW, Deng)
   or it absorbs SOC error (UMich, corr 0.69). RBC-DEKF generalizes to
   fleets only where the plateau separability holds.
3. **Offline vs online is not winner-take-all**: fleet-level offline
   corrections win once (BMW), lose to *nothing at all* once (Deng), and
   lose to online θ where the underlying model is biased (VED) or the
   chemistry is flat (CALCE). Per-vehicle offline calibration (VED
   VehId_0010: 19.8 → 7.7 pp median) suggests the right unit of offline
   calibration is the vehicle, not the fleet — a paper-worthy hybrid
   direction that matches Guo et al.'s own future-work framing.

## 4. Open issues carried to Phase 4

- CALCE/UMich δR0 re-fits are sign-plausible but magnitude-non-physical
  (+121.6 / +219.5 mΩ vs ≈8 mΩ cells); the EKF's calibration sanity gate
  still fires on both. The lab-dataset Mode-A residuals have a genuine
  current-proportional structure the ECM/DFN does not capture.
- VED: generic-cartridge capacity ~1.9× too large (implied ≈42 Ah);
  VehId_0455's ~26 pp floor unexplained (docs/VED_BREAKDOWN.md).
- Coulomb stress-test aggregates carry the clipping artifact (23/45 cells);
  use medians or unclipped subsets for any paper claim.

## 5. Tuning hygiene

Grids/rounds/bound-hits logged per fleet; selection on calibration splits
only; ties prefer smaller Q_θ then R_θ; +20 pp tuning protocol; P0_soc =
max(offset², (2 pp)²); init clipped to [0.02, 0.98]. Scalar-bias variants
use Joseph-form updates per the RBC-DEKF spec (review decision 2026-07-19:
Joseph form lives only in this baseline; a Joseph row for my EKF is
deferred to the Phase 5 ablation).
