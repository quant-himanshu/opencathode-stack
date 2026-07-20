# OpenCATHODE SOC/SOH Estimation — Final Project Report (final-v1)

2026-07-20. This is my complete record of the dual-EKF field study:
the system as built, the final sign-corrected results, the two protocol
defects I found and disclosed (a current-sign inconsistency and an
init-clipping artifact), the failure analyses, the ablations, the
bias-identifiability result, and my honest conclusions about where this
approach works and where it does not. Every number in this report comes
from an actual run of the code in this repository on the actual data, and
every headline pipeline output is protected by abort-on-mismatch
cross-checks and a permanent sign assertion. No result here is estimated,
interpolated, or carried over from a run that no longer reproduces.

---

## 1. The system as I built it

**Fast loop.** A first-order-RC Thevenin EKF over x₁ = [SOC, V_pol] in the
style of Plett's dual-estimation series (Plett 2004, J. Power Sources
134(2), Parts 1–3; Kalman 1960 for the filter itself), implemented in
`diagnosis/dual_ekf_lfp.py`. Voltage model:
V = OCV(SOC) − I·R + V_pol + δV(SOC) + δR0·I.

**The decoupling decision.** The measurement Jacobian is H = [∂OCV/∂SOC, 1]
— the ∂δV/∂SOC term is deliberately excluded. Including it (my Round-2
configuration) destroyed convergence; the final ablation on sign-corrected
data quantifies that choice at 2.5× median RMSE on BMW (17.6 → 43.3 pp)
and 2.0× on VED (24.0 → 47.6 pp). This decoupling is the load-bearing
design decision of the project, and it is the empirical fleet-scale
counterpart of the argument Guo et al. make theoretically for their
residual-bias filter (RBC-DEKF, arXiv:2510.22813).

**Offline corrections.** δV(SOC) is a PCHIP spline (Fritsch & Carlson
1980) over 12 SOC-binned median residuals; δR0 an OLS slope of residual
vs current; both fit on a held-out calibration split only (10% per
vehicle on the fleets; 40%/30% on CALCE/UMich for data-size reasons,
decided before evaluation). OCV curves are fit per fleet from near-rest
samples, falling back to generic tables only if under-populated (the
fallback never fired in the final runs).

**Adaptive process noise.** Q = Q_base · min(1/max(|∂OCV/∂SOC|, 0.02), 50)
· γ, the flat-plateau inflation idea from the LFP literature (e.g. Mikhak
2024); γ tuned per fleet on the calibration split. The built-in fallback
LFP OCV table is Prada 2012.

**Slow loop.** x₂ = [SOH, R_int]: R_int by a separate scalar Kalman update
on the IR term, gated off when the offline δR0 is non-physical
(|δR0| ≥ 50 mΩ — the "calibration sanity gate"); SOH by accumulated
Ah / (ΔSOC · Q_nom) once ≥5 pp of real swing has accumulated,
confidence-weighted. Trips are independent — no cross-trip state.

**Datasets.** Five public sources: BMW i3 field trips (63 held-out; TUM
RDC data, Lüth 2020 / IEEE DataPort), Deng BAIC EU500 fleet charging
sessions (2000-session seeded sample), VED Ann Arbor mixed fleet (408
BEV trips; Oh et al. 2020), CALCE A123 LFP cells under DST/US06/FUDS
(calce.umd.edu/battery-data), and UMich/Ford 3-parallel-cell modules
(Mendeley DOI 10.17632/ssrgfmb8vw.2, arXiv:2604.16769).

**Protocol.** Every trip starts the estimator at SOC_true + 20 pp
(clipped to [2%, 98%]), plus a full sweep over ±30…0 pp. Baselines —
pure coulomb counting and naive OCV inversion — face the identical wrong
init. Metrics are standardized in `validation/metrics.py`: median (IQR)
of per-trip full-window RMSE is primary, mean secondary; convergence is
STRICT (below 5 pp to end of trip), with HOLD-600 and the legacy
30-sample rule reported alongside; outcomes are three-tier
(converged / recovered ≤10 pp at end / diverged).

## 2. Final headline results (sign-corrected, +20 pp, median RMSE in pp)

| Dataset | n | Dual EKF | Coulomb | OCV lookup | my verdict |
|---|---|---|---|---|---|
| BMW i3 | 63 | **17.61** | 19.66¹ | 38.43 | EKF wins on the median; coulomb's mean "win" (16.88) is the clipping artifact¹ |
| Deng BAIC EU500 | 2000 | **9.28** | 14.83 | 6.35 | EKF beats coulomb clearly; naive OCV inversion beats both on 8-s charging data |
| VED | 408 | 23.98 | 19.75¹ | **11.40** | EKF loses — diagnosed as a ~2× cartridge-capacity error + harmful fleet-level δV (see §6) |
| CALCE A123 | 26 | 31.70 | **19.89** | 34.68 | EKF loses — ECM cannot model 1-s dynamic LFP profiles (147 mV unexplained variance) |
| UMich module | 216 | 16.89 | 17.75 | 16.32 | statistical tie |

¹Init-clipping caveat, §4. Strict convergence rates (EKF): BMW 7.9%, Deng
83.9%, VED 16.2%, CALCE 26.9%, UMich 26.9%.

Under **nominal conditions** (offset 0, `results/nominal_accuracy.md`),
correctly-signed coulomb counting wins everywhere except Deng (BMW 0.23,
VED 1.43, CALCE 0.10, UMich 2.53 pp median) — the voltage-based estimators
pay a 3–29 pp model-error floor. The offset sweep
(`figures/offset_sweep.pdf`) shows the honest trade: coulomb's error is
proportional to the initial offset; the EKF's is nearly flat in it. What a
voltage EKF buys on field data is **initialization robustness**, not
unconditional accuracy — it pays off where the OCV channel is informative
(Deng) or when the initial state is untrusted.

## 3. The sign bug — found, fixed, documented (`docs/SIGN_BUG_POSTMORTEM.md`)

While chasing a VED anomaly I computed the implied pack capacity
∫I dt / ΔSOC and got a *negative* number — which unravelled a cross-loader
current-convention inconsistency: the schema is discharge-negative, but
the CALCE/UMich benchmark loaders had flipped raw current the wrong way,
and the coulomb baseline integrated with the opposite sign. Net effect:
every pre-fix EKF-vs-coulomb comparison had one side sign-inverted (the
coulomb column on the three fleets; the EKF family and its calibrations on
the two lab datasets). Empirical audit: BMW 0/65, Deng 0/8368, VED 0/154
segments consistent with discharge-positive vs CALCE 40/40, UMich 221/221.

I fixed the two loaders and the baseline (no estimator changes), added a
**permanent per-segment sign assertion** at the single choke point every
dataset load passes through (`common_schema.make_schema_df`), regenerated
every phase, and verified ten per-segment invariants at exactly 0.0 (the
cells whose code paths were untouched reproduce bit-for-bit; only the
inverted cells moved — 89 changed numbers, all in
`results/sign_fix_before_after.md`). The bug also explained the notorious
"non-physical δR0" values: CALCE −260.5 → +121.6 mΩ, UMich −233.3 →
+219.5 mΩ after the fix — sign healed, magnitude still non-physical, so
the sanity gate still fires and a genuine lab-dataset model mismatch
remains. The detection method (implied-capacity probe + per-segment sign
audit) costs microseconds and would have caught this at first ingestion;
I consider the postmortem itself one of the project's results.

## 4. The clipping disclosure (`results/coulomb_clipping_diagnostic.csv`)

Initial SOC is clipped to [2%, 98%], so trips starting near a rail receive
less than the nominal offset. This materially flatters the coulomb
baseline's *aggregates* in 23 of 45 dataset×offset cells: at +20 pp, 43%
of BMW and 51% of VED trips clip, and clipped-trip coulomb RMSE runs
2–12 pp against ~20 pp unclipped. Medians are far less affected than
means, which is one of two reasons the median is this project's primary
aggregate (the other being divergence-tail robustness). Every generated
table carries this footnote; I quote medians throughout.

## 5. Online scalar bias vs my offline corrections (`docs/PHASE2_FINDINGS.md`)

I implemented an RBC-DEKF-style baseline (Guo et al., arXiv:2510.22813) on
my own ECM: a random-walk scalar bias θ in a second, decoupled filter with
Joseph-form updates, tuned by an adaptive wide grid (Q_θ spanning
1e-18…1, R_θ 1e-10…1) on calibration splits only — plus a coupled
(augmented-state, full-Jacobian) counter-example.

The identifiability prediction I stated before the fleet runs — a scalar
voltage bias is separable from SOC only where the OCV is flat — held up:

- On the steep NMC fleets the honest tuner **switches θ off** (BMW, Deng —
  turning it off improved Deng calibration RMSE 14.2 → 9.3).
- On **CALCE (LFP)** it picks the most aggressive setting of all fleets
  (interior optimum, median |θ_end| = 164 mV) and the scalar-bias variant
  is the best EKF-family method there (26.80 vs my 31.70) — RBC-DEKF
  working in its design regime, now shown on corrected data.
- On steep NCA (UMich) the active θ correlates 0.69 with −slope·SOC-error:
  the predicted ambiguity failure, measured.
- **Coupling harm scales with OCV slope**: the coupled variant degrades
  Deng 2.1× (7.66 → 16.03, convergence 81% → 33%) while being benign on
  the CALCE plateau — a slope-dependent refinement of both my Round-2
  finding and Guo et al.'s theory.

Where does my offline recipe stand? It wins only on BMW (17.61 vs 19.39).
On Deng, carrying *no* correction is better; on VED the online θ partially
undoes a bias my fleet calibration baked in; on CALCE the online bias wins
outright. Chemistry-dependent, as predicted — but the dependence cuts
against my recipe more often than for it.

## 6. Failure analyses (`docs/FAILURE_ANALYSIS.md`, `docs/VED_BREAKDOWN.md`)

**VED.** Not chemistry (NMC-consistent windows; my fitted OCV beats every
generic table; innovations white with zero SOC-structure), not trip
shortness (10% flat-OCV exposure; censoring reclassifies ~15 of 342
failures), not a silent fallback. The primary cause is a **~1.9×
capacity error** in the generic fallback cartridge (80 Ah assumed,
≈42 Ah implied — consistent with a 96s1p pack), which the SOH slow loop
partially re-learns *within every independent trip* (median trip-end SOH
0.63), injecting error while it does: freezing the slow loops improves
VED to 18.41. The fleet-level δV spline is separately harmful (δR0-only:
18.96). Per-vehicle calibration rescues VehId_0010 (19.8 → 7.7 pp) but
not VehId_0455 (~26 pp floor, unexplained — the honest open item).

**CALCE.** The one dataset where the estimator *class* is the binding
constraint: after correction the residual has no SOC structure (2.5 mV)
but 147 mV of variance against an assumed 2 mV measurement noise —
unmodeled rate/hysteresis dynamics under 1-s DST/US06/FUDS profiles — with
59% of samples in the flat plateau. The δR0 magnitude stays non-physical
post-fix and the ablation validates the sanity gate (+5.8 pp if disabled).
A first-order-RC ECM with scalar corrections is simply not enough model
for dynamic LFP lab cycling; Guo et al.'s electrochemical-model choice for
this regime looks vindicated.

## 7. Ablations (`results/ablation.csv`, `figures/ablation.pdf`)

Nine variants × five datasets, the `full` row cross-checked byte-identical
to the committed pipeline. What I learned, component by component:

- **Jacobian decoupling is the one unambiguous win** (2.5×/2.0×
  degradation when reverted on BMW/VED).
- **Joseph form changes nothing** (≤0.06 pp anywhere).
- **The δV+δR0 bundle is optimal on no dataset**: δV-only is best on BMW
  (16.86) and CALCE (28.69); δR0-only on VED (18.96); nothing on Deng
  (7.66). One-size fleet calibration is the recurring failure mode.
- **Adaptive Q helps only BMW and UMich**; constant Q is better on
  Deng/VED/CALCE. I keep the mechanism (it never costs more than ~2 pp
  and the plateau argument stands for LFP), but I would no longer
  headline it.
- **The calibration sanity gate earns its place** (CALCE +5.8 pp when
  disabled); the slow loops as a whole help Deng, hurt VED (capacity
  interaction), and are mixed on CALCE.

## 8. Reproducibility

`requirements-lock.txt` (Python 3.14.2, macOS arm64); seeded sampling
(Deng eval, seed 42); one command per experiment
(`data/soc_baseline_benchmark*.py`, `data/run_main_table.py`,
`data/run_offset_sweep.py`, `analysis/run_ablation.py`,
`analysis/diagnose_dataset.py`, `analysis/ved_hypothesis_test.py`,
`analysis/coulomb_clipping_diagnostic.py`, plotting scripts); every
regeneration cross-checks per-segment values against the committed
reports and aborts on >1e-6 pp mismatch; the discharge-negative
convention is asserted on every dataset load; unit tests cover metrics
(28), the bias filters (structural + known-answer), and the sign
convention (5). Metric definitions: `docs/METRICS.md`. Code map:
`docs/CODE_MAP.md`. Pre-fix artifacts preserved under
`results/pre_sign_fix_snapshot/`.

## 9. Honest conclusions

1. **What this system is good at**: recovering from untrusted
   initialization on fleet data whose OCV channel is informative. On Deng
   it beats coulomb counting at every offset; across all datasets its
   error is nearly flat in the initial offset while coulomb's is
   proportional to it.
2. **What it is not**: a universal upgrade over coulomb counting. With
   correct signs and a clipping-honest protocol, plain coulomb counting
   wins nominal conditions on four of five datasets and wins the stress
   test outright on VED and CALCE. Any claim I make for this estimator is
   a claim about initialization robustness and voltage-informative
   regimes, not about accuracy in general.
3. **The transferable findings** are the decoupling result (offline: my
   Jacobian ablation; online: the coupled-bias collapse on Deng, both now
   at fleet scale), the chemistry-scoped verdict on scalar-bias
   compensation (works on LFP plateaus, self-disables or misbehaves on
   steep chemistries), and the observation that the right unit of offline
   calibration is the vehicle, not the fleet.
4. **The process findings** may matter as much as the estimator: the
   sign-audit assertion, the reproduction-vs-correctness distinction the
   postmortem documents, and the clipping disclosure are all cheap,
   general, and absent from most of the lab-validated-EKF literature I am
   positioned against.
5. **Open items I am not hiding**: VehId_0455's 26 pp floor; the
   still-non-physical lab-dataset δR0 magnitudes; the CALCE model-class
   limit; no cross-trip state or anchor-point resets (a production BMS
   would have both, so these worst-case numbers are a stress floor, not a
   forecast of deployed performance).

### Sources

Plett, J. Power Sources 134(2), 2004 (Parts 1–3) · Kalman, ASME J. Basic
Eng., 1960 · Fritsch & Carlson, SIAM J. Numer. Anal., 1980 (PCHIP) ·
Prada et al., J. Electrochem. Soc., 2012 (LFP OCV) · Mikhak-Beyranvand et
al., 2024 (flat-plateau adaptive Q) · Guo et al., arXiv:2510.22813
(RBC-DEKF) · BMW i3 RDC trips: Lüth 2020, IEEE DataPort · Deng et al.
BAIC EU500 fleet data · Oh et al., 2020 (VED) · CALCE battery data,
calce.umd.edu · UMich/Ford parallel-module dataset, Mendeley DOI
10.17632/ssrgfmb8vw.2 / arXiv:2604.16769. Dataset provenance details:
`SOURCES.md`.
