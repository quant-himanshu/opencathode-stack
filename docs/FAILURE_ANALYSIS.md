# FAILURE ANALYSIS — why the Dual EKF loses where it loses (Phase 4)

Final, sign-corrected data (2026-07-20). Losing datasets at +20 pp, median
full-trip RMSE: **VED** (EKF 23.98 vs coulomb 19.75 vs OCV-lookup 11.40 pp)
and **CALCE** (EKF 31.70 vs coulomb 19.89 pp). BMW is a median win for the
EKF (17.61 vs 19.66), Deng a clear win (9.28 vs 14.83), UMich a tie.

Evidence sources (every number from actual runs):
`results/diagnose_{ved,calce}.json` + per-trip CSVs
(`analysis/diagnose_dataset.py` — innovations, whiteness, flat-OCV
exposure, sensor stats, slow-loop state), `results/ablation.csv`
(Phase 5), `docs/VED_BREAKDOWN.md` (per-vehicle tiers + hypothesis probes),
`results/coulomb_clipping_diagnostic.csv`, `docs/SIGN_BUG_POSTMORTEM.md`.

---

## VED — verdicts

Diagnostics: innovations essentially white (median lag-1 autocorr −0.0005)
with **zero residual SOC-structure after correction (0.5 mV)** but huge
variance (median innovation σ = 57 mV against the assumed R = 1 mV);
flat-OCV exposure only 9.9% of samples; median dt 0.40 s; current
quantization 0.25 A; fleet δR0 = +1.7 mΩ (physical); no OCV fallback;
median trip-end SOH estimate **0.63** (started at 1.0).

| Hypothesis | Verdict | Evidence |
|---|---|---|
| (a) chemistry/OCV-table mismatch | **REJECTED** | per-cell window 3.72–4.11 V = NMC-consistent; fleet-fitted OCV beats every generic table for all 3 vehicles; innovation white and structure-free after correction |
| (b) calibration fit on unrepresentative data | **SUPPORTED (the fleet-level δV spline)** | ablation: removing δV(SOC) improves VED from 23.98 → 18.96 (δR0-only) — the fleet spline, fit across 3 heterogeneous vehicles, is actively harmful; per-vehicle calibration rescues VehId_0010 (19.8 → 7.7 pp median) though not 0455 |
| (c) trips too short/flat for voltage observability | **REJECTED** | 9.9% flat-OCV exposure; median duration 474 s vs median strict t_conv 139 s; censoring reclassifies only ~15/342 failures; the dominant tier is never-approach with median closest approach 19 pp — misdirection, not blindness |
| (d) silent lookup-table fallback | **REJECTED** | empirical PCHIP OCV in use (13 bins, 767 rest points); no fallback string |
| (e) NEW — cartridge capacity error + per-trip SOH transient | **PRIMARY CAUSE** | GENERIC_EV_PACK assumes 80 Ah; the data imply ≈42 Ah (~1.9×); coulomb-vs-BMS drift +9.9…+20 pp/h; the SOH slow loop *learns* the error within each trip (median SOH_end 0.63 → Q_eff → 50 Ah, heading toward truth) but trips are independent so it re-learns from 1.0 every trip, injecting SOC error while it does; freezing the slow loops (ablation `slow_loops_off`) improves VED to **18.41** |

**VED conclusion.** The EKF loses on VED because its process model carries
a ~2× capacity error from a fallback cartridge, and the fleet-level δV
spline adds vehicle-mismatched voltage corrections on top. Both are
data/configuration failures, not estimator-design failures — and both were
only findable with field data. Remaining unknown: VehId_0455's ~26 pp
floor (66% of the never-approach tier), which survives per-vehicle
calibration.

## CALCE — verdicts

Diagnostics: median innovation σ = **147 mV** against assumed R = 2 mV;
lag-1 autocorr −0.15; residual SOC-structure after correction small
(2.5 mV); **59.4% of samples in the flat-OCV region**; dt 1.0 s; δR0
re-fit +121.6 mΩ — sign-plausible post-fix but still ~15× the physical
range, so the calibration sanity gate fires (R_int frozen); median
trip-end SOH estimate **0.59** (clip floor region).

| Hypothesis | Verdict | Evidence |
|---|---|---|
| (a) chemistry/OCV-table mismatch | **REJECTED as bias; REFRAMED as unmodeled dynamics** | the fitted OCV + δV leave near-zero structured residual (2.5 mV) — the table is fine; what remains is 147 mV of *variance* under 1-s DST/US06/FUDS load: rate/hysteresis/diffusion effects the 1-RC ECM cannot represent (cf. LFP hysteresis literature; RBC-DEKF's SPM-based approach targets exactly this regime) |
| (b) calibration fit inadequate | **SUPPORTED** | δR0 magnitude non-physical even with 40% calibration split and corrected signs — the OLS dumps unmodeled rate effects into the current-proportional term; the Round-4 sanity gate is *validated by ablation*: disabling it costs +5.8 pp (31.70 → 37.49) |
| (c) flat-OCV observability | **SUPPORTED as amplifier** | 59% flat exposure starves the voltage channel for most of each window; with the +20 pp init this delays/prevents recovery (strict convergence 26.9%) |
| (d) silent lookup-table fallback | **REJECTED** | empirical PCHIP (20 bins, 5602 rest points) |
| SOH slow loop | **MIXED** | SOH collapses toward 0.59, but freezing the loops makes CALCE *worse* (35.47) — with 59% flat OCV, the shrunken Q_eff accelerates coulomb motion that voltage occasionally corrects; the interaction is real but not the primary lever here |

**CALCE conclusion.** The EKF loses on CALCE because a first-order-RC ECM
with a single δR0 scalar cannot represent dynamic-profile LFP behavior at
1 s resolution — the model-mismatch power (147 mV) is ~75× the assumed
measurement noise, and 59% flat-OCV exposure removes the channel that
would fix it. This is the one dataset where the estimator *class* (simple
ECM + EKF) is the binding constraint, consistent with the scalar-bias
variant (which absorbs DC model error online) beating my offline recipe
there (26.80 vs 31.70) and with Guo et al.'s choice of an electrochemical
model for LFP.

## Cross-cutting findings

1. **The decoupled-Jacobian design decision is re-validated at full
   strength on corrected data**: putting ∂δV/∂SOC into H multiplies median
   RMSE by 2.5× on BMW (17.61→43.34) and 2.0× on VED (23.98→47.59)
   (`results/ablation.csv`, Round-2 row).
2. **No component of the correction recipe is universally good**: δV helps
   BMW/CALCE and hurts VED; δR0 helps VED and hurts BMW/CALCE; adaptive Q
   helps BMW/UMich only. Fleet-level one-size calibration is the
   recurring failure pattern.
3. **The coulomb baseline's stress-test aggregates carry the init-clipping
   artifact** (23/45 sweep cells; medians used throughout this document).
4. **Joseph form changes nothing** (≤0.06 pp anywhere) — the standard
   update is not a source of the losses.
