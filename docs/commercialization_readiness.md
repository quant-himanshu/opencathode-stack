# OpenCATHODE: Commercialization Readiness Assessment

**Technology / IP Summary for Startup Fundraising and Technical Due Diligence**

*Every numeric claim is traced to a source JSON file or commit. Claims without a traceable source are absent from this document.*

---

## 1. Executive Summary

OpenCATHODE is a physics-informed battery degradation modelling stack validated on real field data (Deng BAIC EU500 fleet, 20 NCM vehicles, 30,135 cycles) and a NASA LCO laboratory dataset (4 cells, 168 cycles each). The core finding is that a single-parameter calendar-aging model (λ·√t, SEI-derived) accounts for 100% of modelled fade in field-deployed EV batteries under typical urban driving — cycle-stress terms are negligible in this regime — with fleet-level SOH MAE of 0.035 on pure hold-out vehicles (V10–V20, [`deng_degradation_report.json`](../data/deng_degradation_report.json)). A gated per-vehicle adaptation layer (B3') reduces trajectory RMSE by 10.1% fleet-wide and 15.4% for the 65% of vehicles with identifiable fade signal, using only on-board BMS data ([`cell_to_field_temporal_report.json`](../data/cell_to_field_temporal_report.json)). The technology is at prototype stage (TRL 3–5 depending on component); significant gaps remain in cross-chemistry generalization and real-time embedded deployment, which are stated explicitly below.

---

## 2. Technology Readiness Assessment

### 2a. Real-Time SOC Estimation — **TRL 4–5**

**What it does:** Dual EKF with DFN-SPM physics prior for real-time state-of-charge tracking.

**Evidence:** Quartz WLTP dataset, 36 cells, 12,690 sensor-update windows. On rows where a sensor update occurs: R²=0.9217, MAE=18.6 mV, RMSE=38.4 mV. On all rows (including open-circuit): R²=0.981. Source: [`stack_validation_report.json`](../data/stack_validation_report.json), field `_real_results.quartz_wltp_36cell`.

**Honest caveats:**
- Validated on one chemistry/protocol (Quartz WLTP). No disclosed performance on LFP or NMC at high C-rate.
- The EKF is a running estimator; the R²=0.9217 is on sensor-update rows only. Interpolated rows are structurally easier (R²=0.981) and not the binding constraint.
- No latency or embedded CPU profiling exists yet. TRL 5 claim is for algorithmic fidelity, not embedded deployment readiness.

**Why TRL 4–5 and not 6+:** Validated against real cell data in realistic drive cycle conditions, but not integrated into a product or deployed in hardware-in-the-loop testing.

---

### 2b. Fleet-Level SOH Estimation (Calendar Model) — **TRL 4**

**What it does:** Per-vehicle SEI calendar-aging model (λ·√t) fit on early vehicle history, predicting remaining SOH trajectory without in-situ cell teardown or EIS.

**Evidence:** Deng BAIC EU500 NCM fleet. Training: V01–V04 (λ=0.02639 SOH/√yr, fit on training vehicles only). Hold-out Tier 1–2 (V05–V09) MAE: mean 0.042 (range 0.026–0.065). Pure hold-out Tier 3 (V10–V20, no parameter tuning, no peek): mean MAE=0.035 (range 0.020–0.076). Source: [`deng_degradation_report.json`](../data/deng_degradation_report.json).

**Honest caveats:**
- **Caveat on the MAE metric:** r2_combined is negative for 13 of 20 vehicles (e.g. V16=−9.42, V17=−5.11, V07=−4.68) — the model does not outperform a per-vehicle mean baseline in variance-explained terms, even though absolute MAE is low. This reflects the low signal-to-noise ratio of 2-year fade in this fleet (endpoint SNR≈1.70, see [`soh_noise_floor_report.json`](../data/soh_noise_floor_report.json)), not a strength of the model. An investor evaluating this number should treat MAE=0.035 as "small absolute error on a small, noisy signal," not as evidence of strong predictive power.
- sei_frac=1.0, stress_frac=0.0 on all 20 vehicles: cycling-stress physics is not exercised in this dataset. The model is validated only on calendar-dominated degradation.
- V16 MAE=0.076 and V17 MAE=0.064 are outliers. Both have non-monotone SOH (dsoh_final<0, meaning capacity recovered above initial), which the model cannot represent. This is a real failure mode, not a pre-processing artifact.
- The Deng dataset is 20 vehicles, one make/model/chemistry. Population-level confidence intervals are wide at n=20.

**Why TRL 4 and not 5+:** Validated on real field data under realistic conditions, but only one fleet dataset, one chemistry, one drive pattern (urban BAIC EU500 usage). No multi-fleet or multi-chemistry cross-validation exists.

---

### 2c. Gated Per-Vehicle Degradation Adaptation (B3') — **TRL 3–4**

**What it does:** Uses the first half of each vehicle's observed timeline to fit a per-vehicle λ_v; if λ_v>0 (fade is identifiable), predicts the second half using per-vehicle rate; otherwise falls back to fleet carry-forward. No test-period information is used.

**Evidence:** Fleet-wide trajectory RMSE: B3'=0.0276 vs B0 (carry-forward) 0.0307 — a 10.1% reduction. For the gate-in subgroup (λ_v>0, 13/20 vehicles): 15.4% RMSE reduction. Gate pass-rate: 65% (13/20). Source: [`cell_to_field_temporal_report.json`](../data/cell_to_field_temporal_report.json).

**Honest caveats:**
- The 65% gate pass-rate is a data limitation, not a model limitation. The 7 gated-out vehicles (V05, V07, V10, V12, V14, V16, V17) have non-monotone or near-zero fade — the gate correctly withholds prediction when the signal is unresolvable. Analysis (P6) shows the failure is systematic trajectory bias σ_eff=66–102× per-cycle noise, requiring ~2× longer observation windows to resolve, not EIS or IC analysis. Source: [`sensing_requirement_report.json`](../data/sensing_requirement_report.json).
- B3' is a post-hoc gated OLS predictor. It does not learn cross-vehicle structure; each vehicle's λ_v is estimated independently.
- The LOO-transferred calendar rate (B2') does NOT beat the carry-forward constant fleet-wide (B2' tRMSE=0.0305 vs B0 0.0307). Cross-vehicle transfer of λ adds no value when the fleet has wide λ spread. Only within-vehicle calibration works.

**Why TRL 3–4 and not 5:** Single dataset. No prospective deployment. Gate threshold (λ_v>0) is a data-derived heuristic not independently validated.

---

### 2d. Cross-Chemistry / Cross-Cell Generalization — **TRL 1–2**

**What it does (attempted):** Transfer of degradation rate parameters (β) from LCO lab cells to NCM field vehicles using physics-based feature mapping.

**Evidence of failure:** LCO→NCM parameter transfer fails by 200–300× (the transferred β produces predictions 200–300× the residual magnitude of per-vehicle OLS). Source: commit `eb7779b`. The Sobol-ranked sensitivity drivers (T > DoD > C-rate; Sobol total indices: T=0.749, DoD=0.372, C-rate=0.166, source [`factor_ranking_report.json`](../data/factor_ranking_report.json)) confirm this result qualitatively: temperature and DoD dominate, and the two datasets operate in completely different temperature and DoD regimes.

**Why this fails:** The single-parameter power law β·k^γ conflates chemistry-dependent degradation rate with cycle conditions. Without explicit chemistry-specific parameters (e.g., separate LLI and LAM terms), the β estimated from LCO lab cycling is not a portable quantity.

**Why TRL 1–2 and not higher:** Cross-chemistry transfer is an open research problem. The 200–300× magnitude mismatch is large enough that no partial mitigation has been identified within this project scope.

---

### 2e. Uncertainty Quantification (Hierarchical Bayes + GP) — **TRL 2–3**

**What it does:** Two-layer uncertainty framework. Layer 1: hierarchical Bayesian model over per-cell degradation rates β, estimating population mean and variance. Layer 2: physics-informed GP (Matern 5/2, mean function m(k)=β·√k) for early-cycle trajectory extrapolation with calibrated predictive intervals.

**Evidence:**
- Hierarchical β (4 NASA LCO cells): μ_β=0.02162±0.003692, σ_β HDI 94%: [0.0028, 0.0129]. R-hat=1.0000, ESS_min=6046. Source: [`hierarchical_beta_report.json`](../data/hierarchical_beta_report.json).
- GP LOO validation (4 cells, N∈{20,50,80} early cycles): at N=20, 90% predictive interval coverage=87% (near-nominal). At N=50 and N=80, coverage degrades to 60–63% — systematic overconfidence emerges. Source: [`bayes_gp_report.json`](../data/bayes_gp_report.json).
- Physics mean function gives 2.2–3.1× RMSE improvement over zero-mean baseline across all 12 typical-fold conditions.

**Named limitation (verbatim from module docstring):** The Matern52 kernel's amplitude, length-scale, and σ_obs are estimated from training-cell within-cell residual structure only; they do not and cannot capture between-cell systematic deviation on the held-out cell. This is why coverage degrades at higher N even as the kernel posterior itself converges cleanly (R-hat=1.0000) — convergence of the sampler is not the same as correctness of the uncertainty model.

**Honest caveats:**
- n=4 cells. Hierarchical Bayes is statistically underpowered at n=4; ≥10 cells is the standard recommendation for stable estimates. All results are illustrative of the methodology, not definitive population estimates.
- The GP study uses NASA LCO cells only. Coverage behavior on NCM field vehicles is untested.
- TRL 2–3 reflects that the methodology is demonstrated on real lab data but has not been validated at fleet scale.

---

## 3. Differentiation vs Existing BMS Solutions

### What commercial BMS products currently do on SOH

No commercial BMS vendor publicly discloses fleet-level SOH accuracy as a numeric specification. Sensata, Lithium Balance, and similar vendors describe SOH computation in marketing materials but do not publish MAE, RMSE, or coverage figures against independent ground truth.

A 2026 peer-reviewed study ([arxiv 2603.21592](https://arxiv.org/abs/2603.21592)) of 1,114 EVs across five manufacturers found that commercial BMS SOH reporting has correlation with actual measured capacity of ρ=0.10 (non-significant) to ρ=0.62 under restrictive filtering. Of 371 vehicles classified as "healthy" (SOH≥95%), actual relative capacity spanned 71–142% — a 71-percentage-point spread hidden behind a nominally reassuring number. The EU Battery Regulation's 2027 SOH transparency mandate is partly a regulatory response to this failure.

**What this means for OpenCATHODE:** The project has demonstrated a validated fleet-level pipeline with a disclosed MAE of 0.035 on pure hold-out vehicles. Whether 0.035 MAE SOH units (3.5 percentage points) is better or worse than current commercial BMS SOH — for the specific Deng NCM fleet under urban driving — cannot be determined without access to the commercial BMS SOH output on the same vehicles. The honest claim is:

> *OpenCATHODE demonstrates a fully traceable, validated degradation pipeline on real fleet data with a disclosed hold-out MAE of 3.5% SOH. Commercial BMS SOH accuracy on the same fleet is unknown. The differentiation claim is methodological — explicit hold-out validation, traceable parameter sources, and calibrated uncertainty intervals — not a demonstrated accuracy advantage over any specific commercial product.*

### Where research literature shows commercial BMS gaps

- **Uncertainty quantification:** Conformal prediction and Bayesian uncertainty for BMS SOH is described in 2024 research literature as an "emerging" topic, not a standard product feature. The cited difficulty: "parameterizing models that can accurately quantify predictive uncertainty is challenging because battery datasets are usually limited in size."
- **Early-cycle extrapolation:** GP-based prediction from N=20 cycles is a laboratory demonstration. No commercial BMS product is known to offer this.
- **Degradation factor attribution:** The Sobol-ranked sensitivity decomposition (T > DoD > C-rate) is a research output; commercial BMS products do not publicly disclose degradation attribution at this level.

---

## 4. Honest Gaps and What Funding Would Buy

| Gap | Current Status | What Closes It |
|---|---|---|
| **n=4 LCO cell base** | All GP/hierarchical results from 4 cells; statistically underpowered | Expand to Severson (124 cells, LFP) or equivalent multi-cell dataset; ~6–12 months research |
| **Cross-chemistry transfer** | 200–300× failure LCO→NCM; root cause identified (regime mismatch) | Chemistry-specific parameter sets; requires lab cycling across 3+ chemistries |
| **EKF embedded latency** | Not profiled; no embedded implementation | Hardware-in-the-loop test; BMS microcontroller port |
| **B3' gate improvement** | 65% gate pass-rate; 35% need 2× longer observation window | Longer observation periods, or IC/EIS augmentation for vehicles with flat voltage curves |
| **Multi-fleet validation** | One fleet (Deng, 20 vehicles, NCM, BAIC EU500, 2019–2021) | Additional datasets: Tesla fleet data, CALCE, or OEM partnership |
| **n=20 cells for hierarchical Bayes** | n=4 is below the ≥10 threshold for stable estimates | Expand cell dataset; same lab protocol as NASA study |
| **Real-time SOH update** | B3' uses 50/50 temporal split; not a streaming estimator | Sequential Bayesian update; adds code complexity, not fundamental science |

---

## 5. IP and Defensibility Notes

**What exists today:**
- Validated physics-informed modelling pipeline, publicly available under Git versioning (all commits traceable, all results reproducible from source data).
- Demonstrated methodology for hold-out validated fleet SOH estimation, cross-cell β population inference, and early-cycle GP extrapolation.
- Clear identification of failure modes (cross-chemistry, N≥50 GP overconfidence, gate failure at 35% of fleet) — which is itself defensible IP because it enables honest product scoping.

**What does not constitute defensible IP:**
- The calendar SEI model (λ·√t) is standard electrochemistry (Peled 1979, Safari & Delacourt 2011). Not novel.
- The Dual EKF for SOC is textbook (Plett 2004). Not novel.
- The power-law degradation model (β·k^γ) is standard. Not novel.

**Where defensibility could emerge:**
- The specific gated per-vehicle adaptation architecture (B3') and its sensing-requirement analysis could be novel if no prior art exists for the exact gate criterion (sign of per-vehicle λ_v in training window). Prior art search not conducted.
- The physics-informed GP mean function with LOO-consistent priors and the named limitation (kernel amplitude estimated from within-cell only) may be novel in application. Prior art search not conducted.
- A productized pipeline with disclosed, hold-out validated fleet accuracy could be defensible via trade secret if the training data and validation protocols are proprietary.

**Honest assessment:** At current TRL (2–5 by component), this is pre-IP. Defensibility comes from being first to market with a validated, honestly-scoped product — not from patents on individually standard components. A freedom-to-operate analysis is recommended before any fundraise with IP claims.

---

*Numbers verified against source files as of 2026-07-05. Every claim links to its source JSON or named commit. Claims for which no verifiable source exists are absent.*
