# Viva Prep — OpenCATHODE Stack

---

## 1. Two-Minute Pitch

"We built a physics-grounded battery management stack and validated it end-to-end on five real datasets — not by cherry-picking results, but by pre-registering expected outcomes and reporting failures honestly.

The core result: lab physics works but field physics is hard. The Palmgren-Miner stress-fatigue model captures NASA lab cell degradation with R²=0.9725 [`nasa_degradation_report.json`]. But on the Deng BAIC EU500 fleet — 20 real taxis in Beijing, 30,135 charging sessions, 2.3 years — the cycling damage accumulates to D≈0.002, giving a stress-term contribution of ≈3.5×10⁻⁹ SOH. Negligible. The fleet is SEI/calendar dominated: λ=0.026 SOH/√yr → 4% fade at 2.3 years [`deng_degradation_report.json`]. Any stress model would be predicting noise.

The cell-to-field bridge fails: a naive constant (fleet-mean carry-forward) beats the physics model on the held-out vehicles — RMSE 0.0365 vs 0.0453 [`cell_to_field_report.json`]. The root cause is not a modelling error. It is a data limitation: endpoint signal-to-noise ratio is 1.70, borderline [`soh_noise_floor_report.json`].

The salvage is the interesting part: a gated per-vehicle calendar-rate estimator (B3') recovers 10.1% trajectory RMSE reduction fleet-wide, and 15.4% for the 65% of vehicles with identifiable positive fade [`cell_to_field_temporal_report.json`]. The gate simply refuses to predict vehicles whose fade direction cannot be determined from the training window.

The gate failure mechanism was then quantified: at n≈800 training cycles, per-cycle SE(λ) is already 40× smaller than the identifiability threshold. The systematic trajectory bias in the training window is 48–102× larger than per-cycle noise [`sensing_requirement_report.json`]. You cannot fix it with EIS. You fix it with two more years of data.

Every failure in this project is diagnosed, committed, and cited."

---

## 2. Per-Problem Q&A

### P1 — State Estimation

**Q (Ather framing):** Can a DFN-SPM + Dual EKF stack predict pack voltage and SOC from physics on real field data, without pre-fitted parameters?

**A:** Yes, within the <20 mV precision target. On the Quartz WLTP 36-cell NMC811 pack (634k rows, real drive cycles), the stack achieves MAE 18.6 mV and R²=0.9217 on rows with genuinely new sensor readings. On the Deng BAIC EU500 fleet (20 vehicles, 2000 charging sessions), MAE 15.5 mV.

**Number:** MAE 18.6 mV [Quartz], 15.5 mV [Deng]. Source: `data/validate_quartz.py`.

**Caveat:** R²=0.9217 is foregrounded over R²=0.9810. The 0.9810 figure uses all timestamps — 83% are repeated BMS readings at 6-minute update intervals, trivial to predict. The 0.9217 figure uses only rows with genuinely new sensor measurements (~17% of timestamps). The Deng MAE uses a scale-calibrated OCV (not GITT), which introduces a systematic offset corrected by the adaptive EKF filter.

---

### P2 — Lab Degradation

**Q (Ather framing):** Does the Palmgren-Miner stress-fatigue model correctly capture cell degradation trajectory under controlled conditions?

**A:** Yes — in the regime where it applies. On NASA PCoE cells (B0005/B0006/B0007/B0018; 1C, DoD≈100%, 132–168 cycles each, direct capacity measurement), within-cell R²=0.9725 (mean over 4 cells), MAE=1.26% SOH [`nasa_degradation_report.json`]. The model captures trajectory shape correctly. Cross-cell generalisation fails: R²=−0.68 on held-out cells B0006+B0018 when trained on B0005+B0007. B0006 degrades 45% faster than B0005 at nearly identical cumulative damage D — manufacturing batch variability in the Wöhler β coefficient. Per-cell β calibration is required before deployment.

**Number:** Within-cell R²=0.9725, MAE=1.26% SOH. Cross-cell R²=−0.68. Source: [`nasa_degradation_report.json`](../data/nasa_degradation_report.json).

**Caveat:** "Validated" means model captures trajectory shape in the appropriate regime (mechanical fatigue dominant). It does not mean usable for field prediction without per-cell β calibration, and it does not apply to the field regime (partial DoD, low C-rate, calendar-dominated).

---

### P3 — Cell-to-Field Bridge

**Q (Ather framing):** Can we transfer cell-level stress-fatigue parameters to predict fleet-wide field SOH trajectories?

**A:** No — not on this fleet, and the failure is clean and diagnosable. The physics bridge (B1: calendar only; B2: full; B2α: pack-scale corrected) is beaten by a constant on the held-out Tier 3 vehicles: B0 (fleet-mean) RMSE=0.0365 vs B1=0.0453, a 24% degradation [`cell_to_field_report.json`]. The calibrated scale factor α=−0.382 (degenerate negative) — the OLS is using the cycling term to compensate calendar overshoot, which is physically nonsensical.

**Why it fails:** At 57% DoD, 0.41C, cumulative fatigue damage D≈0.002 across all 20 vehicles. The cycling contribution is ≈3.5×10⁻⁹ SOH — negligible vs 3.8% measured fade. The fleet is SEI/calendar dominated. λ_sei=0.026 SOH/√yr (M2 fit) overestimates fleet-median λ by 1.43×, so even the calendar-only bridge overshoots. Pre-registered expectation: B2≈B1. Confirmed.

**Number:** B0 RMSE=0.0365, B1=0.0453. α=−0.382. Source: [`cell_to_field_report.json`](../data/cell_to_field_report.json).

**Honest framing:** The bridge "failure" is the correct scientific result. It locates the dominant mechanism (calendar/SEI) and disqualifies stress-fatigue for this operating regime. Reporting a good RMSE here would require hiding the noise floor.

---

### P4 — Gated Per-Vehicle Adaptation

**Q (Ather framing):** Given bridge failure, can a per-vehicle gated predictor recover meaningful fleet SOH prediction?

**A:** Partially — for 65% of vehicles. Per-vehicle calendar OLS (λ_v fit on first 50% of each vehicle's timeline) gives the B3' gated predictor: trajectory RMSE −10.1% vs carry-forward across all 20 vehicles, and −15.4% for the 13 vehicles with identifiable positive fade (gate-in group) [`cell_to_field_temporal_report.json`]. The gate is train-window-observable only (no test-period information used): if λ_v>0 in the training half, use per-vehicle calendar prediction; otherwise fall back to carry-forward.

**Trade-off:** Gating trades rank-correlation for accuracy. Endpoint ρ: B2'=+0.802 → B3'=+0.692. Gated-out vehicles receive flat predictions, which reduces discrimination but improves RMSE.

**Number:** B3' trajectory RMSE −10.1% (all 20); −15.4% (λ>0, n=13). Gate pass-rate 65%. Source: [`cell_to_field_temporal_report.json`](../data/cell_to_field_temporal_report.json).

**Gate failure diagnosis (P6):** The 35% gated out are not hardware-limited. SE(λ) at n≈800 cycles is already 40× below the i.i.d. identifiability threshold. The gate failures are systematic trajectory bias — the first-half SOH trajectory of these vehicles is non-monotone (transient plateau or recovery), σ_eff=48–102× per-cycle noise. The fix is a longer observation window (r=2× empirically recovers 3/3 noise-masked vehicles), not EIS or IC upgrades [`sensing_requirement_report.json`]. This is an observation-duration problem.

---

### P5 — Factor Attribution

**Q (Ather framing):** Which operating factor (T, DoD, C-rate, n_cycles) most drives per-vehicle fade rate λ_v?

**A:** Cannot be determined from this fleet. All |ρ| < 0.40 (max=0.268 for DoD_p95), no feature clears Bonferroni significance at n=13 (threshold p<0.0083), LOO R²=−12.83 [`factor_ranking_report.json`]. Pre-registered expectation confirmed: same-city, same-model fleet with T range <1°C, DoD range <2.5%, C-rate range <0.012 does not have enough feature variance to resolve factor attribution.

**Directional (not significant):** Observed rank order by |ρ|: DoD_p95 > n_cycles > C_rate_mean > DoD_mean > T_p95 > T_mean. This is partially consistent with Edge et al. (2021 PCCP 23:8200) T > DoD > C-rate at the ordinal level (DoD is near top, T is at bottom), but all values are indistinguishable at n=13.

**Number:** max |ρ|=0.268 (DoD_p95), LOO R²=−12.83. Source: [`factor_ranking_report.json`](../data/factor_ranking_report.json).

**Caveat:** Permutation importance values are in-sample (optimistic at n=13). SHAP was skipped (package not installed). The indistinguishable result is the pre-registered expected outcome — not a disappointment.

---

## 3. Trap Question Prep

### "Deng et al. got <1.6% error on this dataset. You say it's noise-dominated?"

Different task, different error definition, different data structure.

Deng 2023 predicts the **within-vehicle capacity trajectory** using the vehicle's own historical data (sequence-to-sequence neural network + Gaussian process residuals, trained per vehicle). Their error is: given vehicle V's past 3 months of data, predict V's future capacity sequence. The "error lower than 1.6%" is a within-vehicle trajectory tracking error, where the vehicle's own BMS trend is the primary signal.

We predict **cross-vehicle endpoint ΔSOH** using frozen cell-level parameters (λ_sei, β, γ) calibrated on a disjoint tier of vehicles. Our error is: given fleet-level physics parameters from cells that are NOT this vehicle, what is this vehicle's cumulative SOH fade? This task has no access to the vehicle's own past trajectory. The 3.7% MAE [`deng_degradation_report.json`] is on a strictly harder task. The 1.6% figure does not apply.

The two results are complementary, not contradictory.

---

### "Why didn't Severson's feature (variance of discharge voltage curve) work on your fleet chemistry?"

Two independent reasons, both disqualifying:

1. **Protocol mismatch:** Severson et al. (2019) exploit the variance of dQ/dV between cycles 2–100 during controlled complete discharge at C/5–C/25 rates. This requires slow, full discharge cycles to resolve the fine structure of the voltage curve. The Deng fleet is field charging data — partial SOC windows, variable C-rate, no controlled complete discharge. The feature computation is physically impossible on this data.

2. **Chemistry mismatch:** Severson's result was demonstrated on LFP cylindrical cells. The voltage variance feature exploits the extreme flatness of the LFP plateau (3.2–3.35 V), where small shifts in the plateau boundary are diagnostic of capacity loss. The Deng fleet is CATL NCM 145 Ah (confirmed commit a851620). NMC has a sloping voltage profile throughout — no plateau. The diagnostic sensitivity of the Severson feature does not transfer to NMC without adaptation.

Both conditions (slow full-discharge protocol AND LFP plateau) must hold for the feature to work. Neither holds for the Deng fleet.

---

### "Your bridge lost to a constant — isn't that a failure?"

It is a result, not a failure, and the distinction matters.

A constant (fleet-mean carry-forward, B0) achieves RMSE=0.0365 on Tier-3 held-out vehicles. The physics bridge B1 achieves RMSE=0.0453 — 24% worse [`cell_to_field_report.json`]. The failure has a complete diagnosis: the Deng fleet runs at 57% DoD, 0.41C mean; cumulative fatigue damage D≈0.002; the stress contribution is ≈3.5×10⁻⁹ SOH. The bridge is predicting a calendar-only term (λ·√t) with a λ_sei calibrated on 4 high-fade vehicles that overestimates fleet-median λ by 1.43×. The OLS scale factor α=−0.382 (degenerate) is the system trying to correct this overshoot using the only free parameter available — the cycling term — in the wrong direction.

The B3' temporal salvage (P4) reduces RMSE to 10.1% below the constant for the full fleet, and 15.4% for the gate-in subgroup. The delivered result is: honest diagnosis of why naive transfer fails, plus a gated predictor that beats the constant for 65% of the fleet.

The alternative — tuning parameters to make the bridge win — would have been RMSE-hacking (pre-registered constraint, see Section 5).

---

### "Which stress component does your model use, and why?"

Palmgren-Miner linear damage summation: D = Σ_k (n_k / N_f(DoD_k)) where N_f(DoD_k) = A · DoD_k^(−m) is the Basquin fatigue life for that half-cycle amplitude. Parameters: A=10⁶, m=2.5 (literature defaults). Half-cycles are counted using ASTM E1049 rainflow counting on the per-session SOC profile.

This is a cycle-counting approach, not a maximum-stress criterion. It implies: many small-amplitude cycles can accumulate the same damage as fewer large-amplitude cycles. The Basquin exponent m=2.5 reflects the S-N curve slope: doubling DoD reduces fatigue life by 2^2.5 ≈ 5.7×.

No fracture mechanics (Klinsmann et al. 2016) or electrochemical-mechanical coupling is implemented. This is intentional: those models require per-particle microstructure parameters (Young's modulus, partial molar volume, particle size distribution) not available from BMS data. The Miner rule is the maximum simplification consistent with available observables.

⚠️ *A=10⁶, m=2.5 are literature order-of-magnitude estimates, not calibrated from the Deng data. The cycling term D≈0.002 is too small to resolve against the noise floor — any m in [2, 4] would give the same qualitative conclusion (cycling negligible vs calendar).*

---

### "Why is your within-cell R² 0.9725 but cross-cell R² negative?"

Within-cell (Eval A): fit β and γ per cell using that cell's own complete cycle trajectory. Two free parameters for one curve → the power-law ΔSOH = β·D^γ has enough flexibility to fit any monotone concave trajectory. R²=0.9725 says "this functional form is adequate," not "the parameters generalise."

Cross-cell (Eval B): fit β and γ jointly on B0005+B0007, apply to B0006+B0018. R²=−0.68. The key data: B0006 degrades at 41.7% total fade vs B0005's 28.6%, at nearly identical cumulative damage D (same 1C CC protocol). The β coefficient is a material property of the Wöhler fatigue curve — it captures the cell's resistance to mechanical stress at the electrode particle level. Manufacturing batch variability (dopant distribution, particle size distribution, binder homogeneity) drives cell-to-cell β differences that cannot be predicted from operating protocol alone [`nasa_degradation_report.json`].

Conclusion: the model structure is valid (R²=0.97 within-cell confirms it); the parameter β is cell-specific. Cross-cell deployment requires a per-cell β calibration step (≥20 early cycles at high DoD to resolve β).

---

### "Why didn't you just use ML end-to-end?"

Three reasons:

1. **Sample size:** At n=13 usable gated-in vehicles (P5), any ML model with more than 2 parameters overfits. LOO R²=−12.83 for a 6-feature Ridge regression [`factor_ranking_report.json`]. A neural network or gradient boosting would perform orders of magnitude worse. The physics prior (λ·√t, one parameter per vehicle) is the only approach tractable at this n.

2. **Identifiability:** The key deliverable is not a point prediction but a per-vehicle fade rate λ_v with a sign-identifiable estimate. The gate criterion (λ_v > 0) requires knowing SE(λ_v) — possible analytically from OLS. An ML model gives no such guarantee.

3. **Interpretability:** We need to attribute fade to identifiable physical mechanisms (calendar vs cycling) to know what intervention to recommend. An ML model maps features to a number; we map physics to a mechanism. The finding that β_cycling·D ≈ 0 for this fleet tells Ather that increasing DoD or C-rate within the current operating envelope would not meaningfully accelerate degradation — which is actionable. An RMSE-optimal ML number gives no such guidance.

---

## 4. Four Deep-Physics Answers

### Fatigue vs. maximum-stress

The Palmgren-Miner rule accumulates damage linearly across cycles: D = Σ_k n_k/N_f(σ_k), where N_f(σ) is the number of cycles to failure at stress amplitude σ. This is a **cycle-counting** approach, not a peak-stress criterion.

The key assumption is linearity of damage — each cycle contributes independently, regardless of sequence. This is appropriate for high-cycle fatigue in elastic regimes (electrode particles undergoing small, reversible deformations) and for cases where mean stress effects are secondary. It would be inappropriate for low-cycle fatigue with significant plastic deformation or for sequential loading where larger cycles suppress fatigue life.

In our application, the stress proxy is DoD (a scalar amplitude per half-cycle from rainflow counting). The Basquin exponent m=2.5 gives the S-N curve slope. Higher DoD → shorter fatigue life, steeply (N_f ∝ DoD^{−2.5}).

⚠️ *We use DoD as the mechanical stress proxy. True mechanical stress in electrode particles requires solving the spherical diffusion PDE (σ ∝ E·Ω·∇c, Klinsmann et al. 2016) — not implemented. DoD is a coarser proxy accessible from BMS data.*

---

### σ_t at the particle surface

During lithiation/delithiation, lithium ions intercalate into electrode particles, causing volumetric expansion/contraction (partial molar volume Ω). In a spherical particle with radius R, the concentration profile c(r,t) follows Fick's law. At high C-rate, the surface depletes faster than the bulk can equilibrate, creating a concentration gradient ΔC = c_surface − c_bulk.

The resulting diffusion-induced stress at the particle surface is:

σ_t(surface) = 2EΩ(c_bulk − c_surface) / (9(1−ν))

where E is Young's modulus and ν is Poisson's ratio of the electrode material. When σ_t exceeds the fracture toughness, crack initiation occurs. Crack propagation exposes fresh SEI-free surface, accelerating calendar ageing — the coupling between mechanical and electrochemical degradation.

For the Deng fleet at 0.41C mean C-rate: the concentration gradient ΔC is small (near-equilibrium conditions), so σ_surface is well below fracture thresholds. This is consistent with D≈0.002 (negligible mechanical damage).

⚠️ *σ_surface is not computed in the stack. The DFN-SPM uses the diffusion coefficient D_s from the Chen 2020 LGM50 parameterisation (NMC811). For the Deng fleet's CATL NCM 145 Ah, D_s is unknown. The argument above is qualitative, not computed.*

---

### Log-normal particle radius and the surface-weighted mean

Real electrodes have a particle size distribution, typically log-normal: ln(R) ~ N(μ, σ²). Two means are relevant:

- **Number-weighted mean:** R_n = exp(μ + σ²/2). Most particles are near this size.
- **Surface-area-weighted mean:** R_s = exp(μ + 5σ²/2). Dominates total surface area and reaction kinetics.
- **Volume-weighted mean:** R_v = exp(μ + 7σ²/2). Dominates capacity.

For diffusion-induced stress, the relevant radius is the surface-area-weighted mean, because stress scales as R² for a given current density (more material must be lithiated/delithiated per unit surface). The largest particles accumulate the most stress and crack first — initiating degradation even when the majority of particles are intact.

⚠️ *The stack uses a single effective particle radius from the SPM parameterisation. No particle size distribution is modelled. The Deng fleet CATL NCM cells have unknown particle size distribution — this is an assumption inherited from the Chen 2020 LGM50 cartridge, not a validated cell-specific parameter.*

---

### LLI / LAM / CL population smoothing in DVA/ICA

Differential Voltage Analysis (dV/dQ) and Incremental Capacity Analysis (dQ/dV) decompose the voltage-capacity curve into mechanistic signatures:

- **LLI (Loss of Lithium Inventory):** shifts the SOC window; in DVA, shifts all peaks left (delithiation) or right (lithiation) without changing peak spacing. Primary mechanism: SEI growth consuming Li.
- **LAM_an (Loss of Active Material, anode):** compresses the dV/dQ curve from the left; peak spacing narrows on the anode side.
- **LAM_ca (Loss of Active Material, cathode):** compresses from the right; peak spacing narrows on the cathode side.
- **CL (Contact Loss):** broadens peaks (increased internal resistance → voltage polarisation).

For a fleet, each vehicle produces one ICA/DVA signature. Population smoothing (e.g., median across vehicles) identifies the dominant fleet-wide mechanism. Peaks that shift coherently across vehicles → LLI; peaks that compress → LAM. This requires controlled, low-rate discharge curves (C/10–C/25) to resolve the fine structure.

⚠️ *Full DVA was NOT performed on the Deng field data. The Deng dataset consists of fast partial-charge sessions (BMS-reported SOC, not controlled full discharge at low C-rate). The ICA resolution at field cycling rates is insufficient to separate LLI from LAM. The SEI-dominated finding (P3) comes from the λ·√t model fit, not from DVA peak decomposition. Module 3 DVA was validated on lab reference curves only.*

---

## 5. What I Got Wrong and Fixed

| What | Wrong claim | Fix applied | Commit |
|---|---|---|---|
| RMSE-hacking | Searched for training cell splits that reproduced a target RMSE (≈138); this is target-hacking | Search stopped immediately on identification; no cherry-picked split reported | Pre-registration constraint, session boundary |
| Unverified subgroup claim | "B2' would win clearly for the 13 positive-λ vehicles" — asserted but not computed | Implemented B3' gated predictor; computed actual subgroup RMSE: −15.4% for λ>0 group | `f8bb317` |
| LFP chemistry label | Deng fleet labelled "LFP, 136.2 Ah pack" in 3 committed files (docstring, verdict branch, JSON caveat) | Corrected to CATL NCM (NMC) 145 Ah. Source: Deng 2023 Applied Energy. Confirmed by loader's own detect_chemistry() voltage thresholds | `a851620` |
| Sensing problem framing | P3 deployment note stated gate improvement "requires EIS / incremental capacity or longer windows — a sensing problem" | P6 showed per-cycle SE already 40× below threshold; actual bottleneck is systematic bias 48–102× per-cycle noise; EIS irrelevant; framing corrected to "observation-duration problem" | `42749c3` |
| Dubarry citation venue | "J. Power Sources Adv. 100049" — wrong journal, wrong article ID | Corrected to Front. Energy Res. 10:1023555. DOI 10.3389/fenrg.2022.1023555 | `f341b88` |
| Sulzer DOI | `10.1016/j.joule.2021.08.020` — wrong suffix | Corrected to `10.1016/j.joule.2021.06.005` | `f341b88` |
| Unverified error claim | "~5-8% mode error at 1C" — specific percentage unsourced from Dubarry | Softened to: "mode quantification unreliable at high rate; low-rate (C/20–C/25) data required" | `f341b88` |
