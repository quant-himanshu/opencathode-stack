# OpenCATHODE Stack
**A physics-grounded battery BMS stack validated end-to-end on real data — where every claim survived an adversarial audit, and every failure is diagnosed, not hidden.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MAE](https://img.shields.io/badge/MAE-18.6mV-brightgreen)](data/validate_quartz.py)
[![R²](https://img.shields.io/badge/R²-0.9217-brightgreen)](data/validate_quartz.py)

A six-layer physics-informed BMS stack. The electrochemistry core (DFN-SPM, Dual EKF, EIS) is validated on real laboratory and field data. Pack-management extensions (GraphSAGE GNN, Negative Selection, ACO+Kuramoto) are implemented but not yet validated end-to-end on real data — they are marked **prototype** below.

---

## The Arc

Lab physics works (NASA lab R²=0.9725 [[nasa_degradation_report.json](data/nasa_degradation_report.json)]) → field fade is calendar-dominated, not stress-fatigue (λ·√t accounts for 100% of modelled fade; cycling term ≈3.5×10⁻⁹ SOH [[deng_degradation_report.json](data/deng_degradation_report.json)]) → naive cell-to-field transfer fails (B0 constant beats physics bridge: RMSE 0.0365 vs 0.0453 [[cell_to_field_report.json](data/cell_to_field_report.json)]) → root cause quantified (endpoint SNR=1.70, borderline; trend SNR=3.43 [[soh_noise_floor_report.json](data/soh_noise_floor_report.json)]) → salvage via gated per-vehicle adaptation (B3' −10.1% trajectory RMSE vs carry-forward, all vehicles; −15.4% for the 65% with identifiable fade [[cell_to_field_temporal_report.json](data/cell_to_field_temporal_report.json)]) → limiting factor identified and quantified (gate failure is systematic trajectory bias σ_eff=66–102× per-cycle noise, not precision; window extension r=2× is the operative lever [[sensing_requirement_report.json](data/sensing_requirement_report.json)]).

---

## Ather Problem Scorecard

| Problem | What was done | Key number | Source JSON | Status |
|---|---|---|---|---|
| **P1** State estimation: predict pack voltage and SOC from physics on real fleet data | DFN-SPM + Dual EKF, tested on Quartz WLTP (36-cell NMC811, 634k rows) and 3 real fleets | MAE **18.6 mV** (Quartz), **15.5 mV** (Deng fleet) | `data/validate_quartz.py` | **VALIDATED** |
| **P2** Lab degradation: does the stress-fatigue model capture cell degradation trajectory? | Palmgren-Miner + SEI calendar model on NASA PCoE (B0005/B0006/B0007/B0018; 1C, DoD≈100%) | Within-cell R²=**0.9725**, MAE=1.26% SOH; cross-cell R²=**−0.68** (batch variability) | [`nasa_degradation_report.json`](data/nasa_degradation_report.json) | **VALIDATED in lab regime; cross-cell limited by batch variability** |
| **P3** Cell-to-field bridge: transfer cell parameters to fleet SOH prediction | Palmgren-Miner convolution on 30,135 Deng sessions; 3-tier data separation; α calibration on V05–V09 | B0 (constant) RMSE=**0.0365** beats B1=**0.0453**; α=**−0.382** (degenerate) | [`cell_to_field_report.json`](data/cell_to_field_report.json) | **DIAGNOSED NEGATIVE** — calendar-dominated fleet; bridge does not beat constant |
| **P4** Per-vehicle adaptation: can gated per-vehicle λ recover from bridge failure? | 50/50 temporal split; LOO-transferred vs per-vehicle calendar OLS; gated B3' predictor | B3' trajectory RMSE **−10.1%** vs carry-forward (all 20 vehicles); **−15.4%** for 13 gate-in vehicles | [`cell_to_field_temporal_report.json`](data/cell_to_field_temporal_report.json) | **VALIDATED (gated)** — gate pass-rate 65%; 35% require longer observation window |
| **P5** Factor attribution: which operating factor (T, DoD, C-rate) drives per-vehicle fade rate? | Spearman ρ, Ridge LOO-CV, permutation importance on n=13 gated-in vehicles | All \|ρ\| < 0.40 (max=**0.268**); LOO R²=**−12.83** | [`factor_ranking_report.json`](data/factor_ranking_report.json) | **INDISTINGUISHABLE** — pre-registered expected outcome; fleet too homogeneous |

**P6 Diagnostic (follow-up to P4):** Gate failure mechanism quantified. Per-cycle i.i.d. SE(λ)≈0.0008 (already 40× below identifiability threshold). Systematic trajectory bias σ_eff=**48–102×** per-cycle noise is the actual gate-failure mechanism — not measurement precision. EIS/IC precision upgrades are irrelevant at n≈800 cycles. Operative lever: window extension r=**2×** (full 2.3 yr observation instead of 50/50 train split). Source: [`sensing_requirement_report.json`](data/sensing_requirement_report.json).

---

## Validated Results (real data only)

| Dataset | Layer | Metric | Value | Notes |
|---|---|---|---|---|
| Quartz WLTP (36-cell NMC811 pack) | DFN-SPM + EKF | R² (sensor-update rows) | **0.9217** | primary metric — genuinely new measurements |
| Quartz WLTP (36-cell NMC811 pack) | DFN-SPM + EKF | R² (all rows) | 0.9810 | inflated by repeated BMS readings; see note |
| Quartz WLTP (36-cell NMC811 pack) | DFN-SPM + EKF | MAE | **18.6 mV** | industry target < 20 mV ✅ |
| Quartz WLTP (36-cell NMC811 pack) | DFN-SPM + EKF | RMSE | 38.4 mV | |
| RWTH Aachen EIS (70 real spectra) | 2RC+CPE fit | R² | **0.9999** | offline curve fit on real spectra |
| NASA B0018 (122 held-out discharge cycles) | DualEKF | R² | **0.784** | empirical OCV; see limitation |
| NASA B0018 | DualEKF | MAE | **102 mV** | Sanyo LCO ≠ NMC811 OCP — chemistry mismatch |
| BMW i3 RDC (63 held-out trips) | DualEKF | MAE (scale-cal) | **35.6 mV** | real CAN data |
| BMW i3 RDC | DualEKF | SOC RMSE | 20.8 % | |
| Deng BAIC EU500 (2000 sessions, 20 vehicles) | DualEKF | MAE (scale-cal) | **15.5 mV** | real fleet charging data |
| Deng BAIC EU500 | DualEKF | SOC RMSE | 11.9 % | |
| VED Michigan (38 segments, 30 vehicles) | DualEKF | MAE (scale-cal) | **40.7 mV** | OBD-II resolution limited |
| DFN step time | benchmark | p99 latency | **47 µs/cell** | MacBook M-series single core |
| Deng BAIC EU500 (30,135 sessions, 20 vehicles) | Stress-Fatigue + SEI Degradation (Module 2) | MAE ΔSOH | **3.7% SOH** | SEI dominates; 6/16 held-out vehicles R²>0 |
| NASA PCoE B0005/B0006/B0007/B0018 (lab cycling) | Stress-Fatigue Module 2 (within-cell fit) | R² | **0.9725** | 1C, DoD≈100% lab regime where stress-fatigue applies |

**Note on Quartz R²:** The 0.9217 figure is on rows with genuinely new sensor readings (~17% of timestamps). The 0.9810 uses all rows — 83% are repeated BMS readings (6-min update interval) that are trivial to predict. We foreground 0.9217.

**Note on Module 2 (Stress-Fatigue + SEI Degradation — Deng field):** Two models were compared: stress-only (ΔSOH = β·D^γ) and combined (ΔSOH = β·D^γ + λ·√t). The SEI calendar term dominates: λ=0.026 SOH/√yr → ~4% fade at 2.3 years, while the stress term (D_final≈0.002) contributes ≈3.5×10⁻⁹ ΔSOH — negligible. The key data limitation: 2-year fade signal (~3.8% SOH) is comparable to per-session BMS capacity noise (~2.8% SOH std), giving SNR<1 for 8/20 vehicles and negative R² for most. MAE_ΔSOH ≈ 3.7% for both models — limited by data noise, not model choice. 5 vehicles show apparent capacity recovery (BMS recalibration or seasonal effects) which no monotone model can fit. The finding is the driver: SEI/calendar aging dominates over mechanical stress-fatigue for normal urban BAIC EU500 operation over 2 years.

**Note on Module 2 (Stress-Fatigue — NASA PCoE lab validation):** The same Miner damage model validated on controlled NASA lab cycling (B0005/B0006/B0007/B0018: 1C, DoD≈100%, 132–168 cycles, direct capacity measurement, SNR≈50–99). Within-cell fit R²=0.9725 (mean over 4 cells, MAE=1.3% SOH) confirms the model correctly captures degradation trajectory shape in the appropriate regime. Cross-cell generalisation is limited (R²=−0.68) by manufacturing batch variability: B0006 degrades 45% faster than B0005 at nearly identical cumulative damage D values, requiring per-cell β calibration. Key comparison: stress-fatigue R²=0.97 on controlled lab data (where it is the dominant mechanism) vs. R²=−1.8 on 2-year field data (where SEI/calendar aging dominates). This regime distinction is the central finding, consistent with Sulzer et al. (2021 Joule, DOI 10.1016/j.joule.2021.06.005).

**Note on NASA B0018:** The DualEKF uses an OCV function fitted empirically from 10 calibration discharge cycles (IR-drop compensation, not GITT). This introduces ~27 mV OCV approximation error. The Sanyo LCO chemistry (B0018) differs from the NMC811 DFN cartridge — so the DFN layer is not used directly here. MAE ~102 mV reflects both the OCV approximation error and the chemistry gap.

---

## Honest Limitations

**Modelling:**
- Stress-fatigue (Palmgren-Miner) only applies in the fatigue regime (DoD>60%, C-rate>1C, lab duration). For the Deng field fleet at 57% DoD, 0.41C mean, 2.3 yr, cycling damage is ≈3.5×10⁻⁹ SOH — negligible; calendar (SEI) dominates. R²=0.97 is reported for lab data only.
- Cross-cell β generalisation is limited by manufacturing batch variability (B0006 degrades 45% faster than B0005 at identical D values). Per-cell β calibration is required before deployment.
- λ_sei = 0.02639 SOH/√yr (M2 fit on V01–V04) overestimates fleet median by 1.43× when applied naïvely to all 20 vehicles. Per-vehicle λ_v fit is preferable but requires ≥1 yr of monotone-fade window (see P4 gate).

**Data:**
- Deng fleet: 2.3 yr observation window; vehicles entered the dataset already ~6% degraded (136.2 Ah observed max vs 145 Ah nominal per Deng 2023). Pre-dataset history unknown; absolute SOH reference is approximate.
- Endpoint SNR=1.70 (BORDERLINE) for fleet-aggregate degradation metrics; trend SNR=3.43. Per-vehicle metrics are more reliable than aggregate metrics for this fleet.
- 5 vehicles show apparent capacity recovery within the 2.3 yr window (likely BMS recalibration or seasonal temperature effects); no monotone model fits these.

**Gate (P4):**
- B3' gate pass-rate = 65% (13/20 vehicles). The 35% gated out are NOT hardware-limited. P6 analysis shows gate failure is systematic trajectory bias (σ_eff=48–102× per-cycle noise), not measurement imprecision. The fix is a longer observation window (~2× current), not EIS or IC analysis upgrades.
- 4 of the 7 gated-out vehicles have genuinely negative global SOH fade (V07, V12, V16, V17); these require >3 yr observation for disambiguation and are outside this dataset's scope.

**Prototype layers:**
- GraphSAGE GNN (Layer 2), NSA anomaly detector (Layer 4), ACO+Kuramoto (Layer 5) are implemented and unit-tested but have not been validated on real data. None of these appear in any reported metric.

---

## Architecture (6 Layers)

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 0 │  Segmented Chirp EIS                             │
│           │  8 s sweep · 0.1–1000 Hz                        │
│           │  Status: offline-validated on RWTH spectra       │
│           │          online sweep simulates synthetic cell ⚠ │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 │  2RC + CPE Parameter Extraction                  │
│           │  R_ohm · R_SEI · R_ct · D_s per cell            │
│           │  Status: VALIDATED — R²=0.9999 on 70 RWTH spectra│
├─────────────────────────────────────────────────────────────┤
│  Layer 2 │  GraphSAGE GNN  🔬 prototype                     │
│           │  64 → 32 → 16 · dual edge types                 │
│           │  Status: implemented; unit-tested on random data │
│           │          NOT used in any reported validation      │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 │  DFN Physics Engine + 5 TCOs + Dual EKF         │
│           │  Single-Particle Model · SEI growth · plating    │
│           │  Status: VALIDATED — MAE 18.6 mV, R²=0.9217     │
│           │          (Quartz WLTP 36-cell NMC811 pack)       │
├─────────────────────────────────────────────────────────────┤
│  Layer 4 │  NSA Weakest Cell Detection  🔬 prototype        │
│           │  Negative Selection · min-distance               │
│           │  Status: implemented; disabled in fleet runs     │
├─────────────────────────────────────────────────────────────┤
│  Layer 5 │  ACO + Kuramoto Action Engine  🔬 prototype      │
│           │  Ant Colony current routing                      │
│           │  Kuramoto SOC synchronisation                    │
│           │  Status: implemented; unit-tested only           │
└─────────────────────────────────────────────────────────────┘
```

### Layer 3 — TCO Constraints (Physics Safety)

| TCO | Constraint | Physical Basis |
|---|---|---|
| TCO-1 | Entropy production ≥ 0 | Clausius 2nd Law |
| TCO-2 | OCP drift ≤ 10 mV/step | Nernst tethering |
| TCO-3 | φ_neg ≥ plating limit | Li plating prevention |
| TCO-4 | SEI thickness monotone | SEI irreversibility |
| TCO-5 | Li conservation ≤ 1% | Faraday's law |

---

## Positioning vs Related Tools

> **Note:** PyBaMM is a forward physics *simulator* — a different category of tool. With calibrated parameters it is highly accurate for offline single-cell simulation. The table compares *deployment features* for online field estimation, not simulation fidelity. See `scripts/compare_pybamm_all.py`.

| Feature | PyBaMM | Commercial BMS | **OpenCATHODE Stack** |
|---|---|---|---|
| No BMS dependency (self-predicting EKF) | ❌ | ❌ | ✅ validated |
| ChirpEIS impedance (offline-validated) | ❌ | ❌ | ✅ validated |
| Real-time embedded BMS | ❌ (offline sim) | ✅ | ✅ (47 µs/cell p99) |
| 5-constraint physics safety (TCOs) | ❌ | ❌ | ✅ validated |
| Adaptive EKF (flat-plateau aware) | ❌ | ❌ | ✅ validated |
| Multi-cell GNN state estimation | ❌ | ❌ | 🔬 prototype — not validated |
| Kuramoto SOC synchronisation | ❌ | ❌ | 🔬 prototype — not validated |
| NSA weakest-cell detection | ❌ | ❌ | 🔬 prototype — not validated |
| Open source | ✅ | ❌ | ✅ |

---

## Repository Structure

```
opencathode-stack/
├── core/
│   └── dfn_cell.py          # DFN-SPM physics engine + 5 TCOs  [VALIDATED]
├── stack/
│   ├── gnn_layer.py         # GraphSAGE GNN (dual edge types)  [prototype]
│   └── pack_manager.py      # Pack orchestration + thermal network  [prototype]
├── diagnosis/
│   ├── dual_ekf_lfp.py      # Dual EKF — adaptive Q            [VALIDATED]
│   ├── weakest_cell.py      # NSA anomaly detector             [prototype]
│   └── ica_analysis.py      # Incremental capacity analysis    [prototype]
├── eis/
│   ├── eis_simulator.py     # 2RC+CPE impedance model          [VALIDATED offline]
│   └── chirp_eis.py         # Segmented chirp sweep generator  [offline validated; online = synthetic cell]
├── action/
│   └── policy_engine.py     # ACO routing + Kuramoto sync      [prototype]
├── deploy/
│   └── realtime_bms.py      # 1 Hz real-time BMS loop
├── degradation/
│   ├── deng_loader.py       # Load + clean Deng fleet, chemistry detect  [Module 2]
│   ├── cycle_segmentor.py   # Session segmentation (30-min gap)          [Module 2]
│   ├── stress_model.py      # DIS stress proxy (DoD × C-rate × Arrhenius)[Module 2]
│   ├── fatigue.py           # Rainflow + Palmgren-Miner damage            [Module 2]
│   └── soh_predictor.py     # D → SOH power-law, calibrated on V01–V04  [Module 2]
├── analysis/
│   ├── factor_ranking.py    # P5: Spearman ρ + Ridge LOO-CV + perm. imp. [P5]
│   └── sensing_requirement.py  # P6: Gate-failure mechanism diagnosis     [P6]
├── data/
│   ├── validate_quartz.py   # 36-cell WLTP validation script   [PRIMARY VALIDATION]
│   ├── validate_generic.py  # Fleet validation (VED/BMW/Deng)
│   ├── validate_deng_degradation.py  # Module 2 end-to-end runner
│   └── quartz_wltp/         # 10 × parquet files (634k rows, real data)
│   ├── bmw_i3/              # 70 CSV files, real trips (BMW CAN)
│   ├── deng20/              # 20 CSV files, real 20-vehicle fleet
│   ├── ved/                 # 54 CSV files, real Michigan fleet
│   └── nasa/
│       ├── B0018.mat        # Real NASA B0018 discharge data (132 cycles)
│       └── 5. Battery Data Set/1. BatteryAgingARC-FY08Q4.zip
├── validation/
│   ├── nasa_validator.py              # DualEKF on real NASA B0018 data
│   ├── nasa_degradation_validator.py  # Stress-fatigue model on NASA B0005/B0006/B0007/B0018
│   └── soh_noise_floor.py            # Fleet SOH signal-to-noise diagnostic
├── degradation/
│   ├── cell_to_field_bridge.py    # P3: Palmgren-Miner cell-to-field bridge
│   └── cell_to_field_temporal.py  # P4: Gated per-vehicle λ adaptation
├── SOURCES.md               # Full primary source bibliography
└── main.py                  # Full stack demo + benchmark
```

---

## Quick Start

```bash
git clone https://github.com/quant-himanshu/opencathode-stack.git
cd opencathode-stack
pip install -r requirements.txt

# Run 36-cell Quartz WLTP validation (primary, ~3 min)
python data/validate_quartz.py

# Run DualEKF on real NASA B0018 discharge data (~10 s)
python validation/nasa_validator.py

# Run fleet validation on real BMW i3 data (~30 s)
python data/validate_generic.py --dataset bmw_i3

# Run fleet validation on real Deng 20-vehicle data (~5 min)
python data/validate_generic.py --dataset deng

# Full stack demo
python main.py
```

**Requirements:** `numpy scipy torch pandas pyarrow natsort matplotlib` (see `requirements.txt`)

---

## Data Sources

| Dataset | Location | Size | Real data? |
|---|---|---|---|
| Quartz WLTP NMC811 pack | `data/quartz_wltp/` | 10 parquet files | ✅ |
| BMW i3 RDC trips | `data/bmw_i3/` | 70 CSV files | ✅ |
| Deng BAIC EU500 fleet | `data/deng20/` | 20 CSV files | ✅ |
| VED Michigan fleet | `data/ved/` | 54 CSV files | ✅ |
| NASA B0018 18650 cell | `data/nasa/B0018.mat` | 8 MB | ✅ |
| NASA PCoE B0005/B0006/B0007/B0018 | `data/nasa/5. Battery Data Set/` | zip archive | ✅ |
| RWTH Aachen EIS spectra | `data/rwth/parsed_eis_cells.csv` | 350 rows | ✅ |

---

## Physics References

| Module | Reference |
|---|---|
| DFN-SPM | Doyle M., Fuller T.F., Newman J. (1993) J. Electrochem. Soc. 140(6):1526 |
| SPM approximation | Richardson G. et al. (2020) J. Electrochem. Soc. 167:080542 |
| SEI growth (√t kinetics) | Pinson M.B. & Bazant M.Z. (2013) J. Electrochem. Soc. 160:A243 |
| NMC811 OCP | Chen C.-H. et al. (2020) J. Electrochem. Soc. 167:080534 |
| LFP OCP | Safari M. & Delacourt C. (2011) J. Electrochem. Soc. 158:A562; Prada E. et al. (2012) J. Electrochem. Soc. 159:A1508 |
| Arrhenius R_ohm | Nyman A. et al. (2008) Electrochim. Acta 53:6356 |
| Dual EKF | Plett G.L. (2004) J. Power Sources 134(2):262–276 |
| Adaptive EKF | Mikhak M. et al. (2024) PMC12936157 |
| Degradation modes (ICA/DVA) | Dubarry M. & Anseán D. (2022) Front. Energy Res. 10:1023555. DOI 10.3389/fenrg.2022.1023555 |
| Calendar vs cycle dominance (field) | Sulzer V. et al. (2021) Joule 5(8):1934–1955. DOI 10.1016/j.joule.2021.06.005 |
| Factor ranking (T > DoD > C-rate) | Edge J.S. et al. (2021) Phys. Chem. Chem. Phys. 23(14):8200–8221. DOI 10.1039/D1CP00359C |
| EIS dataset | Schäffer et al. (2024) Zenodo 6405084 (RWTH Aachen) |
| NASA cell dataset | Saha B. & Goebel K. (2009) NASA/TM-2007-214294 |
| Deng fleet dataset | Deng Z. et al. (2023) Applied Energy 339:120954. DOI 10.1016/j.apenergy.2023.120954 |

---

## Cite

```bibtex
@software{opencathode2026,
  title   = {OpenCATHODE Stack: Physics-Informed Battery Pack Management},
  author  = {Sharma, Himanshu},
  year    = {2026},
  url     = {https://github.com/quant-himanshu/opencathode-stack},
  note    = {MAE 18.6 mV, R²=0.9217 (sensor-update rows), 36-cell Quartz WLTP validated.
             Fleet: BMW i3 MAE 35.6 mV, Deng MAE 15.5 mV (real data).
             Prototype layers (GNN, NSA, ACO, Kuramoto) implemented but not end-to-end validated.}
}
```
