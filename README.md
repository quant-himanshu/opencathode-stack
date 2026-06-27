# OpenCATHODE Stack
**Physics-Informed Battery Pack Management System**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MAE](https://img.shields.io/badge/MAE-18.6mV-brightgreen)](data/validate_quartz.py)
[![R²](https://img.shields.io/badge/R²-0.9217-brightgreen)](data/validate_quartz.py)

A six-layer physics-informed BMS stack. The electrochemistry core (DFN-SPM, Dual EKF, EIS) is validated on real laboratory and field data. Pack-management extensions (GraphSAGE GNN, Negative Selection, ACO+Kuramoto) are implemented but not yet validated end-to-end on real data — they are marked **prototype** below.

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
| NASA B0018 | DualEKF | MAE | **102 mV** | Sanyo NMC ≠ NMC811 OCP — chemistry mismatch |
| BMW i3 RDC (63 held-out trips) | DualEKF | MAE (scale-cal) | **35.6 mV** | real CAN data |
| BMW i3 RDC | DualEKF | SOC RMSE | 20.8 % | |
| Deng BAIC EU500 (2000 sessions, 20 vehicles) | DualEKF | MAE (scale-cal) | **15.5 mV** | real fleet charging data |
| Deng BAIC EU500 | DualEKF | SOC RMSE | 11.9 % | |
| VED Michigan (38 segments, 30 vehicles) | DualEKF | MAE (scale-cal) | **40.7 mV** | OBD-II resolution limited |
| DFN step time | benchmark | p99 latency | **47 µs/cell** | MacBook M-series single core |
| Deng BAIC EU500 (30,135 sessions, 20 vehicles) | Stress-Fatigue + SEI Degradation (Module 2) | MAE ΔSOH | **3.7% SOH** | SEI dominates; 6/16 held-out vehicles R²>0 |

**Note on Quartz R²:** The 0.9217 figure is on rows with genuinely new sensor readings (~17% of timestamps). The 0.9810 uses all rows — 83% are repeated BMS readings (6-min update interval) that are trivial to predict. We foreground 0.9217.

**Note on Module 2 (Stress-Fatigue + SEI Degradation):** Two models were compared: stress-only (ΔSOH = β·D^γ) and combined (ΔSOH = β·D^γ + λ·√t). The SEI calendar term dominates: λ=0.026 SOH/√yr → ~4% fade at 2.3 years, while the stress term (D_final≈0.002) contributes ≈3.5×10⁻⁹ ΔSOH — negligible. The key data limitation: 2-year fade signal (~3.8% SOH) is comparable to per-session BMS capacity noise (~2.8% SOH std), giving SNR<1 for 8/20 vehicles and negative R² for most. MAE_ΔSOH ≈ 3.7% for both models — limited by data noise, not model choice. 5 vehicles show apparent capacity recovery (BMS recalibration or seasonal effects) which no monotone model can fit. The finding is the driver: SEI/calendar aging dominates over mechanical stress-fatigue for normal urban BAIC EU500 operation over 2 years.

**Note on NASA B0018:** The DualEKF uses an OCV function fitted empirically from 10 calibration discharge cycles (IR-drop compensation, not GITT). This introduces ~27 mV OCV approximation error. The Sanyo NMC chemistry (B0018) differs from the NMC811 DFN cartridge — so the DFN layer is not used directly here. MAE ~102 mV reflects both the OCV approximation error and the chemistry gap.

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
├── data/
│   ├── validate_quartz.py   # 36-cell WLTP validation script   [PRIMARY VALIDATION]
│   ├── validate_generic.py  # Fleet validation (VED/BMW/Deng)
│   ├── validate_deng_degradation.py  # Module 2 end-to-end runner
│   └── quartz_wltp/         # 10 × parquet files (634k rows, real data)
│   ├── bmw_i3/              # 70 CSV files, real trips (BMW CAN)
│   ├── deng20/              # 20 CSV files, real 20-vehicle fleet
│   ├── ved/                 # 54 CSV files, real Michigan fleet
│   └── nasa/
│       └── B0018.mat        # Real NASA B0018 discharge data (132 cycles)
├── validation/
│   └── nasa_validator.py    # DualEKF on real NASA B0018 data
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
| RWTH Aachen EIS spectra | `data/rwth/parsed_eis_cells.csv` | 350 rows | ✅ |

---

## Physics References

| Module | Reference |
|---|---|
| DFN-SPM | Doyle, Fuller, Newman (1993) J. Electrochem. Soc. 140(6):1526 |
| SPM approximation | Richardson et al. (2020) J. Electrochem. Soc. 167:080542 |
| SEI growth | Pinson & Bazant (2013) J. Electrochem. Soc. 160:A243 |
| NMC811 OCP | Chen et al. (2020) J. Electrochem. Soc. 167:080534 |
| LFP OCP | Safari & Delacourt (2011); Prada et al. (2012) J. Electrochem. Soc. 159:A1508 |
| Arrhenius R_ohm | Nyman et al. (2008) Electrochim. Acta 53:6356 |
| Adaptive EKF | Mikhak et al. (2024) PMC12936157 |
| Degradation modes | Dubarry & Anseán (2022) J. Power Sources Adv. 100049 |
| EIS dataset | Schäffer et al. (2024) Zenodo 6405084 (RWTH Aachen) |
| NASA cell dataset | Saha & Goebel (2009) NASA Ames Research Center |

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
