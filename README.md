# OpenCATHODE Stack
**Physics-Informed Battery Pack Management System**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Validation](https://img.shields.io/badge/MAE-18.6mV-brightgreen)](data/validate_quartz.py)
[![R²](https://img.shields.io/badge/R²-0.9217-brightgreen)](data/validate_quartz.py)

A six-layer physics-informed BMS stack with validated core (DFN-SPM electrochemistry, Dual EKF SOC tracking, EIS impedance calibration) and prototype pack-management extensions (GraphSAGE GNN multi-cell estimation, Negative Selection anomaly detection, ACO+Kuramoto action control) — validated on 12,690 independent 20 s windows (634,450 raw rows) from a 36-cell NMC automotive WLTP pack.

---

## Key Results

| Metric | Value | Notes |
|---|---|---|
| Voltage MAE | **18.6 mV** (RMSE 38.4 mV) | Industry standard < 20 mV ✅ |
| R² (sensor-update rows) | **0.9217** across 36 cells | **primary metric** — genuinely new measurements |
| R² (all rows) | 0.9810 | secondary — inflated by repeated readings (see note) |
| EKF mode | **Self-predicting** — no forced BMS SOC input | — |
| Dataset | **12,690 independent 20 s windows** (from 634,450 raw rows) | Quartz WLTP, 36 cells |
| EIS fit R² | **0.9999** | RWTH Aachen, offline 2RC+CPE fit |
| Step time | **54 µs/cell** p99 | Real-time 1 Hz ✅ |

---

## Architecture (6 Layers)

```
┌─────────────────────────────────────────────────────────┐
│  Layer 0 │  Segmented Chirp EIS                         │
│           │  8 s sweep · 0.1–1000 Hz · online           │
├─────────────────────────────────────────────────────────┤
│  Layer 1 │  2RC + CPE Parameter Extraction              │
│           │  R_ohm · R_SEI · R_ct · D_s per cell        │
├─────────────────────────────────────────────────────────┤
│  Layer 2 │  GraphSAGE GNN                               │
│           │  64 → 32 → 16 · dual edge types             │
│           │  (electrical series · thermal coupling)      │
├─────────────────────────────────────────────────────────┤
│  Layer 3 │  DFN Physics Engine                          │
│           │  Single-Particle Model + 5 TCOs             │
│           │  + Dual EKF SOC (no forced BMS SOC)         │
├─────────────────────────────────────────────────────────┤
│  Layer 4 │  NSA Weakest Cell Detection                  │
│           │  Negative Selection · min-distance          │
├─────────────────────────────────────────────────────────┤
│  Layer 5 │  ACO + Kuramoto Action Engine                │
│           │  Ant Colony current routing                  │
│           │  Kuramoto SOC synchronisation               │
└─────────────────────────────────────────────────────────┘
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

## Validation

### Quartz WLTP — NMC Automotive Pack

| Field | Value |
|---|---|
| Dataset | Quartz WLTP (Universitat Politècnica de Catalunya) |
| Pack topology | 36 cells (3P × 12S) |
| Cell chemistry | NMC811 (confirmed, V_ocv ≈ 4.19 V at SOC 98%) |
| Raw datapoints | 634,450 rows · 70.5 h · 10 WLTP cycles |
| Resampled | 12,690 × 20 s windows |
| **R² (sensor-update rows)** | **0.9217** (primary metric) |
| **R² (all rows)** | 0.9810 (all rows — inflated by repeated readings) |
| **MAE** | **18.6 mV** |
| RMSE | 38.4 mV |
| Cells R² > 0.90 | 36 / 36 |
| EKF converged | 36 / 36 |
| SOC mode | Self-predicting EKF — no BMS SOC forced |
| Temperature gradient ΔT | 5.59 °C (P3S2 = 41.4 °C → P1S7 = 35.8 °C) |

**Note on R²:** The primary metric is **0.9217**, on sensor-update rows (timestamps with genuinely new measurements). The all-rows 0.9810 is reported for completeness but is inflated because ~83% of timestamps are repeated sensor readings (6-min BMS update interval) that are trivial to predict. We foreground the conservative 0.9217.

### EIS — RWTH Aachen

| Field | Value |
|---|---|
| Dataset | RWTH Aachen (Zenodo: 6405084) |
| Real spectra | 70 |
| Frequency range | 0.01 – 10,000 Hz |
| Fit model | 2RC + CPE (Warburg) |
| **R²** | **0.9999** |

---

## Positioning vs Related Tools

> **Note:** PyBaMM is a forward physics *simulator* — a different category of tool. With calibrated parameters it is highly accurate for offline cell simulation (more accurate than OpenCATHODE on clean single-trip SOC). The table compares *deployment* features for online field estimation, not simulation fidelity. See `scripts/compare_pybamm_all.py`.

| Feature | PyBaMM | Commercial BMS | **OpenCATHODE Stack** |
|---|---|---|---|
| Multi-cell GNN state estimation | ❌ | ❌ | 🔬 prototype |
| No BMS dependency (self-predicting EKF) | ❌ | ❌ | ✅ |
| ChirpEIS impedance (offline-validated) | ❌ | ❌ | ✅ |
| Real-time embedded BMS | ❌ (offline simulator) | ✅ | ✅ (54 µs/cell) |
| 5-constraint physics safety (TCOs) | ❌ | ❌ | ✅ |
| Adaptive EKF (flat-plateau aware) | ❌ | ❌ | ✅ |
| Kuramoto SOC synchronisation | ❌ | ❌ | 🔬 prototype |
| Open source | ✅ | ❌ | ✅ |

---

## Repository Structure

```
opencathode-stack/
├── core/
│   └── dfn_cell.py          # DFN-SPM physics engine + 5 TCOs
├── stack/
│   ├── gnn_layer.py         # GraphSAGE GNN (dual edge types)
│   └── pack_manager.py      # Pack orchestration + thermal network
├── diagnosis/
│   ├── dual_ekf_lfp.py      # Dual EKF — adaptive Q, Prada 2012 OCV
│   ├── weakest_cell.py      # NSA anomaly detector
│   └── ica_analysis.py      # Incremental capacity analysis
├── eis/
│   ├── eis_simulator.py     # 2RC+CPE impedance model
│   └── chirp_eis.py         # Segmented chirp sweep generator
├── action/
│   └── policy_engine.py     # ACO routing + Kuramoto sync
├── deploy/
│   └── realtime_bms.py      # 1 Hz real-time BMS loop
├── data/
│   ├── validate_quartz.py   # 36-cell WLTP validation script
│   └── quartz_wltp/         # 10 × parquet files (634k rows)
├── validation/
│   └── nasa_validator.py    # NASA aging dataset validator
└── main.py                  # Full stack demo + benchmark
```

---

## Quick Start

```bash
git clone https://github.com/quant-himanshu/opencathode-stack.git
cd opencathode-stack
pip install -r requirements.txt

# Run full stack demo
python main.py

# Run 36-cell Quartz WLTP validation
python data/validate_quartz.py

# Run standalone EKF validation
python diagnosis/dual_ekf_lfp.py

# Reproduce headline fleet validation (~30 min, requires VED/BMW/Deng data)
python -u data/validate_generic.py --all
python scripts/audit_independent.py
```

**Requirements:** `numpy scipy torch pandas pyarrow natsort matplotlib` (see `requirements.txt`)

---

## Physics References

| Module | Reference |
|---|---|
| DFN-SPM | Doyle, Fuller, Newman (1993) J. Electrochem. Soc. 140(6):1526 |
| SPM approximation | Richardson et al. (2020) J. Electrochem. Soc. 167:080542 |
| SEI growth | Pinson & Bazant (2013) J. Electrochem. Soc. 160:A243 |
| NMC811 OCP | Chen et al. (2020) J. Electrochem. Soc. 167:080534 |
| LFP OCP | Safari & Delacourt (2011); **Prada et al. (2012)** J. Electrochem. Soc. 159:A1508 |
| LFP cell model | Safari (2011) J. Electrochem. Soc. 158:A562 |
| Arrhenius R_ohm | Nyman et al. (2008) Electrochim. Acta 53:6356 |
| Adaptive EKF | **Mikhak et al. (2024)** PMC12936157 — RMSE < 0.15% |
| Degradation modes | **Dubarry & Anseán (2022)** J. Power Sources Adv. 100049 |
| IC-method OCV | **Gao & Onori (2025)** |
| EIS dataset | **Schäffer et al. (2024)** Zenodo 6405084 (RWTH Aachen) |

---

## Cite

```bibtex
@software{opencathode2026,
  title   = {OpenCATHODE Stack: Physics-Informed Battery Pack Management},
  author  = {Sharma, Himanshu},
  year    = {2026},
  url     = {https://github.com/quant-himanshu/opencathode-stack},
  note    = {MAE 18.6 mV, R²=0.9217 (sensor-update rows), 36-cell Quartz WLTP validated}
}
```
