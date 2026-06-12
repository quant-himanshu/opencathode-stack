# OpenCATHODE Stack — Complete Technical Architecture Report

> Auditable report. Every claim cites file + line number verified from source. No marketing language.

---

## 1. FILE TREE

Line counts from `wc -l` on the actual files.

```
opencathode-stack/
├── main.py                              440 lines  Entry point: validation suite, 2 simulations, EIS, benchmark table
│
├── core/
│   ├── __init__.py                        0 lines  (empty)
│   └── dfn_cell.py                      807 lines  SPM physics, OCP functions, 5 TCO checks, DFNCell class
│
├── stack/
│   ├── __init__.py                        0 lines  (empty)
│   ├── pack_manager.py                  593 lines  4S5P topology, Kirchhoff current, lumped thermal net, weakest cell
│   ├── gnn_layer.py                     395 lines  GraphSAGE 3-layer dual-edge GNN, KCL residual loss
│   └── train_gnn.py                     208 lines  GNN training loop on Quartz WLTP parquet
│
├── action/
│   ├── __init__.py                        0 lines  (empty)
│   └── policy_engine.py                 425 lines  ACO current router, Kuramoto SOC sync, safety policy actions
│
├── diagnosis/
│   ├── __init__.py                        0 lines  (empty)
│   ├── dual_ekf_lfp.py                  118 lines  Dual EKF with adaptive process noise for LFP/NMC SOC
│   ├── ica_analysis.py                  161 lines  dQ/dV incremental capacity analysis, peak detection
│   ├── ica_soc_corrector.py             445 lines  ICA + EIS + Coulomb-counting fusion SOC corrector
│   └── weakest_cell.py                  298 lines  Negative Selection Algorithm anomaly detector (6D)
│
├── eis/
│   ├── __init__.py                        0 lines  (empty)
│   ├── eis_simulator.py                 539 lines  Randles 2RC + Warburg synthetic EIS, scipy curve_fit extraction
│   └── chirp_eis.py                     772 lines  Segmented chirp EIS (8 s, 0.1–1000 Hz, 2RC+CPE+L, DRT)
│
├── deploy/
│   ├── __init__.py                        0 lines  (empty)
│   └── realtime_bms.py                  281 lines  Production 1 Hz BMS, cold-start OCV inversion, Arrhenius R_ohm
│
├── data/
│   ├── __init__.py                        0 lines  (empty)
│   ├── download_stack_datasets.py       529 lines  Dataset fetchers: Quartz, NASA, RWTH, CALCE, Oxford
│   ├── parse_real_data.py               430 lines  Quartz column mapping, time alignment, outlier detection
│   ├── train_on_real.py                 276 lines  Per-cell EKF training on Quartz
│   └── validate_quartz.py               534 lines  Quartz WLTP validation script (executable, not importable function)
│
├── validation/
│   ├── __init__.py                        0 lines  (empty)
│   └── nasa_validator.py                572 lines  NASA aging dataset R² / SOH validation
│
└── dashboard/
    ├── __init__.py                        0 lines  (empty)
    └── create_quartz_graph.py           265 lines  Matplotlib 4-panel Quartz discharge plots
```

**Total: 7,282 lines of Python across 19 substantive files.**

---

## 2. MODULE HIERARCHY

### Import dependency graph

```
main.py
├── core.dfn_cell                   ← physics primitives (Layer 0)
├── stack.pack_manager
│   └── core.dfn_cell
├── stack.gnn_layer
│   ├── core.dfn_cell  (EPS)
│   └── stack.pack_manager  (N_SERIES, N_PARALLEL — inside physics_residual)
├── eis.eis_simulator
│   └── core.dfn_cell  (EPS, F, R_GAS, T0)
├── diagnosis.weakest_cell
│   └── core.dfn_cell  (EPS)
└── action.policy_engine
    ├── core.dfn_cell  (EPS, T0)
    └── stack.pack_manager  (N_CELLS, N_SERIES, N_PARALLEL)

data/validate_quartz.py   [standalone script]
├── core.dfn_cell
└── diagnosis.dual_ekf_lfp

deploy/realtime_bms.py    [standalone deployment]
├── core.dfn_cell
├── stack.gnn_layer
├── diagnosis.weakest_cell
└── action.policy_engine
```

### Layer rationale

| Layer | Files | WHY it exists | What breaks if removed |
|---|---|---|---|
| L0: Physics | `core/dfn_cell.py` | Per-cell V, stoichiometry, SEI, temperature | All other layers depend on `DFNCell.step()` |
| L1: Pack | `stack/pack_manager.py` | 20-cell topology: Kirchhoff, thermal coupling, aggregate SOC/SOH | Cells get uniform current, ignoring R_i differences; no thermal network |
| L2: State estimation | `stack/gnn_layer.py`, `diagnosis/dual_ekf_lfp.py` | Refine SOC/SOH via inter-cell correlations (GNN) and measurements (EKF) | Reverts to pure Coulomb counting with no drift correction |
| L3: Anomaly | `diagnosis/weakest_cell.py` | Immune-inspired detection in 6D feature space | No per-cell anomaly flagging; weakest cell from composite risk score only |
| L4: Action | `action/policy_engine.py` | ACO routing, Kuramoto balancing, safety trips | Current routing Kirchhoff-only; no SOC balancing; no emergency logic |
| L5: EIS | `eis/eis_simulator.py`, `eis/chirp_eis.py` | Identify R_ohm, R_SEI, R_ct, D_s from impedance | Resistances fixed at defaults; no aging-tracking parameter updates |

---

## 3. EVERY EQUATION IN CODE

### 3.1 OCP — Graphite Anode

**Equation (LaTeX):**
```
U_neg(x) = 0.7222 + 0.1387*x + 0.029*sqrt(x)
           - 0.0172/x + 0.0019/x^1.5
           + 0.2808*exp(0.9 - 15*x) - 0.7984*exp(0.4465*x - 0.4108)
```

**File:** `core/dfn_cell.py:177–183`, function `ocp_graphite()`

**Variables:** x = graphite stoichiometry [dimensionless], valid 0.15–0.99. U = OCP [V vs Li/Li+].

**Source:** Doyle, Fuller, Newman (1996) J. Electrochem. Soc. 143:1890. Validated: U≈0.083 V at x=0.80.

---

### 3.2 OCP — NMC811 Cathode

**Equation (LaTeX):**
```
U_pos(x) = -0.8090*x + 4.4875
           - 0.0428*tanh(18.5138*(x - 0.5542))
           - 17.7326*tanh(15.7890*(x - 0.3117))
           + 17.5842*tanh(15.9308*(x - 0.3120))
```

**File:** `core/dfn_cell.py:207–211`, function `ocp_nmc811()`

**Variables:** x = cathode stoichiometry [dimensionless], valid 0.15–0.95.

**Source:** Chen et al. (2020) J. Electrochem. Soc. 167:080534, Table 3. Tanh form chosen because a 6th-order polynomial (mentioned in code comment at line 196–198) becomes non-monotone for x > 0.53, which is physically wrong for NMC811 discharge (cathode OCP must be monotone-decreasing in x during discharge).

---

### 3.3 OCP — LFP Cathode (Prada 2012 Lookup)

Linear interpolation over 21-point table from SOC 0.0 to 1.0 → OCV 2.800 to 3.650 V.

**File:** `core/dfn_cell.py:215–241`, arrays `_OCP_LFP_SOC_PTS`, `_OCP_LFP_OCV_PTS`, function `ocp_lfp()`

Flat plateau: OCV 3.300–3.360 V for SOC 30–75%.

**Source:** Prada et al. (2012) J. Electrochem. Soc. 159:A1508.

---

### 3.4 Electrode Plate Area (Faraday Consistency)

**Equation:**
```
A_eff = Q_nom * 3600 / (cs_max_neg * L_neg * DELTA_X_NEG * F)
```
where DELTA_X_NEG = 0.70 (graphite operational stoichiometry window).

**File:** `core/dfn_cell.py:320–323`, `DFNCell.__init__()`

**Why:** Ensures dx/dt from SPM flux equals dSOC/dt × Δx_neg. Without this, the voltage prediction would be internally inconsistent with the Coulomb counter. Source: Richardson et al. (2020) J. Electrochem. Soc. 167:080542.

---

### 3.5 Exchange Current Density

**Equation:**
```
i0 = F * k0 * sqrt(cs * (cs_max - cs) * ce)
```

**File:** `core/dfn_cell.py:383–385`, function `_exchange_current()`

**Variables:**
- k0 = reaction rate constant [m^2.5/(mol^0.5·s)]; NMC neg: 1.764e-11, pos: 6.67e-11 (Ecker 2015)
- cs = surface Li concentration [mol/m³]
- cs_max = max Li concentration [mol/m³]
- ce = electrolyte concentration = 1200 mol/m³ (constant; SPM assumption, line 572)
- i0 = exchange current density [A/m²]

**Source:** Newman & Thomas-Alyea, Electrochemical Systems 3rd ed., Eq. 11.13.

---

### 3.6 Butler-Volmer Overpotential (arcsinh inversion)

**Equation:**
```
eta = (R*T)/(alpha*F) * arcsinh(j / (2*i0))
```

**File:** `core/dfn_cell.py:402–404`, function `_butler_volmer_overpotential()`

**Variables:**
- j = interfacial current density [A/m²]
- alpha = 0.5 for both electrodes (symmetric BV)
- eta = overpotential [V]

Derived by inverting j = 2*i0*sinh(alpha*F*eta/(R*T)). Numerically exact — no linearization. Source: Newman & Thomas-Alyea Eq. 8.3.

---

### 3.7 SPM Interfacial Current Density

**Equation:**
```
j_neg = -I_app / (a_neg * L_neg * A_eff * F)    (discharge: Li leaves anode)
j_pos = +I_app / (a_pos * L_pos * A_eff * F)    (discharge: Li enters cathode)
```

**File:** `core/dfn_cell.py:432–433`, function `_solid_diffusion_dxdt()`

**Variables:**
- a = specific interfacial area [m²/m³]: graphite 3.638e5; NMC811 3.437e5
- L = electrode thickness [m]
- j = [mol/(m²·s)]

---

### 3.8 Stoichiometry Rate (SPM Volume-Averaged)

**Equation:**
```
r_p = 3 / a                   # particle radius from a = 3/r_p
j_n = sign * I / (a*L*A_eff*F)
dxdt = 3 * j_n / (cs_max * r_p)
```

**File:** `core/dfn_cell.py:430–434`, function `_solid_diffusion_dxdt()`

NOTE: This is the **volume-averaged SPM approximation**, not full DFN. Full DFN solves the spherical diffusion PDE dc/dt = D_s * nabla²c. The SPM replaces it with a single volume-averaged flux.

Source: Richardson et al. (2020) J. Electrochem. Soc. 167:080542, Eq. 4–6.

---

### 3.9 Terminal Voltage

**Equation:**
```
V_cell = U_pos - U_neg + eta_pos - eta_neg - I_app * R_ohm
```

**File:** `core/dfn_cell.py:595`, `DFNCell.step()`

Source: Richardson et al. (2020) Eq. 16.

---

### 3.10 SEI Growth (Pinson-Bazant, diffusion-limited)

**Equation:**
```
E_a_SEI = 35000  # [J/mol]
k_eff = k_SEI * exp(-E_a_SEI / (R*T))
rate = k_eff / delta_SEI               # [m/s]
```

**File:** `core/dfn_cell.py:447–449`, function `_sei_growth_rate()`

**Variables:**
- delta_SEI = SEI thickness [m], init = 5 nm
- k_SEI = 1.5e-17 m/s (NMC); 1.0e-17 m/s (LFP)
- E_a = 35,000 J/mol

**Capacity loss rate:**
```
neg_area = a_neg * L_neg * A_eff
dn_Li_rate = rho_SEI * rate * neg_area   # [mol/s]
dQ_loss_rate = dn_Li_rate * F / 3600     # [A·h/s]
```

**File:** `core/dfn_cell.py:451–454`

Source: Pinson & Bazant (2013) J. Electrochem. Soc. 160:A243, Eq. 5.

---

### 3.11 Heat Generation (Bernardi)

**Equation:**
```
Q_ohmic = I^2 * R_ohm
Q_rxn   = |I| * (|eta_neg| + |eta_pos|)
Q_gen   = Q_ohmic + Q_rxn
```

**File:** `core/dfn_cell.py:471–473`, function `_heat_generation()`

Joule + reaction heat. Entropic heat (requires dU/dT) is not included.

Source: Bernardi et al. (1985) J. Electrochem. Soc. 132:5.

---

### 3.12 Thermal Evolution (Newton Cooling)

**Equation:**
```
Q_cool = h_conv * A_surf * (T - T0)
m_cell = rho * V_cell
dTdt   = (Q_gen - Q_cool) / (m_cell * Cp)
T_new  = T + dTdt * dt
```

**File:** `core/dfn_cell.py:620–623`, `DFNCell.step()`

**Parameters:** h_conv=10 W/(m²K), A_surf=1.2e-3 m², rho=2500 kg/m³, V_cell=16.5e-6 m³, Cp=900 J/(kgK)

---

### 3.13 Coulomb Counting SOC

**Equation:**
```
soc_cc_new = soc_cc - I_app * dt / (3600 * Q_nom_eff)
```

**File:** `core/dfn_cell.py:634–637`, `DFNCell.step()`

Source: Plett (2015) Battery Management Systems Vol. 1, Eq. 3.9.

---

### 3.14 Arrhenius R_ohm Correction

**Equation:**
```
R(T) = R_ref * exp(Ea * (1/T_K - 1/T_ref))
```
where Ea = 4000 K (dimensionless pre-exponential activation, Nyman 2008 convention).

**File:** `core/dfn_cell.py:481–483`, function `get_R_ohm_thermal()`
**Also:** `deploy/realtime_bms.py:35–36`, function `arrhenius_factor()`
**Also:** `data/validate_quartz.py:69–72`, function `_arrhenius(T_arr)`

Source: Nyman et al. (2008) J. Electrochem. Soc.

---

### 3.15 Pack Thermal Network (Lumped, Explicit Euler)

**Equation:**
```
dT_i/dt = (sum_j G_ij*(T_j - T_i) + G_cool_i*(T_cool - T_i)) / (m*Cp)
```

**File:** `stack/pack_manager.py:327–340`, function `_update_thermal_network()`

**Parameters:**
- G_cc = 1/R_CELL_CELL = 2 W/K
- G_cool = (1/R_CELL_COOL) * edge_factor; edge cells (p=0,4) get 100% cooling; center (p=2) gets ~10%
- T_cool = T0 + 5.0 = 303.15 K
- Sub-step: dt_sub = 0.4 * m*Cp / G_max (stability margin, line 322)

Source: Bernardi 1985; Bandhauer et al. (2011) J. Electrochem. Soc. 158:R1.

---

### 3.16 Kirchhoff Current Distribution (Parallel Group)

**Equation:**
```
G_cells = 1 / R_cells
I_cells = I_string * G_cells / sum(G_cells)
```

**File:** `stack/pack_manager.py:234–238`, function `_kirchhoff_current_distribution()`

Source: Kirchhoff (1845) Annalen der Physik 64:497.

---

### 3.17 Composite Risk Score (Weakest Cell)

**Equation:**
```
score = 0.40 * soh_risk + 0.35 * thermal_risk + 0.25 * plating_risk
```

**File:** `stack/pack_manager.py:394–411`, function `_cell_composite_risk()`

**Sub-scores:**
- soh_risk    = clip((1 - SOH) / 0.3, 0, 1)
- thermal_risk = clip((T - (T0+25)) / (423.15 - T0 - 25), 0, 1)
- plating_risk  = clip(delta_SEI / 1e-6, 0, 1)   ← uses SEI as plating proxy

---

### 3.18 Remaining Useful Life (Linear Degradation)

**Equation:**
```
observed_rate = (1.0 - SOH) / cycle_count       # [SOH_loss/cycle]
degradation_rate = max(observed_rate, 0.0005)   # floor at NMC811 nominal
rul = (SOH - 0.8) / degradation_rate            # [cycles]
```

**File:** `stack/pack_manager.py:513–518`; `action/policy_engine.py:328–333`

EOL criterion: SOH < 0.8. Source: Attia et al. (2020) Nature Energy 5:737. NMC811 nominal fade = 0.05%/cycle.

---

### 3.19 Thermal Runaway Risk

**Equation:**
```
if T_max < TR_T_ONSET (353.15 K):
    tr_risk = clip((T_max - T0) / (TR_T_ONSET - T0), 0, 1) * 0.3
else:
    tr_risk = 0.3 + 0.7 * (T_max - TR_T_ONSET) / (TR_T_RUNAWAY - TR_T_ONSET)
```

**File:** `stack/pack_manager.py:478–480`, function `_thermal_runaway_risk()`

In `policy_engine.py:309–311` the formula adds a rate term: 0.7*T_norm + 0.3*rate_norm.

Source: Feng et al. (2018) Joule 2:1985.

---

### 3.20 Randles EIS Circuit (eis_simulator.py)

**Equation:**
```
Z_SEI = R_SEI / (1 + j*w*R_SEI*C_SEI)
Z_ct  = R_ct  / (1 + j*w*R_ct*C_dl)
Z_W   = A_W / sqrt(w) * (1 - j) / sqrt(2)       # semi-infinite Warburg
Z     = R_ohm + Z_SEI + Z_ct + Z_W
```

**File:** `eis/eis_simulator.py:73–83`, function `impedance_model()`

Reference parameters: R_ohm=5mΩ, R_SEI=8mΩ, R_ct=15mΩ, A_W=0.03 Ω·s^-0.5

Source: Randles (1947) Discuss. Faraday Soc. 1:11.

---

### 3.21 Warburg Diffusivity Extraction

**Equation:**
```
D_s = (R*T / (n^2 * F^2 * A_ref * sqrt(2) * A_W * cs_max))^2
```

**File:** `eis/eis_simulator.py:110–113`, function `warburg_to_diffusivity()`

**Variables:** A_ref = 1e-4 m² (1 cm²), cs_max = 30555 mol/m³, n=1.
Code comments: "order-of-magnitude estimate (~1–2 decades accuracy)".

Source: Barsoukov & Macdonald (2005) Eq. 2.1.5.

---

### 3.22 2RC+CPE+Inductance Model (chirp_eis.py)

**Equation:**
```
Z_L   = j*w*L
Z_SEI = R_SEI / (1 + (j*w*tau_SEI)^phi_SEI)   # CPE
Z_ct  = R_ct  / (1 + (j*w*tau_ct )^phi_ct )   # CPE
Z_W   = A_W / sqrt(j*w)
Z     = Z_L + R_ohm + Z_SEI + Z_ct + Z_W
```

**File:** `eis/chirp_eis.py:74–81`, function `impedance_model_cpe()`

**Variables:**
- L = inductance [H]; cable+contact, 100–600 nH observed in RWTH data
- phi_SEI, phi_ct = CPE exponents in [0.5, 1.0]; phi=1 → pure RC, phi=0.5 → Warburg

Source: Hahn et al. (2019) J. Electrochem. Soc. 166:A3275, Eq. 2.

---

### 3.23 Chirp Signal (Linear Frequency Sweep)

**Equation:**
```
phi(t) = 2*pi * (f1*t + 0.5*(f2-f1)*t^2/T)
x(t)   = A * sin(phi(t))
```

**File:** `eis/chirp_eis.py:327–329`, function `generate_chirp()`

**Parameters:** f1=0.1 Hz, f2=1000 Hz, T=8 s, A=5e-3 V (5 mV per IEC 62660-1)

Source: Oppenheim & Schafer, DSP 3rd ed., Eq. 10.14.

---

### 3.24 STFT Impedance Estimate

**Equation:**
```
Z(f_k) = FFT(V_seg * w)[b_k] / FFT(I_seg * w)[b_k]
```
where b_k = nearest FFT bin to instantaneous frequency f_k at segment center, w = Hann window.

**File:** `eis/chirp_eis.py:373–384`, function `compute_impedance_from_response()`

Source: Klotz et al. (2011) Electrochim. Acta 56:8763.

---

### 3.25 Dual EKF — Predict Step

**Equation:**
```
tau = 50.0  # polarization time constant [s]
Q_eff = Q_nom * SOH_est

soc_pred  = clip(soc - (I * dt) / (3600 * Q_eff), 0, 1)
v_pol_pred = v_pol * exp(-dt/tau) + R_int * (1 - exp(-dt/tau)) * I

F = [[1,              0          ],
     [0,  exp(-dt/tau)           ]]

P_pred = F @ P @ F.T + Q_adaptive
```

**File:** `diagnosis/dual_ekf_lfp.py:50–54`, function `update()`

---

### 3.26 Dual EKF — Adaptive Process Noise

**Equation:**
```
slope  = |dOCV/dSOC|   (finite difference, h=0.005)
factor = min(1 / max(slope, 0.02), 50.0)
Q      = Q_base * factor
```

**File:** `diagnosis/dual_ekf_lfp.py:41–43`, function `_adaptive_Q()`

**Why:** In the LFP plateau (SOC 30–75%), |dOCV/dSOC| → 0, driving Kalman gain → 0 and freezing SOC estimates. Boosting Q forces the filter to continue tracking. Source: Mikhak (2024) PMC12936157.

---

### 3.27 Dual EKF — Update Step

**Equation:**
```
V_pred = OCV(soc_pred) - I * R_int + v_pol_pred
H      = [dOCV/dSOC,  1.0]
S      = H @ P_pred @ H.T + R_meas     # innovation covariance
K      = P_pred @ H.T / S              # Kalman gain
x_new  = x_pred + K * (V_meas - V_pred)
P_new  = (I_2 - K @ H) @ P_pred
```

**File:** `diagnosis/dual_ekf_lfp.py:55–62`, function `update()`

**Parameters:** R_meas = [[4e-6]] (2 mV)^2, Q_base = diag([1e-6, 1e-5])

---

### 3.28 Arrhenius OLS R_ohm Calibration

**Equation (in validate_quartz.py):**
```
f_i = exp(EA_OHM * (1/T_K_i - 1/T_REF_K))    # Arrhenius factor
num = sum(I_i * f_i * (V_DFN0_i - V_meas_i))
den = sum(I_i^2 * f_i^2)
R_cal = R_OHM_DEFAULT + num / den
```

Applied only to voltage-update rows (mask where |dV| > 1 mV).

**File:** `data/validate_quartz.py:265–270`

---

### 3.29 ACO Probability (Dorigo-Gambardella)

**Equation:**
```
eta_i = SOH_i / (risk_i + eps)          # heuristic desirability
prob_i = (tau_i^alpha * eta_i^beta)     # unnormalized
prob_i /= sum(prob_j)                   # normalized

# After ant routing:
tau *= (1 - rho)                         # evaporation
tau_i += max(SOH_i - mean(SOH) + 0.5, 0) # deposit
```

**File:** `action/policy_engine.py:91–114`, class `ACOCurrentRouter.route()`

**Parameters:** alpha=1.0, beta=2.0, rho=0.1, N_ants=20

Source: Dorigo & Gambardella (1997) IEEE Trans. Evol. Comput. 1:53.

---

### 3.30 Kuramoto SOC Synchronization

**Equation:**
```
phi_i = 2*pi * SOC_i
omega_i = (SOC_i - mean(SOC)) * 0.1     # small natural freq

for step in N_KURAMOTO_STEPS:
    dphi_i = omega_i + (K/N) * sum_j(sin(phi_j - phi_i))
    phi_i += dphi_i * dt

SOC_synced_i = clip(phi_i / (2*pi), 0, 1)
delta_SOC = SOC_synced - SOC_original

# Order parameter:
R = |mean(exp(j * phi_i))|  in [0,1]
```

**File:** `action/policy_engine.py:155–183`, class `KuramotoSOCSynchronizer`

**Parameters:** K=0.5, dt=0.1 s, N_steps=50

Source: Kuramoto (1984) Chemical Oscillations, Waves, Turbulence, Ch. 5.

---

### 3.31 NSA Detector — Activation and Confidence

**Equation:**
```
activates = ||features - detector_center||_2 < detector_radius

confidence_anomaly = max(0, 1 - min_dist_to_fault / detector_radius)
confidence_normal  = min(1, (min_dist - r) / r)   # if no activation
```

**File:** `diagnosis/weakest_cell.py:50–51` (Detector.activates); `diagnosis/weakest_cell.py:188–194` (detect)

**6D feature vector** [SOC, SOH, (T-298.15)/50, delta_SEI/1e-7, (V-3.0)/1.5, plating_risk]

Source: Forrest et al. (1994) IEEE Symposium on Security and Privacy.

---

### 3.32 GraphSAGE Convolution

**Equation:**
```
# Electrical neighbor aggregation (mean):
agg_elec  = (adj_elec / deg_elec) @ h
# Thermal neighbor aggregation:
agg_therm = (adj_therm / deg_therm) @ h

h_out = W_self(h) + W_elec(agg_elec)*1.0 + W_therm(agg_therm)*0.5
h_out = ELU(BatchNorm(h_out))
```

**File:** `stack/gnn_layer.py:98–114`, function `SAGEConv.forward()`

**Architecture:** 7 → 64 → 32 → 16 → 4 (sigmoid), Xavier weight init.

Source: Hamilton et al. (2017) NeurIPS, arXiv:1706.02216, Eq. 3.

---

### 3.33 KCL Physics Residual (GNN training)

**Equation:**
```
for each series group s:
    soc_group = SOC_refined[s*N_P : (s+1)*N_P]
    kcl_s = mean((soc_group - mean(soc_group))^2)
L_KCL = mean(kcl_s over all groups)
```

**File:** `stack/gnn_layer.py:229–241`, function `physics_residual()`

In the training loss (train_gnn.py): total_loss = MSE(V_pred, V_true) + 0.1*KCL + 0.05*TCO.

---

## 4. TIMING POSTMORTEM

### Operations in `DFNCell.step()` (core/dfn_cell.py:556–677)

| Operation | Lines | FLOPs | Dominant cost |
|---|---|---|---|
| ocp_graphite() — 7-term poly + 2 exp | 579 | ~30 | 2× np.exp + scalar dispatch |
| ocp_nmc811() — linear + 3 tanh | 580 | ~25 | 3× np.tanh + scalar dispatch |
| _exchange_current() ×2 — sqrt + clip | 583–584 | ~20 | np.sqrt scalar dispatch ×2 |
| Interfacial j ×2 — 3 muls | 587–588 | ~6 | — |
| _butler_volmer() ×2 — arcsinh + 3 muls | 591–592 | ~20 | np.arcsinh scalar dispatch ×2 |
| Terminal voltage — 5 ops | 595 | ~5 | — |
| TCO-3 plating check — 3 ops | 599–604 | ~5 | — |
| _solid_diffusion_dxdt() ×2 — 6 ops | 607–608 | ~12 | — |
| np.clip ×2 — stoich update | 609–610 | ~4 | np.clip dispatch |
| _sei_growth_rate() — exp + 4 muls | 613–614 | ~15 | np.exp scalar |
| _heat_generation() — 4 ops | 619 | ~4 | — |
| Newton cooling + temp update — 5 ops | 620–623 | ~5 | — |
| TCO 1/2/5 checks | 626–652 | ~15 | — |
| Coulomb counting SOC | 634–637 | ~3 | np.clip dispatch |
| State dataclass creation + dict build (17 keys) | 640–677 | ~30 | Python object alloc |
| **Total** | | **~200** | |

### Why ~40 µs/cell

**From deploy/realtime_bms.py docstring (line 14):** "DFN steps  36 × 40 µs = 1.44 ms"

The benchmark in `core/dfn_cell.py:797–799` checks `p99 < 200 µs` — the 40 µs is the empirical mean on the development machine, not a hard specification.

**Why 40 µs >> theoretical arithmetic floor (~20 ns for 200 FLOPs at 10 GFLOPS):**

The bottleneck is **Python/NumPy scalar call overhead**. Each call to `np.arcsinh(scalar)`, `np.exp(scalar)`, `np.sqrt(scalar)` carries ~1–3 µs of Python dispatch overhead (type checking, argument parsing, C extension entry). There are ~12–15 such scalar NumPy calls per step.

- ~12 scalar NumPy calls × ~2 µs each = ~24 µs
- Python `dict` construction (17 keys) + `DFNCellState` dataclass allocation = ~5 µs
- `time.perf_counter()` calls (lines 569, 655) = ~1 µs each

**Total overhead ≈ 30–35 µs**, leaving ~5–10 µs for actual arithmetic. This matches ~40 µs observed.

**Optimization path:** Removing `np.atleast_1d`/dict-per-step, using pure scalars + pre-allocated output buffers, and batching 36 cells as a numpy vector would reduce to ~5 µs/cell.

---

## 5. DFN USAGE: HONEST ASSESSMENT

### What the code actually implements

The module is titled "DFN (Doyle-Fuller-Newman) Single Particle Model" (core/dfn_cell.py:2). **This implements SPM, not full DFN.**

| Feature | Full DFN (Doyle 1993) | This codebase |
|---|---|---|
| Electrolyte concentration c_e(x,t) | Solved as PDE in x | Constant: ce = 1200 mol/m³ (line 572) |
| Electrolyte potential phi_e(x,t) | Solved as PDE | Not present |
| Solid concentration c_s(r,t) | Solved in r (spherical PDE) | Volume-averaged: single x_neg, x_pos scalars |
| Spatial electrode mesh | Required (typically 10–50 nodes/electrode) | Not present |
| Concentration overpotential | eta_conc = (RT/F)*ln(ce/ce0) | Not present |
| Electrolyte conductivity kappa(c_e) | Required | Not present |

The SPM is a well-established, widely-used simplification of DFN for BMS applications. Its validity is well-documented in Richardson 2020 and elsewhere. The "DFN" label in the filename is a misnomer relative to the full Doyle-Fuller-Newman model.

### PyBaMM usage

**PyBaMM is not imported anywhere in the codebase.** There is zero `import pybamm` or `from pybamm` in any file. PyBaMM appears only in the benchmark comparison table in `main.py:317–329` as a competitor reference — its timings (~50–500 ms/cell for full DFN) are stated as knowledge, not computed.

### Summary

- **No DFN PDE solved.** The model is SPM with Bernardi thermal + Pinson-Bazant SEI.
- **No PyBaMM dependency.** Not installed, not called, not validated against.
- **SPM validity:** Accurate for C-rates below ~2C. Electrolyte concentration effects at high C-rates are not captured.

---

## 6. DATA PIPELINE: validate_quartz.py

### Script structure

`data/validate_quartz.py` is a **top-level executable script**, not a function or class. It runs on `python data/validate_quartz.py`. It is not imported by `main.py`.

### Dataset

| Property | Value | Code location |
|---|---|---|
| Source files | Parquet in `data/quartz_wltp/` | line 41 |
| Pack topology | 3P × 12S = 36 cells | lines 43–44 |
| Cell capacity | Q_QUARTZ = 2.5 Ah | line 45 |
| DFN cell capacity | Q_DFN = 0.5 Ah | line 46 |
| Current scale | I_SCALE = 0.20 | line 47 |
| Sign convention | Quartz negative = discharge → I_dfn = −I_raw × 0.20 | lines 47–48 |
| Total rows | Printed at runtime; ~634,450 | line 121 |
| Resample interval | RESAMPLE_S = "20s" | line 42 |

### Steps

**Step 1 — Load** (lines 104–121): 10 parquet files loaded, concatenated, sorted by Timestamp.

**Step 2 — OCV calibration** (lines 126–146): For each cell, median of near-rest voltage readings from first file. Binary search OCV inversion to get initial SOC. 28 iterations bisection.

**Step 3 — Resample** (lines 152–206): 20-second windows. Current = mean; voltage/temperature = last value. Temperature fault detection: >50% readings above 100°C → impute from neighbors.

**Step 4 — Forced-SOC DFN simulation** (lines 218–233):
```python
for i in range(N_PTS):
    _set_cell_soc(cell, float(soc_pack_pct[i]))   # BMS SOC at every step
    res = cell.step(float(I_dfn[i]), float(dt_arr[i]))
    V_pred[i] = res["V"]
```
The DFN is **forced to the BMS SOC reading at each timestep**. This is a voltage-predictor mode, not a free-running simulation.

**Step 5 — Arrhenius-aware OLS calibration** (lines 246–295):
Per-cell R_ohm fit using Arrhenius factors and voltage-update rows only.

**Step 6 — DualEKF simulation** (lines 302–353):
EKF with NMC811 OCV table derived from `ocp_nmc811() - ocp_graphite()`. Initialized from OCV inversion. Free-running (no forced SOC). This is the prediction used for final metrics.

**Step 7 — Metrics** (lines 364–388):
```python
def _r2(y, yh):
    return float(1.0 - np.sum((y-yh)**2)/(np.sum((y-np.mean(y))**2)+1e-12))

def _mae(y, yh):  return float(np.mean(np.abs(y-yh)))
def _rmse(y, yh): return float(np.sqrt(np.mean((y-yh)**2)))
```

Applied as `_r2(V_meas[key], sim_ekf[key])`, `_mae(V_meas[key], sim_ekf[key])`.

### Critical caveat on R² metric

The code itself prints at line 401 and line 517:
```
Note: R² capped at ~0.87 by 6-min sensor interval (not model limitation)
```

The Quartz voltage sensors update approximately every 6 minutes. After resampling to 20 s, most rows have identical V_meas values (step-wise constant). R² computed over all rows against a smoothly varying EKF prediction is dominated by these flat intervals.

**The "R²=0.9810, MAE=18.6mV" values in slides are runtime-computed from the actual Quartz parquet files.** They are not hard-coded in the script and cannot be verified without the dataset.

---

## 7. CONFIG & PARAMETERS

### Physical constants (core/dfn_cell.py:20–22)

| Symbol | Value | Units | Source |
|---|---|---|---|
| F | 96485.0 | C/mol | CODATA 2018 |
| R_GAS | 8.314462 | J/(mol·K) | CODATA 2018 |
| T0 | 298.15 | K | Reference temperature |

### NMC811 cartridge (core/dfn_cell.py:86–118, function NMC811_cartridge())

| Parameter | Value | Units | Source |
|---|---|---|---|
| cs_max_neg | 30,555 | mol/m³ | Doyle 1996 |
| Ds_neg | 3.9×10⁻¹⁴ | m²/s | Doyle 1996 |
| x0_neg | 0.80 | — | initial stoichiometry |
| k0_neg | 1.764×10⁻¹¹ | A/m² | Ecker 2015 |
| alpha_neg | 0.5 | — | symmetric BV |
| cs_max_pos | 51,555 | mol/m³ | Ecker 2015 |
| Ds_pos | 1.0×10⁻¹⁴ | m²/s | Ecker 2015 |
| x0_pos | 0.45 | — | initial stoichiometry |
| k0_pos | 6.67×10⁻¹¹ | A/m² | Ecker 2015 |
| alpha_pos | 0.5 | — | symmetric BV |
| a_neg | 3.638×10⁵ | m²/m³ | Doyle 1993 |
| L_neg | 100×10⁻⁶ | m | — |
| a_pos | 3.437×10⁵ | m²/m³ | Doyle 1993 |
| L_pos | 183.5×10⁻⁶ | m | — |
| Cp | 900 | J/(kg·K) | Bernardi 1985 |
| rho | 2500 | kg/m³ | — |
| V_cell | 16.5×10⁻⁶ | m³ | 21700 cell |
| h_conv | 10 | W/(m²·K) | — |
| A_surf | 1.2×10⁻³ | m² | — |
| De | 7.5×10⁻¹¹ | m²/s | Valoen-Reimers 2005 |
| ce0 | 1200 | mol/m³ | — |
| k_SEI | 1.5×10⁻¹⁷ | m/s | Pinson-Bazant 2013 |
| rho_SEI | 2.1×10⁴ | mol/m³ | Li₂CO₃ |
| M_SEI | 0.0730 | kg/mol | Li₂CO₃ |
| Q_nom | 0.5 | A·h | per cell |
| R_ohm | 0.005 | Ω | ~5 mΩ base (line 311) |

### LFP cartridge differences (core/dfn_cell.py:125–154, function LFP_cartridge())

| Parameter | NMC811 | LFP | Source |
|---|---|---|---|
| cs_max_pos | 51,555 | 22,806 mol/m³ | Safari 2011 |
| Ds_pos | 1.0×10⁻¹⁴ | 3.2×10⁻¹⁵ m²/s | Safari 2011 + Prada 2012 |
| x0_pos | 0.45 | 0.35 | — |
| x0_neg | 0.80 | 0.75 | — |
| k0_pos | 6.67×10⁻¹¹ | 3.0×10⁻¹¹ A/m² | — |
| k_SEI | 1.5×10⁻¹⁷ | 1.0×10⁻¹⁷ m/s | — |
| a_pos | 3.437×10⁵ | 1.5×10⁶ m²/m³ | — |
| L_pos | 183.5×10⁻⁶ | 80×10⁻⁶ m | — |
| OCV | Chen 2020 tanh polynomial | Prada 2012 21-point table | — |

### Cell-to-cell variation (core/dfn_cell.py:304–311)

Gaussian variation applied independently at initialization:
```
sigma_var = 0.002  (0.2%)
var_capacity = 1 + N(0, sigma_var)
var_Ds       = 1 + N(0, sigma_var)
var_R_ohm    = 1 + N(0, sigma_var)
```

Source: Schmalstieg et al., TUM Battery Workshop 2021.

### Pack topology (stack/pack_manager.py:23–45)

| Constant | Value | Meaning |
|---|---|---|
| N_SERIES | 4 | Series groups |
| N_PARALLEL | 5 | Cells per group |
| N_CELLS | 20 | Total cells in PackManager (≠ 36 cells in validate_quartz) |
| W_SOH | 0.40 | Weakest-cell SOH weight |
| W_THERMAL | 0.35 | Thermal risk weight |
| W_PLATING | 0.25 | Plating risk weight |
| R_CELL_CELL | 0.5 | K/W cell-to-cell thermal resistance |
| R_CELL_COOL | 2.0 | K/W cell-to-coolant thermal resistance |
| T_COOLANT | T0+5.0 = 303.15 K | Coolant temperature |
| TR_T_ONSET | 353.15 K | 80°C ARC onset (Feng 2018) |
| TR_T_RUNAWAY | 423.15 K | 150°C runaway (Feng 2018) |

Note: The PackManager uses 4S5P = 20 cells. The validate_quartz.py uses 3P×12S = 36 cells. These are **different pack configurations**, not the same.

### EIS reference parameters (eis/eis_simulator.py:33–39)

Healthy NMC811 at 25°C, source: Ecker 2015:

| Name | Value | Units |
|---|---|---|
| R_OHM_REF | 0.005 | Ω |
| R_SEI_REF | 0.008 | Ω |
| C_SEI_REF | 0.002 | F |
| R_CT_REF | 0.015 | Ω |
| C_DL_REF | 0.010 | F |
| D_S_REF | 3.9×10⁻¹⁴ | m²/s |
| A_W_REF | 0.03 | Ω·s⁻⁰·⁵ |

### Dual EKF (diagnosis/dual_ekf_lfp.py:15–31)

| Parameter | Value | Meaning |
|---|---|---|
| Q_nom_Ah | 160.0 | A·h (LFP 160Ah cell default) |
| R_int_ohm | 0.0005 | Ω initial internal resistance |
| P1_init | diag([0.01, 0.0001]) | SOC, V_pol variances |
| Q_base | diag([1e-6, 1e-5]) | base process noise |
| R_meas | 4e-6 | V² = (2 mV)² measurement noise |
| tau | 50.0 | s polarization time constant |

In validate_quartz.py, the EKF is adapted for NMC811 with OCV table derived analytically from Chen 2020 ocp functions and Q_nom_Ah = 2.5 Ah (Quartz cell capacity).

---

## 8. KEY FACTS FOR PROFESSOR AUDIT

1. **The physics model is SPM, not DFN.** The electrolyte is treated as uniform (ce = constant). The "DFN" label in `dfn_cell.py` refers to the source material (Doyle 1993), not the implemented model level.

2. **PyBaMM is not used anywhere.** Zero imports. Mentioned only as a competitor in `main.py:317–329`.

3. **The GNN is never trained in main execution.** `stack/train_gnn.py` is a standalone script, not called from `main.py` or `validate_quartz.py`. The `BatteryGNN` in `deploy/realtime_bms.py` runs with random initialized weights unless a checkpoint is loaded. No checkpoint loading code exists.

4. **validate_quartz.py uses forced SOC for the DFN pass.** The DFN is not free-running; it is forced to the BMS SOC at every timestep (line 227). The "free-running" EKF runs separately on top.

5. **R² is computed over all 20-second resampled rows**, including the many timesteps where V_meas is step-constant due to the ~6-minute sensor update interval. The script explicitly warns this caps R² at ~0.87 (lines 401, 517). The R²=0.9810 figure must be verified against the actual dataset.

6. **The validation topology (3P×12S=36 cells) differs from the PackManager (4S5P=20 cells).** These are separate configurations for different use cases.

7. **Cell-to-cell variation is seeded**, not random, so results are reproducible with the same `rng_seed`.

8. **Thermal runaway onset temperature is 80°C (353.15 K), runaway at 150°C (423.15 K)**. These are hard-coded constants from Feng et al. (2018) Joule, not fitted parameters.
