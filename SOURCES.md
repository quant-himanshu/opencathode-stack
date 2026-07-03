# Primary Sources

All claims in this repository trace to one of the citations below.
DOIs are stable links; accessed/verified 2026.

---

## Datasets

| Dataset | Citation |
|---|---|
| Deng BAIC EU500 fleet (20 vehicles, 29 months) | Deng Z., Xu L., Liu H., Hu X., Duan Z., Xu Y. (2023). Prognostics of battery capacity based on charging data and data-driven methods for on-road vehicles. *Applied Energy* 339:120954. https://doi.org/10.1016/j.apenergy.2023.120954 |
| NASA PCoE Battery Aging (B0005/B0006/B0007/B0018) | Saha B. & Goebel K. (2009). Battery Data Set, NASA Ames Prognostics Data Repository. NASA/TM-2007-214294. https://www.nasa.gov/intelligent-systems-division |
| EIS spectra (RWTH Aachen) | Schäffer et al. (2024). Zenodo record 6405084. https://doi.org/10.5281/zenodo.6405084 |

**Deng dataset chemistry note:** BAIC EU500, CATL NCM (NMC) prismatic cells, nominal 145 Ah, 90-cell series pack.
`Q_NOMINAL = 136.2 Ah` in our code is the maximum *observed* available_capacity at dataset entry
(vehicles were ~6% degraded at first reading); Deng 2023 paper nominal is 145 Ah.

---

## Electrochemistry Models

| Model / Parameter | Citation |
|---|---|
| DFN (Doyle-Fuller-Newman) | Doyle M., Fuller T.F., Newman J. (1993). *J. Electrochem. Soc.* 140(6):1526. |
| SPM approximation | Richardson G. et al. (2020). *J. Electrochem. Soc.* 167:080542. |
| SEI √t calendar aging | Pinson M.B. & Bazant M.Z. (2013). *J. Electrochem. Soc.* 160:A243. |
| NMC811 OCP (Chen 2020 LGM50) | Chen C.-H. et al. (2020). *J. Electrochem. Soc.* 167:080534. |
| LFP OCP | Safari M. & Delacourt C. (2011). *J. Electrochem. Soc.* 158:A562. Prada E. et al. (2012). *J. Electrochem. Soc.* 159:A1508. |
| Arrhenius R_ohm | Nyman A. et al. (2008). *Electrochim. Acta* 53:6356. |

---

## Degradation & Fatigue

| Topic | Citation |
|---|---|
| Palmgren-Miner linear damage rule | Miner M.A. (1945). *J. Appl. Mech.* 12(3):A159. |
| Basquin S-N curve (power law) | Basquin O.H. (1910). *Proc. ASTM* 10:625. |
| ASTM E1049 rainflow counting | ASTM International (2017). ASTM E1049-85(2017): Standard Practices for Cycle Counting in Fatigue Analysis. |
| Degradation mode analysis (DVA/ICA) | Dubarry M. & Anseán D. (2022). Best practices for incremental capacity analysis. *Front. Energy Res.* 10:1023555. https://doi.org/10.3389/fenrg.2022.1023555 |
| Factor ranking (T > DoD > C-rate) | Edge J.S. et al. (2021). Lithium ion battery degradation: what you need to know. *Phys. Chem. Chem. Phys.* 23(14):8200–8221. https://doi.org/10.1039/D1CP00359C |
| Calendar vs cycle dominance (field) | Sulzer V. et al. (2021). The challenge and opportunity of battery lifetime prediction from field data. *Joule* 5(8):1934–1955. https://doi.org/10.1016/j.joule.2021.06.005 |

---

## Estimation & Control

| Topic | Citation |
|---|---|
| Dual EKF for SOC + capacity | Plett G.L. (2004). *J. Power Sources* 134(2):262–276. |
| Adaptive EKF (battery) | Mikhak M. et al. (2024). PMC12936157. |

---

## Contribution Framing (P5)

Prior work on the Deng 2023 dataset and its direct follow-ups targets
capacity-trajectory prediction (sequence models, Gaussian process regression),
**not** per-vehicle degradation factor attribution or fade-rate transfer.
To our knowledge, applying permutation importance to per-vehicle λ_v on this
dataset has not been published. The outcome (factors indistinguishable at n=13)
is the pre-registered expected result given the fleet's narrow feature ranges.
