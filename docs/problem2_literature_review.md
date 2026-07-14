# Problem 2 Literature Review: Commonizing a Degradation Model Across Cells and Manufacturers

**Scope.** This project's own work established the problem empirically: `degradation/cross_cell_predictor.py`
gets within-cell R²=0.9725 (β fitted per NASA LCO cell) but cross-cell R²=−0.68 when a population-mean β
is applied to a held-out cell (B0006 fades ~45% faster than B0005 at matched cumulative fatigue damage D).
`degradation/hierarchical_beta.py` (n=4 LCO, partial pooling) explicitly refuses to extrapolate beyond LCO,
citing a measured 200–300x error transferring an LCO-derived β to the Deng et al. NCM fleet, split between a
262x D-scale mismatch and genuine LCO-vs-NCM fatigue-resistance differences that could not be disentangled.
This review asks: does the published literature solve this, and if so, how; if not, what is the honest state
of the art and what should this project do next.

---

## 1. Physics-informed transfer learning across chemistries/manufacturers

Most recent work in this space either (a) transfers within a chemistry family under a different name for
"cross-domain," or (b) genuinely tests cross-chemistry/cross-manufacturer but reports a real, non-trivial
error penalty rather than a solved problem.

- **Early-Cycle Internal Impedance / Current Pulses Enable ML-Based Battery Cycle Life Predictions Across
  Manufacturers** (arXiv:2410.05326). Uses 57 cells from three manufacturers (Samsung INR21700-50E,
  BAK N21700CG, LG INR18650-MH1; NCA vs NMC cathodes, Gr-Si vs graphite-only anodes). Runs a genuine
  **Leave-One-Manufacturer-Out (LOMO)** cross-validation. Result: LOMO MAE ≈160 cycles for EOL85 vs
  ≈100 cycles for within-manufacturer (Leave-One-Triplicate-Out) — a **1.5–2x error increase** just from
  manufacturer, not chemistry, shift. The authors explicitly state: "the cell types in this dataset have in
  common Ni-based cathodes and Gr-based anodes, thus the LOMO strategy does not validate across other
  chemistry families, such as LFP." This is the most rigorous, honestly-scoped cross-manufacturer result
  found — and even it never crosses a real chemistry-family boundary (LFP vs Ni-based).

- **GPT4Battery** (arXiv:2402.00068, "Adapting Amidst Degradation: Cross Domain Li-ion Battery Health
  Estimation via Physics-Guided Test-Time Training"). Trains on five commercial datasets spanning LCO,
  NMC, NCA, LFP (CALCE, SANYO, KOKAM, PANASONIC, GOTION) plus a proprietary 300 Ah dataset. Despite the
  "zero-shot" framing, the method is **not** zero-shot at inference: it performs continuous test-time
  adaptation (TTA) using unlabeled target-domain data collected online as the cell degrades. On the
  hardest held-out set (KOKAM) it reports MAE 7.95 / RMSE 8.01 (SOH %-points) — a usable but non-trivial
  error, achieved only because unlabeled target data is available at deployment, and the authors themselves
  flag that TTA is 10–100x slower at inference than a plain model. This is a meaningfully different problem
  than this project's β-identification-from-≤30-cycles setting (no online unlabeled stream is assumed there).

- **Hybrid Physics-Informed ML degradation model** (Sci. Rep., "calibration-free degradation prediction,"
  s41598-026-56439-z) constrains an LSTM residual with Arrhenius kinetics and Wöhler/Basquin fatigue-stress
  physics, arguing the physics terms (not the learned residual) are what carries across chemistries. This
  matches this project's own architecture (physics core = fatigue damage D + SEI √t, learned/fitted term =
  β) but the paper does not report a genuine leave-one-chemistry-out validation with numeric held-out error;
  it should be read as a design pattern, not evidence the pattern solves cross-chemistry transfer.

- **BatLiNet** (Zhang, H. et al., *Nature Machine Intelligence* 7:270–277, 2025; preprint arXiv:2310.05052,
  "Accurate battery lifetime prediction across diverse aging conditions with deep learning"). The largest
  and most careful multi-chemistry benchmark found: 401 cells, 5 electrode chemistries (LFP, LCO, NCA, NMC,
  and one more), 168 cycling conditions, pooled from MATR-1/2, HUST, CLO, CALCE, RWTH, UL-PUR, SNL, HNEI.
  Uses an **inter-cell** learning mechanism (predicts the lifetime *difference* between a target cell and a
  reference cell rather than an absolute lifetime) which is structurally closer to solving the transfer
  problem than a shared global parameter like β. Reports MAPE 10.0%, RMSE 158±7 cycles on a MIX-100
  benchmark that combines chemistries; MAPE rises to 18.1%, RMSE 201±18 cycles at 20-cycle early prediction
  (MIX-20). Critically, and stated by the authors: **all evaluated chemistries are present in the training
  pool** — this is not leave-one-chemistry-out generalization to a truly unseen chemistry, it is generalization
  to unseen cells/conditions within a chemistry set the model has already seen examples of. The paper is
  explicit that "degradation patterns vary between battery chemistries" and that MIX-20 (data-scarce,
  cross-chemistry-mixed) remains genuinely hard.

- **"Universal" battery degradation foundation model** (arXiv:2601.00862, trained on 16 datasets spanning
  NMC/LCO/LFP/hybrid + sodium-ion/zinc-ion). Marketed as cross-chemistry-generalizing, but on inspection the
  4 nominally "held-out" datasets (EOCV2, SJTU, WZU, Cambridge) are all lithium-ion variants of chemistries
  already represented in training; the sodium-ion and zinc-ion systems it advertises testing on are actually
  **in the training corpus**, not held out. No chemistry-isolated error breakdown is reported. This is a
  useful negative data point: even a paper making an explicit "universal, cross-chemistry" claim, on close
  reading, does not perform the leave-one-chemistry-family-out test this project's problem requires.

**Thread verdict:** No paper found performs a strict leave-one-chemistry-family-out test (train on
{LFP, NMC, NCA}, zero cells of LCO, predict LCO) with a reported, honest error number. The closest rigorous
analogue — LOMO across manufacturers within one chemistry family (arXiv:2410.05326) — already shows a
1.5–2x degradation penalty, and its own authors decline to claim it would hold across chemistry families.

---

## 2. Why β / S-N parameters don't transfer: mechanical grounding

- **Deshpande, R. & Verbrugge, M., "Battery cycle life prediction with coupled chemical degradation and
  fatigue mechanics," J. Electrochem. Soc. 159(10):A1730 (2012).** Couples classical Basquin/Paris fatigue
  mechanics to particle-level lithiation-swing-driven cracking. This is the direct mechanistic ancestor of
  this project's D = Σ(Δσ/A)^m damage-accumulation approach, confirming the physical picture (fatigue
  parameters A, m are material properties of the *electrode particle*, not universal constants) is standard
  in the field — which is exactly why they should not be expected to transfer across chemistries without
  correction.

- **Mechanical property surveys** (ScienceDirect S2542435122001398, "Mechanical properties of cathode
  materials for lithium-ion batteries"; ScienceDirect S221128552500463X on NCM polycrystalline particle
  fracture) report chemistry-dependent, SOC-dependent elastic/fracture behavior: LCO's Young's modulus
  drops substantially with decreasing SOC; NMC532 fracture toughness falls ~50% over 100 cycles at C/20;
  NMC111 vs NMC811 show markedly different volumetric strain on deintercalation (1.2% vs 5.1%). This is a
  quantitative, physically-grounded reason a single (A, m) or β cannot be chemistry-invariant: the
  stress-strain response feeding the Basquin/Miner damage calculation is itself chemistry- and even
  stoichiometry-dependent (NMC111 vs NMC811 differ by ~4x in a driving-strain proxy). No single paper was
  found tabulating ready-to-use A/m constants across LCO, LFP, and NMC in a form directly pluggable into
  this project's fatigue.py — this remains a gap, not a solved lookup problem. The physically correct
  next step (see Section 6) is bounded by data that exists (elastic modulus, fracture toughness by
  chemistry) but not by a directly transferable Basquin (A, m) pair.

- Related: **"Modeling Particle Versus SEI Cracking in Lithium-Ion Battery Degradation: Why Calendar and
  Cycle Aging Cannot Simply be Added"** (IOP, J. Electrochem. Soc., ad76da) reinforces that even the
  additive-damage assumption this project uses (D_cycle + λ√t) is itself a simplification whose error is
  chemistry-dependent — a second, compounding reason a single scalar correction won't cleanly separate
  cross-chemistry β differences from cross-chemistry model-form error, exactly as `hierarchical_beta.py`'s
  docstring already concedes ("BOTH from a D-scale mismatch AND from genuine... differences that could not
  be disentangled").

**Thread verdict:** The mechanical-properties literature explains *why* β/A/m fail to transfer (chemistry-
and SOC-dependent elastic/fracture response) but does not hand this project a ready correction factor —
the numbers exist in scattered single-chemistry papers, not as a cross-chemistry-calibrated table.

---

## 3. Hierarchical/multi-task Bayesian approaches validated on >1 chemistry

- **Hierarchical Bayesian Model for Probabilistic Analysis of EV Battery Degradation** (arXiv:1911.01399)
  and the **second-life hierarchical Bayesian lifetime paper** (Reliability Engineering & System Safety,
  2026, ideas.repec.org/a/eee/reensy) both use two-level partial pooling similar in spirit to this project's
  `hierarchical_beta.py`, but within a single chemistry/operating-condition cluster — clustering is used to
  *avoid* pooling across genuinely different populations, not to pool across them.
- One search summary (not independently verified against full text — flagged as lower confidence) describes
  analysis across **MIT-Stanford MATR (LFP), NASA PCoE (LCO), and CALCE CS2 (LCO)** claiming "partially
  overlapping distributions" of degradation *features* across LCO/NMC/LFP without chemistry-specific
  retuning — but note two of the three cited sources are both LCO, so this is not strong evidence of
  genuine 3-chemistry pooling; it should be treated as a feature-overlap observation, not a validated
  cross-chemistry hierarchical parameter estimate.
- **No paper was found** that builds a single hierarchical model pooling LCO + LFP + NMC cells together
  (with chemistry as a random-effect grouping level) and reports an honest held-out error for a
  never-before-seen chemistry group, in the way `hierarchical_beta.py` reports honest n=4 LCO-only
  shrinkage statistics.

**Thread verdict:** This project's `hierarchical_beta.py` (n=4, single chemistry, radically honest about
its scope) is not behind the state of the art in cross-chemistry hierarchical pooling — because that state
of the art does not appear to exist yet with real validation. The natural extension — a 3-level hierarchy
(chemistry → manufacturer/protocol → cell) fit jointly on NASA+Severson+Deng with chemistry as an explicit
random effect — would be a genuine contribution if done, precisely because nobody found seems to have
published it with an honest held-out-chemistry check.

---

## 4. Feature-based / early-cycle transferable representations

- **Severson, K.A. et al., "Data-driven prediction of battery cycle life before capacity degradation,"
  Nature Energy 4:383–391 (2019).** ΔQ(V) discharge-curve features (variance, min, skewness) computed
  from cycles 2–100, predicting log(cycle life) via elastic net. Validated only within-dataset (124 LFP
  A123 cells, one facility, varied fast-charge protocols) — this is a controlled same-chemistry,
  same-manufacturer generalization test (across *protocols*, not across chemistries or manufacturers).
- **Attia, P.M. et al., "Closed-loop optimization of fast-charging protocols for batteries with machine
  learning," Nature 578:397–402 (2020).** Same ΔQ(V)-style early-prediction machinery used inside a
  Bayesian optimization loop, again single chemistry/facility.
- **BatLiNet / arXiv:2310.05052** (Section 1) is the strongest evidence that features/representations built
  for one chemistry are useful as *auxiliary* signal for another (LFP-target pairs help predict LCO/NCA/NMC
  lifetimes in low-resource settings) but this is inter-cell transfer *within a jointly-trained pool*, not
  zero-shot generalization to a chemistry absent from training.
- Direct web evidence (secondary, from search synthesis rather than full-text read) states plainly: "When
  ML battery prediction models are applied to different chemistries or operating conditions, their
  predictive power diminishes significantly" — consistent with every primary source actually read above.

**Thread verdict:** Severson/Attia-style ΔQ(V) features are validated for early prediction *within* a
chemistry and protocol family, not claimed by their own authors to generalize cross-chemistry. This
project's `severson_gp_predictor.py` already correctly scopes itself this way (explicit statement that
its 124-cell LFP analysis is "SEPARATE from the NASA LCO results and is NOT directly comparable").

---

## 5. Is cross-manufacturer/cross-chemistry transfer a solved problem anywhere?

No. Across ~20 papers surveyed (10+ read in enough depth to extract quantitative claims), the pattern is
consistent:

1. Papers that claim "cross-chemistry" or "universal" generalization, on close reading of their own
   train/test split, either (a) keep all chemistries present in training and test on held-out *cells* or
   *conditions* within that pool (BatLiNet, the "universal foundation model"), or (b) require online
   unlabeled target-domain data at test time (GPT4Battery/TTA), which is a different and easier problem than
   zero-cycle or few-cycle transfer to a truly new chemistry.
2. The one paper found that runs a strict leave-one-group-out test (LOMO, arXiv:2410.05326) reports a real,
   non-trivial 1.5–2x error penalty crossing manufacturers **within one chemistry family**, and its authors
   explicitly refuse to extrapolate that result across chemistry families (Ni-based vs LFP).
3. The mechanical-properties literature (Section 2) supplies a physical reason to expect this: fatigue/
   fracture parameters are set by chemistry- and SOC-dependent elastic/fracture behavior that measurably
   differs even between NMC sub-chemistries (NMC111 vs NMC811), let alone across LCO/LFP/NMC.

**This is a genuinely open research problem.** This project's own honestly-reported cross-cell R²=−0.68 and
200–300x LCO→NCM β transfer error are not an outlier failure — they are consistent with the best-documented
attempts in the published literature, which either avoid the strict test or fail it by a comparable margin
when they run it.

---

## 6. Concrete next step for this project

Given the existing modules and the genuine 3-chemistry dataset (NASA LCO n=4, Severson LFP n=124, Deng NCM
fleet n=20), the literature suggests one implementable, honest next step, ranked by what's actually novel
versus what's just repeating a known-hard test:

**Do this:** Build a **3-level hierarchical model with chemistry as an explicit random-effect grouping
layer** (chemistry → dataset/protocol → cell), fit jointly on all three datasets, and report the **honest
leave-one-chemistry-out** posterior predictive error for the held-out chemistry's population-mean β
(e.g., train hierarchy on {LCO, LFP}, predict NCM population β, compare to Deng's fitted β). This is
directly buildable by extending `hierarchical_beta.py`'s existing PyMC partial-pooling machinery with one
more hierarchy level, and it closes exactly the gap found in Section 3: no paper surveyed reports this
specific test with a numeric error. Expect it to still be bad — the goal is not to "solve" transfer, it is
to replace the current single 200-300x point-estimate anecdote with a properly quantified posterior
predictive interval for a new chemistry's β, which is honestly reportable either way.

**Do this too, as a grounding step for (2):** Before fitting D-scale-mismatched β values, non-dimensionalize
D using a chemistry-specific mechanical normalization pulled from published elastic-modulus/fracture-
toughness ratios (Section 2) — e.g., scale Δσ by a chemistry's reported Young's modulus or fracture
toughness relative to a reference chemistry — to test whether the 262x D-scale component of the LCO→NCM
error (already separated out in `hierarchical_beta.py`'s docstring) shrinks. This uses numbers that exist
in the literature (LCO modulus-vs-SOC, NMC fracture toughness decay) without requiring a fabricated
cross-chemistry Basquin table.

**Do not pursue:** deep meta-learning / foundation-model style approaches (GPT4Battery-style TTA, the 16-
dataset foundation model). They require either (a) far more cells per chemistry than this project has (4,
124, 20 is small even by the surveyed papers' standards — BatLiNet used 401 cells across 5 chemistries and
still didn't achieve genuine leave-one-chemistry-out validation), or (b) an online unlabeled data stream at
deployment that this project's problem framing (predict from limited initial testing before fleet deployment)
does not assume is available.

**Honest bottom line to state in the report:** commonizing a fatigue-based degradation model across
chemistries and manufacturers, validated by genuine leave-one-chemistry-out testing, is not solved in the
published literature as of this review (2026). The best documented attempt at the analogous leave-one-
manufacturer-out test, within a single chemistry family, still shows a 1.5–2x error penalty. This project's
−0.68 cross-cell R² and 200-300x naive cross-chemistry β transfer error are consistent with, not worse than,
the state of the art when the same strict test is actually run elsewhere. The one clearly implementable,
novel contribution available to this project is the honest 3-chemistry leave-one-out hierarchical posterior
described above — nobody found in this review has published that specific result.

---

## Sources consulted (read in enough depth to extract quantitative claims)

1. Deshpande, R. & Verbrugge, M. (2012). "Battery cycle life prediction with coupled chemical degradation
   and fatigue mechanics." *J. Electrochem. Soc.* 159(10):A1730.
2. Early-Cycle Internal Impedance / Current Pulses Enable ML-Based Battery Cycle Life Predictions Across
   Manufacturers. arXiv:2410.05326.
3. GPT4Battery: An LLM-driven Framework for Adaptive SOH Estimation of Raw Li-ion Batteries
   ("Adapting Amidst Degradation: Cross Domain Li-ion Battery Health Estimation via Physics-Guided
   Test-Time Training"). arXiv:2402.00068.
4. Zhang, H. et al. (2025). "Battery lifetime prediction across diverse ageing conditions with inter-cell
   deep learning" (BatLiNet). *Nature Machine Intelligence* 7:270–277; preprint arXiv:2310.05052.
5. Universal Battery Degradation Forecasting Driven by Foundation Model Across Diverse Chemistries and
   Conditions. arXiv:2601.00862.
6. Severson, K.A. et al. (2019). "Data-driven prediction of battery cycle life before capacity
   degradation." *Nature Energy* 4:383–391.
7. Attia, P.M. et al. (2020). "Closed-loop optimization of fast-charging protocols for batteries with
   machine learning." *Nature* 578:397–402.
8. Mechanical properties of cathode materials for lithium-ion batteries. *ScienceDirect* S2542435122001398.
9. Fracture behaviour of NCM polycrystalline particles in lithium-ion batteries under extreme conditions.
   *ScienceDirect* S221128552500463X (abstract/summary level only — full text paywalled, HTTP 403).
10. Modeling Particle Versus SEI Cracking in Lithium-Ion Battery Degradation: Why Calendar and Cycle Aging
    Cannot Simply be Added. *J. Electrochem. Soc.*, IOPscience ad76da.
11. Hierarchical Bayesian Model for Probabilistic Analysis of EV Battery Degradation. arXiv:1911.01399.
12. Hybrid physics-informed machine learning framework for calibration-free degradation prediction of
    lithium-ion batteries. *Scientific Reports*, s41598-026-56439-z (abstract/summary level).

Papers found but not independently confirmed at full-text depth (PDF extraction failed or paywalled;
claims above sourced from search-result synthesis only, flagged inline where used): "Attention towards
chemistry agnostic and explainable battery lifetime prediction" (ARCANA, *npj Computational Materials*,
2024, trained on proprietary BASF dataset — noted but not verified in detail); "Chemistry-aware battery
degradation prediction under simulated real-world cyclic protocols" (arXiv:2504.03701); "Two points are
enough" (arXiv:2408.11872).
