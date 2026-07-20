# Sign-bug postmortem — cross-loader current-convention inconsistency

Status: fixed 2026-07-20, all results regenerated (see
`results/sign_fix_before_after.{csv,md}` for every number that moved).
This document is paper material (Failure Analysis / Threats to Validity),
not internal-only.

## 1. Summary

The project schema defines battery current as **discharge-negative**
(`data/loaders/common_schema.py:enforce_discharge_negative`). Two
independent defects put five datasets under two different conventions and
two estimator families under opposite sign errors:

1. The **CALCE and UMich/Ford benchmark loaders** flipped their raw
   current (`I_A = −Current`) to discharge-positive — raw Arbin/Maccor
   current is negative during discharge, i.e. already schema-conforming —
   while documenting the flip as "verified, not assumed". The verification
   had checked the flip against a wrong statement of the project
   convention (below), so it faithfully installed the inverted sign.
2. The **pure-coulomb baseline** (`data/soc_baseline_benchmark.py:
   coulomb_counting_soc`) integrated with the discharge-positive formula
   `SOC = SOC₀ − ∫I dt/(3600·Q)`, matching its own docstring's incorrect
   claim that "I_cell > 0 = discharge … matches this project's documented
   Deng convention". The EKF chain (`run_mode_b_ekf`) negates schema
   current before the filter and was therefore correct on
   schema-conforming data.

Net effect before the fix:

| Dataset (schema before fix) | EKF family (mine + scalar-bias variants + δ calibration) | Coulomb baseline |
|---|---|---|
| BMW i3, Deng, VED (discharge-negative ✓) | correct | **sign-inverted** |
| CALCE, UMich (discharge-positive ✗) | **sign-inverted** (process model + Mode-A calibration) | accidentally correct (two errors cancelled) |

So *every* pre-fix "EKF vs coulomb" comparison compared a correct
implementation against an inverted one — in one direction on the fleet
datasets and in the other on the lab datasets.

## 2. Mechanism, in each affected component

- **Coulomb baseline on BMW/Deng/VED**: integrated current with inverted
  sign, so SOC moved the wrong way. On Deng charging sessions (large
  monotone SOC rise) the estimate *fell* from its +20 pp handicap instead
  of rising — the pre-fix 40.12 pp mean RMSE is dominated by this, not by
  sensor drift. On short VED trips the inversion is partially masked
  because |ΔSOC| per trip is small.
- **EKF family on CALCE/UMich**: the filter's coulomb-integration predict
  step pushed SOC the wrong way every step, leaving the voltage update to
  fight the process model. Convergence statistics and RMSE on these two
  datasets reflected that fight, not the estimator's design.
- **Mode-A calibration on CALCE/UMich**: the DFN was stepped with inverted
  current, so the fitted current-proportional residual slope δR0 came out
  with the wrong sign and roughly doubled magnitude — the notorious
  **δR0 = −260.5 mΩ (CALCE) / −233.3 mΩ (UMich)** values. After the fix
  they re-fit to **+121.6 mΩ / +219.5 mΩ**: the sign pathology is gone,
  but magnitudes remain far outside the physical per-cell range (≈8 mΩ
  cells), i.e. the sign bug explains the sign, **not** the size — a
  genuine model/data mismatch remains on the lab datasets and is a
  Phase-4 subject. The EKF's Round-4 calibration sanity gate
  (`CAL_DR0_SANITY_THRESHOLD_OHM`) was therefore treating a real symptom
  with a correct instinct but a mis-attributed cause; it still fires
  post-fix on both lab datasets.

## 3. Why three rounds of validation did not catch it

1. **The reproduction discipline reproduces bugs.** Phase 0/1/2 cross-checks
   verified *bit-identical reproduction of the committed pipeline* — they
   prove determinism and provenance, not physical correctness.
2. **Compensating errors.** On CALCE/UMich the loader inversion and the
   coulomb-formula inversion cancelled, producing a plausible (indeed
   excellent) coulomb baseline exactly where ground truth is
   coulomb-derived — the cell most likely to be sanity-checked.
3. **Plausible aggregate numbers.** Every corrupted cell still produced
   RMSEs in a credible 15–40 pp band under the ±20 pp stress protocol;
   nothing looked impossible until a *zero-offset* run existed (Deng
   coulomb at offset 0: 55.7 pp — an integrator cannot do that honestly).
4. **Documentation asserted the wrong convention confidently.** The
   coulomb docstring's "I>0 = discharge" claim became the reference other
   code (the CALCE/UMich loaders' "verified" flips) was checked against.

## 4. Detection chain (what actually caught it)

VED per-vehicle diagnostics → implied pack capacity
`∫I dt / ΔSOC_BMS` came out **negative** (≈ −42 Ah) on all three VED
vehicles → per-segment sign audit across all five datasets
(`sign(∫I dt)` vs BMS ΔSOC): BMW 0/65, Deng 0/8368, VED 0/154 consistent
with discharge-positive vs CALCE 40/40, UMich 221/221 → code trace of both
estimator chains → two independent smoking guns confirmed the map:
Deng offset-0 coulomb RMSE 55.7 pp (inverted) vs CALCE offset-0 coulomb
0.10 pp (correct); BMW implied |Q| = 61.7 Ah vs its 60 Ah cartridge.

## 5. The fix (2026-07-20; no algorithm changes)

1. `data/soc_baseline_benchmark_calce.py` / `_module.py`: keep raw current
   sign (already discharge-negative); docstrings rewritten.
2. `data/soc_baseline_benchmark.py:coulomb_counting_soc`: integration sign
   corrected to the discharge-negative schema
   (`SOC = SOC₀ + ∫I dt/(3600·Q)`); docstring corrected.
3. **Permanent pipeline assertion**: `common_schema.make_schema_df` — the
   single choke point through which every dataset load passes — now runs
   `assert_discharge_negative_consistency` (net ∫I dt must oppose net
   ΔSOC drop whenever |ΔSOC| ≥ 5 pp and |Ah| ≥ 0.1) and raises on
   violation. Unit-tested in `tests/test_sign_convention.py` (correct
   convention passes; inverted data raises; regen-heavy zero-net trips
   never trip it; corrected coulomb baseline tracks a synthetic discharge
   exactly).

The EKF itself (`diagnosis/dual_ekf_lfp.py`) was **not modified**.

## 6. Post-fix verification invariants

Predicted and verified after regeneration:

- CALCE/UMich **coulomb and OCV-lookup columns reproduce the pre-fix values
  exactly** (the loader flip and the formula flip cancelled there, so two
  fixes must cancel too) — aggregate and per-segment.
- BMW/Deng/VED **EKF and OCV-lookup columns reproduce exactly** (their code
  paths are untouched by the fix).
- Only the previously-inverted cells move: coulomb on BMW/Deng/VED; the
  EKF family and calibrations on CALCE/UMich.
- The sign assertion passes on every segment of all five datasets during
  the regeneration loads.

Full magnitude accounting: `results/sign_fix_before_after.md` (changed
cells only) and `.csv` (every compared cell), plus the regenerated
headline reports, `results/main_table.csv`, `results/baseline_comparison.csv`,
`results/offset_sweep.csv`, `results/nominal_accuracy.md`,
`figures/offset_sweep.pdf`.

## 7. Paper implications

- The pre-fix "coulomb counting beats the EKF on VED/CALCE" storyline must
  be restated from the regenerated tables; the failure-analysis section
  gains this postmortem (detection method — the implied-capacity probe and
  the per-segment sign audit — generalizes and is worth reporting).
- The δR0 sanity gate's origin story changes: it was triggered by a data
  defect plus a real (still-open) lab-dataset model mismatch.
- The permanent sign assertion is a reproducibility contribution: the
  audit (`sign(∫I dt)` vs ΔSOC per segment) costs microseconds per load
  and would have caught this class of bug at first data ingestion.
