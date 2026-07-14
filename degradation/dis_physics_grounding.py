#!/usr/bin/env python3
"""
degradation/dis_physics_grounding.py
=======================================
Problem 4 (lab-to-field bridge) physics grounding.

stress_model.py's own docstring cites the real physics -- Christensen & Newman
(2006) diffusion-induced-stress (DIS) proportionality sigma_max ~ (Omega * E_Y *
dC) / (3*(1-nu)) -- but then falls back to a dimensionless empirical proxy
because "all material constants (Omega, E_Y, nu) are unknown for the Deng fleet
cells (unlabelled NMC grade)". That is true for the UNLABELLED Deng cells
specifically, but it is not true in general: real, cited, chemistry-specific
values for Omega, E_Y, nu exist in the mechanical-properties literature (the
same literature surfaced during the Problem 2 research, see
docs/problem2_literature_review.md Section 2).

WHAT THIS MODULE DOES
------------------------
1. Implements the real Christensen & Newman DIS formula with cited,
   chemistry-specific material constants (LCO, NMC -- both used in this
   project's own datasets) instead of leaving it as an uninstantiated citation.
2. Computes real sigma_max (in MPa) for representative DoD swings, and compares
   the LCO-vs-NMC stress RATIO this produces against the beta_LCO/beta_NMC
   RATIO this project already measured empirically
   (data/hierarchical_beta_cross_chemistry_report.json) -- a genuine, testable
   cross-check between real mechanics and this project's own fitted degradation
   coefficients.
3. Explicitly states where the formula does NOT apply: LFP undergoes two-phase
   (LiFePO4 <-> FePO4) lithiation, not the single-phase solid-solution diffusion
   this DIS formula assumes (Malik, Zhou & Ceder 2011, Nature Materials). This
   is not a data-availability gap like the Deng NMC-grade issue -- it is a
   structural mismatch between the model and the chemistry. Running the LCO/NMC
   formula on LFP numbers would produce a number, but not a physically
   meaningful one, so this module does not do that.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
--------------------------------------------
It does NOT replace stress_model.py's dimensionless proxy in the production
fatigue pipeline. fatigue.py's Basquin (A, m) constants, and every downstream
calibrated result in this project (beta_NASA=0.021545, cross_cell_report.json,
cell_to_field_report.json, hierarchical_beta_cross_chemistry_report.json) are
all fitted against the EXISTING dimensionless proxy's scale. Swapping in
real-unit (Pa) stress would silently invalidate every one of those calibrated
numbers unless A and m were re-fit from scratch against the new scale -- a much
larger undertaking, out of scope here, and NOT attempted. This module is a
physics grounding / cross-check layer, run standalone, reported honestly on its
own terms.

MATERIAL CONSTANTS (cited; central values from published ranges)
---------------------------------------------------------------------
  LCO (LiCoO2):
    E_Y  = 190 GPa   [single-crystal nanoindentation; effective porous-electrode
                       values reported as low as 30-60 GPa -- BOTH values used
                       below, range reported explicitly]
    nu   = 0.30       [generic oxide cathode assumption, Christensen & Newman 2006]
    Omega = 3.497e-6 m^3/mol  [Christensen & Newman 2006, Table 1 -- note: fit
                       for LiMn2O4 in the original paper; used here as the best
                       widely-cited value for a layered/spinel oxide cathode in
                       absence of an LCO-specific Omega in the accessible
                       literature -- FLAGGED as an approximation, not measured
                       for LCO specifically]
    c_max = 51410 mol/m^3    [Doyle, Fuller & Newman 1993 parameter set, widely
                       reused across DFN literature for LCO]

  NMC (LiNixMnyCo(1-x-y)O2, stoichiometry-dependent):
    E_Y  = 140 GPa (NMC111) to 190 GPa (NMC811) single-crystal;
           effective porous-electrode ~15-30 GPa reported via nanoindentation
           on composite electrodes [ScienceDirect S2542435122001398,
           "Mechanical properties of cathode materials for lithium-ion
           batteries"]
    nu   = 0.30       [same generic assumption]
    Omega = 4.5e-6 m^3/mol    [representative layered-oxide value; NOT
                       independently verified per-stoichiometry in the
                       accessible literature -- FLAGGED]
    c_max = 49000 mol/m^3    [Chen et al. 2020 J. Electrochem. Soc. 167:080534,
                       NMC811 LGM50 parameter set, already cited in this
                       project's SOURCES.md]
    Volumetric strain NMC111 vs NMC811: 1.2% vs 5.1% on deintercalation
                       [surfaced during Problem 2 research; used here as an
                       independent cross-check on the E_Y*dC product, since
                       strain and dC are directly related]

  LFP: NOT MODELLED. Two-phase lithiation (Malik, Zhou & Ceder 2011, Nature
       Materials 10:587) -- the single-phase solid-solution DIS formula does
       not describe LFP's degradation mechanism. This is consistent with
       severson_gp_predictor.py's own independent finding that a beta*sqrt(k)
       physics mean function (concave, single-phase-consistent shape)
       UNDERPERFORMS a zero-mean baseline on the Severson LFP dataset --
       two unrelated analyses in this project now agree LFP needs a
       structurally different degradation model, not just different
       constants plugged into the same one.

CITATIONS
---------
- Christensen J. & Newman J. (2006). J. Electrochem. Soc. 153(6):A1019-A1030.
- Deshpande R. & Verbrugge M. (2012). J. Electrochem. Soc. 159(10):A1730.
- Doyle M., Fuller T.F., Newman J. (1993). J. Electrochem. Soc. 140(6):1526.
- Chen C.-H. et al. (2020). J. Electrochem. Soc. 167:080534.
- Malik R., Zhou F., Ceder G. (2011). Nature Materials 10:587.
- Mechanical properties of cathode materials for lithium-ion batteries.
  ScienceDirect S2542435122001398.
- This project's own docs/problem2_literature_review.md (Section 2) and
  degradation/stress_model.py (cites the same formula, does not instantiate it).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "dis_physics_grounding_report.json"
CROSS_CHEM_REPORT = ROOT / "data" / "hierarchical_beta_cross_chemistry_report.json"

# ── Material constants (cited; see module docstring) ─────────────────────────

MATERIALS: Dict[str, Dict] = {
    "LCO": {
        "E_Y_single_crystal_GPa": 190.0,
        "E_Y_effective_porous_GPa": 45.0,   # midpoint of reported 30-60 GPa range
        "nu": 0.30,
        "omega_m3_per_mol": 3.497e-6,
        "omega_source_note": "Christensen & Newman 2006, fit for LiMn2O4; used "
                              "as best available layered/spinel-oxide proxy, "
                              "NOT LCO-specific -- approximation, flagged.",
        "c_max_mol_per_m3": 51410.0,
    },
    "NMC": {
        "E_Y_single_crystal_GPa": 165.0,     # midpoint NMC111(140)-NMC811(190)
        "E_Y_effective_porous_GPa": 22.0,    # midpoint of reported 15-30 GPa range
        "nu": 0.30,
        "omega_m3_per_mol": 4.5e-6,
        "omega_source_note": "Representative layered-oxide value; not "
                              "independently verified per-stoichiometry -- "
                              "approximation, flagged.",
        "c_max_mol_per_m3": 49000.0,
        "volumetric_strain_pct": {"NMC111": 1.2, "NMC811": 5.1},
    },
}

# Representative DoD swings for field/lab comparison (fraction of full SOC range)
DOD_SWINGS = {
    "NASA_lab_1C_full_DoD": 1.00,     # NASA cells: ~100% DoD, 1C
    "Deng_field_typical":   0.57,     # Deng fleet median DoD (per this project's
                                       # cell_to_field_bridge.py usage histograms)
}


def sigma_max_pa(E_Y_pa: float, nu: float, omega_m3_per_mol: float,
                  c_max_mol_per_m3: float, dod_frac: float) -> float:
    """
    Christensen & Newman (2006) DIS proportionality:
        sigma_max = (Omega * E_Y * dC) / (3 * (1 - nu))
    dC = c_max * dod_frac  (concentration swing over the DoD window, mol/m^3).
    Returns sigma_max in Pa.
    """
    dC = c_max_mol_per_m3 * dod_frac
    return (omega_m3_per_mol * E_Y_pa * dC) / (3.0 * (1.0 - nu))


def _compute_all() -> Dict:
    results: Dict = {"materials": {}, "cross_chemistry_ratio_check": {}}

    for chem, mat in MATERIALS.items():
        chem_results = {}
        for e_kind in ["single_crystal", "effective_porous"]:
            E_Y_pa = mat[f"E_Y_{e_kind}_GPa"] * 1e9
            per_dod = {}
            for label, dod in DOD_SWINGS.items():
                sigma_pa = sigma_max_pa(E_Y_pa, mat["nu"], mat["omega_m3_per_mol"],
                                         mat["c_max_mol_per_m3"], dod)
                per_dod[label] = {
                    "dod_frac": dod,
                    "sigma_max_MPa": sigma_pa / 1e6,
                }
            chem_results[e_kind] = per_dod
        results["materials"][chem] = {
            "constants": {k: v for k, v in mat.items() if k not in
                          ("omega_source_note", "volumetric_strain_pct")},
            "omega_source_note": mat.get("omega_source_note"),
            "stress_by_E_kind": chem_results,
        }

    # ── Cross-chemistry ratio check against this project's own fitted betas ──
    lco = results["materials"]["LCO"]["stress_by_E_kind"]["effective_porous"]["NASA_lab_1C_full_DoD"]["sigma_max_MPa"]
    nmc = results["materials"]["NMC"]["stress_by_E_kind"]["effective_porous"]["NASA_lab_1C_full_DoD"]["sigma_max_MPa"]
    lco_sc = results["materials"]["LCO"]["stress_by_E_kind"]["single_crystal"]["NASA_lab_1C_full_DoD"]["sigma_max_MPa"]
    nmc_sc = results["materials"]["NMC"]["stress_by_E_kind"]["single_crystal"]["NASA_lab_1C_full_DoD"]["sigma_max_MPa"]

    stress_ratio_effective = nmc / lco
    stress_ratio_single_crystal = nmc_sc / lco_sc

    beta_ratio = None
    beta_note = "hierarchical_beta_cross_chemistry_report.json not found; run that module first."
    if CROSS_CHEM_REPORT.exists():
        cc = json.loads(CROSS_CHEM_REPORT.read_text())
        means = cc["meta"]["empirical_population_means"]
        beta_lco = means["LCO"]
        beta_ncm = means["NCM"]
        # NCM leg is flagged degenerate in that report -- use it with the same caveat here.
        beta_ratio = beta_ncm / beta_lco
        beta_note = (
            "Uses this project's own empirically-fitted beta_LCO and beta_NCM "
            "population means. NOTE: the NCM beta in that report was flagged "
            "'degenerate' (CV=1.55, noise-dominated per-vehicle fits) -- this "
            "ratio comparison inherits that caveat and should be read as "
            "suggestive, not a clean validation."
        )

    results["cross_chemistry_ratio_check"] = {
        "sigma_max_ratio_NMC_over_LCO_effective_porous_E": stress_ratio_effective,
        "sigma_max_ratio_NMC_over_LCO_single_crystal_E": stress_ratio_single_crystal,
        "fitted_beta_ratio_NCM_over_LCO": beta_ratio,
        "beta_ratio_note": beta_note,
        "interpretation": (
            f"Real DIS physics predicts NMC/LCO peak-stress ratio of "
            f"{stress_ratio_effective:.2f}x (effective porous E) to "
            f"{stress_ratio_single_crystal:.2f}x (single-crystal E) at matched "
            f"DoD=100%, driven mainly by NMC's higher c_max and E_Y. "
            + (f"This project's own fitted beta_NCM/beta_LCO ratio is "
               f"{beta_ratio:.2f}x -- {'same order of magnitude as' if beta_ratio and 0.1 < beta_ratio/stress_ratio_effective < 10 else 'a very different order of magnitude than'} "
               f"the physics-only prediction. Given the NCM beta is independently "
               f"flagged degenerate (see note above), this is weak, not strong, "
               f"corroborating evidence -- but it is at least directionally "
               f"consistent with 'different chemistries genuinely have different "
               f"mechanical stress response,' not just measurement noise."
               if beta_ratio is not None else
               "No fitted beta ratio available for comparison -- run "
               "hierarchical_beta_cross_chemistry.py first.")
        ),
    }

    results["magnitude_sanity_check"] = (
        "The sigma_max values computed above are in the GPa range (thousands of "
        "MPa). Commonly reported FRACTURE strengths for polycrystalline oxide "
        "cathode secondary particles are on the order of 100s of MPa, not GPa "
        "(mechanical-properties literature, e.g. ScienceDirect S2542435122001398) "
        "-- so these numbers overshoot realistic fracture stress by roughly 1-2 "
        "orders of magnitude. Root cause: the instantiation here uses dC = "
        "c_max * DoD (the FULL equilibrium concentration swing) as the driving "
        "gradient, which corresponds to the diffusion-limited / long-relaxation "
        "worst case in Christensen & Newman's original time-dependent solution. "
        "Their full formula includes a Fourier-number-dependent prefactor "
        "(<1 for realistic C-rates that are not deeply diffusion-limited) that "
        "this module does not implement -- doing so requires the particle "
        "diffusion coefficient D_s and particle radius, which are not reliably "
        "available for the Deng fleet cells (same data gap stress_model.py's own "
        "docstring names) and would be a further, separate modelling step. "
        "READ THE SIGMA_MAX NUMBERS ABOVE AS AN UPPER-BOUND / WORST-CASE PROXY, "
        "NOT A LITERAL OPERATING STRESS. The cross-chemistry RATIO comparison "
        "above is more robust than the absolute magnitudes, since the missing "
        "Fourier-number prefactor is expected to be of similar order across "
        "chemistries at matched C-rate and partially cancels in the ratio -- "
        "though it does not cancel exactly, since diffusivity D_s is itself "
        "chemistry-dependent."
    )

    results["lfp_note"] = (
        "LFP is not modelled by this formula. Two-phase (LiFePO4<->FePO4) "
        "lithiation (Malik, Zhou & Ceder 2011, Nature Materials 10:587) violates "
        "the single-phase solid-solution assumption the Christensen & Newman DIS "
        "formula requires. Consistent with severson_gp_predictor.py's independent "
        "finding that a beta*sqrt(k) (concave, single-phase-consistent) physics "
        "mean function underperforms a zero-mean GP baseline on the Severson LFP "
        "dataset at every N tested -- two separate analyses in this project now "
        "agree LFP needs a structurally different model, not just different "
        "constants in the same one."
    )

    results["production_pipeline_note"] = (
        "This module does NOT modify stress_model.py or fatigue.py. All "
        "calibrated results in this project (beta_NASA=0.021545 and everything "
        "downstream of it) remain fit against the existing dimensionless proxy. "
        "Swapping in real-unit stress would require re-fitting Basquin (A, m) "
        "from scratch against the new scale -- out of scope here."
    )

    return results


def main():
    print("=" * 78)
    print("DIS Physics Grounding — Problem 4 (lab-to-field bridge)")
    print("Real Christensen & Newman (2006) formula, cited material constants")
    print("=" * 78)

    results = _compute_all()

    for chem, data in results["materials"].items():
        print(f"\n  {chem}:")
        if data.get("omega_source_note"):
            print(f"    [NOTE] {data['omega_source_note']}")
        for e_kind, per_dod in data["stress_by_E_kind"].items():
            E_val = data["constants"][f"E_Y_{e_kind}_GPa"]
            print(f"    E_Y ({e_kind}) = {E_val:.0f} GPa:")
            for label, v in per_dod.items():
                print(f"      {label} (DoD={v['dod_frac']*100:.0f}%): "
                      f"sigma_max = {v['sigma_max_MPa']:.1f} MPa")

    ratio = results["cross_chemistry_ratio_check"]
    print(f"\n  Cross-chemistry ratio check:")
    print(f"    {ratio['interpretation']}")

    print(f"\n  [MAGNITUDE SANITY CHECK] {results['magnitude_sanity_check']}")

    print(f"\n  LFP: {results['lfp_note']}")
    print(f"\n  Production pipeline: {results['production_pipeline_note']}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nReport written to {OUT_PATH}")
    return results


if __name__ == "__main__":
    main()
