"""
stress_model.py — Dimensionless stress proxy for NMC electrode particles.

Physics basis (Christensen & Newman 2006, J. Electrochem. Soc.):
  Diffusion-Induced Stress (DIS) in a spherical electrode particle:
    σ_max ∝ (Ω × E_Y × ΔC) / (3 × (1 - ν))
  where ΔC ∝ DoD (state-of-charge swing).

Since all material constants (Ω, E_Y, ν) are unknown for the Deng fleet
cells (unlabelled NMC grade), we use a dimensionless stress proxy:
    stress = DoD × C_rate_factor × T_factor
  where:
    C_rate_factor = 1 + α_C × (C_rate / C_rate_ref - 1)
    T_factor      = exp(Ea_stress / R × (1/T_ref - 1/T_K))  (Arrhenius)

ASSUMPTIONS:
  - DoD is the primary driver (linear with strain ΔC)
  - C-rate amplifies stress via kinetic polarisation (higher current → steeper
    concentration gradient → higher surface stress)
  - Temperature accelerates degradation via Arrhenius; T_ref = 25 °C
  - α_C = 0.5  (C-rate sensitivity — literature range 0.3–0.8 for NMC)
  - C_rate_ref = 0.3C  (representative urban charging rate)
  - Ea_stress = 6000 K (Arrhenius activation energy; range 5000–8000 for NMC)
  - All constants are nominal; calibration absorbs their product into A and m
    of the S-N curve (see fatigue.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
T_REF_K    = 298.15   # 25 °C reference temperature
EA_STRESS  = 6000.0   # K  (Arrhenius activation energy)
ALPHA_C    = 0.5      # C-rate sensitivity coefficient
C_RATE_REF = 0.3      # reference C-rate (fraction of 1C)


def arrhenius_factor(T_C: float | np.ndarray) -> float | np.ndarray:
    """
    Arrhenius acceleration factor relative to 25 °C.
    factor > 1 for T > T_ref (higher T → more stress damage per cycle).
    """
    T_K = np.asarray(T_C, dtype=float) + 273.15
    return np.exp(EA_STRESS * (1.0 / T_REF_K - 1.0 / T_K))


def c_rate_factor(c_rate: float | np.ndarray) -> float | np.ndarray:
    """
    C-rate amplification of stress.
    At C_RATE_REF: factor = 1.0.
    Increases linearly above ref (simplification of surface-concentration gradient model).
    """
    c = np.asarray(c_rate, dtype=float)
    return 1.0 + ALPHA_C * (c / C_RATE_REF - 1.0)


def compute_stress(
    dod_pct: float | np.ndarray,
    c_rate: float | np.ndarray,
    T_mean_C: float | np.ndarray,
) -> np.ndarray:
    """
    Dimensionless mechanical stress proxy per cycle.

    Args:
        dod_pct  : Depth-of-discharge [%], 0–100.
        c_rate   : C-rate (fraction of 1C, always positive).
        T_mean_C : Mean temperature [°C].

    Returns:
        stress   : Dimensionless, ≥ 0.
    """
    dod = np.asarray(dod_pct, dtype=float) / 100.0   # normalise to [0,1]
    cf  = c_rate_factor(c_rate)
    tf  = arrhenius_factor(T_mean_C)
    stress = dod * np.clip(cf, 0.5, 3.0) * tf        # clip C-rate factor to physical range
    return np.asarray(stress, dtype=float)


def add_stress_column(cycles: pd.DataFrame) -> pd.DataFrame:
    """
    In-place: add 'stress' column to a cycles DataFrame from segment_all().
    C_rate in cycles is already a fraction of 1C (|I_mean| / Q_nominal).
    """
    cycles = cycles.copy()
    cycles["stress"] = compute_stress(
        dod_pct  = cycles["DoD_pct"].values,
        c_rate   = np.abs(cycles["C_rate"].values),
        T_mean_C = cycles["T_mean_C"].values,
    )
    return cycles


if __name__ == "__main__":
    # Quick sanity: check stress increases monotonically with DoD, C-rate, T
    print("=== Stress Model — Sanity Check ===")
    cases = [
        ("baseline (50% DoD, 0.3C, 25°C)",   50, 0.3, 25.0),
        ("high DoD   (80% DoD, 0.3C, 25°C)", 80, 0.3, 25.0),
        ("high C-rate(50% DoD, 0.6C, 25°C)", 50, 0.6, 25.0),
        ("high T     (50% DoD, 0.3C, 45°C)", 50, 0.3, 45.0),
        ("all high   (80% DoD, 0.6C, 45°C)", 80, 0.6, 45.0),
    ]
    for label, dod, cr, T in cases:
        s = compute_stress(dod, cr, T)
        print(f"  {label}: stress = {s:.4f}")
    print("\nExpected: each row > previous row (monotone in each driver)")
