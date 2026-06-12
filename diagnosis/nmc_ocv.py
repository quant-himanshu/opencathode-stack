"""
nmc_ocv.py — empirical OCV(SOC) extractor from fleet near-rest data.

Approach: collect timesteps where |I_cell| < 2 A (quasi-equilibrium),
bin by SOC, take the median voltage per bin, enforce strict monotone
increasing via the Pool Adjacent Violators algorithm, and fit a Pchip
monotone spline.

This directly implements the approach recommended in
docs/literature_survey.md §1.7 ("extract OCV empirically from fleet data
or adapt generic NMC parameterizations") for fleets whose cell-specific
OCV tables are not publicly available (BMW i3 Samsung SDI, BAIC EU500
CATL, VED AESC Leaf).

Falls back to a generic NMC composite table when fewer than 4 SOC bins
contain enough data.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import PchipInterpolator
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# ── Constants ────────────────────────────────────────────────────────────────
_I_REST_A: float = 2.0       # |I_cell| below this → quasi-OCV [A]
_N_BINS: int = 20            # SOC bins spanning [0, 1]
_MIN_BIN_SAMPLES: int = 3    # minimum raw samples per bin


# Generic NMC composite OCV: mean of NMC111 / NMC622 literature at 25 °C.
# Expected error vs cell-specific curve: 20–50 mV (survey §1.7 Table).
_NMC_SOC = np.array([
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
])
_NMC_OCV = np.array([
    3.400, 3.545, 3.618, 3.657, 3.683, 3.705, 3.723, 3.740, 3.757, 3.773,
    3.789, 3.807, 3.827, 3.850, 3.876, 3.907, 3.942, 3.983, 4.028, 4.079, 4.150,
])

# LMO-NMC composite (for Nissan Leaf AESC cells — blend shifts plateau ~30 mV lower)
_LMONMC_SOC = _NMC_SOC.copy()
_LMONMC_OCV = np.array([
    3.380, 3.520, 3.590, 3.628, 3.655, 3.676, 3.695, 3.712, 3.728, 3.744,
    3.760, 3.778, 3.798, 3.820, 3.845, 3.874, 3.908, 3.947, 3.992, 4.042, 4.110,
])


# ── Internal helpers ─────────────────────────────────────────────────────────

def _pool_adjacent_violators(y: np.ndarray) -> np.ndarray:
    """Enforce non-decreasing order via isotonic regression (PAV algorithm)."""
    y = y.copy().astype(float)
    n = len(y)
    i = 0
    while i < n - 1:
        if y[i + 1] < y[i]:
            j = i + 1
            while j < n - 1 and y[j + 1] < y[i]:
                j += 1
            mean_val = float(np.mean(y[i: j + 1]))
            y[i: j + 1] = mean_val
        else:
            i += 1
    return y


def _generic_table(chemistry: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return the appropriate generic OCV table for a chemistry string."""
    chem = chemistry.upper().replace("-", "").replace("_", "")
    if "LMONMC" in chem or "LMONIC" in chem:
        return _LMONMC_SOC, _LMONMC_OCV
    return _NMC_SOC, _NMC_OCV


# ── Public API ───────────────────────────────────────────────────────────────

def extract_ocv_points(
    segs: List[pd.DataFrame],
    n_series: int,
    n_parallel: int,
    i_rest_thresh: float = _I_REST_A,
    n_bins: int = _N_BINS,
    min_bin_samples: int = _MIN_BIN_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect quasi-OCV (SOC, V_cell) pairs from near-rest timesteps across a
    list of segment DataFrames.

    Returns (soc_pts, ocv_pts) after binning and monotone enforcement.
    Returns empty arrays when insufficient data.
    """
    soc_raw, ocv_raw = [], []
    for seg in segs:
        if not all(c in seg.columns for c in ("I_A", "V_V", "SOC_bms")):
            continue
        I_cell = seg["I_A"].values / max(n_parallel, 1)
        V_cell = seg["V_V"].values / max(n_series, 1)
        soc = seg["SOC_bms"].values
        mask = (np.abs(I_cell) < i_rest_thresh) & np.isfinite(V_cell) & np.isfinite(soc)
        if mask.sum() >= 2:
            soc_raw.append(soc[mask])
            ocv_raw.append(V_cell[mask])

    if not soc_raw:
        return np.array([]), np.array([])

    soc_all = np.concatenate(soc_raw)
    ocv_all = np.concatenate(ocv_raw)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (soc_all >= lo) & (soc_all < hi)
        if idx.sum() >= min_bin_samples:
            centers.append(0.5 * (lo + hi))
            medians.append(float(np.median(ocv_all[idx])))

    if len(centers) < 4:
        return np.array([]), np.array([])

    soc_pts = np.array(centers)
    ocv_pts = _pool_adjacent_violators(np.array(medians))
    return soc_pts, ocv_pts


def build_fleet_ocv(
    segs: List[pd.DataFrame],
    n_series: int,
    n_parallel: int,
    fleet_name: str = "",
    chemistry: str = "NMC",
) -> Tuple[Callable[[float], float], str]:
    """
    Build an empirical OCV(SOC) → V_cell [V] callable from fleet near-rest data.

    Returns (ocv_fn, source_description).

    ocv_fn(soc) → float  (V/cell, soc in [0, 1])

    Falls back to generic NMC/LMO-NMC composite table if < 4 SOC bins are
    populated (survey §1.7).
    """
    soc_pts, ocv_pts = extract_ocv_points(segs, n_series, n_parallel)

    if len(soc_pts) >= 4 and _HAVE_SCIPY:
        interp = PchipInterpolator(soc_pts, ocv_pts, extrapolate=False)
        soc_min = float(soc_pts[0])
        soc_max = float(soc_pts[-1])
        _soc_pts_snap = soc_pts
        _ocv_pts_snap = ocv_pts

        def ocv_fn(soc: float) -> float:
            s = float(np.clip(soc, soc_min, soc_max))
            v = float(interp(s))
            if not np.isfinite(v):
                v = float(np.interp(s, _soc_pts_snap, _ocv_pts_snap))
            return v

        n_rest = sum(
            (np.abs(seg["I_A"].values / max(n_parallel, 1)) < _I_REST_A).sum()
            for seg in segs if "I_A" in seg.columns
        )
        src = (
            f"empirical PCHIP: {len(soc_pts)} SOC bins "
            f"({n_rest} near-rest points, |I_cell|<{_I_REST_A:.0f}A) "
            f"from {fleet_name}. Source: fleet data per survey §1.7."
        )
        return ocv_fn, src

    # Fallback: generic table
    soc_tbl, ocv_tbl = _generic_table(chemistry)

    def ocv_fn_fallback(soc: float) -> float:
        return float(np.interp(np.clip(soc, 0.0, 1.0), soc_tbl, ocv_tbl))

    src = (
        f"generic {chemistry} OCV table "
        f"(only {len(soc_pts)} SOC bins from {fleet_name}, need ≥4). "
        "Fallback per survey §1.5."
    )
    return ocv_fn_fallback, src
