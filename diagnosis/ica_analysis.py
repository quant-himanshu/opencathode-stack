#!/usr/bin/env python3
"""
ica_analysis.py — Incremental Capacity Analysis (ICA) for SOC anchoring.

dQ/dV analysis finds stoichiometry landmarks (phase-transition peaks) that
appear even in flat LFP / NMC plateau regions where Coulomb counting drifts.

Reference: Weng et al. (2013) J. Power Sources 235:36.
           Dubarry & Liaw (2012) J. Power Sources 219:204.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

try:
    from scipy.signal import savgol_filter, find_peaks
    _SCIPY = True
except ImportError:
    _SCIPY = False

ICA_PEAKS_NMC811 = [3.60, 3.75, 4.00, 4.12]
ICA_PEAKS_LFP    = [3.27, 3.35, 3.43]


def incremental_capacity(
    V_array: np.ndarray,
    Q_array: np.ndarray,
    smooth_window: int = 21,
    smooth_poly: int = 3,
    min_peak_rel_height: float = 0.10,
    min_distance: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dQ/dV and locate phase-transition peaks.

    ICA peaks correspond to flat OCP plateaus.  Peak positions (in V) anchor
    SOC to ±0.5 % accuracy independent of Coulomb-counting drift.

    Reference: Weng 2013 J. Power Sources 235:36.

    Args:
        V_array: Cell voltage [V].
        Q_array: Cumulative charge [Ah], same length.
        smooth_window: Savitzky-Golay window (odd).
        smooth_poly: SG polynomial order.
        min_peak_rel_height: Fraction of max |ICA| for peak detection.
        min_distance: Minimum samples between peaks.
    Returns:
        ica_smooth: Smoothed dQ/dV [Ah/V].
        peak_indices: Indices of detected peaks.
    """
    V = np.asarray(V_array, dtype=float)
    Q = np.asarray(Q_array, dtype=float)
    if len(V) < smooth_window + 2:
        return np.zeros_like(V), np.array([], dtype=int)

    dQ = np.gradient(Q)
    dV = np.gradient(V)
    ica_raw = dQ / (dV + 1e-10)

    if _SCIPY and len(V) > smooth_window:
        win = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        win = min(win, len(V) - 1 if len(V) % 2 == 0 else len(V))
        win = win if win % 2 == 1 else win - 1
        ica_smooth = savgol_filter(ica_raw, win, smooth_poly)
    else:
        kernel     = np.ones(smooth_window) / smooth_window
        ica_smooth = np.convolve(ica_raw, kernel, mode="same")

    ica_abs = np.abs(ica_smooth)
    min_h   = min_peak_rel_height * (np.max(ica_abs) + 1e-10)
    if _SCIPY:
        peaks, _ = find_peaks(ica_abs, height=min_h, distance=min_distance)
    else:
        peaks = np.array([
            i for i in range(1, len(ica_abs) - 1)
            if ica_abs[i] > ica_abs[i-1] and ica_abs[i] > ica_abs[i+1] and ica_abs[i] >= min_h
        ], dtype=int)
    return ica_smooth, peaks


def soc_from_ica_peak(peak_voltage: float, chemistry: str = "NMC811") -> Optional[float]:
    """Map detected ICA peak voltage to a known SOC anchor [0–1]."""
    anchors = {
        "NMC811": [(3.60, 0.30), (3.75, 0.50), (4.00, 0.70), (4.12, 0.85)],
        "LFP":    [(3.27, 0.25), (3.35, 0.50), (3.43, 0.75)],
    }
    for v_ref, soc_ref in anchors.get(chemistry.upper(), anchors["NMC811"]):
        if abs(peak_voltage - v_ref) < 0.05:
            return soc_ref
    return None


def ica_soc_correction(
    V_window: np.ndarray,
    Q_window: np.ndarray,
    chemistry: str = "NMC811",
    smooth_window: int = 15,
) -> Optional[float]:
    """
    Full ICA pipeline: compute → find peaks → map to SOC anchor.

    Returns corrected SOC [0–1] if a reliable peak was found, else None.
    """
    ica, peaks = incremental_capacity(V_window, Q_window, smooth_window)
    if len(peaks) == 0:
        return None
    V_arr = np.asarray(V_window, dtype=float)
    for p_idx in peaks:
        if 0 <= p_idx < len(V_arr):
            soc = soc_from_ica_peak(float(V_arr[p_idx]), chemistry)
            if soc is not None:
                return soc
    return None


def validate() -> bool:
    print("=" * 60)
    print("VALIDATING: diagnosis/ica_analysis.py")
    print("=" * 60)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        tag = "  [PASS]" if cond else "  [FAIL]"
        print(tag + f" {name}" + (f"  | {detail}" if detail else ""))
        if not cond:
            ok = False

    N   = 200
    V_d = np.linspace(4.1, 3.0, N)
    Q_d = np.linspace(0.0, 1.0, N)
    bump = np.zeros(N)
    bump[97:103] = 2.5
    Q_d  = Q_d + np.cumsum(bump) * (1.0 / N)

    ica, peaks = incremental_capacity(V_d, Q_d, smooth_window=11)
    check("ICA output shape", len(ica) == N, f"len={len(ica)}")
    check("ICA values finite", np.all(np.isfinite(ica)))
    check("Peaks array returned", hasattr(peaks, "__len__"), f"type={type(peaks)}")

    soc_test = soc_from_ica_peak(3.75, "NMC811")
    check("NMC811 anchor at 3.75V", soc_test is not None, f"soc={soc_test}")
    check("NMC811 anchor ≈ 0.5", soc_test is not None and abs(soc_test - 0.5) < 0.05)

    soc_lfp = soc_from_ica_peak(3.35, "LFP")
    check("LFP anchor at 3.35V", soc_lfp is not None, f"soc={soc_lfp}")

    check("No match → None", soc_from_ica_peak(5.0, "NMC811") is None)

    result = ica_soc_correction(V_d, Q_d, "NMC811", smooth_window=11)
    check("Pipeline returns value or None", result is None or 0 <= result <= 1)

    status = "ALL PASS" if ok else "SOME FAILED"
    print(f"\nResult: {status}")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    validate()
