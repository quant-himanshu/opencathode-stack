"""
ica_decomposition.py — Module 3: LLI / LAM / CL decomposition via IC analysis.

WHAT THIS MODULE DOES — AND WHAT IT CANNOT DO
═══════════════════════════════════════════════
Incremental Capacity (IC = dQ/dV) analysis is the standard method for
identifying the three canonical degradation modes in Li-ion cells:

  LLI  Loss of Lithium Inventory   → IC peak POSITION shifts (stoichiometric
                                       window narrows → staging transitions
                                       occur at lower full-cell voltage)
  LAM  Loss of Active Material     → IC peak AREA/HEIGHT shrinks (fewer sites
                                       participate in the phase transition)
  CL   Conductivity Loss           → resistance increase; tracked separately
                                       from impedance spectroscopy (Re, Rct)

CRITICAL FINDING — PEAK AREA IS NEARLY REDUNDANT WITH SOH
══════════════════════════════════════════════════════════
Pre-coding diagnostics on NASA PCoE B0005/B0006/B0007/B0018 showed that
A(n)/A₀ (dominant IC peak area fraction) tracks SOH(n) with r ≈ 0.99 and
OLS R² ≈ 0.97–0.99. This means:

    ΔA_rel = (A₀ − A(n)) / A₀  ≈  ΔSOH = SOH(0) − SOH(n)   [by algebra]

Using ΔA_rel as an independent predictor of ΔSOH in a regression would be
CIRCULAR — it would explain ~99% of variance by essentially predicting ΔSOH
from itself. A reported "LAM fraction %" derived from such a regression would
be a statistical artifact, not a physical finding.

THE ACTUAL QUANTITATIVE DELIVERABLE OF THIS MODULE
═══════════════════════════════════════════════════
What IS new information beyond SOH is the SLOPE of A(n)/A₀ vs SOH(n):

    OLS:  A(n)/A₀ = m · SOH(n) + b

  m = 1.0  → peak shrinks proportionally with capacity: cannot distinguish
              LLI (stoichiometric) from LAM (structural) — uniform fade.
  m > 1.0  → peak shrinks FASTER than capacity: the phase transition is
              selectively losing active sites beyond what proportional
              capacity loss alone explains → genuine LAM-adjacent signal.
  m < 1.0  → peak persists despite capacity loss: resistance-dominated fade
              (IR-drop shifts peaks without destroying phase structure).

For the NASA cells, m ≈ 1.9 across all cells (pre-diagnostic finding).
This means: for each 1% of total capacity lost, the dominant IC peak loses
~1.9% of its initial height. This excess beyond 1.0 is the module's
quantitative contribution — not the raw peak-area percentage, which is
redundant with the SOH value already known from capacity measurement.

FURTHER LIMITATIONS
═══════════════════
• K-COUNT INSTABILITY vs DOMINANT-PEAK STABILITY (two different claims):
  Total K (peak count) varies 3–12 across cells at prom=0.015. This is
  driven by noise peaks that appear/disappear in the 2.7–3.4 V region,
  typically >150 mV (median >380 mV) from the dominant peak. This K
  variation does NOT contaminate the dominant-peak tracker (confirmed
  cycle-by-cycle for all cells — 0 jumps >50 mV in any cell). A spot-
  check at 3 cycles (1, mid, final) with prom=0.02 reported K=5 stable;
  the full-run K distribution was then investigated and the above
  two-part characterization was confirmed. The headline slope_m and Δμ
  depend on dominant-peak tracking quality, not total K count.

• B0005/B0007: K=3–10 and K=2–10 respectively; all extra peaks are noise
  structures far from the dominant peak. Dominant peak: 0 jumps >50 mV,
  0 jumps >25 mV. QUANTITATIVE results stand.

• B0018: K=4–12; extra peaks at high-K cycles (4/132) are noise >150 mV
  from dominant (median separation 389 mV), dominant peak matches
  surrounding trajectory within ≤14 mV at those cycles. Two oscillations
  of ±30 mV at cycle windows 72–77 and 123–129 were investigated:
  tracker bounces among closely-spaced peaks all within ±30 mV of the
  true dominant peak — noise in peak-position estimate, not a jump to a
  different peak. QUANTITATIVE status confirmed.

• B0006: peak count K varies (2–11) with total drift 2× larger than other
  cells (171 mV vs 52–73 mV). New low-voltage peaks emerge and merge over
  its lifetime. Dominant-peak identity genuinely shifts; this is NOT noise-
  peak variation. Peak identity is ambiguous; slope ratio is unreliable.
  B0006 results are QUALITATIVE ONLY.

• B0007: the LLI feature (|Δμ/σ|) and LAM feature (ΔA_rel) have
  r = −0.927 — not from peak identity swapping (verified: smooth monotone
  drift, no jumps > 50 mV confirmed cycle-by-cycle) but from structural
  co-monotonicity: both features are monotone functions of cycle number,
  so they are correlated even with perfect tracking. LLI and LAM cannot
  be quantitatively separated for B0007. Single-factor reporting only.

• No half-cell reference data exists for these cells. True LLI/LAM
  decomposition requires fitting U_pos(x) − U_neg(y) with stoichiometric
  parameters (Dubarry & Anseán 2022). Without it, all mode attributions
  are phenomenological and have ±15–25% uncertainty at minimum.

REFERENCES
══════════
Dubarry M, Anseán D. "Best practices for incremental capacity analysis."
  Front. Energy Res. 10:1023555 (2022). doi:10.3389/fenrg.2022.1023555
Sulzer V, et al. "The challenge and opportunity of battery lifetime
  prediction from field data." Joule 5(8):1934-1955 (2021).
  doi:10.1016/j.joule.2021.06.005
Saha B, Goebel K. NASA Ames Prognostics Data Repository (2007).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress


# ── IC computation ─────────────────────────────────────────────────────────────

def compute_ic(
    V_raw: np.ndarray,
    I_raw: np.ndarray,
    t_raw: np.ndarray,
    v_min: float = 2.70,
    v_max: float = 4.25,
    dv:    float = 0.005,
    sg_w:  int   = 15,
    sg_p:  int   = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IC = dQ/dV from raw discharge time-series.

    Steps:
      1. Coulomb-count Q(t) = ∫|I|dt
      2. Savitzky-Golay smooth V(t) to suppress sensor noise before binning
      3. Bin dQ into uniform voltage grid; divide by ΔV
      4. Interpolate sparse bins; apply second SG smooth

    Returns:
        v_grid : uniform voltage axis [V]
        ic     : smoothed dQ/dV [Ah/V], ≥ 0
    """
    dt  = np.diff(t_raw, prepend=t_raw[0]); dt[0] = 0.0
    dQ  = np.abs(I_raw) * dt / 3600.0
    V_s = savgol_filter(V_raw, sg_w, sg_p) if len(V_raw) > sg_w else V_raw.copy()

    v_grid = np.arange(v_min, v_max, dv)
    ic     = np.zeros(len(v_grid))
    counts = np.zeros(len(v_grid))
    for j in range(len(V_s)):
        bi = int((V_s[j] - v_min) / dv)
        if 0 <= bi < len(ic):
            ic[bi] += dQ[j]
            counts[bi] += 1

    valid = counts > 0
    ic[valid] /= dv
    ic[~valid] = np.nan
    nans = np.isnan(ic)
    if nans.any() and (~nans).sum() > 2:
        ic[nans] = np.interp(v_grid[nans], v_grid[~nans], ic[~nans])

    sg_w2 = min(21, (len(ic) // 3) * 2 + 1)
    if sg_w2 % 2 == 0:
        sg_w2 += 1
    ic = savgol_filter(ic, sg_w2, 3)
    return v_grid, np.clip(ic, 0.0, None)


# ── Peak tracking ──────────────────────────────────────────────────────────────

def _find_all_peaks(
    ic: np.ndarray,
    v_grid: np.ndarray,
    prom: float = 0.015,
    wid:  int   = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (voltages, heights) of all peaks above prominence threshold."""
    pk, _ = find_peaks(ic, prominence=prom, width=wid)
    if len(pk) == 0:
        return np.array([]), np.array([])
    return v_grid[pk], ic[pk]


def track_dominant_peak(
    discs: List[dict],
    prox_tol: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track the dominant IC peak (highest height at cycle 1) across all discharge
    cycles using proximity-based correspondence (nearest peak within ±prox_tol V).

    Proximity tracking prevents rank-swap artifacts: if a new low-voltage peak
    emerges and becomes the new tallest peak, rank-order tracking would
    silently jump to it. Proximity tracking stays anchored to the original peak
    unless it physically disappears.

    Returns:
        mu_arr   : peak voltage per cycle [V], NaN where lost
        A_arr    : peak height per cycle [Ah/V], NaN where lost
        valid    : boolean mask of cycles where peak was found
    """
    d0 = discs[0]["data"]
    vg0, ic0 = compute_ic(
        np.asarray(d0["Voltage_measured"], dtype=float).ravel(),
        np.asarray(d0["Current_measured"], dtype=float).ravel(),
        np.asarray(d0["Time"],             dtype=float).ravel(),
    )
    pvs0, phs0 = _find_all_peaks(ic0, vg0)
    if len(pvs0) == 0:
        raise RuntimeError("No IC peaks found in cycle 1 — check data.")
    dom_idx = int(np.argmax(phs0))
    prev_v  = pvs0[dom_idx]

    mu_arr = np.full(len(discs), np.nan)
    A_arr  = np.full(len(discs), np.nan)

    for n, c in enumerate(discs):
        d    = c["data"]
        V    = np.asarray(d["Voltage_measured"], dtype=float).ravel()
        I    = np.asarray(d["Current_measured"], dtype=float).ravel()
        t    = np.asarray(d["Time"],             dtype=float).ravel()
        vg, ic = compute_ic(V, I, t)
        pvs, phs = _find_all_peaks(ic, vg)
        if len(pvs) > 0:
            dists = np.abs(pvs - prev_v)
            ci    = int(np.argmin(dists))
            if dists[ci] <= prox_tol:
                mu_arr[n] = pvs[ci]
                A_arr[n]  = phs[ci]
                prev_v    = pvs[ci]

    valid = ~np.isnan(mu_arr)
    return mu_arr, A_arr, valid


def count_peaks_per_cycle(discs: List[dict], prom: float = 0.015, wid: int = 3) -> np.ndarray:
    """Return K (peak count) per cycle — used for B0006 stability diagnosis."""
    Ks = []
    for c in discs:
        d  = c["data"]
        vg, ic = compute_ic(
            np.asarray(d["Voltage_measured"], dtype=float).ravel(),
            np.asarray(d["Current_measured"], dtype=float).ravel(),
            np.asarray(d["Time"],             dtype=float).ravel(),
        )
        pvs, _ = _find_all_peaks(ic, vg, prom=prom, wid=wid)
        Ks.append(len(pvs))
    return np.array(Ks)


# ── Slope-ratio analysis (the quantitative deliverable) ────────────────────────

def compute_slope_ratio(
    A_arr:   np.ndarray,
    soh_arr: np.ndarray,
    A0:      float,
) -> Dict:
    """
    OLS regression: A(n)/A₀ = m · SOH(n) + b

    The slope m is the module's primary quantitative output.

      m ≈ 1.0  : uniform degradation — peak loss proportional to capacity loss.
                 LLI and LAM cannot be separated (no selective phase loss).
      m > 1.0  : peak shrinks FASTER than capacity (genuine LAM-adjacent signal).
      m < 1.0  : peak persists relative to capacity (resistance/CL dominated).

    Also computes:
      r(A/A₀, SOH) — confirms circularity when close to 1.0
      Excess LAM rate — (m − 1) × 100%, the fraction of peak loss per unit
        SOH loss that exceeds proportional expectation.
    """
    valid = ~np.isnan(A_arr) & ~np.isnan(soh_arr)
    A_v   = A_arr[valid]
    S_v   = soh_arr[valid]
    ratio = A_v / A0          # A(n)/A₀

    slope, intercept, r_val, p_val, se = linregress(S_v, ratio)
    r2 = r_val ** 2

    return {
        "slope_m"          : round(float(slope),     4),
        "intercept_b"      : round(float(intercept), 4),
        "r2_A_vs_SOH"      : round(r2,               4),
        "r_A_vs_SOH"       : round(float(r_val),     4),
        "n_cycles"         : int(valid.sum()),
        "excess_LAM_rate_pct": round((float(slope) - 1.0) * 100.0, 1),
        "circularity_flag" : bool(r2 > 0.90),
        "slope_interpretation": (
            "peak shrinks FASTER than capacity → genuine LAM-adjacent signal"
            if slope > 1.1 else
            "peak shrinks proportionally → uniform fade, LLI/LAM not separable"
            if 0.9 <= slope <= 1.1 else
            "peak persists relative to capacity → possible CL/resistance dominated"
        ),
    }


# ── EIS / CL extraction ────────────────────────────────────────────────────────

def load_eis_per_discharge(cycs: list) -> Dict:
    """
    Match each discharge cycle to the nearest preceding impedance cycle.
    Returns per-discharge arrays of Re [Ω] and Rct [Ω] (NaN where unmatched).

    NASA stores Re and Rct directly from EIS — no ΔV/I approximation needed.
    B0005/B0006/B0007: first 19 discharges have no preceding EIS (EIS starts
    at cycle index 40). B0018: all 132 discharges have a preceding EIS.

    NOTE on B0018 Rct: the first EIS cycle shows anomalously high Rct
    (95–102 mΩ in first 3 cycles, dropping to ~84 mΩ steady state by cycle 15).
    This appears to be a temperature/conditioning artifact. Use robust delta
    (mean last 10% minus mean first 10% of matched cycles) rather than
    first-vs-last when reporting ΔRct for this cell.
    """
    types = [c["type"] for c in cycs]
    discharge_idx  = [i for i, t in enumerate(types) if t == "discharge"]
    impedance_idx  = [i for i, t in enumerate(types) if t == "impedance"]

    eis_records = []
    for ii in impedance_idx:
        d = cycs[ii]["data"]
        Re  = float(np.asarray(d["Re"]).ravel()[0])
        Rct = float(np.asarray(d["Rct"]).ravel()[0])
        eis_records.append((ii, Re, Rct))

    Re_arr  = np.full(len(discharge_idx), np.nan)
    Rct_arr = np.full(len(discharge_idx), np.nan)
    matched = 0

    for k, di in enumerate(discharge_idx):
        prev_eis = [(ii, Re, Rct) for ii, Re, Rct in eis_records if ii < di]
        if prev_eis:
            _, Re, Rct = prev_eis[-1]
            Re_arr[k]  = Re
            Rct_arr[k] = Rct
            matched += 1

    # Robust delta: mean of first 10% vs mean of last 10% of matched values
    valid_Re  = Re_arr[~np.isnan(Re_arr)]
    valid_Rct = Rct_arr[~np.isnan(Rct_arr)]
    tail      = max(1, len(valid_Re) // 10)
    dRe_robust  = float(np.mean(valid_Re[-tail:])  - np.mean(valid_Re[:tail]))  if len(valid_Re)  > 5 else np.nan
    dRct_robust = float(np.mean(valid_Rct[-tail:]) - np.mean(valid_Rct[:tail])) if len(valid_Rct) > 5 else np.nan

    return {
        "Re_Ohm"        : Re_arr,
        "Rct_Ohm"       : Rct_arr,
        "n_matched"     : matched,
        "n_total"       : len(discharge_idx),
        "coverage_pct"  : round(matched / max(len(discharge_idx), 1) * 100.0, 1),
        "dRe_robust_Ohm"  : dRe_robust,
        "dRct_robust_Ohm" : dRct_robust,
    }
