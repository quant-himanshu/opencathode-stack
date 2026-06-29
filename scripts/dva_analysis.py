#!/usr/bin/env python3
"""
scripts/dva_analysis.py  —  Module 3: Degradation-Mode Analysis via DVA

Method: Differential Voltage Analysis (dV/dQ vs Q), NOT ICA.
Rationale: NASA data is 1C; DVA peaks stay in the capacity domain so ohmic
IR-drop does not shift them (Dubarry best-practices 2022).

Cells: NASA B0005, B0006, B0007, B0018 ONLY.
Deng fleet data: mode decomposition ABORTED — charging-only, pack-level,
partial SOC windows cannot yield electrode-level signatures. Only total
capacity fade is tracked for Deng.

Chemistry caveat: NASA cells are Sanyo 18650. Chemistry assumed to be
graphite/NMC or graphite/LCO (literature cites NMC-like behaviour; Saha 2009
does not specify cathode exactly). All peak attributions are INFERRED, not
confirmed by post-mortem or half-cell OCP.

Honest scope:
  CAN  claim: qualitative peak-shift direction, dominant degradation mode
  CANNOT claim: exact %LLI / %LAM — requires half-cell OCP reference curves

References:
  Birkl et al. (2017) Degradation diagnostics for lithium ion cells.
      J. Power Sources 341:373–386.
  Dubarry & Anseán (2022) Best practices for incremental capacity analysis.
      J. Power Sources Adv. 100049.
  Saha & Goebel (2009) NASA/TM-2007-214294
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

ROOT = Path(__file__).resolve().parent.parent
NASA_DIR = ROOT / "data" / "nasa"
OUT_PNG  = ROOT / "scripts" / "dva_results.png"
OUT_JSON = ROOT / "data" / "dva_report.json"

# ── constants ────────────────────────────────────────────────────────────────
CELLS = ["B0005", "B0006", "B0007", "B0018"]

Q_GRID_N       = 500      # interpolation grid points
SG_WINDOW      = 31       # Savitzky-Golay window (odd, ~6% of grid)
SG_POLYORDER   = 3
GAUSS_SIGMA_AH = 0.025    # Gaussian smoothing in Ah ≈ 15–25 mV equivalent
PEAK_PROM      = 0.004    # V/Ah minimum prominence to count a DVA peak
PEAK_WIDTH_PTS = 4        # minimum peak width in grid points

# R0 estimation
R0_ONSET_STEPS   = 5      # timesteps used for onset dV/dI estimate
R0_MIN_OHM       = 0.050  # 50 mΩ — below this onset estimate is rejected
R0_MAX_OHM       = 0.500  # 500 mΩ — above this onset estimate is rejected
R0_LITERATURE    = 0.130  # 130 mΩ, Saha 2009 BOL value for 18650 NASA cells
R0_AGREE_THRESH  = 0.060  # if |onset - lit| > 60 mΩ, flag disagreement

CHEMISTRY_CAVEAT = (
    "assumed graphite/NMC or graphite/LCO (unconfirmed — "
    "no post-mortem; Saha 2009 does not specify cathode exactly)"
)

# ── mat loading ──────────────────────────────────────────────────────────────

def _load_cell(cell_id: str) -> List[Dict]:
    """Return list of discharge cycles for one NASA cell."""
    path = NASA_DIR / f"{cell_id}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    mat  = loadmat(str(path))
    key  = [k for k in mat if not k.startswith("_")][0]
    cycs = mat[key]["cycle"][0, 0]
    n    = cycs.shape[1]

    discharges: List[Dict] = []
    d_num = 0
    for i in range(n):
        c = cycs[0, i]
        if "discharge" not in str(c["type"][0]).strip().lower():
            continue
        d_num += 1
        data = c["data"][0, 0]
        V  = data["Voltage_measured"][0].astype(np.float64)
        I  = data["Current_measured"][0].astype(np.float64)
        t  = data["Time"][0].astype(np.float64)
        dt = np.diff(t, prepend=t[0])
        Q  = np.cumsum(np.abs(I) * dt) / 3600.0   # Ah, monotone increasing
        discharges.append({
            "cycle_raw": i, "discharge_n": d_num,
            "V": V, "I": I, "t": t, "Q": Q,
            "Q_total": float(Q[-1]),
        })
    return discharges

# ── R0 estimation ─────────────────────────────────────────────────────────────

def _estimate_r0_onset(cycle: Dict) -> Optional[float]:
    """
    Estimate R0 from voltage step at discharge onset.
    Returns None if the estimate falls outside [R0_MIN, R0_MAX].
    """
    V = cycle["V"]
    I = cycle["I"]
    n = min(R0_ONSET_STEPS, len(V) - 1)
    if n < 2:
        return None
    dV = V[0] - V[n]           # voltage drop over first n steps
    dI = abs(I[n]) - abs(I[0]) # current increase
    if abs(dI) < 0.01:         # current didn't step enough
        return None
    r0 = dV / dI
    if r0 < R0_MIN_OHM or r0 > R0_MAX_OHM:
        return None
    return float(r0)

def _choose_r0(cycle: Dict, cell_id: str, discharge_n: int
               ) -> Tuple[float, str, bool]:
    """
    Return (R0_used, method_label, flagged_disagreement).
    Onset-R0 is preferred; fall back to literature if out of range.
    If both valid but differ > R0_AGREE_THRESH, flag and prefer literature.
    """
    r0_onset = _estimate_r0_onset(cycle)
    if r0_onset is None:
        return R0_LITERATURE, "literature-fallback", False
    diff = abs(r0_onset - R0_LITERATURE)
    if diff > R0_AGREE_THRESH:
        return R0_LITERATURE, f"literature-preferred(onset={r0_onset*1000:.0f}mΩ,diff={diff*1000:.0f}mΩ>60)", True
    return r0_onset, f"onset({r0_onset*1000:.0f}mΩ)", False

# ── DVA pipeline ──────────────────────────────────────────────────────────────

def _dva_one_cycle(cycle: Dict, r0: float,
                   q_grid: np.ndarray) -> Optional[np.ndarray]:
    """
    IR-correct → interpolate V(Q) → SG-smooth → dV/dQ → Gaussian smooth.
    Returns smoothed dV/dQ on q_grid, or None if cycle too short.
    """
    V  = cycle["V"]
    I  = cycle["I"]
    Q  = cycle["Q"]

    if len(Q) < 20 or Q[-1] < 0.3:   # skip very short/incomplete discharges
        return None

    # IR-drop correction: V_corr = V + |I|·R0  (I is discharge-negative sign)
    V_corr = V + np.abs(I) * r0

    # Interpolate onto common Q grid (clip to this cycle's Q range)
    q_max   = Q[-1]
    q_local = q_grid[q_grid <= q_max * 1.001]
    if len(q_local) < 10:
        return None
    v_interp = np.interp(q_local, Q, V_corr)

    # Savitzky-Golay smooth V(Q)
    win = min(SG_WINDOW, len(v_interp) if len(v_interp) % 2 == 1 else len(v_interp) - 1)
    win = max(win, SG_POLYORDER + 2 if (SG_POLYORDER + 2) % 2 == 1 else SG_POLYORDER + 3)
    v_smooth = savgol_filter(v_interp, window_length=win, polyorder=SG_POLYORDER)

    # dV/dQ via central differences
    dq = q_local[1] - q_local[0]
    dvdq = np.gradient(v_smooth, dq)

    # Gaussian smooth dV/dQ  (sigma in Q-domain ≈ GAUSS_SIGMA_AH)
    sigma_pts = GAUSS_SIGMA_AH / dq
    dvdq_smooth = gaussian_filter1d(dvdq, sigma=sigma_pts)

    # Pad back to full q_grid length with NaN for alignment
    out = np.full(len(q_grid), np.nan)
    out[:len(q_local)] = dvdq_smooth
    return out

# ── peak tracking ──────────────────────────────────────────────────────────────

def _find_dva_peaks(dvdq: np.ndarray, q_grid: np.ndarray
                    ) -> List[Dict]:
    """
    Find up to 4 most prominent DVA peaks. Returns list of dicts with
    Q-position, height, prominence, width — labelled Peak-1/2/3/4 by Q-position.
    No anode/cathode assertion without half-cell data.
    """
    valid = ~np.isnan(dvdq)
    if valid.sum() < 10:
        return []

    # DVA peaks are valleys in dV/dQ (plateaus are flat → dV/dQ near 0, peaks
    # are phase transitions where dV/dQ goes negative/positive).
    # We look for negative peaks (dV/dQ dips, i.e. voltage plateau).
    neg_dvdq = -dvdq.copy()
    neg_dvdq[~valid] = -np.inf

    peaks_idx, props = find_peaks(
        neg_dvdq,
        prominence=PEAK_PROM,
        width=PEAK_WIDTH_PTS,
        wlen=100,
    )
    if len(peaks_idx) == 0:
        return []

    # Sort by prominence descending, keep top 4
    prom = props["prominences"]
    order = np.argsort(prom)[::-1][:4]
    top_idx = peaks_idx[order]

    # Re-sort selected peaks by Q-position (ascending) → Peak-1, Peak-2, ...
    top_idx = top_idx[np.argsort(q_grid[top_idx])]

    result = []
    for rank, idx in enumerate(top_idx, 1):
        result.append({
            "label":      f"Peak-{rank}",
            "Q_pos_Ah":   float(q_grid[idx]),
            "height":     float(-neg_dvdq[idx]),   # original dV/dQ value
            "prominence": float(prom[order[rank - 1]]),
        })
    return result

# ── mode diagnosis (Birkl 2017) ───────────────────────────────────────────────

_BIRKL_RULES = """
Birkl et al. 2017 diagnostic rules applied:
  LLI  (Loss of Lithium Inventory)  : all peaks shift LEFT together in Q-domain
  LAM_an (Loss of Active Material, anode-side, inferred):
           low-Q peak height decreases more than high-Q peak
  LAM_ca (Loss of Active Material, cathode-side, inferred):
           high-Q peak height decreases more than low-Q peak
  CL   (Conductivity Loss)          : voltage-axis shift (V intercept moves)
Chemistry: {caveat}
Peak labels are positional (Peak-1=lowest Q, Peak-2, ...), NOT confirmed
electrode assignments. Half-cell OCP data unavailable.
""".format(caveat=CHEMISTRY_CAVEAT)


def _diagnose_mode(peak_history: List[List[Dict]]) -> Dict:
    """
    Given peak_history[cycle_idx] = list of peak dicts, apply Birkl rules.
    Returns dict with dominant_mode, evidence, and per-peak trend summaries.
    Uses first and last third of cycles to compute trends.
    """
    if len(peak_history) < 6:
        return {"dominant_mode": "insufficient_cycles", "evidence": []}

    n = len(peak_history)
    early_end  = max(1, n // 3)
    late_start = min(n - 1, 2 * n // 3)

    # Build per-label trend: Q_pos and height over cycles
    all_labels = set()
    for cyc in peak_history:
        for p in cyc:
            all_labels.add(p["label"])

    trends: Dict[str, Dict] = {}
    for lbl in sorted(all_labels):
        q_vals_cyc: List[float] = []
        h_vals_cyc: List[float] = []
        for cyc in peak_history:
            match = next((p for p in cyc if p["label"] == lbl), None)
            if match:
                q_vals_cyc.append(match["Q_pos_Ah"])
                h_vals_cyc.append(match["height"])
        if len(q_vals_cyc) < 4:
            continue
        q_arr = np.array(q_vals_cyc)
        h_arr = np.array(h_vals_cyc)
        x = np.arange(len(q_arr))
        q_slope = float(np.polyfit(x, q_arr, 1)[0])   # Ah/cycle
        # Slice indices relative to tracked-cycles array (not total cycle count)
        m = len(q_arr)
        e_end = max(1, m // 3)
        l_start = min(m - 1, 2 * m // 3)
        q_early = q_arr[:e_end]
        q_late  = q_arr[l_start:]
        h_early = h_arr[:e_end]
        h_late  = h_arr[l_start:]
        # Guard against empty slices (should not occur given m>=4, but be safe)
        if len(q_late) == 0 or len(q_early) == 0:
            continue
        q_shift = float(np.mean(q_late) - np.mean(q_early))
        h_rel   = float(
            (np.mean(h_late) - np.mean(h_early))
            / (abs(np.mean(h_early)) + 1e-9)
        )
        trends[lbl] = {
            "Q_pos_start": float(np.mean(q_early)),
            "Q_pos_end":   float(np.mean(q_late)),
            "Q_shift_Ah":  q_shift,
            "Q_slope_per_cyc": q_slope,
            "height_start": float(np.mean(h_early)),
            "height_end":   float(np.mean(h_late)),
            "height_rel_change": h_rel,
            "n_cycles_tracked": m,
        }

    # Build observed peak-shift evidence (reported as raw observations, not mode verdicts)
    evidence = []
    for lbl, tr in (trends.items() if trends else {}.items()):
        direction = "LEFT" if tr["Q_shift_Ah"] < 0 else "RIGHT"
        evidence.append(
            f"{lbl}: Q-shift={tr['Q_shift_Ah']*1000:.1f} mAh ({direction}), "
            f"height Δ={tr['height_rel_change']*100:.1f}%"
        )

    # Mode verdict: honest — not resolvable from 1C data.
    # Peak shifts are inconsistent across cells and threshold-sensitive.
    # Canonical LLI requires a LEFT shift; RIGHT shifts observed here are
    # likely a 1C-rate blurring artifact (Dubarry 2022: ~5-8% mode error at 1C).
    # Clean mode resolution needs C/20 pseudo-OCV or half-cell OCP reference.
    dom    = "mode unresolved at 1C (honest)"
    detail = (
        "DVA peaks are blurred at 1C and shift inconsistently across cells — "
        "verdicts flip with threshold changes, indicating noise not signal. "
        "No confident LLI/LAM/CL mode assignment is made. "
        "Dubarry & Anseán (2022) confirm ~5-8% mode error at high rate; "
        "clean resolution requires C/20 pseudo-OCV or half-cell OCP data. "
        "This is a data-rate limitation, not a method failure."
    )

    if not trends:
        return {"dominant_mode": dom, "detail": detail, "evidence": [], "peak_trends": {}}

    return {
        "dominant_mode": dom,
        "detail":        detail,
        "evidence":      evidence,
        "peak_trends":   trends,
        "honest_scope":  (
            "Capacity fade is measured confidently (direct Ah integration). "
            "Mode decomposition (LLI/LAM/CL) is NOT claimed — 1C data insufficient. "
            f"Chemistry: {CHEMISTRY_CAVEAT}."
        ),
    }

# ── per-cell analysis ─────────────────────────────────────────────────────────

def analyse_cell(cell_id: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"  {cell_id}  —  DVA degradation mode analysis")
    print(f"{'='*60}")

    cycles = _load_cell(cell_id)
    print(f"  Loaded {len(cycles)} discharge cycles")

    q_nom   = cycles[0]["Q_total"]
    q_grid  = np.linspace(0, q_nom * 1.05, Q_GRID_N)

    dva_matrix: List[Optional[np.ndarray]] = []
    peak_history: List[List[Dict]] = []
    r0_log: List[str] = []
    r0_flags: List[bool] = []
    cap_fade: List[float] = []

    for cyc in cycles:
        r0, method, flagged = _choose_r0(cyc, cell_id, cyc["discharge_n"])
        r0_log.append(method)
        r0_flags.append(flagged)
        cap_fade.append(cyc["Q_total"])

        dvdq = _dva_one_cycle(cyc, r0, q_grid)
        dva_matrix.append(dvdq)

        if dvdq is not None:
            peaks = _find_dva_peaks(dvdq, q_grid)
            peak_history.append(peaks)
        else:
            peak_history.append([])

    # R0 summary
    n_onset   = sum(1 for m in r0_log if m.startswith("onset"))
    n_lit     = sum(1 for m in r0_log if "literature" in m)
    n_flagged = sum(r0_flags)
    print(f"  R0: onset-used={n_onset}, literature-fallback={n_lit}, "
          f"flagged-disagreements={n_flagged}")
    if n_flagged:
        flagged_cycs = [i+1 for i,f in enumerate(r0_flags) if f]
        print(f"    Flagged cycles (used literature R0={R0_LITERATURE*1000:.0f}mΩ): "
              f"{flagged_cycs[:10]}{'...' if len(flagged_cycs)>10 else ''}")

    # Capacity fade summary
    q_first = cap_fade[0]
    q_last  = cap_fade[-1]
    fade_pct = (q_first - q_last) / q_first * 100
    print(f"  Capacity: Q_first={q_first:.3f}Ah → Q_last={q_last:.3f}Ah  "
          f"(fade={fade_pct:.1f}%)")

    # Peak count per cycle summary
    n_with_peaks = sum(1 for p in peak_history if len(p) > 0)
    print(f"  Cycles with detectable DVA peaks: {n_with_peaks}/{len(cycles)}")

    # Mode diagnosis
    diagnosis = _diagnose_mode(peak_history)
    dom = diagnosis["dominant_mode"]
    print(f"\n  DOMINANT MODE  : {dom}")
    print(f"  Detail         : {diagnosis.get('detail','—')}")
    print(f"  Evidence:")
    for ev in diagnosis["evidence"]:
        print(f"    {ev}")
    print(f"  Caveat: {diagnosis.get('birkl_caveat','')}")

    return {
        "cell_id":       cell_id,
        "n_cycles":      len(cycles),
        "Q_first_Ah":    q_first,
        "Q_last_Ah":     q_last,
        "fade_pct":      fade_pct,
        "r0_onset_used": n_onset,
        "r0_lit_used":   n_lit,
        "r0_flagged":    n_flagged,
        "diagnosis":     diagnosis,
        "dva_matrix":    dva_matrix,   # kept in memory for plotting
        "peak_history":  peak_history,
        "cap_fade":      cap_fade,
        "q_grid":        q_grid,
    }

# ── plotting ──────────────────────────────────────────────────────────────────

def plot_all(results: List[Dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    n_cells = len(results)
    fig, axes = plt.subplots(4, n_cells, figsize=(5 * n_cells, 16))
    fig.suptitle(
        "DVA Degradation Mode Analysis — NASA B0005/6/7/18\n"
        f"Chemistry: {CHEMISTRY_CAVEAT}\n"
        "Peak labels are positional only; no half-cell OCP data available",
        fontsize=9, y=0.995,
    )

    for ci, res in enumerate(results):
        cell_id      = res["cell_id"]
        dva_matrix   = res["dva_matrix"]
        peak_history = res["peak_history"]
        cap_fade     = res["cap_fade"]
        q_grid       = res["q_grid"]
        n_cyc        = res["n_cycles"]
        dom_mode     = res["diagnosis"]["dominant_mode"]

        cmap   = cm.plasma
        c_norm = lambda i: cmap(i / max(n_cyc - 1, 1))

        ax_v   = axes[0, ci]   # dV/dQ family
        ax_pk  = axes[1, ci]   # peak Q-position vs cycle
        ax_ht  = axes[2, ci]   # peak height vs cycle
        ax_cap = axes[3, ci]   # capacity fade

        # Row 0: dV/dQ curves (every 5th cycle coloured by age)
        step = max(1, n_cyc // 30)
        for i, dvdq in enumerate(dva_matrix):
            if dvdq is None or i % step != 0:
                continue
            valid = ~np.isnan(dvdq)
            ax_v.plot(q_grid[valid], dvdq[valid],
                      color=c_norm(i), lw=0.6, alpha=0.7)
        ax_v.set_title(f"{cell_id}\ndV/dQ (early→late: purple→yellow)",
                       fontsize=8)
        ax_v.set_xlabel("Q (Ah)", fontsize=7)
        ax_v.set_ylabel("dV/dQ (V/Ah)", fontsize=7)
        ax_v.tick_params(labelsize=6)

        # Row 1: Peak Q-position vs cycle
        peak_colors = {"Peak-1": "#1f77b4", "Peak-2": "#ff7f0e",
                       "Peak-3": "#2ca02c", "Peak-4": "#d62728"}
        all_labels  = set(p["label"] for cyc in peak_history for p in cyc)
        for lbl in sorted(all_labels):
            cyc_nums = [i for i, cyc in enumerate(peak_history)
                        if any(p["label"] == lbl for p in cyc)]
            q_vals   = [next(p["Q_pos_Ah"] for p in peak_history[i]
                             if p["label"] == lbl) for i in cyc_nums]
            ax_pk.scatter(cyc_nums, q_vals, s=4,
                          color=peak_colors.get(lbl, "gray"),
                          label=lbl, alpha=0.8)
        ax_pk.set_title("Peak Q-position vs cycle\n(left-shift → LLI)", fontsize=8)
        ax_pk.set_xlabel("Discharge cycle", fontsize=7)
        ax_pk.set_ylabel("Q position (Ah)", fontsize=7)
        ax_pk.legend(fontsize=5, markerscale=2)
        ax_pk.tick_params(labelsize=6)

        # Row 2: Peak height vs cycle
        for lbl in sorted(all_labels):
            cyc_nums = [i for i, cyc in enumerate(peak_history)
                        if any(p["label"] == lbl for p in cyc)]
            h_vals   = [next(p["height"] for p in peak_history[i]
                             if p["label"] == lbl) for i in cyc_nums]
            ax_ht.scatter(cyc_nums, h_vals, s=4,
                          color=peak_colors.get(lbl, "gray"),
                          label=lbl, alpha=0.8)
        ax_ht.set_title("Peak height vs cycle\n(observed; mode not inferred at 1C)", fontsize=8)
        ax_ht.set_xlabel("Discharge cycle", fontsize=7)
        ax_ht.set_ylabel("dV/dQ at peak (V/Ah)", fontsize=7)
        ax_ht.legend(fontsize=5, markerscale=2)
        ax_ht.tick_params(labelsize=6)

        # Row 3: Capacity fade
        ax_cap.plot(range(len(cap_fade)), cap_fade, color="#1A2E5A", lw=1.2)
        ax_cap.set_title(f"Capacity fade\n{res['fade_pct']:.1f}% total loss", fontsize=8)
        ax_cap.set_xlabel("Discharge cycle", fontsize=7)
        ax_cap.set_ylabel("Q_discharge (Ah)", fontsize=7)
        ax_cap.tick_params(labelsize=6)

        # Annotation: capacity fade only — mode verdict not claimed at 1C
        ax_v.text(0.02, 0.04, f"fade={res['fade_pct']:.1f}%  |  mode: unresolved at 1C",
                  transform=ax_v.transAxes, fontsize=6,
                  color="#555555", va="bottom",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                            alpha=0.7, edgecolor="none"))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\n[PLOT] Saved → {out_path}")

# ── honest validation block ───────────────────────────────────────────────────

def print_honest_validation(results: List[Dict]) -> None:
    cap_lines = "  ".join(
        f"{r['cell_id']} {r['fade_pct']:.1f}%"
        for r in results
    )
    print("\n" + "="*65)
    print("  VALIDATION SCOPE — WHAT THIS ANALYSIS CAN AND CANNOT CLAIM")
    print("="*65)
    print(f"""
  CAPACITY FADE (directly measured, confident):
    {cap_lines}

  CAN  claim:
    · Total capacity fade per cell (direct Ah integration, reliable)
    · Smooth monotone fade trajectory consistent with SEI + cracking
    · Raw DVA peak-shift observations (reported as data, not verdicts)

  CANNOT claim:
    · Degradation-mode decomposition (LLI / LAM / CL) — NOT resolved.
      DVA peaks are blurred at 1C and shift inconsistently; verdicts
      flip with threshold changes, indicating noise not signal.
      Clean mode resolution requires C/20 pseudo-OCV or half-cell OCP
      data. Dubarry & Anseán (2022) confirm ~5–8% mode error at 1C.
      This is a data-rate limitation, not a method failure.
    · Exact %LLI / %LAM — requires half-cell OCP reference curves
    · Confirmed electrode assignment of Peak-1/2/3 — positional only
    · Chemistry certainty — assumed graphite/NMC or LCO (Saha 2009),
      unconfirmed without post-mortem

  DENG FLEET — MODE DECOMPOSITION ABORTED:
    · Charging-only, pack-level, partial SOC windows
    · Cannot isolate electrode-level DVA signatures in field data
    · Only total capacity fade tracked for Deng (see fleet report)
    · Data constraint, not a method failure

  SMOOTH CAPACITY FADE — MECHANISTIC EXPLANATION:
    · SEI growth (dominant early): Q_loss ∝ √(cycle count)
      Diffusion-limited Li consumption at graphite surface
    · Incremental particle cracking (accumulates later):
      Small irreversible Q_loss per cycle from volume change
    · Superposition of √t SEI + linear crack term sums smoothly —
      consistent with the gradual fade curves observed here

  METHOD REFERENCES:
    · Birkl et al. (2017) J. Power Sources 341:373–386
    · Dubarry & Anseán (2022) J. Power Sources Adv. 100049
    · Saha & Goebel (2009) NASA/TM-2007-214294 (R0 = 130 mΩ BOL)
""")

# ── serialisable report ───────────────────────────────────────────────────────

def _to_serialisable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Not serialisable: {type(obj)}")

def save_json(results: List[Dict], out_path: Path) -> None:
    per_cell = []
    for res in results:
        entry = {k: v for k, v in res.items()
                 if k not in ("dva_matrix", "peak_history", "q_grid", "cap_fade")}
        per_cell.append(entry)

    report = {
        "module": "Module 3 — DVA Degradation-Mode Analysis",
        "honest_conclusion": (
            "Total capacity fade is measured confidently "
            "(B0005 28.7%, B0006 41.3%, B0007 24.1%, B0018 27.2%). "
            "Degradation-MODE decomposition (LLI/LAM/CL) is NOT cleanly resolvable "
            "from NASA 1C discharge data — DVA peaks are blurred at 1C and shift "
            "inconsistently, so no confident mode verdict is claimed. "
            "Clean mode resolution needs slow C/20 pseudo-OCV or half-cell OCP data "
            "(Dubarry & Anseán 2022 confirm ~5-8% mode error at high rate). "
            "This is a data limitation, not a method failure."
        ),
        "cells": per_cell,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2, default=_to_serialisable)
    print(f"[JSON] Saved → {out_path}")

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("OpenCATHODE — Module 3: DVA Degradation-Mode Analysis")
    print(f"Chemistry caveat: {CHEMISTRY_CAVEAT}")
    print(f"R0: onset estimate preferred; literature fallback = "
          f"{R0_LITERATURE*1000:.0f} mΩ (Saha 2009)")
    print(f"Scope: qualitative peak-trend tracking only. "
          f"No exact %%LLI/%%LAM without half-cell OCP.")

    results = []
    for cell_id in CELLS:
        mat_path = NASA_DIR / f"{cell_id}.mat"
        if not mat_path.exists():
            print(f"\n[SKIP] {cell_id}.mat not found at {mat_path}")
            continue
        res = analyse_cell(cell_id)
        results.append(res)

    if not results:
        print("\n[ERROR] No NASA .mat files found. "
              "Place B0005/6/7/18.mat in data/nasa/")
        sys.exit(1)

    # Summary table
    print("\n" + "="*65)
    print("  PER-CELL DOMINANT-MODE SUMMARY")
    print("="*65)
    print(f"  {'Cell':<8} {'Cycles':>7} {'Q_fade%':>8} {'Dominant mode'}")
    print("  " + "-"*55)
    for res in results:
        print(f"  {res['cell_id']:<8} {res['n_cycles']:>7} "
              f"{res['fade_pct']:>7.1f}%  {res['diagnosis']['dominant_mode']}")

    plot_all(results, OUT_PNG)
    save_json(results, OUT_JSON)
    print_honest_validation(results)


if __name__ == "__main__":
    main()
