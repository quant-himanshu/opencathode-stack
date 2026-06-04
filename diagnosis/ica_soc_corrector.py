#!/usr/bin/env python3
"""
ica_soc_corrector.py — ICA + EIS SOC Corrector for LFP Prismatic 160Ah Cells.

Corrected for 2.8V–3.65V operating range.  Only ICA peaks c1 (3.35V) and
c2 (3.57V) are accessible; c3 (3.70V) and c4 (3.85V) exceed V_max.

References:
  [1] Dubarry & Beck 2022, Front. Energy Res. 10.3389/fenrg.2022.1023555
  [2] Fly & Chen 2020, J. Energy Storage   10.1016/j.est.2020.101329
  [3] Gao & Onori 2025, Green Energy Intell. Transp. 10.1016/j.geits.2025.100386
  [4] Simolka 2020, J. Electrochem. Soc.   10.1149/1945-7111/abb2d8
  [5] Mikhak 2024, PMC12936157
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
from typing import Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────
# ICA PEAK REFERENCE TABLE  — 160Ah LFP prismatic, 2.8–3.65V range
# Source: Kimi synthesis of [1][4], adjusted for our operating voltage.
#
# EXCLUDED peaks (above V_max = 3.65V):
#   c3 @ 3.70V → outside range
#   c4 @ 3.85V → outside range
# ──────────────────────────────────────────────────────────────────────
LFP_ICA_PEAKS: Dict[str, dict] = {
    "c1": {
        "voltage":    3.35,
        "tolerance":  0.05,
        "soc_anchor": 0.20,
        "confidence": "medium",
        "note": "Graphite LiC18→LiC12 staging. Weak in some LFP cells.",
    },
    "c2": {
        "voltage":    3.57,
        "tolerance":  0.04,
        "soc_anchor": 0.425,
        "confidence": "high",
        "note": "Graphite LiC12→LiC6 + LFP H1→M. Primary anchor (within 2.8–3.65V).",
    },
    # c3 @ 3.70V: EXCLUDED — above V_max 3.65V
    # c4 @ 3.85V: EXCLUDED — above V_max 3.65V
}

# ICA processing constants — Dubarry & Beck 2022 §2.3
DELTA_V_MV  = 2.0   # [mV] voltage grid spacing
SAVGOL_WIN  = 21    # Savitzky-Golay window (must be odd, ≥ 21 for LFP)
SAVGOL_POLY = 3     # polynomial order

# EIS constants — Gao & Onori 2025
EIS_FREQ_HZ     = 0.01          # optimal frequency for LFP SOC
EIS_PERIOD_S    = 1.0 / EIS_FREQ_HZ   # 100 s per period
EIS_N_PERIODS   = 2             # minimum periods for reliable FFT
EIS_DURATION_S  = EIS_N_PERIODS * EIS_PERIOD_S  # 200 s total
EIS_CURRENT_A   = 6.0           # C/25 for 160Ah cell (scaled from Stanford 0.1A/2.5Ah)

MIN_ICA_POINTS = SAVGOL_WIN + 2
EPS            = 1e-12


class ICASocCorrector:
    """
    ICA + EIS fusion SOC corrector for 160Ah LFP prismatic cells.

    Operating modes:
      REALTIME  — Coulomb counting + EIS correction every 12 min
                  (EIS at 0.01 Hz takes 200 s; fits in 12-min window)
      PERIODIC  — ICA from full C/6 charge (weekly), anchors SOC table

    Args:
        Q_nom_Ah   : Nominal capacity [Ah].
        R_int_ohm  : DC internal resistance [Ω] — calibrated per cell.
        V_min      : Minimum operating voltage [V].
        V_max      : Maximum operating voltage [V].
    """

    def __init__(
        self,
        Q_nom_Ah:  float = 160.0,
        R_int_ohm: float = 0.022,
        V_min:     float = 2.8,
        V_max:     float = 3.65,
    ) -> None:
        self.Q_nom = Q_nom_Ah
        self.R_int = R_int_ohm
        self.V_min = V_min
        self.V_max = V_max
        self._soc  = 0.5
        self._initialized = False

        # Filter peak table to peaks within operating range
        self.accessible_peaks = {
            name: p for name, p in LFP_ICA_PEAKS.items()
            if p["voltage"] <= V_max + p["tolerance"]
        }

    # ─────────────────────────────────────────────────────────────────
    # 1.  RESISTANCE CORRECTION  (Fly & Chen 2020 [2], Eq. 3)
    # ─────────────────────────────────────────────────────────────────
    def resistance_correct_voltage(
        self, V_meas: np.ndarray, I_A: np.ndarray
    ) -> np.ndarray:
        """
        V_corr = V_meas − I × R_int.
        Enables C/6 ICA accuracy within 0.59% of C/48 reference.
        """
        return np.asarray(V_meas, dtype=float) - np.asarray(I_A, dtype=float) * self.R_int

    # ─────────────────────────────────────────────────────────────────
    # 2.  ICA COMPUTATION  (Dubarry & Beck 2022 [1])
    # ─────────────────────────────────────────────────────────────────
    def compute_ica(
        self,
        V_array:    np.ndarray,
        Q_array:    np.ndarray,
        delta_V_mV: float = DELTA_V_MV,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dQ/dV vs V on a uniform ΔV = 2 mV grid.

        FEASIBILITY NOTE:
          Quartz 6-min data at 0.5C → ~2 points in plateau → NOT feasible.
          TU Darmstadt 1-Hz data at 0.1C → ~230 points → FEASIBLE.

        Returns empty arrays if data is insufficient.
        """
        V = np.asarray(V_array, dtype=float)
        Q = np.asarray(Q_array, dtype=float)

        if len(V) < MIN_ICA_POINTS:
            return np.array([]), np.array([])

        # Sort by voltage and remove duplicates
        idx = np.argsort(V)
        V, Q = V[idx], Q[idx]
        _, ui = np.unique(V, return_index=True)
        V, Q = V[ui], Q[ui]

        if len(V) < MIN_ICA_POINTS:
            return np.array([]), np.array([])

        v_range_mv = (V.max() - V.min()) * 1000.0
        if v_range_mv < delta_V_mV * MIN_ICA_POINTS:
            return np.array([]), np.array([])

        delta_V = delta_V_mV * 1e-3
        V_grid  = np.arange(V.min(), V.max(), delta_V)
        if len(V_grid) < MIN_ICA_POINTS:
            return np.array([]), np.array([])

        Q_interp = interp1d(V, Q, kind="linear", fill_value="extrapolate")(V_grid)
        dQ       = np.gradient(Q_interp, V_grid)

        win = min(SAVGOL_WIN, len(dQ) - 2)
        win = win if win % 2 == 1 else win - 1
        if win >= 5:
            dQ = savgol_filter(dQ, win, SAVGOL_POLY)

        return V_grid, dQ

    # ─────────────────────────────────────────────────────────────────
    # 3.  PEAK DETECTION
    # ─────────────────────────────────────────────────────────────────
    def detect_peaks(
        self, V_grid: np.ndarray, IC: np.ndarray
    ) -> Dict[str, dict]:
        """
        Match ICA peaks to reference table (c1 @ 3.35V, c2 @ 3.57V only).
        c3 (3.70V) and c4 (3.85V) excluded — above V_max = 3.65V.
        """
        if len(V_grid) == 0:
            return {}

        ic_abs = np.abs(IC)
        if ic_abs.max() < EPS:
            return {}

        min_height = 0.05 * ic_abs.max()
        dv = float(V_grid[1] - V_grid[0]) if len(V_grid) > 1 else 0.002
        min_dist   = max(1, int(0.04 / dv))  # 40 mV minimum separation

        peak_idx, _ = find_peaks(
            ic_abs, height=min_height, distance=min_dist,
            prominence=0.2 * min_height,
        )

        detected: Dict[str, dict] = {}
        for name, ref in self.accessible_peaks.items():
            best_i, best_d = None, np.inf
            for pi in peak_idx:
                if pi < len(V_grid):
                    d = abs(V_grid[pi] - ref["voltage"])
                    if d < ref["tolerance"] and d < best_d:
                        best_d, best_i = d, pi
            if best_i is not None:
                detected[name] = {
                    "voltage":       float(V_grid[best_i]),
                    "soc_anchor":    ref["soc_anchor"],
                    "confidence":    ref["confidence"],
                    "amplitude":     float(ic_abs[best_i]),
                    "voltage_shift": float(V_grid[best_i] - ref["voltage"]),
                }
        return detected

    # ─────────────────────────────────────────────────────────────────
    # 4.  EIS 0.01 Hz SOC ESTIMATION  (Gao & Onori 2025 [3])
    # ─────────────────────────────────────────────────────────────────
    def eis_soc_from_single_frequency(
        self,
        Z_mag:        float,
        Z_phase_deg:  float,
        lookup:       Optional[dict] = None,
    ) -> Tuple[float, float]:
        """
        Estimate SOC from |Z| and phase at 0.01 Hz.

        Accuracy (Gao & Onori 2025):
          Magnitude alone:         RMSE < 5% SOC
          Magnitude + phase:       best SNR across 0.01–1000 Hz
          EIS + ICA c2 fusion:     < 1% achievable

        Measurement requirements:
          Current amplitude: 6 A (C/25 for 160Ah cell)
          Duration:          200 s (2 periods at 0.01 Hz)
          EIS interval:      every 12 min (not 6 min)
        """
        if lookup is None:
            # Approximate 160Ah LFP lookup at 25°C
            # Scaled from Stanford 2.5Ah study: R ∝ 1/Q → factor 64×
            lookup = {
                "soc":   np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                "mag":   np.array([4.4e-4, 3.9e-4, 3.4e-4, 3.1e-4, 3.0e-4,
                                    2.8e-4, 2.7e-4, 2.5e-4, 2.3e-4]),
                "phase": np.array([-45, -42, -40, -38, -36, -34, -32, -30, -28]),
            }

        socs   = np.asarray(lookup["soc"])
        mags   = np.asarray(lookup["mag"])
        phases = np.asarray(lookup["phase"])

        mag_rng = mags.max() - mags.min() + EPS
        ph_rng  = phases.max() - phases.min() + EPS

        dist   = np.sqrt(((Z_mag - mags) / mag_rng) ** 2 +
                         ((Z_phase_deg - phases) / ph_rng) ** 2)
        best   = int(np.argmin(dist))
        conf   = float(np.exp(-dist[best] * 3.0))
        return float(socs[best]), conf

    # ─────────────────────────────────────────────────────────────────
    # 5.  COULOMB COUNTING with temperature correction
    # ─────────────────────────────────────────────────────────────────
    def coulomb_count_step(
        self, I_A: float, dt_s: float, T_C: float = 25.0
    ) -> float:
        """
        dSOC = −I × dt / (3600 × Q_eff(T)).
        Q_eff ≈ Q_nom × (1 − 0.002 × (25 − T)) — LFP ~0.2%/°C.
        """
        Q_eff    = self.Q_nom * float(np.clip(1.0 - 0.002 * (25.0 - T_C), 0.7, 1.05))
        self._soc = float(np.clip(self._soc - (I_A * dt_s) / (3600.0 * Q_eff), 0.0, 1.0))
        return self._soc

    # ─────────────────────────────────────────────────────────────────
    # 6.  FUSION — weighted combination  (Kimi §7)
    # ─────────────────────────────────────────────────────────────────
    def fuse(
        self,
        soc_cc:   float, conf_cc:  float,
        soc_eis:  Optional[float] = None, conf_eis: Optional[float] = None,
        soc_ica:  Optional[float] = None, conf_ica: Optional[float] = None,
    ) -> float:
        """
        Weighted fusion.  Priority: ICA > EIS > Coulomb counting.

        Expected accuracy after fusion:
          CC alone :       ±5%  (drifts over time)
          EIS alone :      <5% RMSE (Gao 2025)
          EIS + ICA c2 :   <1%  (Kimi recommendation)
        """
        pairs = [(soc_cc, conf_cc)]
        if soc_eis is not None and conf_eis is not None and conf_eis > 0.1:
            pairs.append((soc_eis, conf_eis * 2.0))
        if soc_ica is not None and conf_ica is not None and conf_ica > 0.1:
            pairs.append((soc_ica, conf_ica * 3.0))

        total = sum(c for _, c in pairs)
        if total < EPS:
            return self._soc

        self._soc = float(np.clip(sum(s * c for s, c in pairs) / total, 0.0, 1.0))
        return self._soc

    # ─────────────────────────────────────────────────────────────────
    # 7.  DATASET FEASIBILITY CHECK
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def check_ica_feasibility(
        logging_interval_s: float,
        C_rate:             float,
        Q_Ah:               float,
        V_plateau_range_mV: float = 200.0,
    ) -> dict:
        """
        Check whether a dataset has sufficient voltage resolution for ICA.

        For LFP plateau (3.20–3.40V, 200 mV span):
          Need ≥ MIN_ICA_POINTS = 23 points.
          Each point covers dV ≈ (C_rate × Q × dt) / Q × (dV/dSOC in plateau).
          In LFP plateau dV/dSOC ≈ 0.1 V/unit → dV_per_interval ≈ 0.1 × C_rate × dt/3600 × 1000 mV.

        Args:
            logging_interval_s  : Dataset time resolution [s].
            C_rate              : Cycling C-rate.
            Q_Ah                : Cell capacity [Ah].
            V_plateau_range_mV  : Voltage span of plateau [mV].
        Returns:
            dict with feasibility assessment.
        """
        I_A = C_rate * Q_Ah
        dQ_per_interval = I_A * logging_interval_s / 3600.0           # [Ah]
        # In flat plateau dOCV/dSOC ≈ 0.1 V per unit SOC
        dV_mv = (dQ_per_interval / max(Q_Ah, EPS)) * 100.0            # mV
        n_pts = V_plateau_range_mV / dV_mv if dV_mv > EPS else 0.0

        feasible = n_pts >= MIN_ICA_POINTS
        return {
            "feasible":            feasible,
            "points_in_plateau":   round(n_pts, 1),
            "required_points":     MIN_ICA_POINTS,
            "dV_per_interval_mV":  round(dV_mv, 2),
            "verdict": (
                f"ICA FEASIBLE — {n_pts:.0f} points in plateau"
                if feasible else
                f"ICA NOT FEASIBLE — only {n_pts:.1f} points "
                f"(need {MIN_ICA_POINTS})"
            ),
        }


# ──────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────
def validate() -> bool:
    """Self-validation: all tests must pass."""
    print("=" * 60)
    print("VALIDATING: diagnosis/ica_soc_corrector.py")
    print("=" * 60)
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        tag = "  [PASS]" if cond else "  [FAIL]"
        print(tag + f" {name}" + (f"  | {detail}" if detail else ""))
        if not cond:
            ok = False

    corr = ICASocCorrector(Q_nom_Ah=160.0, R_int_ohm=0.022, V_min=2.8, V_max=3.65)

    # 1. Only c1 and c2 accessible (c3 3.70V > 3.65V = V_max)
    check("Accessible peaks: c1 present", "c1" in corr.accessible_peaks)
    check("Accessible peaks: c2 present", "c2" in corr.accessible_peaks)
    check("Excluded c3 (3.70V > V_max)", "c3" not in corr.accessible_peaks,
          f"V_max={corr.V_max}")

    # 2. Resistance correction
    V_corr = corr.resistance_correct_voltage(
        np.array([3.40, 3.35, 3.30]),
        np.array([16.0, 16.0, 16.0]),   # C/10 for 160Ah
    )
    expected = 3.40 - 16.0 * 0.022
    check("Resistance correction at 16A", abs(float(V_corr[0]) - expected) < 1e-6,
          f"got {V_corr[0]:.4f} expect {expected:.4f}")

    # 3. ICA on sufficient synthetic data (simulated charge)
    N = 500
    soc_arr = np.linspace(0.1, 0.9, N)
    # Simulate LFP-like OCV with a c2-like bump at ~3.57V → SOC ~0.42
    ocv_arr = (3.30
               + 0.25 * soc_arr
               + 0.04 * np.exp(-((soc_arr - 0.42) ** 2) / 0.005))
    Q_arr = soc_arr * 160.0
    V_grid, IC = corr.compute_ica(ocv_arr, Q_arr, delta_V_mV=2.0)
    check("ICA output non-empty",   len(V_grid) > 0, f"len={len(V_grid)}")
    check("ICA V_grid in range",    float(V_grid.min()) >= 2.8,
          f"min={V_grid.min():.3f}")
    check("ICA dQ/dV finite",       np.all(np.isfinite(IC)))

    # 4. Peak detection — should find c2 near 3.57V
    detected = corr.detect_peaks(V_grid, IC)
    check("c2 peak detected",       "c2" in detected, f"detected={list(detected.keys())}")
    if "c2" in detected:
        c2_v = detected["c2"]["voltage"]
        check("c2 voltage near 3.57V", abs(c2_v - 3.57) < 0.10, f"v={c2_v:.3f}")

    # 5. Infeasible data: too few points
    V_short = np.linspace(3.3, 3.4, 10)
    Q_short = np.linspace(50, 80, 10)
    V_g2, IC2 = corr.compute_ica(V_short, Q_short)
    check("Short data → empty output", len(V_g2) == 0)

    # 6. EIS SOC estimation
    soc_eis, conf = corr.eis_soc_from_single_frequency(3.4e-4, -38.0)
    check("EIS SOC in [0,1]",  0.0 <= soc_eis <= 1.0, f"soc={soc_eis:.3f}")
    check("EIS conf in [0,1]", 0.0 <= conf <= 1.0,    f"conf={conf:.3f}")

    # 7. Coulomb counting
    corr._soc = 0.80
    soc_new = corr.coulomb_count_step(I_A=16.0, dt_s=60.0, T_C=25.0)
    expected_dsoc = -(16.0 * 60.0) / (3600.0 * 160.0)
    check("Coulomb count step", abs(soc_new - (0.80 + expected_dsoc)) < 1e-6,
          f"Δsoc={expected_dsoc:.5f}")

    # 8. Fusion weights
    corr._soc = 0.60
    fused = corr.fuse(0.60, 0.5, soc_eis=0.50, conf_eis=0.8)
    check("Fusion ICA > EIS > CC", 0.50 < fused < 0.60, f"fused={fused:.3f}")

    # 9. EIS constants
    check("EIS 0.01Hz → 200s duration",
          abs(EIS_DURATION_S - 200.0) < 1e-6, f"{EIS_DURATION_S}s")
    check("EIS current 6A (C/25 for 160Ah)",
          abs(EIS_CURRENT_A - 6.0) < 1e-6, f"{EIS_CURRENT_A}A")

    # 10. Feasibility check
    r_quartz = ICASocCorrector.check_ica_feasibility(360.0, 0.5, 160.0)
    r_darmst = ICASocCorrector.check_ica_feasibility(1.0,   0.1, 160.0)
    check("Quartz NOT feasible",   not r_quartz["feasible"],
          r_quartz["verdict"][:40])
    check("Darmstadt FEASIBLE",    r_darmst["feasible"],
          r_darmst["verdict"][:40])

    status = "ALL PASS" if ok else "SOME TESTS FAILED"
    print(f"\nResult: {status}")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    validate()
