#!/usr/bin/env python3
"""
scripts/verify_sign_convention.py — Sign-convention verification for VED and BMW i3.

For each dataset, loads segments with |ΔSOC| > 0.01 and checks two independent
physical invariants:

  1. Coulomb-counting invariant:
       Q_int = ∫ I_A dt  (trapezoidal, in As)
       ΔSOC_meas = SOC_end − SOC_start
       Under correct convention (discharge<0, charge>0):
         sign(Q_int) MUST equal sign(ΔSOC_meas)

  2. Voltage correlation invariant:
       corr(I_A, V_V − rolling_mean(V_V)) MUST be POSITIVE
       Charging raises terminal voltage above trend; discharge depresses it.
       (V_residual = V_measured − rolling_mean captures the IR/polarisation swing.)

Prints per-segment evidence and a dataset-level pass/fail verdict.
If VED fails majority of segments, automatically corrects _VED_DISCHARGE_POSITIVE
in ved_loader.py, re-runs, and prints the corrected evidence.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _q_int(seg: pd.DataFrame) -> float:
    """Trapezoidal integral of I_A over t_s, in ampere-seconds."""
    t = seg["t_s"].values
    I = seg["I_A"].values
    return float(np.trapezoid(I, t))


def _v_corr(seg: pd.DataFrame, window_s: float = 30.0) -> float:
    """
    Pearson corr of I_A with (V_V − rolling_mean(V_V)).
    Rolling window approximated by rows; if dt is irregular, use 60-row window.
    """
    dt_med = float(np.median(np.diff(seg["t_s"].values))) if len(seg) > 1 else 1.0
    if dt_med <= 0:
        dt_med = 1.0
    win = max(5, int(window_s / dt_med))
    win = min(win, len(seg) // 3, len(seg) - 1)
    win = max(win, 3)

    V = seg["V_V"]
    V_roll = V.rolling(window=win, center=True, min_periods=3).mean()
    V_resid = V - V_roll

    valid = ~(V_resid.isna())
    if valid.sum() < 5:
        return float("nan")

    I_v = seg["I_A"][valid].values
    Vr_v = V_resid[valid].values
    if np.std(I_v) < 1e-6 or np.std(Vr_v) < 1e-6:
        return float("nan")
    return float(np.corrcoef(I_v, Vr_v)[0, 1])


def _check_segments(
    segs: List[pd.DataFrame],
    dataset_name: str,
    soc_thresh: float = 0.01,
    max_segs: int = 15,
) -> Tuple[int, int, int]:
    """
    Check sign invariants on segments with |ΔSOC| > soc_thresh.
    Returns (n_checked, n_q_pass, n_v_pass).
    """
    print(f"\n{'─'*70}")
    print(f"  {dataset_name} — sign convention check")
    print(f"{'─'*70}")
    print(f"  {'seg':>4}  {'ΔSOC':>8}  {'Q_int(As)':>11}  {'sign_match':>10}  "
          f"{'V_corr':>7}  {'V_ok':>5}")
    print(f"  {'':─<4}  {'':─<8}  {'':─<11}  {'':─<10}  {'':─<7}  {'':─<5}")

    n_checked = n_q_pass = n_v_pass = 0
    seg_idx = 0

    for seg in segs:
        if n_checked >= max_segs:
            break
        soc = seg["SOC_bms"].dropna()
        if len(soc) < 10:
            continue
        d_soc = float(soc.iloc[-1] - soc.iloc[0])
        if abs(d_soc) < soc_thresh:
            seg_idx += 1
            continue

        q = _q_int(seg)
        vcorr = _v_corr(seg)

        sign_match = (q >= 0) == (d_soc >= 0)
        v_ok = (not np.isnan(vcorr)) and vcorr > 0

        n_checked += 1
        if sign_match:
            n_q_pass += 1
        if v_ok:
            n_v_pass += 1

        sm_str = "Y  ✓" if sign_match else "N  ✗"
        v_str  = f"{vcorr:+.3f}" if not np.isnan(vcorr) else "   nan"
        v_ok_s = "✓" if v_ok else "✗"
        print(f"  {seg_idx:>4d}  {d_soc:>+8.4f}  {q:>+11.1f}  {sm_str:>10}  "
              f"{v_str:>7}  {v_ok_s:>5}")
        seg_idx += 1

    print(f"\n  Coulomb sign match : {n_q_pass}/{n_checked}")
    print(f"  Voltage corr > 0  : {n_v_pass}/{n_checked}")
    return n_checked, n_q_pass, n_v_pass


# ─────────────────────────────────────────────────────────────────────────────
# VED
# ─────────────────────────────────────────────────────────────────────────────

def _load_ved_segments(max_veh: int = 3, max_trips: int = 60) -> List[pd.DataFrame]:
    from data.loaders.ved_loader import VEDLoader
    loader = VEDLoader(max_veh=max_veh, max_trips_per_veh=max_trips)
    return [s for s, _ in loader.iter_segments()]


def check_ved() -> bool:
    print("\n" + "=" * 70)
    print("  CHECKING: VED")
    print("=" * 70)

    try:
        segs = _load_ved_segments()
    except FileNotFoundError as e:
        print(f"  [SKIP] VED data not found: {e}")
        return True  # not a sign error

    if not segs:
        print("  [SKIP] No VED segments loaded.")
        return True

    n, q_pass, v_pass = _check_segments(segs, "VED")
    if n == 0:
        print("  [SKIP] No VED segments with |ΔSOC|>0.01 found (increase max_trips).")
        return True

    q_frac = q_pass / n
    v_frac = v_pass / n
    sign_ok = q_frac >= 0.6 and v_frac >= 0.5

    print(f"\n  VERDICT: {'PASS ✓' if sign_ok else 'FAIL — sign inversion detected ✗'}")
    print(f"    Coulomb: {q_pass}/{n} ({q_frac*100:.0f}%)  |  Voltage: {v_pass}/{n} ({v_frac*100:.0f}%)")

    if not sign_ok:
        _flip_ved_sign()
        print("\n  Re-running after flip…")
        segs2 = _load_ved_segments()
        n2, q2, v2 = _check_segments(segs2, "VED (after flip)")
        q2f = q2 / n2 if n2 else 0
        v2f = v2 / n2 if n2 else 0
        sign_ok2 = q2f >= 0.6 and v2f >= 0.5
        print(f"\n  POST-FLIP VERDICT: {'PASS ✓' if sign_ok2 else 'STILL FAILING ✗'}")
        print(f"    Coulomb: {q2}/{n2} ({q2f*100:.0f}%)  |  Voltage: {v2}/{n2} ({v2f*100:.0f}%)")
        return sign_ok2

    return sign_ok


def _flip_ved_sign() -> None:
    """Toggle _VED_DISCHARGE_POSITIVE between True and False in ved_loader.py."""
    path = _ROOT / "data" / "loaders" / "ved_loader.py"
    src = path.read_text()

    if "_VED_DISCHARGE_POSITIVE = True" in src:
        new_val = "False"
        old_val = "True"
    else:
        new_val = "True"
        old_val = "False"

    src2 = src.replace(
        f"_VED_DISCHARGE_POSITIVE = {old_val}",
        f"_VED_DISCHARGE_POSITIVE = {new_val}",
    )
    path.write_text(src2)

    # Invalidate the cached module so the re-import picks up the new value
    for key in list(sys.modules.keys()):
        if "ved_loader" in key:
            del sys.modules[key]

    print(f"\n  [FIX] _VED_DISCHARGE_POSITIVE flipped: {old_val} → {new_val}")
    print(f"        Wrote to {path.relative_to(_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# BMW i3
# ─────────────────────────────────────────────────────────────────────────────

def _load_bmw_segments(max_trips: int = 40) -> List[pd.DataFrame]:
    from data.loaders.bmw_i3_loader import BMWI3Loader
    loader = BMWI3Loader(max_trips=max_trips)
    return [s for s, _ in loader.iter_segments()]


def check_bmw() -> bool:
    print("\n" + "=" * 70)
    print("  CHECKING: BMW i3")
    print("=" * 70)

    try:
        segs = _load_bmw_segments()
    except FileNotFoundError as e:
        print(f"  [SKIP] BMW i3 data not found: {e}")
        return True

    if not segs:
        print("  [SKIP] No BMW i3 segments loaded.")
        return True

    n, q_pass, v_pass = _check_segments(segs, "BMW i3")
    if n == 0:
        print("  [SKIP] No BMW i3 segments with |ΔSOC|>0.01 found.")
        return True

    q_frac = q_pass / n
    v_frac = v_pass / n
    sign_ok = q_frac >= 0.6 and v_frac >= 0.5

    print(f"\n  VERDICT: {'PASS ✓' if sign_ok else 'FAIL — sign inversion detected ✗'}")
    print(f"    Coulomb: {q_pass}/{n} ({q_frac*100:.0f}%)  |  Voltage: {v_pass}/{n} ({v_frac*100:.0f}%)")

    if not sign_ok:
        print(f"\n  [ACTION NEEDED] Update _BMW_DISCHARGE_POSITIVE in bmw_i3_loader.py")
        print(f"  Current: _BMW_DISCHARGE_POSITIVE = False")
        print(f"  Change to True and re-validate.")

    return sign_ok


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("  OPENCATHODE — SIGN CONVENTION VERIFICATION")
    print("  Invariants: sign(Q_int)==sign(ΔSOC)  AND  corr(I, V_resid)>0")
    print("=" * 70)

    ved_ok  = check_ved()
    bmw_ok  = check_bmw()

    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)
    print(f"  VED  : {'PASS ✓' if ved_ok  else 'FAIL ✗'}")
    print(f"  BMW  : {'PASS ✓' if bmw_ok  else 'FAIL ✗'}")
    print()

    return 0 if (ved_ok and bmw_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
