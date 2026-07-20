"""
tests/test_sign_convention.py — the permanent discharge-negative sign
assertion (sign-bug postmortem, 2026-07-20) and the corrected coulomb
baseline. Run: python tests/test_sign_convention.py (or pytest).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loaders.common_schema import (
    assert_discharge_negative_consistency, make_schema_df,
)


def _mk(I_sign: float, n=600, dt=10.0, q_ah=60.0):
    """Synthetic 100 A discharge from SOC 0.9; I sign controlled."""
    t = np.arange(n) * dt
    I = np.full(n, I_sign * 100.0)
    soc = 0.9 - (100.0 * dt * np.arange(n)) / (3600.0 * q_ah)  # true discharge
    V = np.full(n, 3.7)
    return t, I, V, soc


def test_correct_convention_passes():
    t, I, V, soc = _mk(-1.0)  # discharge-negative: OK
    df = make_schema_df(t, I, V, None, soc, source="selftest")
    assert len(df) == 600


def test_inverted_convention_raises():
    t, I, V, soc = _mk(+1.0)  # discharge-positive: must fail loudly
    try:
        make_schema_df(t, I, V, None, soc, source="selftest")
        raise AssertionError("expected ValueError for discharge-positive data")
    except ValueError as e:
        assert "SIGN-CONVENTION VIOLATION" in str(e)


def test_small_net_change_never_trips():
    # regen-heavy trip: big currents both ways, ~zero net ΔSOC → no assertion
    t = np.arange(600) * 10.0
    I = 100.0 * np.sin(np.arange(600) / 5.0)
    soc = np.full(600, 0.5)
    make_schema_df(t, I, np.full(600, 3.7), None, soc, source="selftest")


def test_charging_direction_also_checked():
    # SOC rises while ∫I dt < 0 → violation (charge must be positive current)
    t = np.arange(600) * 10.0
    I = np.full(600, -100.0)
    soc = 0.2 + (100.0 * 10.0 * np.arange(600)) / (3600.0 * 60.0)
    try:
        assert_discharge_negative_consistency(t, I, soc, source="selftest")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_coulomb_baseline_decreases_soc_on_discharge():
    from data.soc_baseline_benchmark import coulomb_counting_soc
    from data.validate_generic import ValidationConfig
    t, I, V, soc = _mk(-1.0)
    seg = pd.DataFrame({"t_s": t, "I_A": I, "V_V": V, "SOC_bms": soc,
                        "T_degC": np.nan})
    cfg = ValidationConfig(dataset_name="selftest", n_series=1, n_parallel=1,
                           q_cell_ah=60.0, ekf_soc_offset=0.0)
    est = coulomb_counting_soc(seg, cfg)
    # correct-sign integration tracks the true discharge exactly
    assert est[0] > est[-1], "coulomb SOC must fall during discharge"
    assert float(np.max(np.abs(est - np.clip(soc, 0, 1)))) < 1e-9


def _run_all():
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
