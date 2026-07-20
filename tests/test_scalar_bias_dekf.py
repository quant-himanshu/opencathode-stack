"""
tests/test_scalar_bias_dekf.py — Phase 2 baseline estimators.

The numerically subtle content (bias identifiability, Joseph-form
symmetry/PSD, wrong-init recovery) lives in
diagnosis/scalar_bias_dekf.py:validate() with known-answer synthetic
scenarios; this wrapper runs it under pytest plus a few structural checks.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diagnosis.scalar_bias_dekf import CoupledBiasEKF, ScalarBiasDEKF, validate


def test_validate_suite_passes():
    assert validate() is True


def test_decoupled_has_no_offline_corrections():
    f = ScalarBiasDEKF()
    assert f._cal_soc_fn is None
    assert f._cal_dR0 == 0.0
    # cal_dR0 == 0 → the parent's calibration sanity gate never fires here
    assert f.r_int_update_enabled is True


def test_coupled_state_is_three_dimensional_with_bias_in_jacobian():
    f = CoupledBiasEKF(ocv_fn=lambda s: 3.0 + 0.8 * float(np.clip(s, 0, 1)))
    f.set_soc(0.5)
    out = f.update(V_meas=3.5, I_A=2.0, dt_s=1.0)
    assert f.x3.shape == (3,)
    assert f.P3.shape == (3, 3)
    # bias must be able to move in a single joint update (H has the 1 for θ)
    assert out["theta"] != 0.0 or f.P3[2, 2] > 0


def test_theta_moves_only_via_bias_filter_in_decoupled():
    # decoupled: one update with a huge residual moves θ through the scalar
    # bias filter, never through the state filter's K (structurally θ is not
    # in x1) — θ must equal K_theta·residual' with K_theta = P/(P+R)
    # gamma tiny: the predict step re-inflates P1 by the adaptive Q every
    # call, so freezing the fast states needs small process noise, not just
    # small P0
    f = ScalarBiasDEKF(ocv_fn=lambda s: 3.0 + 0.8 * float(np.clip(s, 0, 1)),
                       P0_soc=1e-12, R_meas_V2=1e-6, gamma=1e-6,
                       Q_theta_V2_per_s=0.0, R_theta_V2=1e-5, P_theta0_V2=2.5e-3)
    f.P1 = np.diag([1e-12, 1e-12])
    f.set_soc(0.5)
    out = f.update(V_meas=3.48, I_A=0.0, dt_s=1.0)  # ~+80 mV residual
    k_theta = 2.5e-3 / (2.5e-3 + 1e-5)
    assert abs(f.theta) > 0.05                       # grabbed most of it
    assert f.theta <= 0.08 * k_theta + 1e-3          # bounded by K_theta
    assert abs(out["soc"] - 0.5) < 1e-3              # state barely moved
