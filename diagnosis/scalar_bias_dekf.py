"""
diagnosis/scalar_bias_dekf.py — RBC-DEKF-style online scalar-bias baselines
(Phase 2).

Two variants of THIS PROJECT's estimator (diagnosis/dual_ekf_lfp.py:
identical RC Thevenin model, identical fleet-fitted OCV curve, identical
adaptive-Q scaling and gated SOH/R_int slow loops) in which the OFFLINE
structured corrections δV(SOC) + δR0·I are REPLACED by an ONLINE scalar
voltage-bias state θ, in the spirit of Guo et al., "Residual Bias
Compensation Filter for Physics-Based SOC Estimation in Lithium Iron
Phosphate Batteries" (RBC-DEKF, arXiv:2510.22813):

  ScalarBiasDEKF (decoupled, the RBC-DEKF analogue)
      θ_{k+1} = θ_k + w_θ,  w_θ ~ N(0, Q_θ·dt)   (random walk, per-second Q)
      z_k = V_model(x_k) + θ_k + v_k
      SEQUENTIAL two-filter update: the state filter treats the latest θ as
      a known constant (θ absent from its Jacobian H — the same decoupling
      this project applies to δV(SOC)); the bias filter then updates θ from
      the residual of the LATEST state estimate with H_θ = 1.
      Covariance updates in JOSEPH FORM in both filters (per RBC-DEKF spec;
      2026-07-19 review decision (a): Joseph form lives ONLY in this
      baseline — the project's own estimator keeps its standard update, and
      a Joseph-form row is deferred to the Phase 5 ablation).

  CoupledBiasEKF (the coupling counter-example)
      Same bias state AUGMENTED into ONE joint filter, x = [SOC, V_pol, θ],
      FULL Jacobian H = [∂OCV/∂SOC, 1, 1]. The shared gain couples θ to SOC
      through cross-covariances — the joint-estimation alternative that both
      Guo et al. (theoretically) and this project's Round-2 ∂δV/∂SOC failure
      (empirically) argue against. Run on the same fleet data to test the
      coupling-degradation claim. Also Joseph form, so the ONLY structural
      difference vs ScalarBiasDEKF is the coupling itself.

Both variants keep the slow SOH/R_int loops of the parent class unchanged;
in the R_int residual the online θ plays the role the offline δV correction
plays in the parent (cal_dR0 = 0, so the parent's calibration sanity gate
never disables the R_int loop here — disclosed).

Tuning: Q_θ and R_θ are fleet-tuned by a small grid search on the
CALIBRATION split only (same 10/40/30% splits as the δ fits), never on
held-out trips — see data/run_offset_sweep.py, which logs the grid and the
chosen values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diagnosis.dual_ekf_lfp import DualEKF_LFP, EPS


class ScalarBiasDEKF(DualEKF_LFP):
    """Decoupled online scalar-bias DEKF (RBC-DEKF analogue on this
    project's ECM). See module docstring."""

    def __init__(
        self,
        Q_nom_Ah: float = 160.0,
        R_int_ohm: float = 0.0005,
        ocv_fn=None,
        R_meas_V2: float = 4e-6,
        P0_soc: float = 0.04,
        gamma: float = 1.0,
        Q_theta_V2_per_s: float = 1e-8,
        R_theta_V2: float = 1e-5,
        theta0_V: float = 0.0,
        P_theta0_V2: float = 2.5e-3,   # (50 mV)² initial bias uncertainty
    ):
        # No offline corrections: cal_soc_fn=None, cal_dR0=0 by construction.
        super().__init__(
            Q_nom_Ah=Q_nom_Ah, R_int_ohm=R_int_ohm, ocv_fn=ocv_fn,
            R_meas_V2=R_meas_V2, P0_soc=P0_soc, gamma=gamma,
            cal_soc_fn=None, cal_dR0=0.0,
        )
        self.theta = float(theta0_V)
        self.P_theta = float(P_theta0_V2)
        self.Q_theta_per_s = float(Q_theta_V2_per_s)
        self.R_theta = float(R_theta_V2)

    # V_model WITHOUT the bias term (bias added explicitly by callers)
    def _v_model(self, soc: float, v_pol: float, I_A: float, R_use: float) -> float:
        return self._ocv(soc) - I_A * R_use + v_pol

    def update(self, V_meas: float, I_A: float, dt_s: float, T_C: float = 25.0) -> dict:
        soc, v_pol = self.x1
        Q_eff = self.Q_nom * float(self.x2[0])
        R_use = float(self.x2[1])
        tau = 50.0

        # ── state-filter predict (identical to parent) ──────────────────────
        soc_p = float(np.clip(soc - (I_A * dt_s) / (3600.0 * Q_eff), 0.0, 1.0))
        v_pol_p = v_pol * np.exp(-dt_s / tau) + R_use * (1.0 - np.exp(-dt_s / tau)) * I_A
        x_p = np.array([soc_p, v_pol_p])
        F = np.array([[1.0, 0.0], [0.0, np.exp(-dt_s / tau)]])
        P_p = F @ self.P1 @ F.T + self._adaptive_Q(soc)

        # ── state-filter update: LATEST θ treated as a known constant ──────
        V_pred = self._v_model(x_p[0], x_p[1], I_A, R_use) + self.theta
        H = np.array([[self._docv_dsoc(x_p[0]), 1.0]])   # θ NOT in H (decoupled)

        S = (H @ P_p @ H.T + self._R_meas)[0, 0] + EPS
        K = (P_p @ H.T) / S
        innov = V_meas - V_pred
        self.x1 = x_p + K.flatten() * innov
        self.x1[0] = float(np.clip(self.x1[0], 0.0, 1.0))
        IKH = np.eye(2) - np.outer(K.flatten(), H)
        # Joseph form (RBC-DEKF spec)
        self.P1 = IKH @ P_p @ IKH.T + np.outer(K.flatten(), K.flatten()) * self._R_meas[0, 0]

        # ── bias-filter update: random walk predict, then update from the
        #    LATEST state estimate (sequential order per RBC-DEKF) ──────────
        self.P_theta += self.Q_theta_per_s * max(dt_s, 0.0)
        v_model_now = self._v_model(float(self.x1[0]), float(self.x1[1]), I_A, R_use)
        innov_theta = V_meas - v_model_now - self.theta
        S_theta = self.P_theta + self.R_theta + EPS
        K_theta = self.P_theta / S_theta
        self.theta = float(self.theta + K_theta * innov_theta)
        # Joseph form, scalar: (1−K)²P + K²R
        self.P_theta = float((1.0 - K_theta) ** 2 * self.P_theta
                             + K_theta ** 2 * self.R_theta)

        # ── slow loops unchanged; online θ plays the offline δV role ───────
        self._update_r_int(V_meas, I_A, cal_off=self.theta, r0_off=0.0)
        self._update_soh(dt_s, I_A)

        return {
            "soc":        float(self.x1[0]),
            "soh":        float(self.x2[0]),
            "r_int":      float(self.x2[1]),
            "theta":      float(self.theta),
            "V_pred":     float(V_pred),
            "innovation": float(innov),
        }


class CoupledBiasEKF(DualEKF_LFP):
    """Coupled counter-example: same bias state augmented into ONE joint
    filter with the FULL Jacobian H = [∂OCV/∂SOC, 1, 1]. See module
    docstring."""

    def __init__(
        self,
        Q_nom_Ah: float = 160.0,
        R_int_ohm: float = 0.0005,
        ocv_fn=None,
        R_meas_V2: float = 4e-6,
        P0_soc: float = 0.04,
        gamma: float = 1.0,
        Q_theta_V2_per_s: float = 1e-8,
        R_theta_V2: float = 1e-5,   # unused (single joint R = R_meas); kept
                                    # for a call-signature identical to the
                                    # decoupled variant
        theta0_V: float = 0.0,
        P_theta0_V2: float = 2.5e-3,
    ):
        super().__init__(
            Q_nom_Ah=Q_nom_Ah, R_int_ohm=R_int_ohm, ocv_fn=ocv_fn,
            R_meas_V2=R_meas_V2, P0_soc=P0_soc, gamma=gamma,
            cal_soc_fn=None, cal_dR0=0.0,
        )
        self.Q_theta_per_s = float(Q_theta_V2_per_s)
        # Joint state x3 = [SOC, V_pol, θ]; x1 kept as a live view of the
        # first two components so the parent's slow loops / set_soc work.
        self.x3 = np.array([float(self.x1[0]), float(self.x1[1]), float(theta0_V)])
        self.P3 = np.zeros((3, 3))
        self.P3[:2, :2] = self.P1
        self.P3[2, 2] = float(P_theta0_V2)

    @property
    def theta(self) -> float:
        return float(self.x3[2])

    def set_soc(self, soc_init: float) -> None:
        super().set_soc(soc_init)
        self.x3[0] = float(self.x1[0])

    def update(self, V_meas: float, I_A: float, dt_s: float, T_C: float = 25.0) -> dict:
        soc, v_pol, theta = self.x3
        Q_eff = self.Q_nom * float(self.x2[0])
        R_use = float(self.x2[1])
        tau = 50.0

        # ── joint predict: state dynamics as parent, θ random walk ─────────
        soc_p = float(np.clip(soc - (I_A * dt_s) / (3600.0 * Q_eff), 0.0, 1.0))
        v_pol_p = v_pol * np.exp(-dt_s / tau) + R_use * (1.0 - np.exp(-dt_s / tau)) * I_A
        x_p = np.array([soc_p, v_pol_p, theta])
        F = np.diag([1.0, np.exp(-dt_s / tau), 1.0])
        Q3 = np.zeros((3, 3))
        Q3[:2, :2] = self._adaptive_Q(soc)
        Q3[2, 2] = self.Q_theta_per_s * max(dt_s, 0.0)
        P_p = F @ self.P3 @ F.T + Q3

        # ── joint update: FULL Jacobian including the bias (the coupling) ──
        V_pred = self._ocv(x_p[0]) - I_A * R_use + x_p[1] + x_p[2]
        H = np.array([[self._docv_dsoc(x_p[0]), 1.0, 1.0]])

        S = (H @ P_p @ H.T + self._R_meas)[0, 0] + EPS
        K = (P_p @ H.T) / S
        innov = V_meas - V_pred
        self.x3 = x_p + K.flatten() * innov
        self.x3[0] = float(np.clip(self.x3[0], 0.0, 1.0))
        IKH = np.eye(3) - np.outer(K.flatten(), H)
        self.P3 = IKH @ P_p @ IKH.T + np.outer(K.flatten(), K.flatten()) * self._R_meas[0, 0]

        # keep parent's x1/P1 views coherent for slow loops & callers
        self.x1 = self.x3[:2].copy()
        self.P1 = self.P3[:2, :2].copy()

        self._update_r_int(V_meas, I_A, cal_off=float(self.x3[2]), r0_off=0.0)
        self._update_soh(dt_s, I_A)

        return {
            "soc":        float(self.x3[0]),
            "soh":        float(self.x2[0]),
            "r_int":      float(self.x2[1]),
            "theta":      float(self.x3[2]),
            "V_pred":     float(V_pred),
            "innovation": float(innov),
        }


def validate() -> bool:
    print("=" * 55)
    print("VALIDATING: diagnosis/scalar_bias_dekf.py")
    print("=" * 55)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  | {detail}" if detail else ""))

    # Steep synthetic OCV (NMC-like slope). NOTE on identifiability: with a
    # steep OCV a constant voltage bias and a SOC offset are near-equivalent
    # explanations of the same residual, and whichever filter is given the
    # larger process noise wins the residual. The tests below therefore use
    # scenarios where exactly one explanation is available: (a) correct SOC
    # init + SOC pinned by coulomb counting (tiny gamma) → θ must learn the
    # bias; (b) no bias + wrong init → SOC must recover. On real LFP-plateau
    # data (RBC-DEKF's regime) the flat OCV itself provides this separation.
    ocv = lambda s: 3.0 + 0.8 * float(np.clip(s, 0, 1))
    rng = np.random.default_rng(1)

    # Simulated truth is OCV + bias only (no IR/RC term): the parent model's
    # −I·R + v_pol cancels at steady state (v_pol → R·I), i.e. the ECM
    # carries no DC IR drop, so a DC-consistent truth must not either — this
    # keeps the expected θ exactly equal to the injected bias.
    def simulate(filt, bias_V, init_offset, n_steps=3000):
        soc_true, q = 0.90, 2.0
        filt.set_soc(float(np.clip(soc_true + init_offset, 0.02, 0.98)))
        out = None
        for _ in range(n_steps):
            I, dt = 2.0, 1.0
            soc_true = max(0.0, soc_true - I * dt / (3600.0 * q))
            v = ocv(soc_true) + bias_V + rng.normal(0, 1e-3)
            out = filt.update(v, I, dt)
        return soc_true, out

    # (a) bias identifiable: correct init, SOC pinned (gamma≈0) → θ → bias.
    # Slow loops frozen for this scenario ONLY, to test the bias filter in
    # isolation: a constant measurement bias corrupts the residuals the
    # R_int/SOH loops learn from (on real data the parent's cal_dR0 sanity
    # gate guards the worst of this), which would fold slow-loop error into
    # θ and break the known answer. Fleet runs keep both loops active.
    true_bias = 0.080
    # P0 tiny on BOTH fast states: init is correct AND the filter knows it
    # (the project's own convention is P0_soc = offset², i.e. ≈0 at zero
    # offset). With the defaults (P0_soc=0.04, P0_vpol=(10 mV)²) the state
    # filter absorbs the whole bias into SOC/v_pol at the first update
    # before θ ever sees it — three states competing for one residual is
    # exactly the ambiguity this scenario is built to exclude. R_int=0 so
    # the RC branch carries nothing.
    f = ScalarBiasDEKF(Q_nom_Ah=2.0, R_int_ohm=0.0, ocv_fn=ocv,
                       R_meas_V2=1e-6, P0_soc=1e-10, gamma=1e-4,
                       Q_theta_V2_per_s=1e-7, R_theta_V2=1e-5)
    f.P1 = np.diag([1e-10, 1e-10])
    f.r_int_update_enabled = False
    f._update_soh = lambda dt_s, I_A: None
    soc_true, out = simulate(f, bias_V=true_bias, init_offset=0.0)
    chk("decoupled: theta learns injected +80mV bias (identifiable case)",
        abs(f.theta - true_bias) < 0.02, f"theta={f.theta*1000:.1f}mV")
    chk("decoupled: SOC stays accurate while theta absorbs bias",
        abs(out["soc"] - soc_true) < 0.05,
        f"err={abs(out['soc']-soc_true)*100:.2f}pp")

    # (b) SOC recovery: no bias, +20pp wrong init, normal gamma
    f2 = ScalarBiasDEKF(Q_nom_Ah=2.0, R_int_ohm=0.010, ocv_fn=ocv,
                        R_meas_V2=1e-6, gamma=1.0,
                        Q_theta_V2_per_s=1e-8, R_theta_V2=1e-5)
    soc_true2, out2 = simulate(f2, bias_V=0.0, init_offset=0.20)
    chk("decoupled: SOC recovers from +20pp wrong init (no bias)",
        abs(out2["soc"] - soc_true2) < 0.05,
        f"err={abs(out2['soc']-soc_true2)*100:.2f}pp")

    chk("decoupled: Joseph P1 symmetric",
        np.allclose(f2.P1, f2.P1.T, atol=1e-14))
    chk("decoupled: Joseph P1 PSD", np.all(np.linalg.eigvalsh(f2.P1) > -1e-15))
    chk("decoupled: P_theta positive", f.P_theta > 0)

    fc = CoupledBiasEKF(Q_nom_Ah=2.0, R_int_ohm=0.010, ocv_fn=ocv,
                        R_meas_V2=1e-6, Q_theta_V2_per_s=1e-8)
    soc_true_c, out_c = simulate(fc, bias_V=true_bias, init_offset=0.20)
    chk("coupled: runs, SOC finite and in [0,1]",
        np.isfinite(out_c["soc"]) and 0 <= out_c["soc"] <= 1)
    chk("coupled: Joseph P3 symmetric", np.allclose(fc.P3, fc.P3.T, atol=1e-14))
    chk("coupled: theta state live", np.isfinite(fc.theta))

    print(f"\nResult: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


if __name__ == "__main__":
    validate()
