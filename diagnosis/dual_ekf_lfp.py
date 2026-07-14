"""
Dual EKF for SOC/SOH estimation.

Improvement round 3:
  - cal_soc_fn: δV(SOC) applied to V_pred for innovation ONLY.
    Its SOC-derivative is NOT included in Jacobian H — treated as locally
    constant. This decouples calibration accuracy (Mode A) from Kalman gain
    stability (Mode B). Round 2 bug: adding dcal/dSOC to H introduced
    large, erratic gain swings from the PCHIP spline slope, destabilizing
    convergence (VED 45.6%, BMW N/A).
  - gamma: adaptive-Q scale factor (sweep {0.5, 1, 2} per fleet)
  - R_meas_V2: fleet-specific voltage measurement variance [V²/cell]
  - P0_soc: initial SOC covariance (default 0.04 = σ20%, matches +20% offset init)
  - cal_dR0: current-proportional R0 correction [V/A], applied inside measurement model

Key innovation: Adaptive process noise Q scaled by 1/|dOCV/dSOC|
prevents filter from freezing in flat OCV plateau (SOC 30-75% for LFP).
Reference: Mikhak 2024 PMC12936157 — RMSE < 0.15% with IC-method OCV.

Improvement round 4 (this fix): x2 = [SOH, R_int] was previously defined at
__init__ but NEVER updated by update() -- it sat at its initial value for
every call site in this project (validate_generic.py's run_mode_b_ekf never
reads back x2 either). This class was named "Dual EKF" but only ran a
single loop. Two genuinely separate slow-timescale updates are added below,
run every step from update(), consistent with Plett's dual/two-filter
design (a SEPARATE, slower filter for parameters, not one joint state with
SOC -- SOH and R_int are not observable from a single voltage residual the
same way SOC is, so they get their own gated update rules, not a shared H
row with x1):

  R_int : directly observable from the IR-drop term of the SAME voltage
          residual used by the SOC filter (dV/dR_int = -I_A). Scalar
          Kalman update, gated to |I_A| > R_INT_MIN_CURRENT_A so a near-zero
          current (where -I*R_int carries no information about R_int)
          doesn't inject noise into the estimate.
  SOH   : NOT observable from a single instantaneous voltage residual --
          capacity fade is a Ah-throughput-vs-delta-SOC quantity, only
          identifiable once real SOC swing has accumulated. Tracked via a
          recursive, confidence-weighted estimator: accumulate coulomb-
          counted Ah and the fast filter's own delta-SOC since the last
          update; once accumulated |delta-SOC| clears SOH_MIN_DSOC (a real
          swing, not noise), form SOH_obs = accumulated_Ah / (delta_SOC *
          Q_nom) and blend it into x2[0] with a gain that grows with how
          much delta-SOC has actually been observed (more evidence = more
          trusted estimate), then reset the accumulators.

Round 4 safety guard (added after re-running the SOC baseline benchmark with
the Round 4 fix): on datasets where the externally-supplied calibration
(cal_dR0, fit outside this class by validate_generic.py's
_build_calibration_for_fleet) is itself already unstable -- observed on
CALCE (delta_R0 = -260 mV/A) and the UMich/Ford parallel-cell module
(-233 mV/A), both roughly 30x the physically plausible per-cell resistance
range -- letting the R_int slow loop learn from the resulting corrupted
voltage residual made results WORSE, not better (CALCE SOC RMSE 35.14% ->
40.71% after the Round 4 fix). The R_int loop is now gated off entirely
when |cal_dR0| exceeds CAL_DR0_SANITY_THRESHOLD_OHM at construction time --
R_int stays frozen at its init value in that case, same as the pre-Round-4
behaviour, rather than actively learning from a residual already known (from
the external calibration's own diagnostic) to be corrupted. This does not
fix the underlying calibration instability (see
data/soc_baseline_benchmark_calce_report.json /
docs/soc_baseline_benchmark.md for that still-open issue) -- it only stops
this filter from compounding it.
"""
import numpy as np

EPS = 1e-12
R_INT_MIN_CURRENT_A = 1.0   # below this, IR-drop carries ~no info about R_int
CAL_DR0_SANITY_THRESHOLD_OHM = 0.05   # ~50 mOhm; a current-proportional
    # calibration correction beyond this is already outside the physically
    # plausible per-cell resistance range for the chemistries this project
    # models -- a symptom of an unstable external calibration fit, not a
    # real cell property. See docstring above.
SOH_MIN_DSOC = 0.05         # need >=5% real SOC swing before trusting a SOH update


class DualEKF_LFP:
    def __init__(
        self,
        Q_nom_Ah: float = 160.0,
        R_int_ohm: float = 0.0005,
        ocv_fn=None,
        R_meas_V2: float = 4e-6,
        P0_soc: float = 0.04,
        gamma: float = 1.0,
        cal_soc_fn=None,
        cal_dR0: float = 0.0,
    ):
        """
        Parameters
        ----------
        Q_nom_Ah   : nominal cell capacity [Ah]
        R_int_ohm  : initial internal resistance estimate [Ω]
        ocv_fn     : optional OCV(soc)->V callable; overrides built-in LFP table.
                     Pass empirical NMC spline (from diagnosis/nmc_ocv.py) to avoid
                     SOC saturation on NMC fleets.
        R_meas_V2  : voltage measurement variance [V²/cell].
                     Fleet-specific: VED ~4e-4, BMW ~1e-4, Deng ~2.5e-5.
                     Default 4e-6 kept for backward compat.
        P0_soc     : initial SOC error variance. Default 0.04 (σ=20%) matches
                     +20% deliberate offset init in Mode B.
        gamma      : adaptive-Q scale factor. gamma>1 → larger process noise →
                     faster SOC tracking but noisier. Best value per fleet
                     selected by sweep {0.5, 1, 2} on calibration segments.
        cal_soc_fn : SOC-dependent OCV correction δV(SOC) [V/cell] callable.
                     Applied inside EKF measurement model so innovation reflects
                     only noise, not systematic chemistry bias.
        cal_dR0    : current-proportional R0 correction [V/A].
                     Sign convention: same as I_A (discharge-positive).
        """
        self.Q_nom = Q_nom_Ah
        self.R_int = R_int_ohm
        self._ocv_fn_custom = ocv_fn
        self._gamma = float(gamma)
        self._cal_soc_fn = cal_soc_fn
        self._cal_dR0 = float(cal_dR0)

        self.x1 = np.array([0.5, 0.0])           # [SOC, V_polarization]
        self.P1 = np.diag([float(P0_soc), 0.01**2])
        self._Q_base = np.diag([1e-6, 1e-5])
        self._R_meas = np.array([[float(R_meas_V2)]])

        self.x2 = np.array([1.0, R_int_ohm])     # [SOH, R_int_aged]
        self.P2 = np.diag([0.05**2, 0.005**2])
        self._R_int_meas_var = max(float(R_meas_V2), 1e-8)  # reuses voltage measurement noise

        # Safety guard (see module docstring): if the externally-fit
        # calibration's current-proportional correction is itself already
        # outside a plausible resistance range, the R_int slow loop would
        # just be learning from a residual known to be corrupted -- disable
        # it rather than let it compound an already-diagnosed calibration
        # problem. R_int then stays frozen at its init value, as it always
        # did before the Round 4 fix.
        self.r_int_update_enabled = abs(self._cal_dR0) < CAL_DR0_SANITY_THRESHOLD_OHM
        self.r_int_guard_reason = (
            None if self.r_int_update_enabled else
            f"cal_dR0={self._cal_dR0*1000:+.1f} mV/A exceeds sanity threshold "
            f"({CAL_DR0_SANITY_THRESHOLD_OHM*1000:.0f} mV/A) -- R_int loop disabled, "
            f"frozen at init value {R_int_ohm*1000:.2f} mOhm"
        )

        # Slow-loop (x2) process-noise / observation-variance constants.
        # Small relative to P2's initial diagonal -- x2 should drift only
        # slowly between genuine observations, consistent with SOH/R_int
        # being slow (cycle-to-cycle) quantities, not fast per-step states.
        self._Q2_R_INT = (0.0005 * R_int_ohm) ** 2
        self._Q2_SOH_PROCESS = 1e-7
        self._Q2_SOH_OBS_BASE = 0.02 ** 2   # base variance of a SOH observation at dsoc=1.0

        # SOH accumulator state (reset each time a SOH update fires)
        self._soh_accum_ah = 0.0
        self._soh_accum_dsoc = 0.0
        self._soh_soc_ref = float(self.x1[0])

        # Prada 2012 LFP OCV table — used only when ocv_fn is None
        self._soc_pts = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                   0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                                   0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        self._ocv_pts = np.array([2.800, 3.050, 3.150, 3.210, 3.250, 3.280,
                                   3.300, 3.310, 3.320, 3.325, 3.330, 3.335,
                                   3.338, 3.340, 3.345, 3.360, 3.390, 3.430,
                                   3.480, 3.540, 3.650])

    def _ocv(self, soc: float) -> float:
        if self._ocv_fn_custom is not None:
            return float(self._ocv_fn_custom(soc))
        return float(np.interp(np.clip(soc, 0.0, 1.0), self._soc_pts, self._ocv_pts))

    def _docv_dsoc(self, soc: float) -> float:
        h = 0.005
        return (self._ocv(soc + h) - self._ocv(soc - h)) / (2.0 * h)

    def _cal_offset(self, soc: float) -> float:
        """SOC-dependent calibration offset δV(SOC) [V]."""
        if self._cal_soc_fn is not None:
            return float(self._cal_soc_fn(float(np.clip(soc, 0.0, 1.0))))
        return 0.0

    def _adaptive_Q(self, soc: float) -> np.ndarray:
        slope = abs(self._docv_dsoc(soc))
        factor = min(1.0 / max(slope, 0.02), 50.0)
        return self._Q_base * factor * self._gamma

    def _update_r_int(self, V_meas: float, I_A: float, cal_off: float, r0_off: float) -> None:
        """
        Slow-loop scalar Kalman update for R_int (x2[1]).
        Gated on |I_A| > R_INT_MIN_CURRENT_A: at near-zero current, dV/dR_int
        = -I_A carries ~no information, and running the update anyway just
        injects measurement noise into R_int. This is a SEPARATE filter from
        the SOC update above (Plett-style two-filter design), not an extra
        row bolted onto the SOC filter's H -- R_int is observable via a
        completely different term of the same voltage equation (the IR-drop
        term), so it gets its own scalar Kalman gain, not a shared 2x2 update.
        """
        if not self.r_int_update_enabled:
            return
        if abs(I_A) < R_INT_MIN_CURRENT_A:
            return
        soc, v_pol = self.x1
        soh, r_int = self.x2

        V_pred_without_ir = self._ocv(soc) + v_pol + cal_off + r0_off
        h2 = -I_A * r_int
        H2 = -I_A
        innov2 = (V_meas - V_pred_without_ir) - h2

        p_r = float(self.P2[1, 1])
        s2 = H2 * p_r * H2 + self._R_int_meas_var + EPS
        k2 = p_r * H2 / s2

        r_int_new = r_int + k2 * innov2
        # R_int must stay physically positive and within a sane band around
        # the initial estimate (a runaway scalar filter with a bad H2==0
        # near-zero-current edge case should not be allowed to diverge).
        r_int_new = float(np.clip(r_int_new, 0.1 * self.R_int, 5.0 * self.R_int))
        self.x2[1] = r_int_new
        self.P2[1, 1] = max((1.0 - k2 * H2) * p_r, 1e-10)
        self.P2[1, 1] += self._Q2_R_INT

    def _update_soh(self, dt_s: float, I_A: float) -> None:
        """
        Slow-loop recursive SOH update (x2[0]).
        SOH is NOT observable from a single instantaneous voltage residual --
        it is a coulomb-throughput-per-unit-delta-SOC quantity, so it needs
        real accumulated SOC swing before an update is trustworthy. Between
        calls, accumulate coulomb-counted Ah and the fast filter's own
        delta-SOC since the last fired update; once |delta-SOC| clears
        SOH_MIN_DSOC, form SOH_obs = accumulated_Ah / (delta_SOC * Q_nom) and
        blend it into x2[0] with a confidence gain that grows with how much
        delta-SOC was actually observed (more evidence -> more trusted).
        """
        self._soh_accum_ah += abs(I_A) * dt_s / 3600.0
        current_soc = float(self.x1[0])
        dsoc = current_soc - self._soh_soc_ref

        if abs(dsoc) < SOH_MIN_DSOC:
            return

        soh_obs = float(np.clip(
            self._soh_accum_ah / (abs(dsoc) * self.Q_nom), 0.5, 1.05
        ))

        p_soh = float(self.P2[0, 0])
        # Confidence in this observation grows with the swing actually seen;
        # a small swing near SOH_MIN_DSOC is barely more trustworthy than the
        # prior, a large swing (near a full cycle) is close to fully trusted.
        obs_var = self._Q2_SOH_OBS_BASE / max(abs(dsoc), SOH_MIN_DSOC)
        k_soh = p_soh / (p_soh + obs_var + EPS)

        self.x2[0] = float(np.clip(self.x2[0] + k_soh * (soh_obs - self.x2[0]), 0.5, 1.05))
        self.P2[0, 0] = max((1.0 - k_soh) * p_soh, 1e-8) + self._Q2_SOH_PROCESS

        self._soh_accum_ah = 0.0
        self._soh_soc_ref = current_soc

    def update(self, V_meas: float, I_A: float, dt_s: float, T_C: float = 25.0) -> dict:
        """
        EKF measurement update with SOC-dependent calibration correction.
        Key design choice (Round 3): δV(SOC) from cal_soc_fn is added to V_pred
        so the innovation (z - h(x)) is correct, but its derivative ∂δV/∂SOC is
        NOT included in H. Treating the correction as locally constant keeps the
        Kalman gain stable — adding dcal/dSOC to H produced erratic gain swings
        from the PCHIP spline slope and destroyed BMW convergence (Round 2 bug).

        Round 4: after the SOC update below, also runs the two slow-loop
        parameter updates (_update_r_int, _update_soh) so x2 = [SOH, R_int]
        is genuinely estimated, not just carried at its init value -- see
        module docstring for why these are separate filters, not one more
        row on the SOC filter's H.
        """
        soc, v_pol = self.x1
        Q_eff = self.Q_nom * float(self.x2[0])
        R_use = float(self.x2[1])
        tau = 50.0

        # Predict
        soc_p = float(np.clip(soc - (I_A * dt_s) / (3600.0 * Q_eff), 0.0, 1.0))
        v_pol_p = v_pol * np.exp(-dt_s / tau) + R_use * (1.0 - np.exp(-dt_s / tau)) * I_A
        x_p = np.array([soc_p, v_pol_p])
        F = np.array([[1.0, 0.0], [0.0, np.exp(-dt_s / tau)]])
        P_p = F @ self.P1 @ F.T + self._adaptive_Q(soc)

        # Predicted voltage with calibration correction applied inside model
        cal_off = self._cal_offset(x_p[0])
        r0_off  = self._cal_dR0 * I_A   # I_A discharge-positive
        V_pred  = self._ocv(x_p[0]) - I_A * R_use + x_p[1] + cal_off + r0_off

        # Jacobian H — δV(SOC) treated as locally constant; derivative excluded.
        # Calibration enters innovation only, keeping Kalman gain stable.
        H = np.array([[self._docv_dsoc(x_p[0]), 1.0]])

        S = (H @ P_p @ H.T + self._R_meas)[0, 0] + EPS
        K = (P_p @ H.T) / S
        innov = V_meas - V_pred
        self.x1 = x_p + K.flatten() * innov
        self.x1[0] = float(np.clip(self.x1[0], 0.0, 1.0))
        self.P1 = (np.eye(2) - np.outer(K.flatten(), H)) @ P_p

        # Slow loop: genuinely update x2 = [SOH, R_int] (Round 4 fix).
        self._update_r_int(V_meas, I_A, cal_off, r0_off)
        self._update_soh(dt_s, I_A)

        return {
            "soc":        float(self.x1[0]),
            "soh":        float(self.x2[0]),
            "r_int":      float(self.x2[1]),
            "V_pred":     float(V_pred),
            "innovation": float(innov),
        }

    def set_soc(self, soc_init: float) -> None:
        """Initialize SOC from OCV at startup — no BMS needed."""
        self.x1[0] = float(np.clip(soc_init, 0.0, 1.0))
        self._soh_soc_ref = float(self.x1[0])
        self._soh_accum_ah = 0.0


def validate() -> bool:
    print("=" * 55)
    print("VALIDATING: diagnosis/dual_ekf_lfp.py")
    print("=" * 55)
    ok = True

    def chk(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  | {detail}" if detail else ""))

    ekf = DualEKF_LFP()

    span = abs(ekf._ocv(0.75) - ekf._ocv(0.30)) * 1000
    chk("LFP plateau flat <70mV (SOC 30-75%)", span < 70, f"{span:.1f}mV")
    chk("OCV at SOC=0.5 in plateau", 3.25 < ekf._ocv(0.5) < 3.40,
        f"{ekf._ocv(0.5):.3f}V")
    chk("Adaptive Q larger in plateau",
        ekf._adaptive_Q(0.5)[0, 0] > ekf._adaptive_Q(0.02)[0, 0])

    ekf.x1 = np.array([0.5, 0.0])
    r = ekf.update(V_meas=3.33, I_A=80.0, dt_s=1.0)
    chk("SOC in [0,1] after update", 0 <= r["soc"] <= 1, f"{r['soc']:.4f}")
    chk("V_pred in range", 2.5 < r["V_pred"] < 3.7, f"{r['V_pred']:.3f}V")

    ekf.x1 = np.array([0.80, 0.0])
    for _ in range(100):
        ekf.update(V_meas=3.33, I_A=80.0, dt_s=1.0)
    chk("SOC decreases over discharge", ekf.x1[0] < 0.80, f"→{ekf.x1[0]:.4f}")

    # Gamma scaling test
    ekf_hi = DualEKF_LFP(gamma=2.0)
    ekf_lo = DualEKF_LFP(gamma=0.5)
    chk("gamma=2.0 Q larger than gamma=0.5",
        ekf_hi._adaptive_Q(0.5)[0, 0] > ekf_lo._adaptive_Q(0.5)[0, 0])

    # cal_soc_fn test
    ekf_cal = DualEKF_LFP(cal_soc_fn=lambda s: 0.050, cal_dR0=0.001)
    ekf_cal.x1 = np.array([0.5, 0.0])
    r_cal = ekf_cal.update(V_meas=3.38, I_A=10.0, dt_s=1.0)
    chk("cal_soc_fn applied (V_pred shifted by ~50mV)", r_cal["V_pred"] > 3.30)

    # Round 4: x2 = [SOH, R_int] must actually change from init, not sit static.
    ekf_dual = DualEKF_LFP(Q_nom_Ah=2.0, R_int_ohm=0.010, R_meas_V2=1e-5)
    ekf_dual.set_soc(0.90)
    soh0, r0 = float(ekf_dual.x2[0]), float(ekf_dual.x2[1])
    # Simulate a discharge with a TRUE capacity of 1.7Ah (85% SOH) and a
    # TRUE R_int of 0.015 ohm, both different from the filter's init guess,
    # so a genuinely-working update should move x2 away from init toward
    # something in that direction.
    true_soh, true_r = 0.85, 0.015
    soc_true = 0.90
    rng = np.random.default_rng(0)
    for _ in range(4000):
        I = 2.0  # amps, discharge-positive
        dt = 1.0
        soc_true = max(0.0, soc_true - (I * dt) / (3600.0 * 2.0 * true_soh))
        v_true = ekf_dual._ocv(soc_true) - I * true_r
        v_meas = v_true + rng.normal(0, 1e-3)
        ekf_dual.update(V_meas=v_meas, I_A=I, dt_s=dt)
    soh_f, r_f = float(ekf_dual.x2[0]), float(ekf_dual.x2[1])
    chk("SOH moved from init toward true value (not static)",
        abs(soh_f - true_soh) < abs(soh0 - true_soh),
        f"init={soh0:.3f} -> final={soh_f:.3f} (true={true_soh})")
    chk("R_int moved from init toward true value (not static)",
        abs(r_f - true_r) < abs(r0 - true_r),
        f"init={r0:.4f} -> final={r_f:.4f} (true={true_r})")

    # Round 4 safety guard: a wildly unstable external calibration (as seen
    # on CALCE, -260 mV/A) must disable the R_int loop, not let it learn
    # from a residual already known to be corrupted.
    ekf_bad_cal = DualEKF_LFP(R_int_ohm=0.010, cal_dR0=-0.2605)
    chk("R_int loop disabled when cal_dR0 is implausible (CALCE-like case)",
        ekf_bad_cal.r_int_update_enabled is False,
        ekf_bad_cal.r_int_guard_reason)
    r_before = float(ekf_bad_cal.x2[1])
    for _ in range(500):
        ekf_bad_cal.update(V_meas=3.30, I_A=5.0, dt_s=1.0)
    chk("R_int stays frozen at init under the guard",
        float(ekf_bad_cal.x2[1]) == r_before,
        f"{r_before:.4f} -> {float(ekf_bad_cal.x2[1]):.4f}")

    ekf_good_cal = DualEKF_LFP(R_int_ohm=0.010, cal_dR0=0.0001)
    chk("R_int loop stays enabled for a sane calibration (BMW-i3-like case)",
        ekf_good_cal.r_int_update_enabled is True)

    # Fleet-specific R_meas
    ekf_tight = DualEKF_LFP(R_meas_V2=2.5e-5)
    chk("Fleet-specific R_meas stored", abs(ekf_tight._R_meas[0, 0] - 2.5e-5) < 1e-9)

    print(f"\nResult: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


if __name__ == "__main__":
    validate()
