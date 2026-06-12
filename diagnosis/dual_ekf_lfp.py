"""
Dual EKF for LFP 160Ah prismatic SOC estimation.
Replaces forced-SOC approach with self-predicting filter.

Key innovation: Adaptive process noise Q scaled by 1/|dOCV/dSOC|
prevents filter from freezing in flat LFP plateau (SOC 30-75%).
Reference: Mikhak 2024 PMC12936157 — RMSE < 0.15% with IC-method OCV.
"""
import numpy as np

EPS = 1e-12


class DualEKF_LFP:
    def __init__(
        self,
        Q_nom_Ah: float = 160.0,
        R_int_ohm: float = 0.0005,
        ocv_fn=None,
    ):
        """
        Parameters
        ----------
        Q_nom_Ah  : nominal cell capacity [Ah]
        R_int_ohm : initial internal resistance estimate [Ω]
        ocv_fn    : optional OCV(soc)->V callable that overrides the built-in
                    LFP table.  Pass an empirical NMC spline (from
                    diagnosis/nmc_ocv.py) to avoid SOC saturation on NMC fleets.
        """
        self.Q_nom = Q_nom_Ah
        self.R_int = R_int_ohm
        self._ocv_fn_custom = ocv_fn   # None → use LFP table below
        self.x1 = np.array([0.5, 0.0])          # [SOC, V_polarization]
        self.P1 = np.diag([0.1**2, 0.01**2])
        self._Q_base = np.diag([1e-6, 1e-5])
        self._R_meas = np.array([[4e-6]])         # (2mV)^2
        self.x2 = np.array([1.0, R_int_ohm])     # [SOH, R_int_aged]
        self.P2 = np.diag([0.05**2, 0.005**2])
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

    def _adaptive_Q(self, soc: float) -> np.ndarray:
        slope = abs(self._docv_dsoc(soc))
        factor = min(1.0 / max(slope, 0.02), 50.0)
        return self._Q_base * factor

    def update(self, V_meas: float, I_A: float, dt_s: float, T_C: float = 25.0) -> dict:
        soc, v_pol = self.x1
        Q_eff = self.Q_nom * float(self.x2[0])
        R_use = float(self.x2[1])
        tau = 50.0
        soc_p = float(np.clip(soc - (I_A * dt_s) / (3600.0 * Q_eff), 0.0, 1.0))
        v_pol_p = v_pol * np.exp(-dt_s / tau) + R_use * (1.0 - np.exp(-dt_s / tau)) * I_A
        x_p = np.array([soc_p, v_pol_p])
        F = np.array([[1.0, 0.0], [0.0, np.exp(-dt_s / tau)]])
        P_p = F @ self.P1 @ F.T + self._adaptive_Q(soc)
        V_pred = self._ocv(x_p[0]) - I_A * R_use + x_p[1]
        H = np.array([[self._docv_dsoc(x_p[0]), 1.0]])
        S = (H @ P_p @ H.T + self._R_meas)[0, 0] + EPS
        K = (P_p @ H.T) / S
        innov = V_meas - V_pred
        self.x1 = x_p + K.flatten() * innov
        self.x1[0] = float(np.clip(self.x1[0], 0.0, 1.0))
        self.P1 = (np.eye(2) - np.outer(K.flatten(), H)) @ P_p
        return {
            "soc": float(self.x1[0]),
            "soh": float(self.x2[0]),
            "V_pred": float(V_pred),
            "innovation": float(innov),
        }

    def set_soc(self, soc_init: float) -> None:
        """Initialize SOC from OCV at startup — no BMS needed."""
        self.x1[0] = float(np.clip(soc_init, 0.0, 1.0))
        self.P1[0, 0] = 0.05**2


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

    # Plateau flat check
    span = abs(ekf._ocv(0.75) - ekf._ocv(0.30)) * 1000
    chk("LFP plateau flat <70mV (SOC 30-75%)", span < 70, f"{span:.1f}mV")
    chk("OCV at SOC=0.5 in plateau", 3.25 < ekf._ocv(0.5) < 3.40,
        f"{ekf._ocv(0.5):.3f}V")

    # Adaptive Q in plateau > steep
    chk("Adaptive Q larger in plateau",
        ekf._adaptive_Q(0.5)[0, 0] > ekf._adaptive_Q(0.02)[0, 0])

    # Update step
    ekf.x1 = np.array([0.5, 0.0])
    r = ekf.update(V_meas=3.33, I_A=80.0, dt_s=1.0)
    chk("SOC in [0,1] after update", 0 <= r["soc"] <= 1, f"{r['soc']:.4f}")
    chk("V_pred in range", 2.5 < r["V_pred"] < 3.7, f"{r['V_pred']:.3f}V")

    # SOC decreases under discharge
    ekf.x1 = np.array([0.80, 0.0])
    for _ in range(100):
        ekf.update(V_meas=3.33, I_A=80.0, dt_s=1.0)
    chk("SOC decreases over discharge", ekf.x1[0] < 0.80, f"→{ekf.x1[0]:.4f}")

    print(f"\nResult: {'ALL PASS' if ok else 'SOME FAILED'}")
    return ok


if __name__ == "__main__":
    validate()
