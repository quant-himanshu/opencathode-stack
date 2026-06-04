"""
Policy Engine: ACO current routing + Kuramoto SOC synchronization.

ACO reference: Dorigo & Gambardella (1997) IEEE Trans. Evol. Comput. 1:53.
Kuramoto reference: Kuramoto (1984) Chemical Oscillations, Waves, Turbulence.
TR risk: Feng et al. (2018) Joule 2:1985.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.dfn_cell import EPS, T0
from stack.pack_manager import N_CELLS, N_SERIES, N_PARALLEL

# =============================================================================
# POLICY CONSTANTS
# =============================================================================
# ACO parameters (Dorigo & Gambardella 1997)
ACO_ALPHA: float = 1.0      # Pheromone exponent [dimensionless]
ACO_BETA: float = 2.0       # Heuristic exponent [dimensionless]
ACO_RHO: float = 0.1        # Pheromone evaporation rate [1/step]
ACO_Q: float = 1.0          # Pheromone deposit amount [dimensionless]
N_ANTS: int = 20            # Number of ants per optimization step

# Kuramoto parameters (Kuramoto 1984)
KURAMOTO_K: float = 0.5     # Coupling strength [1/s]
KURAMOTO_DT: float = 0.1    # Phase update timestep [s]
N_KURAMOTO_STEPS: int = 50  # Synchronization steps per call

# Policy action thresholds
TR_RISK_EMERGENCY: float = 0.95    # TR risk -> emergency stop
TR_RISK_COOLING: float = 0.50      # TR risk -> activate cooling
SOH_FLAG_THRESHOLD: float = 0.75   # SOH < this -> flag cell
PLATING_CHARGE_LIMIT: float = 0.4  # Plating risk -> reduce charge rate

# RUL end-of-life threshold [cycles]
RUL_EOL_THRESHOLD: float = 50.0


class ACOCurrentRouter:
    """
    Ant Colony Optimization for current routing in parallel groups.

    Models current distribution as a path selection problem.
    Ants route current preferentially through healthier cells (lower risk).
    Pheromone trails reinforce paths through cells with higher SOH.

    Reference: Dorigo & Gambardella (1997) IEEE Trans. Evol. Comput. 1:53.
    """

    def __init__(self, n_parallel: int = N_PARALLEL, rng_seed: int = 0) -> None:
        self.n_parallel = n_parallel
        self._rng = np.random.default_rng(rng_seed)
        # Pheromone matrix: one value per cell [dimensionless]
        self._pheromones = np.ones(n_parallel, dtype=np.float64)

    def __repr__(self) -> str:
        return f"ACOCurrentRouter(n_parallel={self.n_parallel}, pheromones={self._pheromones})"

    def _heuristic(self, cell_soh: np.ndarray, cell_risk: np.ndarray) -> np.ndarray:
        """
        Heuristic desirability: prefer high SOH, low risk cells.
        eta_i = SOH_i / (risk_i + eps)  [dimensionless]
        """
        return cell_soh / (cell_risk + EPS)

    def route(
        self,
        I_total: float,
        cell_soh: np.ndarray,
        cell_risk: np.ndarray,
    ) -> np.ndarray:
        """
        Optimize current distribution across parallel group via ACO.

        Args:
            I_total: Total current to distribute [A].
            cell_soh: SOH per cell in group, shape (n_parallel,).
            cell_risk: Composite risk per cell, shape (n_parallel,).
        Returns:
            I_cells: Optimized current per cell [A], shape (n_parallel,).
        """
        eta = self._heuristic(cell_soh, cell_risk)  # (n_parallel,)

        # ACO probability for each cell
        # P_i = (tau_i^alpha * eta_i^beta) / sum_j(tau_j^alpha * eta_j^beta)
        tau = np.maximum(self._pheromones, EPS)  # (n_parallel,)
        prob = (tau**ACO_ALPHA) * (eta**ACO_BETA)
        prob = np.maximum(prob, EPS)
        prob /= prob.sum()  # Normalize to probability distribution

        # Simulate N_ANTS ants routing current fractions
        # Each ant allocates its share proportional to selection probability
        current_allocation = np.zeros(self.n_parallel, dtype=np.float64)

        for _ in range(N_ANTS):
            # Ant selects cell probabilistically
            choice = int(self._rng.choice(self.n_parallel, p=prob))
            current_allocation[choice] += 1.0 / N_ANTS

        # Scale to total current
        I_cells = I_total * current_allocation

        # Pheromone update: evaporate + deposit on selected paths
        # Delta_tau_i = Q * (SOH_i - mean_SOH) if selected
        self._pheromones *= (1.0 - ACO_RHO)  # Evaporation
        for i in range(self.n_parallel):
            if current_allocation[i] > 0:
                deposit = ACO_Q * (cell_soh[i] - cell_soh.mean() + 0.5)
                self._pheromones[i] += max(deposit, 0.0)

        self._pheromones = np.clip(self._pheromones, 0.01, 10.0)
        return I_cells


class KuramotoSOCSynchronizer:
    """
    Kuramoto coupled oscillators for series string SOC synchronization.

    Models each cell's SOC as a phase oscillator. Cells with higher SOC
    are "ahead" in phase. The coupling forces SOC convergence.

    d(phi_i)/dt = omega_i + (K/N) * sum_j sin(phi_j - phi_i)

    Here phi_i ~ SOC_i (mapped to [0, 2*pi]).
    Synchronization recommendations feed into charge rate adjustments.

    Reference: Kuramoto (1984) Chemical Oscillations, Waves, Turbulence. Ch. 5.
    """

    def __init__(self, n_series: int = N_SERIES, rng_seed: int = 0) -> None:
        self.n_series = n_series
        self._rng = np.random.default_rng(rng_seed)
        # Natural frequencies (small disorder from SOC differences)
        self._omega = np.zeros(n_series, dtype=np.float64)

    def __repr__(self) -> str:
        return f"KuramotoSOCSynchronizer(n_series={self.n_series}, K={KURAMOTO_K})"

    def synchronize(self, soc_groups: np.ndarray) -> np.ndarray:
        """
        Run Kuramoto dynamics on series group SOCs.
        Returns recommended SOC adjustment per group.

        Args:
            soc_groups: Mean SOC per series group, shape (N_SERIES,).
        Returns:
            delta_soc: Recommended adjustment [fractional], shape (N_SERIES,).
        """
        # Map SOC to phase: phi = 2*pi*SOC
        phi = 2.0 * np.pi * soc_groups.copy()

        # Natural frequencies proportional to SOC deviation from mean
        soc_mean = soc_groups.mean()
        self._omega = (soc_groups - soc_mean) * 0.1  # Small forcing [rad/s]

        # Kuramoto integration (N_KURAMOTO_STEPS steps)
        for _ in range(N_KURAMOTO_STEPS):
            dphi = np.zeros(self.n_series, dtype=np.float64)
            for i in range(self.n_series):
                coupling = KURAMOTO_K / self.n_series * np.sum(
                    np.sin(phi - phi[i])
                )
                dphi[i] = self._omega[i] + coupling
            phi += dphi * KURAMOTO_DT

        # Map back to SOC
        soc_synced = np.clip(phi / (2.0 * np.pi), 0.0, 1.0)
        delta_soc = soc_synced - soc_groups
        return delta_soc

    def order_parameter(self, soc_groups: np.ndarray) -> float:
        """
        Kuramoto order parameter R = |mean(exp(i*phi))| in [0,1].
        R=1: perfect synchronization; R=0: incoherence.
        Reference: Kuramoto (1984) Eq. 5.2.
        """
        phi = 2.0 * np.pi * soc_groups
        R = float(np.abs(np.mean(np.exp(1j * phi))))
        return R


class PolicyEngine:
    """
    Top-level policy engine integrating ACO routing and Kuramoto synchronization.

    Actions:
        reduce_charge_rate: Limit charge current for at-risk cells.
        activate_cooling: Increase heat extraction.
        flag_cell: Mark cell for inspection.
        emergency_stop: Halt all charging/discharging.

    Also provides:
        pack_rul_estimate: Remaining useful life [cycles].
        tr_risk_score: Thermal runaway risk [0,1].
    """

    def __init__(self, rng_seed: int = 0) -> None:
        self._aco_routers = [
            ACOCurrentRouter(N_PARALLEL, rng_seed=rng_seed * N_SERIES + s)
            for s in range(N_SERIES)
        ]
        self._kuramoto = KuramotoSOCSynchronizer(N_SERIES, rng_seed)
        self._action_log: List[Dict] = []
        self._step: int = 0

    def __repr__(self) -> str:
        return f"PolicyEngine(step={self._step}, actions_logged={len(self._action_log)})"

    def compute_actions(self, pack_state: dict) -> Dict:
        """
        Compute policy actions given current pack state.

        Args:
            pack_state: Dict with keys:
                tr_risk [0,1], weakest_cell [0..19],
                weakest_score [0,1], SOC_pack [0,1],
                SOH_pack [0,1], cell_soh [20,], cell_risk [20,],
                cell_soc_groups [4,] (mean SOC per series group).
        Returns:
            Dict with actions and routing recommendations.
        """
        self._step += 1

        tr_risk = float(pack_state.get("tr_risk", 0.0))
        weakest_cell = int(pack_state.get("weakest_cell", 0))
        weakest_score = float(pack_state.get("weakest_score", 0.0))
        SOH_pack = float(pack_state.get("SOH_pack", 1.0))

        # Default cell arrays if not provided
        cell_soh = np.array(pack_state.get("cell_soh",
                            [1.0] * N_CELLS), dtype=np.float64)
        cell_risk = np.array(pack_state.get("cell_risk",
                             [0.0] * N_CELLS), dtype=np.float64)
        soc_groups = np.array(pack_state.get("cell_soc_groups",
                              [0.7] * N_SERIES), dtype=np.float64)

        actions = []
        charge_rate_factor = 1.0  # Multiplier on charge current [dimensionless]

        # --- Emergency stop (TCO-3 / TR risk) ---
        if tr_risk >= TR_RISK_EMERGENCY:
            actions.append("emergency_stop")
            charge_rate_factor = 0.0

        # --- Cooling activation ---
        elif tr_risk >= TR_RISK_COOLING:
            actions.append("activate_cooling")
            charge_rate_factor = 0.7

        # --- Cell flagging (SOH threshold) ---
        if SOH_pack < SOH_FLAG_THRESHOLD or weakest_score > 0.7:
            actions.append(f"flag_cell_{weakest_cell}")

        # --- Charge rate reduction (plating risk) ---
        if weakest_score > PLATING_CHARGE_LIMIT and "emergency_stop" not in actions:
            actions.append("reduce_charge_rate")
            charge_rate_factor = min(charge_rate_factor, 0.5)

        # --- ACO current routing per series group ---
        aco_routing = []
        for s in range(N_SERIES):
            g_start = s * N_PARALLEL
            g_end = (s + 1) * N_PARALLEL
            I_recommended = self._aco_routers[s].route(
                I_total=2.0,  # Reference 2A discharge
                cell_soh=cell_soh[g_start:g_end],
                cell_risk=cell_risk[g_start:g_end],
            )
            aco_routing.append(I_recommended)

        # --- Kuramoto SOC synchronization ---
        delta_soc = self._kuramoto.synchronize(soc_groups)
        sync_order = self._kuramoto.order_parameter(soc_groups)

        if not actions:
            actions.append("nominal")

        result = {
            "actions": actions,
            "charge_rate_factor": charge_rate_factor,
            "aco_routing": [r.tolist() for r in aco_routing],
            "soc_delta": delta_soc.tolist(),
            "sync_order_parameter": sync_order,
            "tr_risk": tr_risk,
            "step": self._step,
        }
        self._action_log.append(result)
        return result

    def tr_risk_score(self, T_max_K: float, rate_dTdt: float = 0.0) -> float:
        """
        Thermal runaway (TR) risk score in [0,1].
        Combines temperature level and heating rate.
        Reference: Feng et al. (2018) Joule 2:1985, Fig. 3.

        Args:
            T_max_K: Maximum cell temperature [K].
            rate_dTdt: Heating rate [K/s].
        Returns:
            TR risk [0,1].
        """
        T_onset = 353.15    # [K] 80 C ARC onset
        T_runaway = 423.15  # [K] 150 C runaway
        T_norm = np.clip((T_max_K - T_onset) / (T_runaway - T_onset + EPS), 0.0, 1.0)
        rate_norm = np.clip(rate_dTdt / 5.0, 0.0, 1.0)  # 5 K/s = critical rate
        return float(0.7 * T_norm + 0.3 * rate_norm)

    def pack_rul(self, soh_now: float, soh_initial: float, cycle_count: float) -> float:
        """
        Pack Remaining Useful Life [cycles] via linear degradation extrapolation.
        EOL criterion: SOH < 0.8.
        Reference: Attia et al. (2020) Nature Energy 5:737, linear model.

        Args:
            soh_now: Current pack SOH [0,1].
            soh_initial: Initial SOH (typically 1.0).
            cycle_count: Current cycle count.
        Returns:
            RUL [cycles], 0 if at EOL.
        """
        if soh_now <= 0.8:
            return 0.0
        observed_rate = (soh_initial - soh_now) / (cycle_count + EPS)
        # Floor at NMC811 nominal 0.05%/cycle (Attia 2020 Nature Energy 5:737)
        # Cap RUL at 2000 cycles (realistic upper bound for NMC811 at EOL criterion SOH=0.8)
        NMC_NOMINAL_RATE: float = 0.0005  # [SOH_loss/cycle]
        degradation_per_cycle = max(observed_rate, NMC_NOMINAL_RATE)
        rul = (soh_now - 0.8) / (degradation_per_cycle + EPS)
        return float(np.clip(rul, 0.0, 2000.0))


def validate() -> bool:
    """Self-test suite for action/policy_engine.py."""
    print("=" * 60)
    print("VALIDATING: action/policy_engine.py")
    print("=" * 60)
    all_pass = True

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f"  | {detail}" if detail else ""))
        if not condition:
            all_pass = False

    # ACO Router
    aco = ACOCurrentRouter(n_parallel=5, rng_seed=42)
    check("ACO created", True, str(aco))
    cell_soh = np.array([0.95, 0.90, 0.85, 0.92, 0.88])
    cell_risk = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
    I_cells = aco.route(10.0, cell_soh, cell_risk)
    check("ACO output shape (5,)", I_cells.shape == (5,))
    check("ACO current sums to total", abs(I_cells.sum() - 10.0) < 0.01,
          f"sum={I_cells.sum():.4f}")
    check("ACO all positive", bool(np.all(I_cells > 0)))

    # Kuramoto
    kur = KuramotoSOCSynchronizer(n_series=4, rng_seed=42)
    check("Kuramoto created", True, str(kur))
    soc_groups = np.array([0.80, 0.75, 0.78, 0.72])
    delta = kur.synchronize(soc_groups)
    check("Kuramoto delta shape (4,)", delta.shape == (4,))
    check("Kuramoto delta finite", bool(np.all(np.isfinite(delta))))
    R = kur.order_parameter(soc_groups)
    check("Order parameter in [0,1]", 0.0 <= R <= 1.0, f"R={R:.4f}")

    # Policy engine
    engine = PolicyEngine(rng_seed=42)
    check("PolicyEngine created", True, str(engine))

    # Normal state
    result = engine.compute_actions({
        "tr_risk": 0.1,
        "weakest_cell": 3,
        "weakest_score": 0.2,
        "SOH_pack": 0.95,
        "SOC_pack": 0.7,
        "cell_soh": np.ones(N_CELLS) * 0.95,
        "cell_risk": np.ones(N_CELLS) * 0.1,
        "cell_soc_groups": soc_groups,
    })
    check("Actions returned", "actions" in result)
    check("ACO routing returned", "aco_routing" in result)
    check("Charge rate in [0,1]", 0.0 <= result["charge_rate_factor"] <= 1.0,
          f"rate={result['charge_rate_factor']}")

    # Emergency stop trigger
    result_emerg = engine.compute_actions({
        "tr_risk": 0.96,
        "weakest_cell": 5,
        "weakest_score": 0.9,
        "SOH_pack": 0.70,
        "SOC_pack": 0.3,
        "cell_soh": np.ones(N_CELLS) * 0.70,
        "cell_risk": np.ones(N_CELLS) * 0.9,
        "cell_soc_groups": soc_groups,
    })
    check("Emergency stop triggered", "emergency_stop" in result_emerg["actions"],
          str(result_emerg["actions"]))
    check("Emergency: charge_rate=0", result_emerg["charge_rate_factor"] == 0.0)

    # TR risk score
    tr = engine.tr_risk_score(T_max_K=360.0, rate_dTdt=0.5)
    check("TR risk in [0,1]", 0.0 <= tr <= 1.0, f"tr={tr:.3f}")
    tr_safe = engine.tr_risk_score(T_max_K=300.0, rate_dTdt=0.0)
    check("Safe T -> TR risk=0", tr_safe == 0.0, f"tr={tr_safe:.3f}")

    # RUL
    rul = engine.pack_rul(soh_now=0.90, soh_initial=1.0, cycle_count=100.0)
    check("RUL positive", rul > 0, f"RUL={rul:.1f} cycles")
    rul_eol = engine.pack_rul(soh_now=0.79, soh_initial=1.0, cycle_count=500.0)
    check("EOL: RUL=0", rul_eol == 0.0, f"RUL={rul_eol:.1f}")

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    validate()
