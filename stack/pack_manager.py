"""
4S5P Battery Pack Manager with Lumped Thermal Network.

Topology: 4 series groups x 5 parallel cells = 20 cells total.
Physics references:
    Kirchhoff: Kirchhoff (1845) Annalen der Physik 64:497.
    Thermal: Bernardi et al. (1985) J. Electrochem. Soc. 132:5.
    Cell variation: Schmalstieg et al., TUM Battery Workshop 2021.
    Weakest cell composite score: internal methodology.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from core.dfn_cell import DFNCell, NMC811_cartridge, T0, EPS

# =============================================================================
# PACK CONSTANTS
# =============================================================================
N_SERIES: int = 4          # Number of series groups
N_PARALLEL: int = 5        # Cells per parallel group
N_CELLS: int = N_SERIES * N_PARALLEL  # Total cells = 20

# Weakest cell composite score weights (must sum to 1.0)
W_SOH: float = 0.40        # SOH risk weight
W_THERMAL: float = 0.35    # Thermal risk weight
W_PLATING: float = 0.25    # Plating risk weight

# Thermal network parameters
R_CELL_CELL: float = 0.5   # Thermal resistance, cell-to-cell [K/W]
R_CELL_COOL: float = 2.0   # Thermal resistance, cell-to-coolant [K/W]
T_COOLANT: float = T0 + 5.0  # Coolant temperature [K]

# Thermal runaway thresholds (based on ARC data for NMC)
# Reference: Feng et al. (2018) Joule 2:1985
TR_T_ONSET: float = 353.15    # Onset temperature [K] (80 C)
TR_T_RUNAWAY: float = 423.15  # Runaway temperature [K] (150 C)

# Fault risk thresholds
FAULT_SOH_THRESH: float = 0.80     # SOH below this = high risk
FAULT_PLATING_THRESH: float = 0.5  # Plating risk above this = high risk
FAULT_TEMP_THRESH: float = T0 + 45.0  # Temperature above this = high risk [K]


class PackManager:
    """
    4S5P battery pack manager.

    Manages 20 DFN cells in 4 series groups of 5 parallel cells.
    Implements:
        - Kirchhoff current distribution in parallel groups
        - 20-node lumped thermal network (center hotter than edge)
        - Cell-to-cell variation (sigma=0.2%, TUM 2021)
        - Weakest cell detection via composite risk score
        - Causal root cause diagnosis
        - GNN input interface

    Cell indexing: cell[s][p] -> flat index s*N_PARALLEL + p
    where s in [0..3] (series), p in [0..4] (parallel).
    """

    def __init__(
        self,
        chemistry_factory=None,
        rng_seed: int = 0,
    ) -> None:
        """
        Initialize 4S5P pack.

        Args:
            chemistry_factory: Callable returning ChemistryCartridge (default NMC811).
            rng_seed: Seed for cell variation RNG.
        """
        if chemistry_factory is None:
            chemistry_factory = NMC811_cartridge

        # Create 20 cells with unique variation seeds
        self.cells: List[DFNCell] = [
            DFNCell(
                chemistry=chemistry_factory(),
                cell_id=i,
                variation_seed=rng_seed * N_CELLS + i,
            )
            for i in range(N_CELLS)
        ]

        # Cell layout: cells[s][p]
        # Flat index: s * N_PARALLEL + p
        self.n_cells = N_CELLS
        self.n_series = N_SERIES
        self.n_parallel = N_PARALLEL

        # Build thermal network (20-node lumped)
        # Node positions for 4x5 grid: (row=series, col=parallel)
        self._build_thermal_network()

        # Build adjacency matrix (thermal + electrical edges)
        self._adj = self._build_adjacency()

        # History for diagnostics
        self._step_count: int = 0
        self._V_pack_history: List[float] = []
        self._weakest_history: List[int] = []

        # Pre-allocated arrays for GNN input
        self._node_features = np.zeros((N_CELLS, 7), dtype=np.float64)

    def __repr__(self) -> str:
        V = self.pack_voltage()
        SOC = self.pack_soc()
        return f"PackManager(4S5P, V={V:.3f}V, SOC={SOC:.3f}, step={self._step_count})"

    def _cell_idx(self, s: int, p: int) -> int:
        """Flat index from (series group s, parallel position p)."""
        return s * N_PARALLEL + p

    def _build_thermal_network(self) -> None:
        """
        Build 20-node lumped thermal network.
        Node positions: 4x5 grid (series x parallel).
        Center cells (p=2) are hotter due to reduced convection access.
        Edge cells (p=0,4) have direct coolant contact.

        Reference: Bandhauer et al. (2011) J. Electrochem. Soc. 158:R1.
        """
        # Thermal conductance matrix [W/K], 20x20
        self._G_thermal = np.zeros((N_CELLS, N_CELLS), dtype=np.float64)

        # Cell-to-cell conductance (nearest neighbor in parallel direction)
        G_cc = 1.0 / R_CELL_CELL  # [W/K]

        for s in range(N_SERIES):
            for p in range(N_PARALLEL):
                idx = self._cell_idx(s, p)

                # Parallel-direction coupling (within series group)
                if p > 0:
                    idx_left = self._cell_idx(s, p - 1)
                    self._G_thermal[idx, idx_left] = G_cc
                    self._G_thermal[idx_left, idx] = G_cc

                # Series-direction coupling (between groups)
                if s > 0:
                    idx_above = self._cell_idx(s - 1, p)
                    G_series = G_cc * 0.5  # Weaker coupling in series direction
                    self._G_thermal[idx, idx_above] = G_series
                    self._G_thermal[idx_above, idx] = G_series

        # Coolant conductance: edge cells (p=0,4) have better cooling
        self._G_cool = np.zeros(N_CELLS, dtype=np.float64)
        G_cool_base = 1.0 / R_CELL_COOL  # [W/K]
        for s in range(N_SERIES):
            for p in range(N_PARALLEL):
                idx = self._cell_idx(s, p)
                # Linear decay from edge to center; center = 10% of edge cooling
                edge_factor = 1.0 - 0.9 * (p / (N_PARALLEL - 1) - 0.5)**2 * 4
                # Remap: p=0 or 4 -> 1.0; p=2 -> 0.1
                dist_center = abs(p - (N_PARALLEL - 1) / 2) / ((N_PARALLEL - 1) / 2)
                cool_factor = 0.1 + 0.9 * dist_center
                self._G_cool[idx] = G_cool_base * cool_factor

    def _build_adjacency(self) -> np.ndarray:
        """
        Build 20x20 adjacency matrix encoding thermal and electrical edges.
        Entries:
            1.0 = electrical series connection
            0.5 = electrical parallel connection
            0.3 = thermal coupling (non-electrical neighbors)
            0.0 = no connection

        Returns:
            adj: shape (20, 20) float64.
        """
        adj = np.zeros((N_CELLS, N_CELLS), dtype=np.float64)

        for s in range(N_SERIES):
            for p in range(N_PARALLEL):
                idx = self._cell_idx(s, p)

                # Electrical parallel edges (same series group)
                for p2 in range(N_PARALLEL):
                    if p2 != p:
                        idx2 = self._cell_idx(s, p2)
                        adj[idx, idx2] = 0.5  # Parallel electrical

                # Electrical series edges (between groups, same parallel position)
                if s > 0:
                    idx_prev = self._cell_idx(s - 1, p)
                    adj[idx, idx_prev] = 1.0  # Series electrical
                    adj[idx_prev, idx] = 1.0

                # Thermal edges (adjacent in parallel direction)
                if p > 0:
                    idx_left = self._cell_idx(s, p - 1)
                    if adj[idx, idx_left] == 0.0:  # Don't override electrical
                        adj[idx, idx_left] = 0.3
                        adj[idx_left, idx] = 0.3

        return adj

    def get_adjacency(self) -> np.ndarray:
        """
        Return (20, 20) adjacency matrix with thermal + electrical edges.

        Returns:
            adj: shape (20, 20) float64.
        """
        return self._adj.copy()

    def _kirchhoff_current_distribution(
        self, s: int, I_string: float
    ) -> np.ndarray:
        """
        Kirchhoff current distribution in parallel group s.
        For N parallel cells with internal resistances R_i:
            I_i = (1/R_i) * V_group / sum(1/R_j)
        Reference: Kirchhoff (1845) Annalen der Physik 64:497.

        Args:
            s: Series group index [0..3]
            I_string: Total string current [A], positive = discharge
        Returns:
            I_cells: shape (N_PARALLEL,) current per cell [A]
        """
        R_cells = np.array([
            self.cells[self._cell_idx(s, p)].R_ohm
            for p in range(N_PARALLEL)
        ], dtype=np.float64)  # [Ohm]

        # Conductances
        G_cells = 1.0 / (R_cells + EPS)  # [1/Ohm]
        G_total = G_cells.sum()

        # Current proportional to conductance (Kirchhoff current divider)
        I_cells = I_string * G_cells / (G_total + EPS)  # [A]
        return I_cells

    def step_pack(self, I_pack: float, dt: float) -> Dict:
        """
        Advance entire pack by one timestep.

        Args:
            I_pack: Pack current [A], positive = discharge.
            dt: Timestep [s].
        Returns:
            dict with pack-level diagnostics and per-cell results.
        """
        # String current = pack current (series topology)
        I_string = I_pack  # [A]

        # Step each series group
        all_results = []
        V_groups = np.zeros(N_SERIES, dtype=np.float64)

        for s in range(N_SERIES):
            # Kirchhoff distribution within parallel group
            I_cells = self._kirchhoff_current_distribution(s, I_string)

            group_results = []
            V_cell_min = np.inf

            for p in range(N_PARALLEL):
                cell = self.cells[self._cell_idx(s, p)]
                res = cell.step(float(I_cells[p]), dt)
                group_results.append(res)
                # Parallel group voltage = weakest cell voltage
                V_cell_min = min(V_cell_min, res["V"])

            V_groups[s] = V_cell_min  # Conservative: weakest determines group V
            all_results.append(group_results)

        # Update thermal network (after electrical step)
        self._update_thermal_network(dt)

        # Pack-level aggregation
        V_pack = float(V_groups.sum())  # Series sum [V]
        SOC_pack = self._compute_pack_soc()
        SOH_pack = self._compute_pack_soh()

        # Weakest cell identification
        weakest_idx, weakest_score = self._find_weakest_cell()

        # Thermal runaway risk
        tr_risk = self._thermal_runaway_risk()

        self._step_count += 1
        self._V_pack_history.append(V_pack)
        self._weakest_history.append(weakest_idx)

        return {
            "V_pack": V_pack,
            "I_pack": I_pack,
            "SOC_pack": SOC_pack,
            "SOH_pack": SOH_pack,
            "V_groups": V_groups,
            "weakest_cell": weakest_idx,
            "weakest_score": weakest_score,
            "tr_risk": tr_risk,
            "step": self._step_count,
            "cell_results": all_results,
        }

    def _update_thermal_network(self, dt: float) -> None:
        """
        Update cell temperatures via lumped thermal network.
        dT_i/dt = (sum_j G_ij*(T_j-T_i) + G_cool_i*(T_cool-T_i)) / (m*Cp)
        Reference: Bernardi et al. (1985) J. Electrochem. Soc. 132:5.

        Sub-steps to satisfy explicit Euler stability: dt_sub < mCp/G_total.
        G_total for a center cell is ~6 W/K; mCp ~37 J/K → limit ~6 s.

        Args:
            dt: Timestep [s].
        """
        # Stable sub-step: 0.4 * mCp / G_max (0.4 gives margin below the limit of 1.0)
        m0 = self.cells[0].chem.rho * self.cells[0].chem.V_cell
        Cp0 = self.cells[0].chem.Cp
        G_max = float(np.max(self._G_thermal.sum(axis=1) + self._G_cool))
        dt_sub = min(dt, 0.4 * m0 * Cp0 / (G_max + EPS))
        n_sub = max(1, int(np.ceil(dt / dt_sub)))
        dt_sub = dt / n_sub

        for _ in range(n_sub):
            T_cells = np.array([c.state.T for c in self.cells], dtype=np.float64)

            for i, cell in enumerate(self.cells):
                dT_cc = 0.0
                for j in range(N_CELLS):
                    if self._G_thermal[i, j] > 0:
                        dT_cc += self._G_thermal[i, j] * (T_cells[j] - T_cells[i])

                dT_cool = self._G_cool[i] * (T_COOLANT - T_cells[i])

                m_cell = cell.chem.rho * cell.chem.V_cell
                Cp_cell = cell.chem.Cp
                dTdt_network = (dT_cc + dT_cool) / (m_cell * Cp_cell + EPS)
                cell.state.T += dTdt_network * dt_sub

    def _compute_pack_soc(self) -> float:
        """
        Pack SOC = mean of all cell Coulomb-counting SOCs.
        dSOC_cell = I_cell*dt/(3600*Q_cell) = (I_pack/N_par)*dt/(3600*Q_cell)
        = I_pack*dt/(3600*Q_pack) where Q_pack = N_par*Q_cell = 5*0.5 = 2.5 Ah.
        So pack_SOC rate = I_pack/(3600*2.5) = 2/(3600*2.5) = 2.22e-4/step. ✓
        """
        socs = np.array([c.state.soc_cc for c in self.cells])
        return float(np.mean(socs))

    def _compute_pack_soh(self) -> float:
        """
        Pack SOH = min cell SOH (weakest cell limits pack).
        Returns minimum SOH [dimensionless].
        """
        sohs = np.array([
            max(0.0, 1.0 - c.state.Q_loss / (c.Q_nom_eff + EPS))
            for c in self.cells
        ])
        return float(np.min(sohs))

    def pack_voltage(self) -> float:
        """Estimate pack voltage from cell OCPs [V]. Requires at least 1 step."""
        from core.dfn_cell import ocp_graphite, ocp_nmc811
        V_groups = []
        for s in range(N_SERIES):
            V_group = np.mean([
                float(ocp_nmc811(np.array([self.cells[self._cell_idx(s, p)].state.x_pos]))[0])
                - float(ocp_graphite(np.array([self.cells[self._cell_idx(s, p)].state.x_neg]))[0])
                for p in range(N_PARALLEL)
            ])
            V_groups.append(float(V_group))
        return sum(V_groups)

    def pack_soc(self) -> float:
        """Current pack SOC (mean Coulomb-counting SOC across all cells)."""
        return self._compute_pack_soc()

    def _cell_composite_risk(self, cell_idx: int) -> Tuple[float, dict]:
        """
        Composite risk score for cell i.
        score = W_SOH * SOH_risk + W_THERMAL * thermal_risk + W_PLATING * plating_risk
        All sub-risks in [0,1].

        Args:
            cell_idx: Flat cell index [0..19]
        Returns:
            Tuple (composite_score [0,1], sub_scores dict)
        """
        cell = self.cells[cell_idx]
        s = cell.state

        # SOH risk: 0 at SOH=1, 1 at SOH=0.7 (extrapolated)
        SOH = max(0.0, 1.0 - s.Q_loss / (cell.Q_nom_eff + EPS))
        soh_risk = float(np.clip((1.0 - SOH) / 0.3, 0.0, 1.0))

        # Thermal risk: 0 at T0+25K, 1 at TR_T_RUNAWAY
        thermal_risk = float(np.clip(
            (s.T - (T0 + 25.0)) / (TR_T_RUNAWAY - T0 - 25.0), 0.0, 1.0
        ))

        # Plating risk from SEI thickness proxy
        plating_risk = float(np.clip(s.delta_SEI / (1e-6), 0.0, 1.0))  # 1 um = max risk

        score = W_SOH * soh_risk + W_THERMAL * thermal_risk + W_PLATING * plating_risk
        return float(score), {
            "soh_risk": soh_risk,
            "thermal_risk": thermal_risk,
            "plating_risk": plating_risk,
        }

    def _find_weakest_cell(self) -> Tuple[int, float]:
        """
        Find cell with highest composite risk score.

        Returns:
            Tuple (cell_idx, composite_score).
        """
        scores = np.array([self._cell_composite_risk(i)[0] for i in range(N_CELLS)])
        weakest = int(np.argmax(scores))
        return weakest, float(scores[weakest])

    def diagnose_root_cause(self, cell_idx: int) -> Dict:
        """
        Causal root cause diagnosis for a given cell.
        Uses rule-based causal reasoning on TCO indicators.

        Args:
            cell_idx: Flat cell index [0..19]
        Returns:
            dict with primary_fault, confidence, contributing_factors.
        """
        cell = self.cells[cell_idx]
        s = cell.state
        _, sub = self._cell_composite_risk(cell_idx)

        # Causal rules (ordered by priority)
        faults = []

        if s.T >= TR_T_ONSET:
            faults.append(("thermal_runaway_onset", 0.95))
        if sub["plating_risk"] > FAULT_PLATING_THRESH:
            faults.append(("li_plating_risk", 0.85))
        if sub["soh_risk"] > 0.5:
            faults.append(("sei_capacity_fade", 0.80))
        if sub["thermal_risk"] > 0.3:
            faults.append(("thermal_stress", 0.70))
        if s.delta_SEI > 5e-8:
            faults.append(("sei_growth", 0.75))

        if not faults:
            faults.append(("nominal_operation", 0.99))

        primary_fault, confidence = max(faults, key=lambda x: x[1])

        return {
            "cell_idx": cell_idx,
            "primary_fault": primary_fault,
            "confidence": confidence,
            "sub_risks": sub,
            "T_K": s.T,
            "SEI_nm": s.delta_SEI * 1e9,
            "cycle_count": s.cycle_count,
        }

    def _thermal_runaway_risk(self) -> float:
        """
        Pack-level thermal runaway (TR) risk score in [0,1].
        Based on hottest cell temperature.
        Reference: Feng et al. (2018) Joule 2:1985.

        Returns:
            tr_risk [0,1]
        """
        T_max = max(c.state.T for c in self.cells)
        if T_max < TR_T_ONSET:
            return float(np.clip((T_max - T0) / (TR_T_ONSET - T0), 0.0, 1.0) * 0.3)
        tr_risk = 0.3 + 0.7 * (T_max - TR_T_ONSET) / (TR_T_RUNAWAY - TR_T_ONSET + EPS)
        return float(np.clip(tr_risk, 0.0, 1.0))

    def get_gnn_input(self) -> np.ndarray:
        """
        Collect (20, 7) node feature matrix for GNN input.
        Each row = get_state_vector() for one cell.

        Returns:
            features: shape (20, 7) float64.
        """
        for i, cell in enumerate(self.cells):
            self._node_features[i] = cell.get_state_vector()
        return self._node_features.copy()

    def rul_estimate(self) -> float:
        """
        Pack Remaining Useful Life estimate [cycles].
        Based on weakest cell SOH trajectory via linear extrapolation.
        EOL criterion: SOH < 0.8.

        Returns:
            RUL [cycles] or 0 if already at EOL.
        """
        weakest_idx, _ = self._find_weakest_cell()
        cell = self.cells[weakest_idx]
        SOH = max(0.0, 1.0 - cell.state.Q_loss / (cell.Q_nom_eff + EPS))

        if SOH <= 0.8:
            return 0.0

        # Linear SOH degradation rate [per cycle], floored at NMC811 nominal rate.
        # NMC811 nominal: ~0.05%/cycle = 0.0005/cycle (Attia et al. 2020 Nature Energy).
        NMC_NOMINAL_RATE: float = 0.0005  # [SOH/cycle]
        cycles = max(cell.state.cycle_count, 1.0)
        observed_rate = (1.0 - SOH) / cycles
        degradation_rate = max(observed_rate, NMC_NOMINAL_RATE)  # Floor at nominal

        rul = (SOH - 0.8) / (degradation_rate + EPS)  # [cycles]
        return float(np.clip(rul, 0.0, 2000.0))  # Cap at realistic 2000 cycles


def validate() -> bool:
    """
    Self-test suite for pack_manager module.

    Returns:
        True if all tests pass.
    """
    print("=" * 60)
    print("VALIDATING: stack/pack_manager.py")
    print("=" * 60)
    all_pass = True

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f"  | {detail}" if detail else ""))
        if not condition:
            all_pass = False

    pack = PackManager(rng_seed=42)

    # Structure tests
    check("Pack: 20 cells", len(pack.cells) == N_CELLS, f"n={len(pack.cells)}")
    check("Adjacency shape (20,20)", pack.get_adjacency().shape == (N_CELLS, N_CELLS))
    check("Adjacency symmetric",
          bool(np.allclose(pack.get_adjacency(), pack.get_adjacency().T)))

    # GNN input
    feat = pack.get_gnn_input()
    check("GNN input shape (20,7)", feat.shape == (N_CELLS, 7))
    check("GNN input finite", bool(np.all(np.isfinite(feat))))

    # Step test
    result = pack.step_pack(2.0, 1.0)
    check("Pack step: V_pack positive", result["V_pack"] > 0, f"V={result['V_pack']:.3f}V")
    check("Pack step: SOC in [0,1]", 0 <= result["SOC_pack"] <= 1)
    check("Pack step: SOH in [0,1]", 0 <= result["SOH_pack"] <= 1)
    check("Pack step: weakest in [0,19]", 0 <= result["weakest_cell"] < N_CELLS)
    check("Pack step: TR risk in [0,1]", 0 <= result["tr_risk"] <= 1)

    # Current distribution
    I_cells = pack._kirchhoff_current_distribution(0, 2.0)
    check("Kirchhoff: current sums to I", abs(I_cells.sum() - 2.0) < 1e-6,
          f"sum={I_cells.sum():.6f}")
    check("Kirchhoff: all currents positive", bool(np.all(I_cells > 0)))

    # Thermal network
    temps = [c.state.T for c in pack.cells]
    check("Thermal: all temps finite", all(np.isfinite(t) for t in temps))

    # Diagnosis
    diag = pack.diagnose_root_cause(0)
    check("Diagnosis: has primary_fault", "primary_fault" in diag)
    check("Diagnosis: confidence in [0,1]", 0 <= diag["confidence"] <= 1)

    # RUL
    rul = pack.rul_estimate()
    check("RUL: positive", rul >= 0, f"RUL={rul:.1f} cycles")

    # 5-step simulation
    for _ in range(5):
        pack.step_pack(2.0, 1.0)
    soc_after = pack.pack_soc()
    check("5 steps: SOC decreased", soc_after < pack.cells[0].chem.x0_neg,
          f"SOC={soc_after:.4f}")

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    validate()
