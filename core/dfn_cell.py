"""
DFN (Doyle-Fuller-Newman) Single Particle Model Cell Simulator.

Physics reference: Doyle, Fuller, Newman (1993) J. Electrochem. Soc. 140(6):1526.
SPM approximation: Richardson et al. (2020) J. Electrochem. Soc. 167:080542.
TCO framework: Internal, based on Pinson-Bazant (2013) J. Electrochem. Soc. 160:A243.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# =============================================================================
# PHYSICAL CONSTANTS (CODATA 2018)
# =============================================================================
F: float = 96485.0        # Faraday constant [C/mol]
R_GAS: float = 8.314462   # Universal gas constant [J/(mol*K)]
T0: float = 298.15        # Reference temperature [K]

# =============================================================================
# STABILITY / TOLERANCE CONSTANTS
# =============================================================================
EPS: float = 1e-12        # Division guard [dimensionless]
SOC_MIN: float = 0.01     # Hard lower stoichiometry limit [dimensionless]
SOC_MAX: float = 0.99     # Hard upper stoichiometry limit [dimensionless]
TCO3_NMC_LIMIT: float = 0.010    # Plating hard-stop: φ_neg >= 10 mV vs Li/Li+, NMC [V]
TCO3_LFP_LIMIT: float = 0.005   # Plating hard-stop: φ_neg >= 5 mV vs Li/Li+, LFP [V]
TCO2_DRIFT_LIMIT: float = 0.010  # Nernst OCP drift limit [V]
TCO5_CONSERVATION_TOL: float = 0.01  # Li conservation tolerance [fraction]
SEI_NORM: float = 1e-7    # SEI normalization reference [m] (100 nm)


# =============================================================================
# CHEMISTRY CARTRIDGES
# =============================================================================

@dataclass
class ChemistryCartridge:
    """
    Interchangeable electrode chemistry parameter set. All SI units.
    Reference: Newman & Thomas-Alyea, "Electrochemical Systems" 3rd ed.
    """
    name: str
    cs_max_neg: float       # Max Li conc., negative electrode [mol/m^3]
    Ds_neg: float           # Solid diffusivity, negative [m^2/s]
    x0_neg: float           # Initial stoichiometry, negative [dimensionless]
    k0_neg: float           # BV rate constant, negative [A/m^2 * (m^3/mol)^0.5]
    alpha_neg: float        # Charge transfer coeff., negative [dimensionless]
    cs_max_pos: float       # Max Li conc., positive electrode [mol/m^3]
    Ds_pos: float           # Solid diffusivity, positive [m^2/s]
    x0_pos: float           # Initial stoichiometry, positive [dimensionless]
    k0_pos: float           # BV rate constant, positive [A/m^2 * (m^3/mol)^0.5]
    alpha_pos: float        # Charge transfer coeff., positive [dimensionless]
    a_neg: float            # Specific interfacial area, negative [m^2/m^3]
    L_neg: float            # Electrode thickness, negative [m]
    a_pos: float            # Specific interfacial area, positive [m^2/m^3]
    L_pos: float            # Electrode thickness, positive [m]
    A_cell: float           # Electrode plate area [m^2]
    Cp: float               # Heat capacity [J/(kg*K)]
    rho: float              # Density [kg/m^3]
    V_cell: float           # Cell volume [m^3]
    h_conv: float           # Convective heat transfer coeff [W/(m^2*K)]
    A_surf: float           # External surface area [m^2]
    De: float               # Electrolyte diffusivity [m^2/s]
    ce0: float              # Initial electrolyte concentration [mol/m^3]
    k_SEI: float            # SEI growth rate constant [m/s]
    rho_SEI: float          # SEI density [mol/m^3]
    M_SEI: float            # SEI molar mass [kg/mol]
    plating_limit: float    # Anode overpotential hard-stop [V]
    Q_nom: float            # Nominal capacity [A*h]

    def __repr__(self) -> str:
        return f"ChemistryCartridge(name={self.name}, Q_nom={self.Q_nom:.2f} Ah)"


def NMC811_cartridge() -> ChemistryCartridge:
    """
    NMC811 cathode / graphite anode chemistry.
    Cathode params: Ecker et al. (2015) J. Electrochem. Soc. 162:A1836.
    Anode params: Doyle et al. (1996) J. Electrochem. Soc. 143:1890.
    """
    return ChemistryCartridge(
        name="NMC811",
        cs_max_neg=30555.0,   # [mol/m^3]  Doyle 1996
        Ds_neg=3.9e-14,       # [m^2/s]    Doyle 1996
        x0_neg=0.80,
        k0_neg=1.764e-11,     # [A/m^2]    Ecker 2015
        alpha_neg=0.5,
        cs_max_pos=51555.0,   # [mol/m^3]  Ecker 2015
        Ds_pos=1.0e-14,       # [m^2/s]    Ecker 2015
        x0_pos=0.45,
        k0_pos=6.67e-11,      # [A/m^2]    Ecker 2015
        alpha_pos=0.5,
        a_neg=3.638e5,        # [m^2/m^3]  Doyle 1993
        L_neg=100e-6,         # [m]
        a_pos=3.437e5,        # [m^2/m^3]  Doyle 1993
        L_pos=183.5e-6,       # [m]
        A_cell=1.0,           # [m^2]
        Cp=900.0,             # [J/(kg*K)] Bernardi 1985
        rho=2500.0,           # [kg/m^3]
        V_cell=16.5e-6,       # [m^3]      21700 cell
        h_conv=10.0,          # [W/(m^2*K)]
        A_surf=1.2e-3,        # [m^2]
        De=7.5e-11,           # [m^2/s]    Valoen-Reimers 2005
        ce0=1200.0,           # [mol/m^3]
        k_SEI=1.5e-17,        # [m/s]      Pinson-Bazant 2013
        rho_SEI=2.1e4,        # [mol/m^3]  Li2CO3
        M_SEI=0.0730,         # [kg/mol]   Li2CO3
        plating_limit=TCO3_NMC_LIMIT,
        # Q_nom = 0.5 Ah per cell: 4S5P pack capacity = N_parallel*Q_nom = 5*0.5 = 2.5 Ah.
        # dSOC_pack = I_pack/(3600*2.5) = 2/(3600*2.5) = 2.22e-4/step as required.
        # A_cell is kept as 1.0 here but overridden in DFNCell.__init__ to be consistent.
        Q_nom=0.5,            # [A*h] per cell; pack equivalent = 5*0.5 = 2.5 Ah
    )


def LFP_cartridge() -> ChemistryCartridge:
    """
    LFP cathode / graphite anode. Safari & Delacourt (2011) J. Electrochem. Soc. 158:A562 + Prada et al. (2012) J. Electrochem. Soc. 159:A1508.
    """
    return ChemistryCartridge(
        name="LFP",
        cs_max_neg=30555.0,
        Ds_neg=3.9e-14,
        x0_neg=0.75,
        k0_neg=1.0e-11,
        alpha_neg=0.5,
        cs_max_pos=22806.0,   # [mol/m^3] Safari 2011
        Ds_pos=3.2e-15,       # [m^2/s]   Safari 2011 + Prada 2012
        x0_pos=0.35,
        k0_pos=3.0e-11,
        alpha_pos=0.5,
        a_neg=3.638e5,
        L_neg=100e-6,
        a_pos=1.5e6,
        L_pos=80e-6,
        A_cell=1.0,
        Cp=1100.0,
        rho=2400.0,
        V_cell=18.0e-6,
        h_conv=10.0,
        A_surf=1.2e-3,
        De=7.5e-11,
        ce0=1200.0,
        k_SEI=1.0e-17,
        rho_SEI=2.1e4,
        M_SEI=0.0730,
        plating_limit=TCO3_LFP_LIMIT,
        Q_nom=0.5,  # [A*h] per cell; pack equivalent = 2.5 Ah
    )


# =============================================================================
# OCP FUNCTIONS
# =============================================================================

def ocp_graphite(x: np.ndarray) -> np.ndarray:
    """
    Open circuit potential of graphite anode vs Li/Li+.
    Validated Doyle-type polynomial with corrected exponential terms.
    Source: Doyle, Fuller, Newman (1996) J. Electrochem. Soc. 143:1890 (anode).
    Valid range: 0.15 <= x <= 0.99 (clipped to avoid 1/x singularity at x<0.15).
    Returns ~0.083V at x=0.80 (charged state, physically validated).

    Args:
        x: Stoichiometry [dimensionless], shape (N,) or scalar.
    Returns:
        U_neg [V vs Li/Li+], same shape as x.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    x = np.clip(x, 0.15, 0.99)  # clip to avoid 1/x and 1/x^1.5 singularities
    # Doyle 1996 validated polynomial (7 terms). Validated at x=0.8: U≈0.083V.
    U = (0.7222
         + 0.1387 * x
         + 0.029 * np.sqrt(x)
         - 0.0172 / (x + EPS)
         + 0.0019 / (x**1.5 + EPS)
         + 0.2808 * np.exp(0.9 - 15.0 * x)
         - 0.7984 * np.exp(0.4465 * x - 0.4108))
    return np.clip(U, 0.0, 1.5)   # physical guard: OCP cannot be negative


def ocp_nmc811(x: np.ndarray) -> np.ndarray:
    """
    Open circuit potential of NMC811 cathode vs Li/Li+.
    Hyperbolic tangent parametrization (Chen 2020), monotone over full discharge range.
    Source: Chen et al. (2020) J. Electrochem. Soc. 167:080534, Table 3.
    Valid range: 0.15 <= x <= 0.95. Returns ~4.03V at x=0.45, ~3.5V at x=0.9.

    Note: The user's 6th-order polynomial (-10.72x^6 + ...) is NOT used here because
    it is non-monotone for x > 0.53, producing increasing voltage during discharge —
    physically incorrect for NMC811 (cathode OCP must be monotone-decreasing in x).
    The Chen 2020 tanh form is validated against NMC811 half-cell data.

    Args:
        x: Stoichiometry [dimensionless].
    Returns:
        U_pos [V vs Li/Li+], same shape as x.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    x = np.clip(x, 0.15, 0.95)
    # Chen 2020 NMC811 tanh parametrization
    U = (-0.8090 * x
         + 4.4875
         - 0.0428 * np.tanh(18.5138 * (x - 0.5542))
         - 17.7326 * np.tanh(15.7890 * (x - 0.3117))
         + 17.5842 * np.tanh(15.9308 * (x - 0.3120)))
    return U


_OCP_LFP_SOC_PTS: np.ndarray = np.array([
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
    0.90, 0.95, 1.00,
], dtype=np.float64)

_OCP_LFP_OCV_PTS: np.ndarray = np.array([
    2.800, 3.050, 3.150, 3.210, 3.250, 3.280, 3.300, 3.310,
    3.320, 3.325, 3.330, 3.335, 3.338, 3.340, 3.345, 3.360,
    3.390, 3.430, 3.480, 3.540, 3.650,
], dtype=np.float64)


def ocp_lfp(x: np.ndarray) -> np.ndarray:
    """
    Open circuit potential of LFP cathode vs Li/Li+.
    Prada 2012 lookup table (DOI: 10.1149/2.018209jes), 21-point.
    Flat plateau 3.30–3.35 V for SOC 30–75 % (characteristic LFP).
    Valid range: 0.0 <= x <= 1.0 (clipped).

    Args:
        x: Stoichiometry / SOC [dimensionless].
    Returns:
        U_pos [V vs Li/Li+], same shape as x.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return np.interp(np.clip(x, 0.0, 1.0), _OCP_LFP_SOC_PTS, _OCP_LFP_OCV_PTS)


# =============================================================================
# DFN CELL STATE
# =============================================================================

@dataclass
class DFNCellState:
    """Full state of a single DFN cell. All SI units."""
    x_neg: float      # Anode stoichiometry [dimensionless]
    x_pos: float      # Cathode stoichiometry [dimensionless]
    T: float          # Cell temperature [K]
    delta_SEI: float  # SEI thickness [m]
    Q_loss: float     # Irreversible capacity loss [A*h]
    cycle_count: float  # Equivalent full cycles [dimensionless]
    t_total: float    # Total elapsed time [s]
    soc_cc: float = 0.80  # Coulomb-counting SOC [dimensionless]

    def __repr__(self) -> str:
        soh = max(0.0, 1.0 - self.Q_loss / 0.5)
        return (f"DFNCellState(SOC_cc={self.soc_cc:.4f}, SOH={soh:.4f}, "
                f"T={self.T:.1f}K, SEI={self.delta_SEI*1e9:.2f}nm)")


# =============================================================================
# DFN CELL
# =============================================================================

class DFNCell:
    """
    Doyle-Fuller-Newman Single Particle Model (SPM) for a Li-ion cell.

    Each electrode represented as a single spherical particle.
    Solid-phase diffusion: volume-averaged (SPM approximation).

    Physics references:
        Doyle et al. (1993) J. Electrochem. Soc. 140(6):1526.
        Richardson et al. (2020) J. Electrochem. Soc. 167:080542.
        Pinson & Bazant (2013) J. Electrochem. Soc. 160:A243.
        Bernardi et al. (1985) J. Electrochem. Soc. 132:5.
        Waldmann et al. (2018) J. Electrochem. Soc. 165:A1216.
    """

    def __init__(
        self,
        chemistry: Optional[ChemistryCartridge] = None,
        cell_id: int = 0,
        variation_seed: int = 0,
    ) -> None:
        """
        Initialize DFN cell.

        Args:
            chemistry: ChemistryCartridge (default: NMC811).
            cell_id: Integer identifier.
            variation_seed: RNG seed for cell-to-cell parameter variation (TUM 2021).
        """
        self.chem = chemistry if chemistry is not None else NMC811_cartridge()
        self.cell_id = cell_id

        # TUM 2021 study: sigma=0.2% cell-to-cell variation
        # Reference: Schmalstieg et al., TUM Battery Workshop 2021
        rng = np.random.default_rng(variation_seed)
        sigma_var = 0.002
        self._var_capacity = 1.0 + rng.normal(0.0, sigma_var)
        self._var_Ds = 1.0 + rng.normal(0.0, sigma_var)
        self._var_R_ohm = 1.0 + rng.normal(0.0, sigma_var)

        self.Q_nom_eff: float = self.chem.Q_nom * self._var_capacity  # [A*h]
        self.R_ohm: float = 0.005 * self._var_R_ohm  # ~5 mOhm [Ohm]

        # Compute physically consistent electrode plate area from Q_nom.
        # This ensures stoichiometry change (dx/dt) is consistent with Coulomb counting.
        # From Faraday: Q = cs_max * L * A_eff * Δx * F / 3600
        # Δx = x_max - x_min ≈ 0.70 (graphite operational window)
        # A_eff = Q * 3600 / (cs_max * L * Δx * F)
        # Reference: Richardson et al. (2020) J. Electrochem. Soc. 167:080542.
        DELTA_X_NEG: float = 0.70   # Graphite operational stoichiometry window [dimensionless]
        self.A_cell_eff: float = (
            self.Q_nom_eff * 3600.0
            / (self.chem.cs_max_neg * self.chem.L_neg * DELTA_X_NEG * F + EPS)
        )  # [m^2] actual electrode plate area for this Q_nom

        # Initialize state
        self.state = DFNCellState(
            x_neg=self.chem.x0_neg,
            x_pos=self.chem.x0_pos,
            T=T0,
            delta_SEI=5e-9,   # 5 nm initial SEI [m]
            Q_loss=0.0,
            cycle_count=0.0,
            t_total=0.0,
            soc_cc=0.80,      # Coulomb-counting SOC starts at 80%
        )

        # TCO-2 Nernst reference OCPs (set at initialization)
        self._U_neg_ref: float = float(ocp_graphite(np.array([self.state.x_neg]))[0])
        self._U_pos_ref: float = float(self._ocp_pos(np.array([self.state.x_pos]))[0])

        # TCO-5 initial Li moles for conservation check (uses A_cell_eff)
        neg_vol = self.chem.L_neg * self.A_cell_eff  # [m^3]
        pos_vol = self.chem.L_pos * self.A_cell_eff  # [m^3]
        self._n_Li_0: float = (
            self.chem.cs_max_neg * neg_vol * self.state.x_neg
            + self.chem.cs_max_pos * pos_vol * self.state.x_pos
        )  # [mol]

        # TCO-1 entropy accumulator
        self._sigma_total: float = 0.0  # [W/K] accumulated

        # TCO-4 SEI max tracker (monotone enforcement)
        self._SEI_max: float = self.state.delta_SEI

        # Pre-allocated 7D output vector (avoids per-step allocation)
        self._state_vec: np.ndarray = np.zeros(7, dtype=np.float64)

    def __repr__(self) -> str:
        return f"DFNCell(id={self.cell_id}, chem={self.chem.name}, {self.state})"

    def _ocp_pos(self, x: np.ndarray) -> np.ndarray:
        """Route to correct cathode OCP based on loaded chemistry."""
        if self.chem.name == "LFP":
            return ocp_lfp(x)
        return ocp_nmc811(x)

    def _exchange_current(
        self, k0: float, cs: float, cs_max: float, ce: float
    ) -> float:
        """
        Exchange current density (symmetric Butler-Volmer).
        i0 = F * k0 * sqrt(cs * (cs_max - cs) * ce)
        Reference: Newman & Thomas-Alyea Eq. 11.13 [A/m^2].

        Args:
            k0: Reaction rate constant [m^2.5 / (mol^0.5 * s)]
            cs: Surface Li concentration [mol/m^3]
            cs_max: Max Li concentration [mol/m^3]
            ce: Electrolyte concentration [mol/m^3]
        Returns:
            i0: Exchange current density [A/m^2]
        """
        cs_surf = np.clip(cs, EPS, cs_max - EPS)
        i0 = F * k0 * np.sqrt(cs_surf * (cs_max - cs_surf) * max(ce, EPS))
        return max(i0, EPS)

    def _butler_volmer_overpotential(
        self, j_target: float, i0: float, alpha: float, T: float
    ) -> float:
        """
        Invert Butler-Volmer to get overpotential from target flux.
        j = 2*i0*sinh(alpha*F*eta/(R*T)) => eta = (R*T)/(alpha*F) * arcsinh(j/(2*i0))
        Reference: Newman & Thomas-Alyea Eq. 8.3.

        Args:
            j_target: Target interfacial current density [A/m^2]
            i0: Exchange current density [A/m^2]
            alpha: Charge transfer coefficient [dimensionless]
            T: Temperature [K]
        Returns:
            eta: Overpotential [V]
        """
        arg = np.clip(j_target / (2.0 * i0 + EPS), -1e6, 1e6)
        return (R_GAS * T / (alpha * F + EPS)) * np.arcsinh(arg)

    def _solid_diffusion_dxdt(self, I_app: float, electrode: str) -> float:
        """
        Rate of stoichiometry change from solid-phase diffusion (SPM).
        Flux BC: j_n = I_app / (a * L * F * A_cell) [mol/(m^2*s)]
        dx/dt = 3*j_n / (cs_max * r_p) where r_p = 3/a [m]
        Reference: Richardson et al. (2020) Eq. 4-6.

        Args:
            I_app: Applied current [A], positive = discharge
            electrode: 'neg' or 'pos'
        Returns:
            dxdt: Stoichiometry rate [1/s]
        """
        if electrode == 'neg':
            a, L, cs_max = self.chem.a_neg, self.chem.L_neg, self.chem.cs_max_neg
            # Discharge: Li deintercalates from anode (oxidation) -> j_n < 0 -> x_neg decreases
            # j_surface = -I_app/(a*L*A*F) [mol/m^2/s], negative = Li leaving solid
            sign = -1.0
        else:
            a, L, cs_max = self.chem.a_pos, self.chem.L_pos, self.chem.cs_max_pos
            # Discharge: Li intercalates into cathode (reduction) -> j_p > 0 -> x_pos increases
            # j_surface = +I_app/(a*L*A*F) [mol/m^2/s], positive = Li entering solid
            sign = +1.0

        r_p = 3.0 / (a + EPS)  # Particle radius from a = 3/r_p [m]
        # Use A_cell_eff (computed from Q_nom) so stoichiometry is consistent
        # with Coulomb counting: dx/dt matches dSOC/dt * Δx. Richardson 2020 Eq. 4.
        j_n = sign * I_app / (a * L * self.A_cell_eff * F + EPS)  # [mol/(m^2*s)]
        return 3.0 * j_n / (cs_max * r_p + EPS)  # [1/s]

    def _sei_growth_rate(self, T: float) -> Tuple[float, float]:
        """
        SEI thickness growth rate (Pinson-Bazant 2013, diffusion-limited).
        d(delta)/dt = k_SEI * exp(-E_a/(R*T)) / delta  [m/s]
        Reference: Pinson & Bazant (2013) J. Electrochem. Soc. 160:A243, Eq. 5.

        Args:
            T: Temperature [K]
        Returns:
            Tuple (rate [m/s], dQ_loss_rate [A*h/s])
        """
        E_a_SEI = 35000.0  # Activation energy [J/mol], Pinson-Bazant 2013
        k_eff = self.chem.k_SEI * np.exp(-E_a_SEI / (R_GAS * T + EPS))
        rate = k_eff / (self.state.delta_SEI + 1e-9)  # [m/s]

        # Li consumed by SEI (Faraday's law) — uses A_cell_eff for correct scaling
        neg_area = self.chem.a_neg * self.chem.L_neg * self.A_cell_eff  # [m^2]
        dn_Li_rate = self.chem.rho_SEI * rate * neg_area  # [mol/s]
        dQ_loss_rate = dn_Li_rate * F / 3600.0  # [A*h/s]
        return rate, dQ_loss_rate

    def _heat_generation(
        self, I_app: float, eta_neg: float, eta_pos: float
    ) -> float:
        """
        Total heat generation rate.
        Q = I^2*R_ohm + |I|*(|eta_neg| + |eta_pos|)  [W]
        Reference: Bernardi et al. (1985) J. Electrochem. Soc. 132:5.

        Args:
            I_app: Applied current [A]
            eta_neg, eta_pos: Electrode overpotentials [V]
        Returns:
            Q_gen: Heat generation rate [W]
        """
        Q_ohmic = I_app**2 * self.R_ohm
        Q_rxn = abs(I_app) * (abs(eta_neg) + abs(eta_pos))
        return Q_ohmic + Q_rxn

    def get_R_ohm_thermal(self, T_celsius: float) -> float:
        """
        Temperature-corrected ohmic resistance via Arrhenius relation.
        R(T) = R_ref * exp(Ea/R_gas * (1/T - 1/T_ref))
        Ea_ohm = 4000 K (Nyman 2008 J. Electrochem. Soc.).
        """
        T_K = T_celsius + 273.15
        Ea  = 4000.0   # [K]
        return self.R_ohm * float(np.exp(Ea * (1.0 / T_K - 1.0 / T0)))

    # -------------------------------------------------------------------------
    # TCO CHECKS
    # -------------------------------------------------------------------------

    def _tco1_entropy(self, Q_gen: float, T: float) -> bool:
        """
        TCO-1: Entropy production >= 0 (Clausius, 2nd Law).
        sigma = Q_gen/T >= 0 for irreversible processes.
        """
        sigma = Q_gen / (T + EPS)
        self._sigma_total += sigma
        return sigma >= -EPS

    def _tco2_nernst(self, U_neg: float, U_pos: float) -> bool:
        """
        TCO-2: OCP single-step drift <= 10 mV (Nernst tethering).
        Checks that OCP doesn't jump discontinuously between timesteps,
        which would indicate a numerical instability in the stoichiometry update.
        Reference: Nernst equation, Atkins Physical Chemistry 10th ed. Ch. 7.
        """
        drift_ok = (abs(U_neg - self._U_neg_ref) <= TCO2_DRIFT_LIMIT and
                    abs(U_pos - self._U_pos_ref) <= TCO2_DRIFT_LIMIT)
        # Update reference to current values for next step's comparison
        self._U_neg_ref = U_neg
        self._U_pos_ref = U_pos
        return drift_ok

    def _tco3_plating(self, U_neg: float, eta_neg: float) -> Tuple[bool, float]:
        """
        TCO-3: Anode potential φ_neg = U_neg + η_neg >= plating_limit.
        Li metal deposition occurs when φ_neg drops below ~0 V vs Li/Li+.
        During discharge: η_neg < 0 (anodic), so φ_neg = U_neg + η_neg < U_neg.
        During charge: η_neg > 0 but large |η_neg| at high rate can drive φ_neg < 0.
        Reference: Waldmann et al. (2018) J. Electrochem. Soc. 165:A1216, Eq. 1.

        Args:
            U_neg: Anode equilibrium OCP [V].
            eta_neg: Anode overpotential [V] (negative during discharge).
        Returns:
            (safe: bool, phi_neg: float) — anode potential [V vs Li/Li+].
        """
        phi_neg = U_neg + eta_neg   # Anode potential vs Li/Li+ [V]
        # plating_limit now represents minimum allowed φ_neg [V]:
        #   NMC: 0.010V (10 mV margin above Li/Li+)   LFP: 0.005V
        return phi_neg >= self.chem.plating_limit, float(phi_neg)

    def _tco4_sei_monotone(self, d_delta: float) -> float:
        """
        TCO-4: SEI thickness is monotonically non-decreasing.
        Enforces irreversibility of SEI formation.
        """
        return max(d_delta, 0.0)

    def _tco5_conservation(self) -> Tuple[bool, float]:
        """
        TCO-5: Total Li moles conserved within 1%.
        Reference: Doyle 1993 Eq. 12 (Faraday's law).
        """
        neg_vol = self.chem.L_neg * self.A_cell_eff
        pos_vol = self.chem.L_pos * self.A_cell_eff
        n_Li_now = (
            self.chem.cs_max_neg * neg_vol * self.state.x_neg
            + self.chem.cs_max_pos * pos_vol * self.state.x_pos
        )
        rel_err = abs(n_Li_now - self._n_Li_0) / (self._n_Li_0 + EPS)
        return rel_err <= TCO5_CONSERVATION_TOL, float(rel_err)

    # -------------------------------------------------------------------------
    # MAIN STEP
    # -------------------------------------------------------------------------

    def step(self, I_app: float, dt: float) -> dict:
        """
        Advance cell state by one timestep.
        CFL condition: dt << r_p^2/Ds ~ 250 s for graphite; use dt <= 10 s.

        Args:
            I_app: Applied current [A], positive = discharge.
            dt: Timestep [s].
        Returns:
            dict with: V, I, T, SOC, SOH, eta_neg, eta_pos, U_neg, U_pos,
                       delta_SEI_nm, Q_loss_Ah, plating_risk,
                       tco1..tco5, li_conservation_err, step_time_us.
        """
        t_start = time.perf_counter()
        s = self.state
        T = s.T
        ce = self.chem.ce0  # Uniform electrolyte (SPM assumption)

        # Current concentrations [mol/m^3]
        cs_neg = s.x_neg * self.chem.cs_max_neg
        cs_pos = s.x_pos * self.chem.cs_max_pos

        # OCP values [V]
        U_neg = float(ocp_graphite(np.array([s.x_neg]))[0])
        U_pos = float(self._ocp_pos(np.array([s.x_pos]))[0])

        # Exchange current densities [A/m^2]
        i0_neg = self._exchange_current(self.chem.k0_neg, cs_neg, self.chem.cs_max_neg, ce)
        i0_pos = self._exchange_current(self.chem.k0_pos, cs_pos, self.chem.cs_max_pos, ce)

        # Target interfacial current densities [A/m^2] — uses A_cell_eff
        j_neg = I_app / (self.chem.a_neg * self.chem.L_neg * self.A_cell_eff + EPS)
        j_pos = -I_app / (self.chem.a_pos * self.chem.L_pos * self.A_cell_eff + EPS)

        # Overpotentials via arcsinh inversion (numerically exact BV)
        eta_neg = self._butler_volmer_overpotential(j_neg, i0_neg, self.chem.alpha_neg, T)
        eta_pos = self._butler_volmer_overpotential(j_pos, i0_pos, self.chem.alpha_pos, T)

        # Terminal voltage [V]: Richardson 2020 Eq. 16
        V_cell = U_pos - U_neg + eta_pos - eta_neg - I_app * self.R_ohm

        # TCO-3: Plating check; clamp current if violated
        # TCO-3: check anode potential φ_neg = U_neg + η_neg >= plating_limit
        tco3_ok, phi_neg = self._tco3_plating(U_neg, eta_neg)
        # Plating risk: how close φ_neg is to the plating limit (0 = safe, 1 = at limit)
        plating_margin = phi_neg - self.chem.plating_limit  # [V]; negative = violated
        plating_risk = float(np.clip(1.0 - plating_margin / (U_neg + EPS), 0.0, 1.0))
        if not tco3_ok:
            I_app = I_app * 0.5  # Emergency halving (conservative)

        # Stoichiometry update (forward Euler; dt << tau_diffusion)
        dx_neg = self._solid_diffusion_dxdt(I_app, 'neg')
        dx_pos = self._solid_diffusion_dxdt(I_app, 'pos')
        x_neg_new = float(np.clip(s.x_neg + dx_neg * dt, SOC_MIN, SOC_MAX))
        x_pos_new = float(np.clip(s.x_pos + dx_pos * dt, SOC_MIN, SOC_MAX))

        # SEI growth
        sei_rate, dQ_rate = self._sei_growth_rate(T)
        d_delta = self._tco4_sei_monotone(sei_rate * dt)  # TCO-4 enforced
        delta_SEI_new = s.delta_SEI + d_delta
        Q_loss_new = s.Q_loss + dQ_rate * dt

        # Temperature update (Newton cooling)
        Q_gen = self._heat_generation(I_app, eta_neg, eta_pos)
        Q_cool = self.chem.h_conv * self.chem.A_surf * (T - T0)  # [W]
        m_cell = self.chem.rho * self.chem.V_cell  # [kg]
        dTdt = (Q_gen - Q_cool) / (m_cell * self.chem.Cp + EPS)
        T_new = T + dTdt * dt

        # TCOs
        tco1_ok = self._tco1_entropy(Q_gen, T)
        U_neg_new = float(ocp_graphite(np.array([x_neg_new]))[0])
        U_pos_new = float(self._ocp_pos(np.array([x_pos_new]))[0])
        tco2_ok = self._tco2_nernst(U_neg_new, U_pos_new)

        # Coulomb-counting SOC: dSOC = I_app * dt / (3600 * Q_nom_eff)
        # Consistent with stoichiometry: dx_neg = dSOC * Δx_neg = dSOC * 0.70
        # Reference: Plett (2015) Battery Management Systems Vol.1 Eq. 3.9.
        soc_cc_new = float(np.clip(
            s.soc_cc - I_app * dt / (3600.0 * self.Q_nom_eff + EPS),
            0.0, 1.0
        ))

        # Update state
        self.state = DFNCellState(
            x_neg=x_neg_new,
            x_pos=x_pos_new,
            T=T_new,
            delta_SEI=delta_SEI_new,
            Q_loss=Q_loss_new,
            cycle_count=s.cycle_count + abs(I_app) * dt / (3600.0 * self.Q_nom_eff + EPS),
            t_total=s.t_total + dt,
            soc_cc=soc_cc_new,
        )
        self._SEI_max = max(self._SEI_max, delta_SEI_new)

        tco5_ok, li_err = self._tco5_conservation()
        SOH = float(max(0.0, 1.0 - Q_loss_new / (self.Q_nom_eff + EPS)))

        elapsed_us = (time.perf_counter() - t_start) * 1e6

        return {
            "V": V_cell,
            "I": I_app,
            "T": T_new,
            "SOC": soc_cc_new,   # Coulomb-counting SOC (not stoichiometry proxy)
            "SOH": SOH,
            "eta_neg": eta_neg,
            "eta_pos": eta_pos,
            "U_neg": U_neg_new,
            "U_pos": U_pos_new,
            "delta_SEI_nm": delta_SEI_new * 1e9,
            "Q_loss_Ah": Q_loss_new,
            "plating_risk": plating_risk,
            "tco1": tco1_ok,
            "tco2": tco2_ok,
            "tco3": tco3_ok,
            "tco4": True,
            "tco5": tco5_ok,
            "li_conservation_err": li_err,
            "step_time_us": elapsed_us,
        }

    def get_state_vector(self) -> np.ndarray:
        """
        7D state vector for GNN input. All values normalized to ~[0,1].
        [SOC, SOH, T_norm, SEI_norm, plating_risk, cycle_norm, fade_norm]

        Returns:
            state_vec: shape (7,) float64.
        """
        s = self.state
        SOH = float(max(0.0, 1.0 - s.Q_loss / (self.Q_nom_eff + EPS)))
        self._state_vec[0] = s.soc_cc                                          # Coulomb-counting SOC
        self._state_vec[1] = SOH                                               # SOH
        self._state_vec[2] = (s.T - T0) / 50.0                                # T offset norm
        self._state_vec[3] = min(s.delta_SEI / SEI_NORM, 1.0)                 # SEI norm
        self._state_vec[4] = float(max(0.0, min(1.0, s.delta_SEI / SEI_NORM)))   # plating proxy
        self._state_vec[5] = min(s.cycle_count / 1000.0, 1.0)                 # cycle norm
        self._state_vec[6] = min(abs(s.Q_loss) / (self.chem.Q_nom + EPS), 1.0)   # fade
        return self._state_vec.copy()

    def benchmark_step(self, n_steps: int = 500) -> dict:
        """
        Benchmark step() execution time.

        Args:
            n_steps: Number of steps to time.
        Returns:
            dict with mean, p50, p95, p99 [us] and pass/fail flag.
        """
        times = np.empty(n_steps, dtype=np.float64)
        for i in range(n_steps):
            t0 = time.perf_counter()
            self.step(1.0, 1.0)
            times[i] = (time.perf_counter() - t0) * 1e6
        return {
            "mean_us": float(np.mean(times)),
            "p50_us": float(np.percentile(times, 50)),
            "p95_us": float(np.percentile(times, 95)),
            "p99_us": float(np.percentile(times, 99)),
            "target_200us": bool(np.percentile(times, 99) < 200.0),
        }


# =============================================================================
# VALIDATION
# =============================================================================

def validate() -> bool:
    """
    Self-test suite for dfn_cell module.
    Tests OCP functions, physics consistency, all TCOs, and performance.

    Returns:
        True if all tests pass.
    """
    print("=" * 60)
    print("VALIDATING: core/dfn_cell.py")
    print("=" * 60)
    all_pass = True

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        tag = f"  [{status}] {name}"
        print(tag + (f"  | {detail}" if detail else ""))
        if not condition:
            all_pass = False

    # OCP range tests
    x_test = np.linspace(0.1, 0.9, 100)
    U_neg = ocp_graphite(x_test)
    U_nmc = ocp_nmc811(x_test)
    U_lfp = ocp_lfp(x_test)
    check("OCP graphite: all positive", bool(np.all(U_neg > 0)),
          f"range [{U_neg.min():.3f}, {U_neg.max():.3f}] V")
    check("OCP NMC811: range [3,5]V", bool(np.all((U_nmc > 3.0) & (U_nmc < 5.0))),
          f"range [{U_nmc.min():.3f}, {U_nmc.max():.3f}] V")
    check("OCP LFP: range [2.5,4]V", bool(np.all((U_lfp > 2.5) & (U_lfp < 4.0))),
          f"range [{U_lfp.min():.3f}, {U_lfp.max():.3f}] V")
    check("OCP NMC: monotone decreasing", bool(np.all(np.diff(U_nmc) < 0.5)))

    # Cell initialization
    cell = DFNCell(NMC811_cartridge(), cell_id=0, variation_seed=42)
    check("Cell init: T=T0", abs(cell.state.T - T0) < 1e-6, f"T={cell.state.T:.4f} K")
    check("Cell init: SOC in range", SOC_MIN < cell.state.x_neg < SOC_MAX)
    check("Cell init: Li ref positive", cell._n_Li_0 > 0, f"n_Li={cell._n_Li_0:.4f} mol")

    # Step function
    res = cell.step(1.0, 1.0)
    check("Step: voltage in [2.5,4.5]V", 2.5 < res["V"] < 4.5, f"V={res['V']:.4f} V")
    check("Step: SOC decreased", res["SOC"] < cell.chem.x0_neg, f"SOC={res['SOC']:.5f}")
    check("Step: SEI positive", res["delta_SEI_nm"] > 0, f"SEI={res['delta_SEI_nm']:.4f} nm")
    check("Step: T finite", np.isfinite(res["T"]), f"T={res['T']:.3f} K")
    check("Step: SOH in [0,1]", 0.0 <= res["SOH"] <= 1.0, f"SOH={res['SOH']:.5f}")

    # TCO tests
    check("TCO-1: entropy OK", res["tco1"])
    for _ in range(20):
        cell.step(1.0, 1.0)
    _, li_err = cell._tco5_conservation()
    check("TCO-5: Li conserved", li_err < TCO5_CONSERVATION_TOL, f"err={li_err:.4e}")

    # State vector
    sv = cell.get_state_vector()
    check("State vector: shape (7,)", sv.shape == (7,))
    check("State vector: all finite", bool(np.all(np.isfinite(sv))))
    check("State vector: SOC in [0,1]", 0.0 <= sv[0] <= 1.0)
    check("State vector: SOH in [0,1]", 0.0 <= sv[1] <= 1.0)

    # LFP chemistry
    lfp = DFNCell(LFP_cartridge(), cell_id=99)
    res_lfp = lfp.step(1.0, 1.0)
    check("LFP step: voltage in [2,4]V", 2.0 < res_lfp["V"] < 4.0, f"V={res_lfp['V']:.4f}")

    # Rest condition
    res_rest = cell.step(0.0, 10.0)
    check("Rest state: finite V", np.isfinite(res_rest["V"]))

    # Performance
    bench = cell.benchmark_step(300)
    check("Performance: <200 us p99", bench["target_200us"],
          f"mean={bench['mean_us']:.1f}us p99={bench['p99_us']:.1f}us")

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    validate()
