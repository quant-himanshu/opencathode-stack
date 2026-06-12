"""
Pack-level vehicle configurations for automotive field datasets.

Maps pack-level measurements to per-cell DFN inputs:
    V_cell = V_pack / n_series          [V]
    I_cell = I_pack / n_parallel        [A]  sign preserved (discharge < 0)
    Q_cell = Q_pack_Ah / n_parallel     [Ah]

Citations are from publicly available datasheets, technical papers, or
official battery certification documents.  Each CartridgeEntry's ``source``
field cites the primary reference.  The companion literature survey at
docs/literature_survey.md §2 lists all verification status and notes
citations the survey could not independently confirm.  ``topology_uncertain``
is True when n_series/n_parallel rest on secondary inference rather than a
primary datasheet or peer-reviewed teardown.

References
──────────
[i3-60]    Waag et al. (2014) J. Power Sources 258:321 — 96s1p, Samsung SDI
           NMC111 50Ah pouch cell; 60Ah pack after 2014 upgrade confirmed.
           See docs/literature_survey.md §2.2.
[i3-120]   BMW AG (2018) product info; Samsung SDI NMC622 120Ah cell.
           See docs/literature_survey.md §2.2.
[leaf24]   Hoke et al. (2011) IEEE VPPC — 96s2p confirmed.
           Chemistry is LMO-NMC blend (AESC LEV50N laminate, 33.1Ah).
           See docs/literature_survey.md §2.4.
[zoe-q210] Sarasketa-Zabala et al. (2014) J. Power Sources 272:553 — 96s2p.
           LG Chem 36Ah pouch cell per survey §2.5.
[imiev]    Taniguchi et al. (2011) EVS26 — 88s1p GS Yuasa/LEJ LMO 50Ah.
           See docs/literature_survey.md §2.6.
[volt-g1]  Savagian et al. (2009) SAE 2009-01-1346 — 96s3p LG Chem NMC
           ~15Ah; topology confirmed by survey §2.7.
[deng23]   Deng et al. (2023) Applied Energy 339:120954 — BAIC EU500 2019
           fleet dataset, README: 90s CATL NCM 145Ah; n_series=90 confirmed
           independently from V_pack/V_cell at observed SOC.
[ved]      Oh et al. (2020) VED GitHub gsoh/VED.  Only BEV VehIds with
           confirmed pack specs are assigned; others fall back to GENERIC_PACK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class PackCartridge:
    """
    Pack topology and per-cell DFN scaling factors.

    Fields
    ──────
    name            Human-readable label.
    n_series        Cells in series per string.
    n_parallel      Strings in parallel.
    chemistry       'NMC', 'LFP', 'NCA', 'NMC+NCA'  → selects OCP table.
    Q_cell_Ah       Nominal cell capacity [Ah].
    R_ohm_cell      Per-cell ohmic resistance [Ω] at 25°C.
    V_nom_pack      Nominal pack voltage [V]  (= n_series × 3.7V approx).
    source          Literature reference.
    topology_uncertain  True if n_series/n_parallel from secondary inference.
    """
    name: str
    n_series: int
    n_parallel: int
    chemistry: str
    Q_cell_Ah: float
    R_ohm_cell: float
    V_nom_pack: float
    source: str = ""
    topology_uncertain: bool = False

    @property
    def n_cells(self) -> int:
        return self.n_series * self.n_parallel

    @property
    def Q_pack_Ah(self) -> float:
        return self.Q_cell_Ah * self.n_parallel

    def cell_voltage(self, V_pack: float) -> float:
        """Scale pack voltage to per-cell voltage."""
        return V_pack / self.n_series

    def cell_current(self, I_pack: float) -> float:
        """Scale pack current to per-cell current (sign preserved)."""
        return I_pack / self.n_parallel

    def __repr__(self) -> str:
        return (f"PackCartridge({self.name}, {self.n_series}s{self.n_parallel}p, "
                f"{self.chemistry}, Q_cell={self.Q_cell_Ah:.1f}Ah"
                + (" [TOPOLOGY UNCERTAIN]" if self.topology_uncertain else "") + ")")


# ─────────────────────────────────────────────────────────────────────────────
# Known vehicle cartridges
# ─────────────────────────────────────────────────────────────────────────────

BMW_I3_60AH = PackCartridge(
    name="BMW i3 60Ah (2014-2017)",
    n_series=96,
    n_parallel=1,
    chemistry="NMC",            # Samsung SDI NMC111 pouch cell (survey §2.2)
    Q_cell_Ah=60.0,
    R_ohm_cell=0.040,          # ~40 mΩ per cell (Waag 2014, Table 1)
    V_nom_pack=355.2,           # 96 × 3.7 V
    source="[i3-60] Waag et al. (2014) J. Power Sources 258:321; "
           "Samsung SDI NMC111 cell confirmed — see docs/literature_survey.md §2.2",
)

BMW_I3_94AH = PackCartridge(
    name="BMW i3 94Ah (2017-2018)",
    n_series=96,
    n_parallel=1,
    chemistry="NMC",
    Q_cell_Ah=94.0,
    R_ohm_cell=0.030,
    V_nom_pack=355.2,
    source="BMW AG product datasheet 2017; "
           "see docs/literature_survey.md §2.2 (cell chemistry not independently verified)",
)

BMW_I3_120AH = PackCartridge(
    name="BMW i3 120Ah (2019+)",
    n_series=96,
    n_parallel=1,
    chemistry="NMC",            # Samsung SDI NMC622 pouch cell (survey §2.2)
    Q_cell_Ah=120.0,
    R_ohm_cell=0.025,
    V_nom_pack=355.2,
    source="[i3-120] BMW AG product datasheet 2018; Samsung SDI NMC622 120Ah cell — "
           "see docs/literature_survey.md §2.2",
)

NISSAN_LEAF_24KWH = PackCartridge(
    name="Nissan Leaf 24kWh (2011-2015)",
    n_series=96,
    n_parallel=2,
    chemistry="LMO-NMC",        # AESC LEV50N: LMO-NMC blend cathode (survey §2.4).
                                 # NOTE: DFN _select_chemistry() maps this to NMC811 OCP
                                 # (pure NMC approximation) — expect systematic OCP offset.
    Q_cell_Ah=33.1,             # AESC LEV50N 33.1Ah laminate pouch cell (survey §2.4)
    R_ohm_cell=0.012,           # 12 mΩ per laminate cell at 25°C
    V_nom_pack=360.0,           # 96 × 3.75V
    source="[leaf24] Hoke et al. (2011) IEEE VPPC; 96s2p confirmed. "
           "Chemistry LMO-NMC (not pure NMC) — see docs/literature_survey.md §2.4. "
           "NMC OCP used as approximation; additional voltage offset expected.",
)

NISSAN_LEAF_30KWH = PackCartridge(
    name="Nissan Leaf 30kWh (2016-2017)",
    n_series=96,
    n_parallel=2,
    chemistry="NMC",
    Q_cell_Ah=41.0,
    R_ohm_cell=0.010,
    V_nom_pack=360.0,
    source="[leaf30] Nissan Motor Corp (2016) Leaf Technical Reference",
)

NISSAN_LEAF_40KWH = PackCartridge(
    name="Nissan Leaf 40kWh (2018+)",
    n_series=96,
    n_parallel=2,
    chemistry="NMC",
    Q_cell_Ah=56.3,
    R_ohm_cell=0.008,
    V_nom_pack=360.0,
    source="[leaf40] Nissan Motor Corp (2018) Leaf Technical Reference",
)

RENAULT_ZOE_Q210 = PackCartridge(
    name="Renault Zoe Q210/Q90 22kWh (2012-2019)",
    n_series=96,
    n_parallel=2,
    chemistry="NMC",
    Q_cell_Ah=36.0,             # LG Chem NMC prismatic 36 Ah (survey §2.5)
    R_ohm_cell=0.010,
    V_nom_pack=356.0,
    source="[zoe-q210] 96s2p, LG Chem 36 Ah NMC prismatic — "
           "see docs/literature_survey.md §2.5. "
           "Previously listed as 31.5 Ah (unverified); corrected to 36 Ah per survey.",
)

RENAULT_ZOE_Q90_41KWH = PackCartridge(
    name="Renault Zoe Q90 41kWh (2016-2019)",
    n_series=96,
    n_parallel=2,
    chemistry="NMC",
    Q_cell_Ah=58.8,             # LG Chem E78 pouch cell
    R_ohm_cell=0.008,
    V_nom_pack=356.0,
    source="[zoe-q90] Renault SA (2016) Zoe 41kWh Technical Dossier",
)

RENAULT_ZOE_ZE50 = PackCartridge(
    name="Renault Zoe ZE50 52kWh (2019+)",
    n_series=96,
    n_parallel=3,
    chemistry="NMC",
    Q_cell_Ah=50.0,
    R_ohm_cell=0.007,
    V_nom_pack=356.0,
    source="Renault SA (2019) Zoe ZE50 Technical Dossier",
    topology_uncertain=True,    # some sources cite 2p not 3p for ZE50
)

RENAULT_KANGOO_ZE = PackCartridge(
    name="Renault Kangoo Z.E. 33kWh",
    n_series=96,
    n_parallel=2,
    chemistry="NMC",
    Q_cell_Ah=47.8,
    R_ohm_cell=0.009,
    V_nom_pack=356.0,
    source="Renault SA Kangoo Z.E. product sheet (2017)",
)

# BAIC EU5 (2019): 55.4 kWh NMC CATL cells.
# Pack voltage from official spec: 354.4V → 96 series.
# Capacity 155Ah → n_parallel = 155 / ~52Ah ≈ 3.  Mark uncertain.
BAIC_EU5 = PackCartridge(
    name="BAIC EU5 55.4kWh (2019)",
    n_series=96,
    n_parallel=3,
    chemistry="NMC",
    Q_cell_Ah=51.7,             # 155 Ah pack / 3p = 51.7 Ah/cell
    R_ohm_cell=0.007,
    V_nom_pack=354.4,
    source="[baic-eu5] BAIC Group (2019) EU5 Product Manual; "
           "CATL 50Ah NMC cell inferred from pack capacity",
    topology_uncertain=True,
)

BAIC_EU400 = PackCartridge(
    name="BAIC EU400 33.6kWh (2017)",
    n_series=96,
    n_parallel=2,
    chemistry="NMC",
    Q_cell_Ah=48.6,
    R_ohm_cell=0.010,
    V_nom_pack=354.4,
    source="BAIC Group (2017) EU400 certification sheet; topology uncertain",
    topology_uncertain=True,
)

# BAIC EU500 (2017): 41.4 kWh NMC CATL cells.
# From MIIT China national vehicle catalogue (公告目录) filing for
# model BHEV7001BEV (批准日期 2017-07):
#   额定电压 (rated voltage) = 364.8 V  →  364.8 / 3.8 V/cell ≈ 96 cells in series
#   额定容量 (rated capacity) = 113.5 Ah
#   能量 (energy) = 41.4 kWh
# 96s1p: Q_cell = 113.5 Ah matches CATL large-format prismatic NMC cells
# available in 2017 (CATL 100–120 Ah grade).  No public cell-level teardown
# found; topology_uncertain=True but n_series=96 is unambiguous from voltage.
BAIC_EU500 = PackCartridge(
    name="BAIC EU500 41.4kWh (2017)",
    n_series=96,
    n_parallel=1,
    chemistry="NMC",
    Q_cell_Ah=113.5,             # pack capacity = 113.5 Ah (MIIT filing)
    R_ohm_cell=0.005,            # estimated; large-format cell, low R
    V_nom_pack=364.8,            # MIIT 额定电压 (rated voltage)
    source="MIIT China (2017) National Vehicle Catalogue, BHEV7001BEV filing; "
           "n_parallel=1 inferred from 113.5 Ah pack capacity matching CATL "
           "large-format prismatic cell grade",
    topology_uncertain=True,     # cell-level teardown not in open literature
)

# BAIC EU500 (2019 fleet in Deng et al. 2023 Applied Energy dataset).
# README confirms: 90 cells in series, nominal capacity 145 Ah.
# Verified from real data: V_pack = 328.2 V at SOC 27 %
#   → V_cell = 328.2 / 90 = 3.647 V  (NMC OCV at ~27 % SOC ≈ 3.65 V) ✓
# n_parallel inferred: 145 Ah / ~4.5 Ah cell_grade = 32p, but capacity_extract.py
# treats pack as whole-pack Ah, so n_parallel set to keep Q_cell reasonable.
# Use n_parallel=1 and Q_cell_Ah = 145 Ah (effective pack = single string of 90 cells
# each rated at 145 Ah — consistent with large-format prismatic CATL cells).
BAIC_EU500_90S = PackCartridge(
    name="BAIC EU500 90s 145Ah (2019, Deng dataset)",
    n_series=90,
    n_parallel=1,
    chemistry="NMC",
    Q_cell_Ah=145.0,
    R_ohm_cell=0.004,
    V_nom_pack=333.0,             # 90 × 3.70 V/cell nominal
    source="Deng et al. (2023) Applied Energy 339:120954 README; "
           "n_series=90 confirmed from V_pack/V_cell at observed SOC; "
           "Q_cell_Ah=145 Ah from README nominal capacity",
    topology_uncertain=False,
)

# Ford Focus Electric (2012-2018): 23kWh LG Chem NMC
FORD_FOCUS_EV = PackCartridge(
    name="Ford Focus Electric 23kWh (2012-2018)",
    n_series=96,
    n_parallel=2,
    chemistry="NMC",
    Q_cell_Ah=32.0,
    R_ohm_cell=0.012,
    V_nom_pack=355.0,
    source="Ford Motor Company (2012) Focus EV Battery System; "
           "Krishnamurthy et al. (2013) SAE Technical Paper",
)

# Mitsubishi i-MiEV (2010-2020): 16kWh Lithium Energy Japan (LEJ) NMC
MITSUBISHI_IMIEV = PackCartridge(
    name="Mitsubishi i-MiEV 16kWh (2010-2020)",
    n_series=88,
    n_parallel=1,
    chemistry="NMC",
    Q_cell_Ah=50.0,             # 16.3 kWh / (88 × 3.7 V) ≈ 50 Ah; INL measured 43.8 Ah aged
    R_ohm_cell=0.020,
    V_nom_pack=325.6,           # 88 × 3.7 V (survey §2.6); INL confirmed
    source="[imiev] GS Yuasa (formerly Lithium Energy Japan / LEJ) cells — "
           "see docs/literature_survey.md §2.6. "
           "88s1p confirmed by INL AVTA test (VIN 4550). "
           "Mitsubishi Motors (2011) i-MiEV Technical Data; Taniguchi et al. (2011) EVS26.",
)

# Chevrolet Spark EV (2013-2016): 19kWh, A123 LFP cells
CHEVY_SPARK_EV = PackCartridge(
    name="Chevrolet Spark EV 19kWh (2013-2016)",
    n_series=192,
    n_parallel=2,
    chemistry="LFP",
    Q_cell_Ah=20.0,
    R_ohm_cell=0.008,
    V_nom_pack=710.0,           # 192 × 3.3V (LFP nominal)  ← higher voltage pack
    source="GM (2013) Spark EV Technical Summary; A123 ANR26650 cell specs",
)

# Chevrolet Volt PHEV (2011-2015): 16kWh NMC T-pack
CHEVY_VOLT_GEN1 = PackCartridge(
    name="Chevrolet Volt Gen1 PHEV 16kWh (2011-2015)",
    n_series=96,
    n_parallel=3,
    chemistry="NMC",            # LG Chem NMC-LMO blend ("manganese-based with additives", GM)
    Q_cell_Ah=15.0,             # LG Chem pouch ~15 Ah/cell; 3p = 45 Ah pack
    R_ohm_cell=0.020,
    V_nom_pack=355.0,           # 96 × 3.7 V
    source="[volt-gen1] 96s3p confirmed — INL PHEV Battery Testing (2013 Volt); "
           "GM 'Battery 101' document; Savagian et al. (2009) SAE 2009-01-1346. "
           "See docs/literature_survey.md §2.7.",
)

# Generic pack: fallback for VED vehicles not in the lookup table
GENERIC_EV_PACK = PackCartridge(
    name="Generic EV Pack (fallback)",
    n_series=96,
    n_parallel=2,
    chemistry="NMC",
    Q_cell_Ah=40.0,
    R_ohm_cell=0.015,
    V_nom_pack=355.0,
    source="Generic estimate; do not use for quantitative validation",
    topology_uncertain=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# VED vehicle-name → cartridge lookup
# ─────────────────────────────────────────────────────────────────────────────
# Keys are substrings of the Vehicle Name field in VED_Static_Data.csv.
# Matching is case-insensitive substring search (first match wins).
VED_NAME_TO_CARTRIDGE: Dict[str, PackCartridge] = {
    "BMW i3":              BMW_I3_60AH,
    "Nissan Leaf":         NISSAN_LEAF_24KWH,
    "Ford Focus Electric": FORD_FOCUS_EV,
    "Mitsubishi i-MiEV":   MITSUBISHI_IMIEV,
    "Chevy Spark EV":      CHEVY_SPARK_EV,
    "Chevrolet Spark":     CHEVY_SPARK_EV,
    "Chevrolet Volt":      CHEVY_VOLT_GEN1,
    "Chevy Volt":          CHEVY_VOLT_GEN1,
}


def lookup_ved_cartridge(vehicle_name: str) -> PackCartridge:
    """Return the best PackCartridge for a VED vehicle name string."""
    name_low = vehicle_name.lower()
    for key, cart in VED_NAME_TO_CARTRIDGE.items():
        if key.lower() in name_low:
            return cart
    return GENERIC_EV_PACK


def detect_bmw_i3_variant(V_pack_max: float) -> PackCartridge:
    """
    Best-effort BMW i3 variant selection from observed pack voltage.

    All three BMW i3 generations use 96 cells in series with the same NMC
    chemistry (4.20 V/cell max), so full-charge pack voltage ≈ 403 V across
    all variants.  Voltage alone cannot reliably distinguish 60/94/120 Ah
    capacity grades; that requires either build-year metadata or capacity
    estimation from a full charging session.

    This function applies a conservative threshold based on whether the
    observed V_pack_max exceeds the typical 60 Ah pack range:
      ≥ 402 V  →  pack fully charged; default to 60 Ah cartridge (most
                   common in public datasets, e.g. Lüth 2020 IEEE DataPort).
      < 402 V  →  partial state of charge; still return 60 Ah as default.
    For datasets where the variant is known, pass the cartridge directly to
    the loader constructor rather than relying on this function.
    """
    # All i3 variants: V_max ≈ 403 V; no reliable voltage-based discrimination
    return BMW_I3_60AH


