"""
data/loaders — OpenCATHODE automotive field-dataset loaders.

All loaders produce a common schema DataFrame:
  t_s       float64  elapsed time [s] from segment start
  I_A       float64  pack current [A],  discharge < 0
  V_V       float64  pack terminal voltage [V]
  T_degC    float64  temperature [°C]  — NaN when unavailable
  SOC_bms   float64  BMS SOC [0..1]

Pack-level datasets (VED, BMW i3, Renault Zoe, Deng) report total pack voltage
and require "average-cell mode" in validate_generic.py:
  V_cell_avg = V_pack / n_series
  I_cell     = I_pack / n_parallel
Per-cell features (GNN, weakest-cell, P3S10) are disabled with a logged notice
in validate_generic.py for these datasets — they are NOT silently skipped.

Exception: the 300-EV Nature Comms dataset (not yet included here) provides
highest/lowest/individual cell voltages, enabling the full per-cell + GNN
pipeline identical to the Quartz validation path.
"""

from data.loaders.common_schema import (
    SCHEMA_COLS,
    REQUIRED_COLS,
    MIN_SEGMENT_ROWS,
    GAP_THRESH_S,
    SegmentMeta,
    validate_schema,
    split_segments,
    normalise_soc,
    enforce_discharge_negative,
    resample_to_uniform_dt,
    make_schema_df,
)

from data.loaders.pack_cartridge import (
    PackCartridge,
    BMW_I3_60AH,
    BMW_I3_94AH,
    BMW_I3_120AH,
    NISSAN_LEAF_24KWH,
    NISSAN_LEAF_30KWH,
    NISSAN_LEAF_40KWH,
    RENAULT_ZOE_Q210,
    RENAULT_ZOE_Q90_41KWH,
    RENAULT_ZOE_ZE50,
    RENAULT_KANGOO_ZE,
    BAIC_EU5,
    BAIC_EU400,
    BAIC_EU500,
    FORD_FOCUS_EV,
    MITSUBISHI_IMIEV,
    CHEVY_SPARK_EV,
    CHEVY_VOLT_GEN1,
    GENERIC_EV_PACK,
    VED_NAME_TO_CARTRIDGE,
    lookup_ved_cartridge,
    detect_bmw_i3_variant,
)

from data.loaders.ved_loader import VEDLoader
from data.loaders.bmw_i3_loader import BMWI3Loader
from data.loaders.renault_zoe_loader import RenaultZoeLoader
from data.loaders.deng_charging_loader import DengChargingLoader, SOHTrajectory

__all__ = [
    # Schema
    "SCHEMA_COLS", "REQUIRED_COLS", "MIN_SEGMENT_ROWS", "GAP_THRESH_S",
    "SegmentMeta", "validate_schema", "split_segments",
    "normalise_soc", "enforce_discharge_negative", "resample_to_uniform_dt",
    "make_schema_df",
    # Cartridges
    "PackCartridge",
    "BMW_I3_60AH", "BMW_I3_94AH", "BMW_I3_120AH",
    "NISSAN_LEAF_24KWH", "NISSAN_LEAF_30KWH", "NISSAN_LEAF_40KWH",
    "RENAULT_ZOE_Q210", "RENAULT_ZOE_Q90_41KWH", "RENAULT_ZOE_ZE50",
    "RENAULT_KANGOO_ZE",
    "BAIC_EU5", "BAIC_EU400", "BAIC_EU500",
    "FORD_FOCUS_EV", "MITSUBISHI_IMIEV", "CHEVY_SPARK_EV", "CHEVY_VOLT_GEN1",
    "GENERIC_EV_PACK",
    "VED_NAME_TO_CARTRIDGE", "lookup_ved_cartridge", "detect_bmw_i3_variant",
    # Loaders
    "VEDLoader",
    "BMWI3Loader",
    "RenaultZoeLoader",
    "DengChargingLoader",
    "SOHTrajectory",
]
