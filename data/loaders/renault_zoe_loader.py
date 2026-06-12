"""
Renault Zoe / Kangoo Z.E. CAN dataset loader.

Source
──────
Renault SA / IFSTTAR. (2018). CAN bus recordings from Renault Zoe and Kangoo
Z.E. in real-world driving conditions. IEEE DataPort,
doi:10.21227/s8dz-cn76.

Dataset structure
──────────────────
  renault_zoe/
    zoe_trip_001.csv
    zoe_trip_002.csv
    ...
    kangoo_trip_001.csv
    kangoo_trip_002.csv
    ...
    vehicle_info.csv   (optional metadata: model, registration date, mileage)

Column name variants observed across Renault CAN exports:
  Time[s]  /  Timestamp[s]  /  t
  VBatt[V]  /  Voltage[V]  /  Pack_V[V]  /  Ubat_V
  IBatt[A]  /  Current[A]  /  Pack_I[A]  /  Ibat_A
  SOC[%]  /  BMS_SOC[%]  /  SoC_Pcent
  T_batt[degC]  /  Temperature[degC]  /  T_avg[degC]

Sign convention
───────────────
Renault CAN bus exports current as discharge-positive.
Loader flips to discharge-negative to match OpenCATHODE schema.

Pack topologies
───────────────
  Zoe Q210 / Q90 (2012–2019):  96s2p, NMC LG Chem, Q_cell ≈ 31.5 Ah
  Zoe Q90 41kWh  (2016–2019):  96s2p, NMC LG Chem, Q_cell ≈ 58.8 Ah
  Zoe ZE50       (2019+):      96s3p, NMC (uncertain), Q_cell ≈ 50 Ah
  Kangoo Z.E. 33kWh:           96s2p, NMC, Q_cell ≈ 47.8 Ah

The loader auto-detects model from filenames ("zoe" / "kangoo") and
selects the appropriate cartridge. Override via `cartridge` parameter.

Average-cell mode
─────────────────
Only total pack voltage is available from CAN.  Loader returns pack-level
V_V and I_A.  validate_generic.py applies:
  V_cell_avg = V_pack / n_series  (96 for all Renault EVs above)
  I_cell     = I_pack / n_parallel
Per-cell features (weakest-cell, GNN) are DISABLED with a logged notice by
validate_generic.py; they are not silently skipped.

Download
────────
  https://ieee-dataport.org/open-access/renault-zoe-driving-cycles
  Extract to: data/renault_zoe/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from data.loaders.common_schema import (
    GAP_THRESH_S, MIN_SEGMENT_ROWS, SegmentMeta,
    enforce_discharge_negative, make_schema_df,
    normalise_soc, resample_to_uniform_dt,
    split_segments, validate_schema,
    _loader_selftest_fixture,
)
from data.loaders.pack_cartridge import (
    PackCartridge, RENAULT_ZOE_Q210, RENAULT_ZOE_Q90_41KWH,
    RENAULT_ZOE_ZE50, RENAULT_KANGOO_ZE, GENERIC_EV_PACK,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "renault_zoe"

# Renault CAN: discharge-positive convention
_RENAULT_DISCHARGE_POSITIVE: bool = True

_TIME_COLS = ["Time[s]", "Timestamp[s]", "t", "time_s", "t[s]", "Time_s"]
_VOLT_COLS = ["VBatt[V]", "Voltage[V]", "Pack_V[V]", "Ubat_V", "U_batt[V]",
              "Battery_Voltage[V]", "V[V]"]
_CURR_COLS = ["IBatt[A]", "Current[A]", "Pack_I[A]", "Ibat_A", "I_batt[A]",
              "Battery_Current[A]", "I[A]"]
_SOC_COLS  = ["SOC[%]", "BMS_SOC[%]", "SoC_Pcent", "State_of_Charge[%]",
              "SOC", "soc_pct"]
_TEMP_COLS = ["T_batt[degC]", "Temperature[degC]", "T_avg[degC]",
              "T[degC]", "Temp_batt[degC]", "Battery_Temp[degC]"]


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    low_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low_map:
            return low_map[c.lower()]
    return None


def _detect_vehicle_cartridge(filename: str) -> PackCartridge:
    """Heuristic: infer Renault variant from file/directory name."""
    name = filename.lower()
    if "kangoo" in name:
        return RENAULT_KANGOO_ZE
    if "ze50" in name or "ze_50" in name or "52kwh" in name:
        return RENAULT_ZOE_ZE50
    if "41kwh" in name or "q90" in name:
        return RENAULT_ZOE_Q90_41KWH
    return RENAULT_ZOE_Q210   # default: most common Zoe in this dataset


# ─────────────────────────────────────────────────────────────────────────────
# Single-file parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_trip_csv(
    path: Path,
    cart: Optional[PackCartridge] = None,
    discharge_positive: bool = _RENAULT_DISCHARGE_POSITIVE,
) -> Tuple[Optional[pd.DataFrame], Optional[SegmentMeta]]:
    """Parse one Renault Zoe/Kangoo trip CSV into common schema."""
    try:
        df_raw = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        log.warning("Renault: cannot read %s: %s", path.name, exc)
        return None, None

    time_col = _find_col(df_raw, _TIME_COLS)
    volt_col = _find_col(df_raw, _VOLT_COLS)
    curr_col = _find_col(df_raw, _CURR_COLS)
    soc_col  = _find_col(df_raw, _SOC_COLS)
    temp_col = _find_col(df_raw, _TEMP_COLS)

    if any(c is None for c in [time_col, volt_col, curr_col, soc_col]):
        log.warning(
            "Renault: %s missing required columns "
            "(time=%s volt=%s curr=%s soc=%s)",
            path.name, time_col, volt_col, curr_col, soc_col,
        )
        return None, None

    df = df_raw.dropna(subset=[time_col, volt_col, curr_col, soc_col]).copy()
    if len(df) < MIN_SEGMENT_ROWS:
        return None, None

    df = df.sort_values(time_col).reset_index(drop=True)

    t_s   = df[time_col].values.astype(np.float64)
    t_s   = t_s - t_s[0]
    I_raw = df[curr_col].values.astype(np.float64)
    V_raw = df[volt_col].values.astype(np.float64)
    soc   = normalise_soc(df[soc_col]).values

    I_A = enforce_discharge_negative(pd.Series(I_raw), discharge_positive).values

    T_degC = (
        df[temp_col].values.astype(np.float64)
        if temp_col is not None else None
    )

    seg_df = make_schema_df(t_s, I_A, V_raw, T_degC, soc)
    warns  = validate_schema(seg_df, f"Renault/{path.name}")

    if cart is None:
        cart = _detect_vehicle_cartridge(path.name)

    meta = SegmentMeta(
        dataset="Renault_Zoe",
        vehicle_id=_detect_vehicle_cartridge(path.name).name.replace(" ", "_"),
        segment_id=path.stem,
        n_rows=len(seg_df),
        dt_s_median=float(np.median(np.diff(t_s))) if len(t_s) > 1 else 0.0,
        duration_s=float(t_s[-1]),
        soc_start=float(soc[0]),
        soc_end=float(soc[-1]),
        I_mean_A=float(np.mean(I_A)),
        V_mean_V=float(np.mean(V_raw)),
        T_mean_degC=float(np.mean(T_degC)) if T_degC is not None else float("nan"),
        has_temperature=T_degC is not None,
        notes=warns + [
            f"cart={cart.name}",
            "avg_cell_mode: V_cell = V_pack/96 (no per-cell telemetry)",
        ],
    )
    return seg_df, meta


# ─────────────────────────────────────────────────────────────────────────────
# Loader class
# ─────────────────────────────────────────────────────────────────────────────

class RenaultZoeLoader:
    """
    Loads the Renault Zoe / Kangoo Z.E. CAN driving dataset.

    Parameters
    ──────────
    data_dir        Directory containing trip CSV files.
    cartridge       Override PackCartridge for all files (None → auto-detect).
    discharge_positive  True if source current > 0 during discharge.
    max_trips       Limit trip files.
    resample_dt_s   Resample to uniform timestep [s] if set.
    """

    DATASET_NAME = "Renault_Zoe"

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        cartridge: Optional[PackCartridge] = None,
        discharge_positive: bool = _RENAULT_DISCHARGE_POSITIVE,
        max_trips: Optional[int] = None,
        resample_dt_s: Optional[float] = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.cartridge = cartridge
        self.discharge_positive = discharge_positive
        self.max_trips = max_trips
        self.resample_dt_s = resample_dt_s

    def iter_segments(
        self,
    ) -> Generator[Tuple[pd.DataFrame, SegmentMeta], None, None]:
        """Yield (segment_df, meta) for each trip CSV."""
        csvs = sorted(self.data_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}.\n"
                "Download: https://ieee-dataport.org/open-access/renault-zoe-driving-cycles\n"
                "Extract to: data/renault_zoe/"
            )

        count = 0
        for csv_path in csvs:
            seg_df, meta = parse_trip_csv(
                csv_path,
                cart=self.cartridge,
                discharge_positive=self.discharge_positive,
            )
            if seg_df is None:
                continue

            sub_count = 0
            for sub_df, sub_meta in split_segments(
                seg_df,
                dataset=self.DATASET_NAME,
                vehicle_id=meta.vehicle_id,
                gap_thresh_s=GAP_THRESH_S,
            ):
                if self.resample_dt_s:
                    sub_df = resample_to_uniform_dt(sub_df, self.resample_dt_s)
                sub_meta.notes.extend([n for n in meta.notes if n not in sub_meta.notes])
                yield sub_df, sub_meta
                sub_count += 1

            if sub_count == 0:
                if self.resample_dt_s:
                    seg_df = resample_to_uniform_dt(seg_df, self.resample_dt_s)
                yield seg_df, meta

            count += 1
            if self.max_trips and count >= self.max_trips:
                break

    def load_all(self) -> Tuple[List[pd.DataFrame], List[SegmentMeta]]:
        segs, metas = [], []
        for seg, meta in self.iter_segments():
            segs.append(seg)
            metas.append(meta)
        return segs, metas


# ─────────────────────────────────────────────────────────────────────────────
# Validate
# ─────────────────────────────────────────────────────────────────────────────

def validate() -> bool:
    print("=" * 60)
    print("VALIDATING: data/loaders/renault_zoe_loader.py")
    print("=" * 60)
    ok = True

    def chk(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  | {detail}" if detail else ""))

    chk("Zoe Q210 n_series=96",  RENAULT_ZOE_Q210.n_series == 96)
    chk("Zoe Q210 n_parallel=2", RENAULT_ZOE_Q210.n_parallel == 2)
    chk("Kangoo n_series=96",    RENAULT_KANGOO_ZE.n_series == 96)
    chk("detect kangoo",
        _detect_vehicle_cartridge("kangoo_trip_01.csv").name == RENAULT_KANGOO_ZE.name)
    chk("detect ze50",
        _detect_vehicle_cartridge("zoe_ze50_trip.csv").name == RENAULT_ZOE_ZE50.name)
    chk("detect default Zoe Q210",
        _detect_vehicle_cartridge("zoe_trip_001.csv").name == RENAULT_ZOE_Q210.name)

    # Synthetic round-trip
    syn = _loader_selftest_fixture(n_rows=200, V_nom=356.0, I_discharge_A=-60.0)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tmp_path = Path(f.name)

    syn_r = syn.rename(columns={
        "t_s": "Time[s]", "I_A": "IBatt[A]", "V_V": "VBatt[V]",
        "T_degC": "T_batt[degC]", "SOC_bms": "SOC[%]",
    })
    syn_r["SOC[%]"] *= 100.0
    syn_r["IBatt[A]"] *= -1.0          # discharge-positive convention
    syn_r.to_csv(tmp_path, index=False)

    seg_df, meta = parse_trip_csv(tmp_path, discharge_positive=True)
    tmp_path.unlink(missing_ok=True)

    chk("Synthetic Renault CSV parsed", seg_df is not None)
    if seg_df is not None:
        chk("Schema columns correct",
            list(seg_df.columns) == ["t_s","I_A","V_V","T_degC","SOC_bms"])
        chk("Discharge negative after flip", float(seg_df["I_A"].mean()) < 0)

    loader = RenaultZoeLoader(max_trips=2)
    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.csv")):
        print(f"  [SKIP] Renault Zoe data not found at {DATA_DIR}")
        print("  Download: https://ieee-dataport.org/open-access/renault-zoe-driving-cycles")
    else:
        segs, metas = loader.load_all()
        chk("Renault: at least 1 segment", len(segs) > 0, f"n={len(segs)}")

    print(f"\nResult: {'ALL PASS' if ok else 'SOME FAILED'}")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    validate()
