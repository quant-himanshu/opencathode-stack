"""
BMW i3 Real Driving Cycles loader.

Source
──────
Steinstraeter, M., Buberger, J., Trifonov, D. (2021).
"Discharge data of an electric vehicle and a portable battery pack."
TU München / Mendeley Data, doi:10.17632/tb9m2t28n5.1
Dataset ID: tb9m2t28n5 (BMW i3 60Ah, 96s1p NMC Samsung SDI)

Real dataset structure (inspected June 2024 from data/bmw_i3/)
────────────────────────────────────────────────────────────────
  data/bmw_i3/
    TripA01.csv … TripA32.csv   — summer driving (28 cols)
    TripB01.csv … TripB38.csv   — winter driving  (48 cols, full thermal)
    Overview.xlsx                — trip metadata
    readin.m                     — MATLAB read script

File format: SEMICOLON-delimited, latin-1 encoding, 0.1 s sampling.

Verified column names (from TripB01.csv, 48 columns):
  Time [s]                 ← already seconds from trip start
  Battery Voltage [V]      ← pack voltage (96s)
  Battery Current [A]      ← DISCHARGE-NEGATIVE (already correct sign)
  Battery Temperature [°C] ← cell/pack temperature
  SoC [%]                  ← BMS SoC in percent → loader normalises to [0,1]

Category A vs B:
  TripA: 28 columns (missing extended HVAC/thermal sensors).
         All 4 required battery channels ARE present in TripA.
         No trips need to be skipped for missing battery data.
  TripB: 48 columns (full thermal instrumentation).

Sign convention
───────────────
Battery Current [A] is DISCHARGE-NEGATIVE in this dataset (BMS CAN sign).
  TripB01 row 0: I = -19.06 A at SoC 86.1 % while decelerating → confirmed.
_BMW_DISCHARGE_POSITIVE = False  (no sign flip needed)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

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
    PackCartridge, BMW_I3_60AH,
    detect_bmw_i3_variant,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "bmw_i3"

# Discharge is already NEGATIVE in this dataset (BMS CAN convention).
_BMW_DISCHARGE_POSITIVE: bool = False

# Real column names (verified on TripB01.csv).  Aliases kept for tolerance.
_TIME_COLS  = [
    "Time [s]",           # verified real name (semicolon CSV, latin-1)
    "Time[s]", "Time", "timestamp_s", "time_s", "t[s]",
]
_VOLT_COLS  = [
    "Battery Voltage [V]",   # verified
    "Voltage[V]", "Voltage", "Pack_Voltage[V]", "Battery Voltage[V]",
    "V[V]", "U[V]",
]
_CURR_COLS  = [
    "Battery Current [A]",   # verified
    "Current[A]", "Current", "Pack_Current[A]", "Battery Current[A]",
    "I[A]",
]
_TEMP_COLS  = [
    "Battery Temperature [°C]",   # verified (latin-1 degree sign)
    "Battery Temperature [\xb0C]",
    "Temperature[degC]", "Temperature[°C]", "Temperature",
    "T_cell[degC]", "T[degC]",
]
_SOC_COLS   = [
    "SoC [%]",             # verified real name
    "SOC[%]", "SOC", "State_of_Charge[%]", "BMS_SOC[%]", "soc_pct",
]

# File encoding and separator for this dataset
_CSV_SEP      = ";"
_CSV_ENCODING = "latin-1"


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    low_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low_map:
            return low_map[c.lower()]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Single-file parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_trip_csv(
    path: Path,
    cart: Optional[PackCartridge] = None,
    discharge_positive: bool = _BMW_DISCHARGE_POSITIVE,
) -> Tuple[Optional[pd.DataFrame], Optional[SegmentMeta]]:
    """
    Parse one trip CSV from the BMW i3 RDC dataset.

    Returns (segment_df, meta) in common schema, or (None, None) if the file
    lacks required columns or is too short.
    """
    try:
        df_raw = pd.read_csv(path, sep=_CSV_SEP, encoding=_CSV_ENCODING,
                             low_memory=False)
    except Exception as exc:
        log.warning("BMW i3: cannot read %s: %s", path.name, exc)
        return None, None

    time_col = _find_col(df_raw, _TIME_COLS)
    volt_col = _find_col(df_raw, _VOLT_COLS)
    curr_col = _find_col(df_raw, _CURR_COLS)
    soc_col  = _find_col(df_raw, _SOC_COLS)
    temp_col = _find_col(df_raw, _TEMP_COLS)   # optional

    if any(c is None for c in [time_col, volt_col, curr_col, soc_col]):
        log.warning(
            "BMW i3: %s missing required columns "
            "(time=%s volt=%s curr=%s soc=%s)",
            path.name, time_col, volt_col, curr_col, soc_col,
        )
        return None, None

    df = df_raw.dropna(subset=[time_col, volt_col, curr_col, soc_col]).copy()
    if len(df) < MIN_SEGMENT_ROWS:
        return None, None

    df = df.sort_values(time_col).reset_index(drop=True)

    t_s   = df[time_col].values.astype(np.float64)
    t_s   = t_s - t_s[0]                                 # re-zero
    I_raw = df[curr_col].values.astype(np.float64)
    V_raw = df[volt_col].values.astype(np.float64)
    soc   = normalise_soc(df[soc_col]).values

    # Sign convention
    I_A = enforce_discharge_negative(pd.Series(I_raw), discharge_positive).values

    T_degC = (
        df[temp_col].values.astype(np.float64)
        if temp_col is not None else None
    )

    seg_df = make_schema_df(t_s, I_A, V_raw, T_degC, soc)
    warns  = validate_schema(seg_df, f"BMW_i3/{path.name}")

    # Auto-detect pack variant from observed max voltage
    if cart is None:
        cart = detect_bmw_i3_variant(float(V_raw.max()))

    meta = SegmentMeta(
        dataset="BMW_i3",
        vehicle_id="bmw_i3",
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

class BMWI3Loader:
    """
    Loads the BMW i3 Real Driving Cycles dataset.

    Parameters
    ──────────
    data_dir        Path to the directory containing trip_*.csv files.
    cartridge       Override PackCartridge (None → auto-detect per file).
    discharge_positive  Set to True if source current > 0 during discharge.
    max_trips       Limit number of trip files processed.
    resample_dt_s   If set, resample each segment to uniform timestep [s].
    """

    DATASET_NAME = "BMW_i3"

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        cartridge: Optional[PackCartridge] = None,
        discharge_positive: bool = _BMW_DISCHARGE_POSITIVE,
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
        """Yield (segment_df, meta) for each trip CSV in data_dir.

        Segment ID convention:
          - Single segment per trip  →  segment_id = trip filename stem
                                        e.g. "TripA01", "TripB38"
          - Multiple sub-segments    →  segment_id = "<stem>_s1", "<stem>_s2", …
                                        e.g. "TripA01_s1", "TripA01_s2"
        All yielded segment_ids within a loader run are asserted unique.
        """
        # Trip*.csv only — skip Overview.xlsx, readin.m, metadata.csv
        csvs = sorted(self.data_dir.glob("Trip*.csv"))
        if not csvs:
            csvs = sorted(self.data_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}.\n"
                "Dataset: Steinstraeter et al. (2021) TUM BMW i3, doi:10.17632/tb9m2t28n5.1\n"
                "Extract TripA*.csv and TripB*.csv to: data/bmw_i3/"
            )

        seen_ids: set = set()
        count = 0
        for csv_path in csvs:
            trip_stem = csv_path.stem   # unique per file: "TripA01", "TripB01", …
            seg_df, meta = parse_trip_csv(
                csv_path,
                cart=self.cartridge,
                discharge_positive=self.discharge_positive,
            )
            if seg_df is None:
                continue

            # Collect sub-segments (split on ignition-off gaps within a file)
            sub_segs = list(split_segments(
                seg_df,
                dataset=self.DATASET_NAME,
                vehicle_id="bmw_i3",
                gap_thresh_s=GAP_THRESH_S,
            ))

            if not sub_segs:
                # All sub-splits dropped (too short) → yield raw trip with stem ID
                meta.segment_id = trip_stem
                if self.resample_dt_s:
                    seg_df = resample_to_uniform_dt(seg_df, self.resample_dt_s)
                assert trip_stem not in seen_ids, \
                    f"Duplicate segment_id {trip_stem!r} — filename collision?"
                seen_ids.add(trip_stem)
                yield seg_df, meta

            elif len(sub_segs) == 1:
                # Exactly one segment per trip (normal BMW case — no internal gaps)
                sub_df, sub_meta = sub_segs[0]
                sub_meta.segment_id = trip_stem
                sub_meta.notes.extend([n for n in meta.notes if n not in sub_meta.notes])
                if self.resample_dt_s:
                    sub_df = resample_to_uniform_dt(sub_df, self.resample_dt_s)
                assert trip_stem not in seen_ids, \
                    f"Duplicate segment_id {trip_stem!r} — filename collision?"
                seen_ids.add(trip_stem)
                yield sub_df, sub_meta

            else:
                # Multiple sub-segments: suffix _s1, _s2, …
                for i, (sub_df, sub_meta) in enumerate(sub_segs, start=1):
                    sid = f"{trip_stem}_s{i}"
                    sub_meta.segment_id = sid
                    sub_meta.notes.extend([n for n in meta.notes if n not in sub_meta.notes])
                    if self.resample_dt_s:
                        sub_df = resample_to_uniform_dt(sub_df, self.resample_dt_s)
                    assert sid not in seen_ids, \
                        f"Duplicate segment_id {sid!r} — gap-split produced collision?"
                    seen_ids.add(sid)
                    yield sub_df, sub_meta

            count += 1
            if self.max_trips and count >= self.max_trips:
                break

    def load_all(self) -> Tuple[List[pd.DataFrame], List[SegmentMeta]]:
        """Load all trip segments into memory."""
        segs, metas = [], []
        for seg, meta in self.iter_segments():
            segs.append(seg)
            metas.append(meta)
        return segs, metas


# ─────────────────────────────────────────────────────────────────────────────
# Validate (runs on synthetic data when real data absent)
# ─────────────────────────────────────────────────────────────────────────────

def validate() -> bool:
    print("=" * 60)
    print("VALIDATING: data/loaders/bmw_i3_loader.py")
    print("=" * 60)
    ok = True

    def chk(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  | {detail}" if detail else ""))

    # detect_bmw_i3_variant always returns 60Ah: all i3 variants share
    # the same ~403 V full-charge voltage (96 × 4.20 V); capacity grade
    # cannot be inferred from pack voltage alone.
    from data.loaders.pack_cartridge import BMW_I3_60AH, BMW_I3_94AH, BMW_I3_120AH
    chk("detect_bmw_i3_variant returns 60Ah for any V",
        detect_bmw_i3_variant(400.0) is BMW_I3_60AH and
        detect_bmw_i3_variant(403.0) is BMW_I3_60AH and
        detect_bmw_i3_variant(415.0) is BMW_I3_60AH,
        "all i3 variants share V_max ≈ 403 V (96×4.20V)")

    chk("BMW i3 60Ah n_series=96",   BMW_I3_60AH.n_series == 96)
    chk("BMW i3 60Ah n_parallel=1",  BMW_I3_60AH.n_parallel == 1)
    chk("BMW i3 cell_voltage(355.2)", abs(BMW_I3_60AH.cell_voltage(355.2) - 355.2/96) < 0.01)
    chk("BMW i3 cell_current(-80A)",  abs(BMW_I3_60AH.cell_current(-80.0) - (-80.0)) < 0.01,
        "1p: I_cell = I_pack")

    # Synthetic CSV round-trip — use real column names and format
    syn_pack = _loader_selftest_fixture(n_rows=300, V_nom=355.0, I_discharge_A=-80.0)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tmp_path = Path(f.name)

    # Simulate the real TripB format: semicolon, latin-1, discharge-negative
    syn_bmw = syn_pack.rename(columns={
        "t_s":     "Time [s]",
        "I_A":     "Battery Current [A]",
        "V_V":     "Battery Voltage [V]",
        "T_degC":  "Battery Temperature [°C]",
        "SOC_bms": "SoC [%]",
    })
    syn_bmw["SoC [%]"] *= 100.0    # schema stores [0,1]; real CSV stores [%]
    # Current is already discharge-negative in schema — keep it (no flip)
    syn_bmw.to_csv(tmp_path, index=False, sep=";", encoding="latin-1")

    seg_df, meta = parse_trip_csv(tmp_path, discharge_positive=False)
    tmp_path.unlink(missing_ok=True)

    chk("Synthetic BMW CSV parsed", seg_df is not None)
    if seg_df is not None:
        chk("Schema columns correct", list(seg_df.columns) == ["t_s","I_A","V_V","T_degC","SOC_bms"])
        chk("Discharge negative (already correct sign)", float(seg_df["I_A"].mean()) < 0,
            f"mean={seg_df['I_A'].mean():.2f}")
        chk("SOC in [0,1]", seg_df["SOC_bms"].between(0, 1).all())

    # Real data check
    trip_csvs = list(DATA_DIR.glob("Trip*.csv"))
    if not DATA_DIR.exists() or not trip_csvs:
        print(f"  [SKIP] BMW i3 data not found at {DATA_DIR}")
        print("  Dataset: doi:10.17632/tb9m2t28n5.1 — extract TripA/B*.csv to data/bmw_i3/")
    else:
        loader = BMWI3Loader()   # no limits — load all 70 trips
        segs, metas = loader.load_all()
        n_trip_csvs = len(list(DATA_DIR.glob("Trip*.csv")))
        chk("BMW i3: at least 1 segment", len(segs) > 0, f"n={len(segs)}")
        if segs:
            s0 = segs[0]
            chk("Schema columns correct", list(s0.columns) == ["t_s","I_A","V_V","T_degC","SOC_bms"])
            chk("discharge negative", float(s0["I_A"].mean()) < 0,
                f"mean I={s0['I_A'].mean():.1f}A")
            dt_med = float(np.median(np.diff(s0["t_s"].values)))
            chk("dt ~0.1 s", abs(dt_med - 0.1) < 0.05, f"dt_median={dt_med:.3f}s")
            chk("SOC in [0,1]", s0["SOC_bms"].between(0,1).all())
            # Uniqueness assertion (belt-and-suspenders — iter_segments asserts inline too)
            all_sids = [m.segment_id for m in metas]
            chk("All BMW segment_ids unique",
                len(all_sids) == len(set(all_sids)),
                f"{len(all_sids)} segments, {len(set(all_sids))} unique")
            chk("BMW segment_ids use trip filename (not seg_NNNN)",
                all(not sid.startswith("seg_") for sid in all_sids),
                f"first={all_sids[0]!r}" if all_sids else "no segs")
            # Derive originating trip stems: "TripA01_s1" → "TripA01", "TripA01" → "TripA01"
            import re as _re
            trip_stems_seen = {
                _re.sub(r"_s\d+$", "", sid) for sid in all_sids
            }
            n_skipped = n_trip_csvs - len(trip_stems_seen)
            print(f"\n  Real data sample ({metas[0].segment_id}):")
            print(f"    trip_files={n_trip_csvs}  total_segments={len(segs)}")
            print(f"    n_rows={len(s0)}  dt_med={dt_med:.3f}s  "
                  f"SOC {s0['SOC_bms'].iloc[0]:.3f}→{s0['SOC_bms'].iloc[-1]:.3f}")
            print(f"    V_pack_mean={s0['V_V'].mean():.1f}V  I_mean={s0['I_A'].mean():.1f}A  "
                  f"T_mean={s0['T_degC'].mean():.1f}°C")
            print(s0.head(5).to_string())

    print(f"\nResult: {'ALL PASS' if ok else 'SOME FAILED'}")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    validate()
