"""
VED (Vehicle Energy Dataset) loader.

Source
──────
Oh, S., & Kim, M. (2020). Vehicle Energy Dataset (VED): A Large-Scale Dataset
for Vehicle Energy Consumption Research. GitHub: github.com/gsoh/VED.
Also archived at: IEEE DataPort, doi:10.21227/6jbv-3188.

Real dataset structure (inspected 2024-06 from git repo gsoh/VED master)
──────────────────────────────────────────────────────────────────────────
  data/ved/Data/
    VED_Static_Data_PHEV&EV.xlsx   — 27 PHEV/EV vehicles
        columns: VehId, EngineType, Vehicle Class,
                 Engine Configuration & Displacement, Transmission,
                 Drive Wheels, Generalized_Weight
        EngineType values: 'EV', 'PHEV'
        BEV vehicles (EngineType=='EV'): VehIds 10, 455, 541

    VED_Static_Data_ICE&HEV.xlsx   — 357 ICE/HEV vehicles (not used here)

    VED_DynamicData_Part1.7z  (82.7 MB 7z archive)
    VED_DynamicData_Part2.7z  (93.7 MB 7z archive)
        Extract with: 7z x VED_DynamicData_Part1.7z -o data/ved/
        Produces weekly CSVs: VED_mmddyy_week.csv
        Columns (verified on extracted files):
          DayNum, VehId, Trip, Timestamp(ms),
          Latitude[deg], Longitude[deg], VehicleSpeed[mph],
          MAFRate[g/sec], EngineRPM[rpm], AbsoluteLoad[%],
          OutdoorTemp[degF], FuelEconomy[mpg],
          HVBatt_SOC[%], HVBatt_Volt[V], HVBatt_Curr[A]
          (OBD PIDs present only for vehicles that logged them;
           HV battery columns are NaN for ICE-only vehicles)

Download / extract instructions
────────────────────────────────
  # Download files directly (clone fails on large 7z blobs):
  curl -L https://github.com/gsoh/VED/raw/master/Data/VED_DynamicData_Part1.7z \\
       -o data/ved/Data/VED_DynamicData_Part1.7z
  curl -L https://github.com/gsoh/VED/raw/master/Data/VED_DynamicData_Part2.7z \\
       -o data/ved/Data/VED_DynamicData_Part2.7z
  brew install p7zip   # or: apt-get install p7zip-full
  7z x data/ved/Data/VED_DynamicData_Part1.7z -o data/ved/
  7z x data/ved/Data/VED_DynamicData_Part2.7z -o data/ved/

Sign convention
───────────────
VED HVBatt_Curr[A]: positive during discharge (confirmed Oh 2020 §III-A).
The loader flips to discharge-negative to match the OpenCATHODE schema.

Note on vehicle models
──────────────────────
The static XLSX does not record make/model (no 'Nissan Leaf', 'BMW i3' etc.).
All three BEV VehIds (10, 455, 541) are classified as 'Car ELECTRIC'.
The loader assigns GENERIC_EV_PACK to all VED BEVs.  If you know the vehicle
model from an external source, pass the cartridge to VEDLoader(cartridge=...).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from data.loaders.common_schema import (
    GAP_THRESH_S, MIN_SEGMENT_ROWS, SegmentMeta,
    make_schema_df, normalise_soc, split_segments, validate_schema,
    _loader_selftest_fixture,
)
from data.loaders.pack_cartridge import (
    PackCartridge, GENERIC_EV_PACK, lookup_ved_cartridge,
)

# ─────────────────────────────────────────────────────────────────────────────
# The git repo stores data in a Data/ subdirectory.
# Extracted weekly CSVs land in the parent ved/ directory.
DATA_DIR      = Path(__file__).parent.parent / "ved"
DATA_SUBDIR   = DATA_DIR / "Data"    # location of XLSX and 7z archives

# Column name candidates (verified on extracted VED_mmddyy_week.csv files, 22 columns):
#   DayNum, VehId, Trip, Timestamp(ms), Latitude[deg], Longitude[deg],
#   Vehicle Speed[km/h], MAF[g/sec], Engine RPM[RPM], Absolute Load[%],
#   OAT[DegC], Fuel Rate[L/hr], Air Conditioning Power[kW],
#   Air Conditioning Power[Watts], Heater Power[Watts],
#   HV Battery Current[A], HV Battery SOC[%], HV Battery Voltage[V], ...
_SOC_COLS   = [
    "HV Battery SOC[%]",       # verified real column name (weekly CSV)
    "HVBatt_SOC[%]",
    "BatterySOC[%]",
    "Battery SOC[%]",
    "HVBatt_SoC_Pcent",
]
_VOLT_COLS  = [
    "HV Battery Voltage[V]",   # verified real column name
    "HVBatt_Volt[V]",
    "BatteryVoltage[V]",
    "Battery Voltage[V]",
    "HVBatt_BattVoltage(V)",
]
_CURR_COLS  = [
    "HV Battery Current[A]",   # verified real column name
    "HVBatt_Curr[A]",
    "BatteryCurrent[A]",
    "Battery Current[A]",
    "HVBatt_CurrBattCurr(Amp)",
]
_TIME_COLS  = ["Timestamp(ms)", "Timestamp[ms]", "time_ms"]

# VED HV Battery Current[A]: positive during DISCHARGE (Oh 2020 §III-A).
# Loader flips to discharge-negative to match schema.
_VED_DISCHARGE_POSITIVE = False


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    low_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low_map:
            return low_map[c.lower()]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# n_series inference for unknown VED BEV vehicles
# ─────────────────────────────────────────────────────────────────────────────

def infer_n_series(
    V_pack_series: "pd.Series",
    soc_series: "pd.Series",
    v_cell_lo: float = 2.5,
    v_cell_hi: float = 4.25,
    soc_high_thresh: float = 95.0,
    target_v_cell_full: tuple = (3.9, 4.25),
) -> Tuple[int, str]:
    """
    Infer integer n_series from pack voltage envelope.

    Algorithm:
      1. Find all n in [84, 111] where V_pack_min/n >= v_cell_lo AND
         V_pack_max/n <= v_cell_hi  (hard electrochemical limits).
      2. Among those, prefer n where V_pack at SOC>=95% / n falls in
         target_v_cell_full (3.9–4.25 V), which is expected for NMC near-full.
      3. Within the valid set, pick n=96 if it satisfies both conditions
         (96 is by far the most common EV series count in the VED era fleet).

    Returns (n_series, evidence_string).
    """
    V = V_pack_series.dropna()
    SOC = soc_series.dropna()
    V_max = float(V.max())
    V_min = float(V.min())

    valid_range = [
        n for n in range(84, 112)
        if V_min / n >= v_cell_lo and V_max / n <= v_cell_hi
    ]
    if not valid_range:
        return 96, f"no n in [84,111] satisfies [2.5,4.25]V/cell; defaulting to 96 (V_max={V_max:.1f}V)"

    # Check near-full SOC voltage
    mask_high = SOC >= soc_high_thresh
    valid_full = []
    if mask_high.any():
        # Align V and SOC by index then apply mask
        common_idx = V.index.intersection(SOC[mask_high].index)
        if len(common_idx):
            V_high_mean = float(V.loc[common_idx].mean())
            for n in valid_range:
                vc = V_high_mean / n
                if target_v_cell_full[0] <= vc <= target_v_cell_full[1]:
                    valid_full.append(n)

    candidates = valid_full if valid_full else valid_range
    # Prefer n=96 if it's a candidate (most common EV pack in this fleet era)
    n_best = 96 if 96 in candidates else candidates[len(candidates) // 2]

    vc_max = V_max / n_best
    vc_min = V_min / n_best
    evidence = (
        f"n_series={n_best} inferred from voltage envelope: "
        f"V_pack=[{V_min:.1f},{V_max:.1f}]V → V_cell=[{vc_min:.3f},{vc_max:.3f}]V; "
        f"valid range=[{valid_range[0]}..{valid_range[-1]}]; "
        f"n=96 selected (most common EV architecture in Ann Arbor fleet era)"
    )
    return n_best, evidence


# ─────────────────────────────────────────────────────────────────────────────
# Static data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_static_metadata(
    static_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load VED static metadata from the XLSX file in Data/ subdirectory.

    The real VED repo (gsoh/VED master) stores static data as:
      Data/VED_Static_Data_PHEV&EV.xlsx  — columns: VehId, EngineType, ...
    BEV identification: EngineType == 'EV'
    No make/model column exists; VehicleName is set to 'EV_Car' as placeholder.

    Returns DataFrame with columns [VehId, VehicleName, IsBEV].
    """
    # Try the real XLSX path first, then legacy CSV fallbacks.
    # static_path may be None (use defaults) or a concrete file path.
    candidates = [
        static_path,
        DATA_SUBDIR / "VED_Static_Data_PHEV&EV.xlsx",
        DATA_DIR    / "VED_Static_Data_PHEV&EV.xlsx",
        DATA_DIR    / "VED_Static_Data_ver1.3.csv",
        DATA_DIR    / "VED_Static_Data.csv",
    ]
    path: Optional[Path] = None
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.is_file():          # must be a file, not a directory
            path = p
            break
    if path is None:
        raise FileNotFoundError(
            f"VED static metadata not found in {DATA_SUBDIR} or {DATA_DIR}.\n"
            "Download: curl -L https://github.com/gsoh/VED/raw/master/Data/"
            "VED_Static_Data_PHEV%26EV.xlsx -o data/ved/Data/VED_Static_Data_PHEV&EV.xlsx"
        )

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "-q"])
            df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path, low_memory=False)

    veh_id_col = next((c for c in df.columns if "vehid" in c.lower()), None)
    if veh_id_col is None:
        raise KeyError(f"Cannot find VehId column in {path}. Columns: {list(df.columns)}")

    # Real XLSX: EngineType column ('EV', 'PHEV')
    engine_col = next((c for c in df.columns if "engine" in c.lower() and "type" in c.lower()), None)
    # Legacy CSV: VehicleName column
    name_col = next((c for c in df.columns if "vehicle" in c.lower() and "name" in c.lower()), None)

    df = df.rename(columns={veh_id_col: "VehId"})
    if engine_col:
        df["IsBEV"] = df[engine_col].str.upper().str.strip() == "EV"
        df["VehicleName"] = df[engine_col].str.upper().str.strip().map(
            lambda t: "EV_Car" if t == "EV" else t
        )
    elif name_col:
        df = df.rename(columns={name_col: "VehicleName"})
        df["IsBEV"] = df["VehicleName"].str.upper().str.contains("BEV|^EV$", na=False, regex=True)
    else:
        raise KeyError(
            f"Cannot find EngineType or VehicleName column in {path}. "
            f"Columns: {list(df.columns)}"
        )

    return df[["VehId", "VehicleName", "IsBEV"]].copy()


def bev_veh_ids(static_meta: pd.DataFrame) -> Dict[int, str]:
    """Return dict of {VehId: VehicleName} for BEV-only vehicles."""
    bev = static_meta[static_meta["IsBEV"]]
    return dict(zip(bev["VehId"].astype(int), bev["VehicleName"].astype(str)))


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_dynamic_parts(data_dir: Path) -> pd.DataFrame:
    """
    Load VED dynamic data CSVs and concatenate.

    Search order:
    1. Weekly CSVs extracted from 7z archives (VED_mmddyy_week.csv) in data_dir
    2. Legacy monolithic part files (VED_DynamicData_Part*.csv) in data_dir
    3. Same patterns inside data_dir/Data/ subdirectory
    """
    search_dirs = [data_dir, data_dir / "Data"]
    parts: List[Path] = []

    for sd in search_dirs:
        if not sd.exists():
            continue
        # Weekly extracted CSVs (primary format after 7z extraction)
        weekly = sorted(sd.glob("VED_*week.csv")) + sorted(sd.glob("VED_*_week.csv"))
        if weekly:
            parts = weekly
            break
        # Legacy monolithic CSVs
        mono = sorted(sd.glob("VED_DynamicData_Part*.csv")) + sorted(sd.glob("VED_Dynamic*.csv"))
        if mono:
            parts = mono
            break

    if not parts:
        # Check if 7z archives are present but not yet extracted
        archives = list((data_dir / "Data").glob("*.7z")) if (data_dir / "Data").exists() else []
        archives += list(data_dir.glob("*.7z"))
        if archives:
            raise FileNotFoundError(
                f"VED 7z archives found ({[a.name for a in archives]}) "
                f"but not yet extracted to {data_dir}.\n"
                "Extract with:\n"
                f"  7z x '{archives[0]}' -o {data_dir}\n"
                "  7z x '...Part2.7z' -o {data_dir}"
            )
        raise FileNotFoundError(
            f"No VED dynamic CSV files found in {data_dir} or {data_dir / 'Data'}.\n"
            "Download and extract:\n"
            "  curl -L https://github.com/gsoh/VED/raw/master/Data/VED_DynamicData_Part1.7z"
            f" -o {data_dir / 'Data' / 'VED_DynamicData_Part1.7z'}\n"
            "  7z x VED_DynamicData_Part1.7z -o {data_dir}"
        )

    frames = []
    for p in parts:
        df = pd.read_csv(p, low_memory=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _parse_trip_segment(
    trip_df: pd.DataFrame,
    veh_id: int,
    day_num: int,
    trip_num: int,
    vehicle_name: str,
    cart: PackCartridge,
) -> Optional[Tuple[pd.DataFrame, SegmentMeta]]:
    """
    Convert one (VehId, DayNum, Trip) group into a schema DataFrame.
    Returns None if required columns are missing or segment is too short.
    """
    time_col = _find_col(trip_df, _TIME_COLS)
    soc_col  = _find_col(trip_df, _SOC_COLS)
    volt_col = _find_col(trip_df, _VOLT_COLS)
    curr_col = _find_col(trip_df, _CURR_COLS)

    if any(c is None for c in [time_col, soc_col, volt_col, curr_col]):
        return None

    df = trip_df[[time_col, curr_col, volt_col, soc_col]].copy()
    df = df.dropna(subset=[time_col, soc_col, volt_col, curr_col])
    if len(df) < MIN_SEGMENT_ROWS:
        return None

    df = df.sort_values(time_col).reset_index(drop=True)
    # VED records multiple OBD PIDs at the same millisecond → deduplicate
    df = df.drop_duplicates(subset=[time_col]).reset_index(drop=True)
    if len(df) < MIN_SEGMENT_ROWS:
        return None

    # t_s: convert ms to seconds, re-zero
    t_s = (df[time_col].values.astype(np.float64)) / 1000.0
    t_s = t_s - t_s[0]

    # Current: flip discharge-positive → discharge-negative
    I_pack = df[curr_col].values.astype(np.float64)
    if _VED_DISCHARGE_POSITIVE:
        I_pack = -I_pack

    # Voltage: pack-level
    V_pack = df[volt_col].values.astype(np.float64)

    # SOC: normalise to [0,1]
    soc = normalise_soc(df[soc_col]).values

    # Infer n_series from this trip's voltage envelope (no model name known)
    n_series, n_evidence = infer_n_series(
        pd.Series(V_pack), pd.Series(soc * 100.0)  # soc is [0,1]; infer_n_series expects %
    )
    # Build a per-vehicle cartridge with inferred topology
    from data.loaders.pack_cartridge import PackCartridge as _PC
    inferred_cart = _PC(
        name=f"VED_VehId{veh_id}_inferred_{n_series}s1p",
        n_series=n_series,
        n_parallel=1,       # single string assumed (cannot determine from pack data)
        chemistry="NMC",    # most common EV chemistry in 2017 Ann Arbor fleet
        Q_cell_Ah=40.0,     # placeholder; irrelevant for V_cell_avg = V_pack/n_series
        R_ohm_cell=0.015,
        V_nom_pack=float(np.mean(V_pack)),
        source="n_series inferred from voltage envelope (see notes)",
        topology_uncertain=True,
    )

    seg_df = make_schema_df(t_s, I_pack, V_pack, None, soc)

    warns = validate_schema(seg_df, f"VED VehId={veh_id} day={day_num} trip={trip_num}")
    meta = SegmentMeta(
        dataset="VED",
        vehicle_id=f"VehId_{veh_id:04d}",
        segment_id=f"day{day_num:04d}_trip{trip_num:02d}",
        n_rows=len(seg_df),
        dt_s_median=float(np.median(np.diff(t_s))) if len(t_s) > 1 else 0.0,
        duration_s=float(t_s[-1]),
        soc_start=float(soc[0]),
        soc_end=float(soc[-1]),
        I_mean_A=float(np.mean(I_pack)),
        V_mean_V=float(np.mean(V_pack)),
        T_mean_degC=float("nan"),
        has_temperature=False,
        notes=warns + [
            f"vehicle={vehicle_name}",
            f"cart={inferred_cart.name}",
            f"n_series_inference: {n_evidence}",
            "T_degC not available in VED (no BEV battery-temp OBD PID)",
        ],
    )
    return seg_df, meta


class VEDLoader:
    """
    Loads the VED dataset, filters to BEV vehicles, and yields per-trip segments.

    Parameters
    ──────────
    data_dir        Path to the directory containing VED CSV files.
    max_veh         Limit number of vehicles processed (None = all).
    max_trips_per_veh  Limit trips per vehicle.
    resample_dt_s   If set, resample all segments to this uniform timestep.
    """

    DATASET_NAME = "VED"

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        max_veh: Optional[int] = None,
        max_trips_per_veh: Optional[int] = None,
        resample_dt_s: Optional[float] = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.max_veh = max_veh
        self.max_trips_per_veh = max_trips_per_veh
        self.resample_dt_s = resample_dt_s
        self._static_meta: Optional[pd.DataFrame] = None
        self._bev_ids: Optional[Dict[int, str]] = None

    def _ensure_metadata(self) -> None:
        if self._static_meta is None:
            # Pass None to use DATA_SUBDIR defaults; the loader will look in
            # data_dir/Data/ first, then data_dir/ for the XLSX file.
            self._static_meta = load_static_metadata(None)
            self._bev_ids = bev_veh_ids(self._static_meta)

    def iter_segments(
        self,
    ) -> Generator[Tuple[pd.DataFrame, SegmentMeta], None, None]:
        """
        Yield (segment_df, meta) for every BEV trip in the dataset.
        Skips trips where required battery columns are absent.
        """
        self._ensure_metadata()
        assert self._bev_ids is not None

        dyn = _load_dynamic_parts(self.data_dir)

        veh_id_col = next((c for c in dyn.columns if "vehid" in c.lower()), "VehId")
        day_col    = next((c for c in dyn.columns if "daynum" in c.lower()), "DayNum")
        trip_col   = next((c for c in dyn.columns if "trip" == c.lower()), "Trip")

        bev_mask = dyn[veh_id_col].isin(self._bev_ids.keys())
        dyn_bev = dyn[bev_mask].copy()

        veh_count = 0
        for veh_id, veh_df in dyn_bev.groupby(veh_id_col):
            veh_id = int(veh_id)
            if veh_id not in self._bev_ids:
                continue
            vehicle_name = self._bev_ids[veh_id]
            cart = lookup_ved_cartridge(vehicle_name)

            trip_count = 0
            for (day_num, trip_num), trip_df in veh_df.groupby([day_col, trip_col]):
                result = _parse_trip_segment(
                    trip_df, veh_id, int(day_num), int(trip_num),
                    vehicle_name, cart,
                )
                if result is None:
                    continue

                seg_df, meta = result
                if self.resample_dt_s:
                    from data.loaders.common_schema import resample_to_uniform_dt
                    seg_df = resample_to_uniform_dt(seg_df, self.resample_dt_s)

                yield seg_df, meta
                trip_count += 1
                if self.max_trips_per_veh and trip_count >= self.max_trips_per_veh:
                    break

            veh_count += 1
            if self.max_veh and veh_count >= self.max_veh:
                break

    def load_all(self) -> Tuple[List[pd.DataFrame], List[SegmentMeta]]:
        """Load all segments into memory. Use iter_segments() for large datasets."""
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
    print("VALIDATING: data/loaders/ved_loader.py")
    print("=" * 60)
    ok = True

    def chk(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        if not cond: ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  | {detail}" if detail else ""))

    # Test schema helpers with synthetic data
    syn = _loader_selftest_fixture(n_rows=500, V_nom=355.0, I_discharge_A=-80.0)
    chk("Synthetic segment columns", list(syn.columns) == ["t_s","I_A","V_V","T_degC","SOC_bms"])
    chk("Synthetic t_s starts at 0", float(syn["t_s"].iloc[0]) == 0.0)
    chk("Synthetic I_A discharge < 0", float(syn["I_A"].mean()) < 0, f"mean={syn['I_A'].mean():.1f}")
    chk("Synthetic SOC in [0,1]", syn["SOC_bms"].between(0,1).all())

    warns = validate_schema(syn, "synthetic")
    chk("Schema validation passes on synthetic", len(warns) == 0, str(warns))

    # Test pack cartridge scaling
    from data.loaders.pack_cartridge import BMW_I3_60AH
    chk("BMW i3 cell_voltage(350V)", abs(BMW_I3_60AH.cell_voltage(350.0) - 350/96) < 0.01)
    chk("BMW i3 cell_current(-80A)", abs(BMW_I3_60AH.cell_current(-80.0) - (-80.0)) < 0.01)

    # Test VED lookup
    cart = lookup_ved_cartridge("Nissan Leaf BEV")
    chk("VED name lookup Leaf", cart.n_series == 96 and cart.n_parallel == 2)

    # Real data check (graceful if absent)
    xlsx_ok = (DATA_SUBDIR / "VED_Static_Data_PHEV&EV.xlsx").exists()
    if not xlsx_ok:
        print(f"  [SKIP] VED static XLSX not found at {DATA_SUBDIR}")
        print("  curl -L 'https://github.com/gsoh/VED/raw/master/Data/VED_Static_Data_PHEV%26EV.xlsx'"
              f" -o '{DATA_SUBDIR}/VED_Static_Data_PHEV&EV.xlsx'")
    else:
        try:
            loader2 = VEDLoader()   # no limits — load all trips from all 3 BEVs
            segs, metas = loader2.load_all()
            n_bev_ids = len({m.vehicle_id for m in metas})
            chk("VED: at least 1 BEV segment loaded", len(segs) > 0, f"n={len(segs)}")
            if segs:
                s0 = segs[0]
                chk("VED: schema columns correct",
                    list(s0.columns) == ["t_s","I_A","V_V","T_degC","SOC_bms"])
                has_discharge_events = any(s["I_A"].min() < -1.0 for s in segs)
                n_neg_mean = sum(1 for s in segs if s["I_A"].mean() < 0)
                chk("VED: discharge events present (some I_A<-1A)", has_discharge_events,
                    f"{n_neg_mean}/{len(segs)} segs net-discharge; "
                    f"min I_A={min(s['I_A'].min() for s in segs):.1f}A")
                chk("VED: SOC in [0,1]", s0["SOC_bms"].between(0,1).all())
                dt_med = float(np.median(np.diff(s0["t_s"].values)))
                chk("VED: dt_median is seconds-level", 0.5 < dt_med < 5.0,
                    f"dt_med={dt_med:.2f}s")
                note_cart = next((n for n in metas[0].notes if "n_series_inference" in n), "")
                print(f"\n  Segment sample ({metas[0].vehicle_id}/{metas[0].segment_id}):")
                print(f"    bev_vehicles={n_bev_ids}  total_segments={len(segs)}  "
                      f"n_rows_first={len(s0)}  dt_med={dt_med:.2f}s")
                print(f"    SOC {s0['SOC_bms'].iloc[0]:.3f}→{s0['SOC_bms'].iloc[-1]:.3f}  "
                      f"V_mean={s0['V_V'].mean():.1f}V  I_mean={s0['I_A'].mean():.1f}A")
                if note_cart:
                    print(f"    {note_cart}")
                print(s0.head(5).to_string())
        except FileNotFoundError as exc:
            msg = str(exc)
            if "not yet extracted" in msg:
                print(f"  [SKIP] VED: 7z archives present but not extracted.")
                print("  Run: 7z x data/ved/Data/VED_DynamicData_Part1.7z -o data/ved/")
            else:
                print(f"  [SKIP] VED dynamic data not found: {exc}")

    print(f"\nResult: {'ALL PASS' if ok else 'SOME FAILED'}")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    validate()
