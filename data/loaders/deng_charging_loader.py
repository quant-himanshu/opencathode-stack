"""
Deng et al. 20-vehicle real-world EV charging dataset loader.

Source
──────
Deng Z, Xu L, Liu H, Hu X, Duan Z, Xu Y. "Prognostics of battery capacity
based on charging data and data-driven methods for on-road vehicles."
Applied Energy, 2023, 339:120954. doi:10.1016/j.apenergy.2023.120954

Dataset repository
──────────────────
https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles
  Shallow clone into data/deng20/:
    git clone --depth 1 https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles.git data/deng20_tmp/
    unar data/deng20_tmp/#1.rar -o data/deng20/  # repeat for #1..#20

Real file layout (inspected 2026-06 from extracted archives)
─────────────────────────────────────────────────────────────
  data/deng20/
    #1.csv  #2.csv  …  #20.csv   (one file per vehicle, UTF-8 BOM)

  Column names (actual, with unit suffixes):
    Unnamed: 0           — row index (skip)
    record_time          — integer YYYYMMDDHHMMSS (e.g. 20190726200235)
    soc                  — BMS SOC in percent [0–100]
    pack_voltage (V)     — pack terminal voltage [V]
    charge_current (A)   — NEGATIVE during charging (discharge-negative CAN)
    max_cell_voltage (V) — max cell voltage across pack
    min_cell_voltage (V) — min cell voltage across pack
    max_temperature (℃)  — max cell temperature [°C]
    min_temperature (℃)  — min cell temperature [°C]
    available_energy (kw)
    available_capacity (Ah)

Vehicle platform
────────────────
BAIC EU500 (2019 production onwards): CATL NCM, 90s1p, 145 Ah nominal.
  Confirmed: V_pack = 328.2 V at SOC 27 % → V_cell = 3.647 V (NMC OCV ✓)
  32 temperature sensors, ~8 s CAN sampling.

Sign convention
───────────────
charge_current (A) is NEGATIVE during grid charging (discharge-negative CAN).
  Verified: charge_current = -52.2 A while SOC rises from 27 % to 77 %.
  Loader negates: I_schema = -I_raw  → I_schema > 0 during charging ✓

Session splitting
─────────────────
Each vehicle CSV concatenates all charging sessions with gaps > 10 s between
sessions.  Loader splits on time gaps > _SESSION_GAP_S = 30 s.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from natsort import natsorted, ns

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from data.loaders.common_schema import (
    MIN_SEGMENT_ROWS, SegmentMeta,
    make_schema_df, normalise_soc, validate_schema,
    _loader_selftest_fixture,
)
from data.loaders.pack_cartridge import PackCartridge, BAIC_EU500_90S

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "deng20"

# charge_current is NEGATIVE during charging → negate for schema (charging > 0)
_DENG_CHARGING_NEGATIVE: bool = True

# Session gap threshold: gaps larger than this indicate a new charging session
_SESSION_GAP_S: float = 30.0

# SOH: only sessions spanning this SOC window are used for capacity estimation
SOC_WINDOW_LO: float = 0.10
SOC_WINDOW_HI: float = 0.90
MIN_SOC_SPAN:  float = 0.20

Q_NOMINAL_AH: float = 145.0   # BAIC EU500 nominal pack capacity (README)

# Real column names (with unit suffixes, as found in extracted #N.csv files)
_TIME_COL  = "record_time"          # integer YYYYMMDDHHMMSS
_VOLT_COLS = ["pack_voltage (V)", "Pack_Voltage[V]", "Voltage[V]"]
_CURR_COLS = ["charge_current (A)", "Current[A]", "I[A]"]
_SOC_COLS  = ["soc", "SOC[%]", "BMS_SOC[%]"]
_TMAX_COLS = ["max_temperature (℃)", "max_temperature", "Temperature[degC]", "T_cell[degC]"]
_TMIN_COLS = ["min_temperature (℃)", "min_temperature"]


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def _parse_record_time(series: pd.Series) -> pd.Series:
    """Convert integer YYYYMMDDHHMMSS timestamps to pandas Timestamps."""
    return pd.to_datetime(series.astype(str), format="%Y%m%d%H%M%S", errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# Session capacity estimation (Coulomb counting)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_session_capacity(
    t_s: np.ndarray,
    I_A: np.ndarray,
    soc: np.ndarray,
    min_soc_span: float = MIN_SOC_SPAN,
) -> Optional[float]:
    """
    Estimate pack capacity [Ah] as Q_int / ΔSOC — consistent with the authors'
    capacity_extract.py (label_Ca = trapz(current) / delta_SOC).

    I_A must be schema-convention (positive = charging).
    Returns None if ΔSOC < min_soc_span or session is not charging.
    """
    if len(soc) < 2 or soc[-1] <= soc[0]:
        return None  # not a charging session
    delta_soc = float(soc[-1] - soc[0])
    if delta_soc < min_soc_span:
        return None
    Q_int_Ah = float(np.trapezoid(I_A, t_s) / 3600.0)
    if Q_int_Ah <= 0:
        return None
    return Q_int_Ah / delta_soc


# ─────────────────────────────────────────────────────────────────────────────
# Single vehicle file parser  →  yields per-session DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def _iter_sessions_in_file(
    path: Path,
    vehicle_id: str,
    cart: PackCartridge,
) -> Generator[Tuple[pd.DataFrame, SegmentMeta, Optional[float]], None, None]:
    """
    Load one vehicle CSV, split into charging sessions by time gap,
    and yield (seg_df, meta, Q_Ah_estimate) per session.
    """
    try:
        df_raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:
        log.warning("Deng: cannot read %s: %s", path.name, exc)
        return

    volt_col = _find_col(df_raw, _VOLT_COLS)
    curr_col = _find_col(df_raw, _CURR_COLS)
    soc_col  = _find_col(df_raw, _SOC_COLS)
    tmax_col = _find_col(df_raw, _TMAX_COLS)
    tmin_col = _find_col(df_raw, _TMIN_COLS)

    if any(c is None for c in [volt_col, curr_col, soc_col]):
        log.warning("Deng: %s missing required columns (volt=%s curr=%s soc=%s)",
                    path.name, volt_col, curr_col, soc_col)
        return
    if _TIME_COL not in df_raw.columns:
        log.warning("Deng: %s missing record_time column", path.name)
        return

    df_raw = df_raw.dropna(subset=[_TIME_COL, volt_col, curr_col, soc_col]).copy()
    df_raw = df_raw.sort_values(_TIME_COL).reset_index(drop=True)

    # Parse timestamps → elapsed seconds from file start
    timestamps = _parse_record_time(df_raw[_TIME_COL])
    elapsed_s  = (timestamps - timestamps.iloc[0]).dt.total_seconds().values

    # Find session boundaries: gaps > _SESSION_GAP_S
    dt_arr = np.diff(elapsed_s)
    gap_idx = np.where(dt_arr > _SESSION_GAP_S)[0]  # indices of last row before gap
    session_starts = np.concatenate([[0], gap_idx + 1])
    session_ends   = np.concatenate([gap_idx + 1, [len(df_raw)]])

    for sess_num, (i0, i1) in enumerate(zip(session_starts, session_ends)):
        sess = df_raw.iloc[i0:i1].copy().reset_index(drop=True)
        if len(sess) < MIN_SEGMENT_ROWS:
            continue

        t_abs  = elapsed_s[i0:i1]
        t_s    = t_abs - t_abs[0]                        # re-zero per session

        I_raw  = sess[curr_col].values.astype(np.float64)
        I_A    = -I_raw if _DENG_CHARGING_NEGATIVE else I_raw   # negate: charging → positive

        V_raw  = sess[volt_col].values.astype(np.float64)
        soc    = normalise_soc(sess[soc_col]).values

        T_degC: Optional[np.ndarray] = None
        if tmax_col and tmin_col:
            T_degC = ((sess[tmax_col].values.astype(np.float64) +
                       sess[tmin_col].values.astype(np.float64)) / 2.0)
        elif tmax_col:
            T_degC = sess[tmax_col].values.astype(np.float64)

        # Only keep sessions where current is actually positive (charging) or
        # where the session contains a significant charge window
        I_mean = float(np.mean(I_A))
        if I_mean < 0:
            # Discharge-dominated session → skip (dataset is charging-only)
            continue

        seg_df = make_schema_df(t_s, I_A, V_raw, T_degC, soc)
        warns  = validate_schema(seg_df, f"Deng/{vehicle_id}/sess{sess_num:03d}")

        # Date from first timestamp of session
        sess_date = timestamps.iloc[i0].strftime("%Y-%m-%d") if not pd.isna(timestamps.iloc[i0]) else "unknown"

        Q_Ah = estimate_session_capacity(t_s, I_A, soc)

        dt_med = float(np.median(np.diff(t_s))) if len(t_s) > 1 else 8.0
        meta = SegmentMeta(
            dataset="Deng_Charging",
            vehicle_id=vehicle_id,
            segment_id=f"sess{sess_num:03d}_{sess_date}",
            n_rows=len(seg_df),
            dt_s_median=dt_med,
            duration_s=float(t_s[-1]),
            soc_start=float(soc[0]),
            soc_end=float(soc[-1]),
            I_mean_A=float(np.mean(I_A)),
            V_mean_V=float(np.mean(V_raw)),
            T_mean_degC=float(np.mean(T_degC)) if T_degC is not None else float("nan"),
            has_temperature=T_degC is not None,
            notes=warns + [
                f"cart={cart.name}",
                f"Q_session_Ah={Q_Ah:.2f}" if Q_Ah is not None else "Q_session_Ah=N/A",
                f"session_date={sess_date}",
                "avg_cell_mode: V_cell = V_pack/90 (pack-level voltage)",
            ],
        )
        yield seg_df, meta, Q_Ah


def _extract_vehicle_num(name: str) -> int:
    """Extract integer vehicle number from filename '#N.csv' → N."""
    m = re.search(r"#?(\d+)", name)
    return int(m.group(1)) if m else 9999


# ─────────────────────────────────────────────────────────────────────────────
# SOH trajectory
# ─────────────────────────────────────────────────────────────────────────────

class SOHTrajectory:
    EOL_CAPACITY = 0.80

    def __init__(self, vehicle_id: str) -> None:
        self.vehicle_id = vehicle_id
        self.sessions: List[Tuple[int, float]] = []  # (ordinal_day, C_norm)
        self.Q_nominal: Optional[float] = None
        self.q_nominal_flagged: bool = False  # True if >15% from spec
        self.rul_alpha: Optional[float] = None
        self.rul_months: Optional[float] = None

    def set_nominal(self, Q_nom: float, spec_Ah: float = Q_NOMINAL_AH) -> None:
        """Set Q_nominal explicitly from first-month median; flag if >15% from spec."""
        self.Q_nominal = Q_nom
        self.q_nominal_flagged = abs(Q_nom - spec_Ah) / spec_Ah > 0.15

    def add_session(self, date_str: str, Q_est_Ah: float) -> None:
        """Add a session. Q_est_Ah should be Q_int/ΔSOC (authors' capacity estimate).
        If set_nominal() was not called, Q_nominal is set from the first session."""
        try:
            ordinal = (datetime.fromisoformat(date_str).toordinal()
                       if date_str != "unknown" else len(self.sessions))
        except ValueError:
            ordinal = len(self.sessions)

        if self.Q_nominal is None:
            self.Q_nominal = Q_est_Ah if Q_est_Ah > 0 else Q_NOMINAL_AH

        C_norm = (Q_est_Ah / self.Q_nominal
                  if self.Q_nominal and self.Q_nominal > 0 else float("nan"))
        self.sessions.append((ordinal, C_norm))

    def fit_rul(self) -> None:
        if len(self.sessions) < 3:
            return
        self.sessions.sort(key=lambda x: x[0])
        t_arr = np.array([s[0] for s in self.sessions], dtype=float)
        c_arr = np.array([s[1] for s in self.sessions], dtype=float)
        valid = np.isfinite(c_arr)
        if valid.sum() < 3:
            return
        t_v, c_v = t_arr[valid], c_arr[valid]
        t_months  = (t_v - t_v[0]) / 30.44
        A = np.column_stack([np.ones_like(t_months), -t_months])
        result, *_ = np.linalg.lstsq(A, c_v, rcond=None)
        C_0_fit, self.rul_alpha = float(result[0]), float(result[1])
        self.rul_months = (
            float((C_0_fit - self.EOL_CAPACITY) / self.rul_alpha)
            if self.rul_alpha > 1e-6 else float("inf")
        )

    def summary(self) -> Dict[str, object]:
        return {
            "vehicle_id":           self.vehicle_id,
            "n_sessions":           len(self.sessions),
            "Q_nominal_Ah":         self.Q_nominal,
            "q_nominal_flagged":    self.q_nominal_flagged,
            "C_norm_first":         self.sessions[0][1]  if self.sessions else None,
            "C_norm_last":          self.sessions[-1][1] if self.sessions else None,
            "rul_alpha_per_mo":     self.rul_alpha,
            "rul_months_to_eol":    self.rul_months,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loader class
# ─────────────────────────────────────────────────────────────────────────────

class DengChargingLoader:
    """
    Loads the Deng et al. 20-vehicle BAIC EU500 charging dataset.

    File layout: data/deng20/#1.csv … #20.csv (one CSV per vehicle,
    containing all charging sessions concatenated, split by time gaps).
    """

    DATASET_NAME = "Deng_Charging"

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        cartridge: Optional[PackCartridge] = None,
        max_vehicles: Optional[int] = None,
        max_sessions_per_vehicle: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.cartridge = cartridge or BAIC_EU500_90S
        self.max_vehicles = max_vehicles
        self.max_sessions_per_vehicle = max_sessions_per_vehicle

    def _vehicle_files(self) -> List[Tuple[str, Path]]:
        """Return [(vehicle_id, path), …] sorted by vehicle number."""
        csvs = list(self.data_dir.glob("#*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No '#N.csv' files found in {self.data_dir}.\n"
                "Clone and extract from: "
                "https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles\n"
                "  git clone --depth 1 <url> data/deng20_tmp/\n"
                "  for i in $(seq 1 20); do unar data/deng20_tmp/#${i}.rar -o data/deng20/; done"
            )
        csvs = sorted(csvs, key=lambda p: _extract_vehicle_num(p.name))
        if self.max_vehicles:
            csvs = csvs[:self.max_vehicles]
        return [(f"vehicle_{_extract_vehicle_num(p.name):02d}", p) for p in csvs]

    def iter_segments(
        self,
    ) -> Generator[Tuple[pd.DataFrame, SegmentMeta], None, None]:
        for veh_id, path in self._vehicle_files():
            sess_count = 0
            for seg_df, meta, _ in _iter_sessions_in_file(path, veh_id, self.cartridge):
                yield seg_df, meta
                sess_count += 1
                if self.max_sessions_per_vehicle and sess_count >= self.max_sessions_per_vehicle:
                    break

    def soh_trajectories(self) -> Dict[str, SOHTrajectory]:
        """Build per-vehicle SOH trajectories using Q_int/ΔSOC capacity estimation.

        Q_nominal = median of first month's qualifying sessions (ΔSOC > 0.3).
        Vehicles whose Q_nominal deviates >15% from the 145 Ah spec are flagged.
        Consistent with the authors' capacity_extract.py method.
        """
        trajs: Dict[str, SOHTrajectory] = {}
        for veh_id, path in self._vehicle_files():
            # Collect (date_str, delta_soc, Q_est) for all sessions
            all_raw: List[Tuple[str, float, float]] = []
            sess_count = 0
            for _, meta, Q_est in _iter_sessions_in_file(path, veh_id, self.cartridge):
                date_str = next(
                    (n.split("=", 1)[-1] for n in meta.notes if "session_date=" in n),
                    "unknown",
                )
                delta_soc = meta.soc_end - meta.soc_start
                if Q_est is not None and Q_est > 0:
                    all_raw.append((date_str, delta_soc, Q_est))
                sess_count += 1
                if self.max_sessions_per_vehicle and sess_count >= self.max_sessions_per_vehicle:
                    break

            # Determine Q_nominal: median of first month's deep charges (ΔSOC > 0.3)
            Q_nominal: float = Q_NOMINAL_AH
            if all_raw:
                try:
                    first_date = datetime.fromisoformat(all_raw[0][0])
                    first_month_q = [
                        q for d, ds, q in all_raw
                        if ds > 0.3 and d != "unknown"
                        and (datetime.fromisoformat(d) - first_date).days <= 30
                    ]
                except (ValueError, TypeError):
                    first_month_q = [q for _, ds, q in all_raw[:10] if ds > 0.3]
                if first_month_q:
                    Q_nominal = float(np.median(first_month_q))
                else:
                    Q_nominal = Q_NOMINAL_AH

            traj = SOHTrajectory(veh_id)
            traj.set_nominal(Q_nominal)
            if traj.q_nominal_flagged:
                log.warning(
                    "Vehicle %s: Q_nominal=%.1f Ah deviates %.0f%% from spec %.0f Ah — "
                    "flagged as outlier (partial first-month charges?)",
                    veh_id, Q_nominal,
                    abs(Q_nominal - Q_NOMINAL_AH) / Q_NOMINAL_AH * 100,
                    Q_NOMINAL_AH,
                )

            for date_str, delta_soc, Q_est in all_raw:
                if delta_soc > 0.05:  # exclude trivial partial sessions from trajectory
                    traj.add_session(date_str, Q_est)

            traj.fit_rul()
            trajs[veh_id] = traj
        return trajs

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
    print("VALIDATING: data/loaders/deng_charging_loader.py")
    print("=" * 60)
    ok = True

    def chk(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  | {detail}" if detail else ""))

    # SOH math smoke test
    traj = SOHTrajectory("test_veh")
    for month, cap in [(0, 145.0), (6, 141.0), (12, 137.0), (18, 133.0), (24, 129.0), (29, 126.0)]:
        yr = 2019 + (month // 12)
        mo = 1 + (month % 12)
        traj.add_session(f"{yr}-{mo:02d}-01", cap)
    traj.fit_rul()
    chk("SOHTrajectory: 6 sessions stored",  len(traj.sessions) == 6)
    chk("SOHTrajectory: Q_nominal = 145 Ah", traj.Q_nominal == 145.0,
        f"Q_nom={traj.Q_nominal}")
    chk("SOHTrajectory: rul_alpha > 0",      traj.rul_alpha is not None and traj.rul_alpha > 0,
        f"alpha={traj.rul_alpha:.5f}")
    chk("SOHTrajectory: RUL > 0 months",     traj.rul_months is not None and traj.rul_months > 0,
        f"RUL={traj.rul_months:.1f} mo")

    # Capacity estimation from synthetic charging session
    n_pts = 500
    t_syn  = np.arange(n_pts) * 8.0
    soc_syn = np.linspace(0.05, 0.95, n_pts)
    I_syn  = np.full(n_pts, 52.0)     # 52 A (positive = charging in schema)
    Q_est  = estimate_session_capacity(t_syn, I_syn, soc_syn)
    chk("Capacity estimate > 0 on synthetic charging", Q_est is not None and Q_est > 0,
        f"Q={Q_est:.2f} Ah" if Q_est else "None")

    # Pack cartridge
    chk("BAIC_EU500_90S n_series=90",   BAIC_EU500_90S.n_series == 90)
    chk("BAIC_EU500_90S Q_cell=145 Ah", BAIC_EU500_90S.Q_cell_Ah == 145.0)

    # Real data check
    csvs = list(DATA_DIR.glob("#*.csv"))
    if not DATA_DIR.exists() or not csvs:
        print(f"  [SKIP] Deng data not found at {DATA_DIR}")
        print("  git clone --depth 1 https://github.com/BatICM/"
              "battery-charging-data-of-on-road-electric-vehicles.git data/deng20_tmp/")
        print("  for i in $(seq 1 20); do unar data/deng20_tmp/#${i}.rar -o data/deng20/; done")
    else:
        print(f"\n  Found {len(csvs)} vehicle CSV(s) in {DATA_DIR}")
        loader = DengChargingLoader(max_vehicles=3, max_sessions_per_vehicle=5)
        try:
            segs, metas = loader.load_all()
            chk("Deng: at least 1 segment", len(segs) > 0, f"n={len(segs)}")
            if segs:
                s0 = segs[0]
                chk("Schema columns correct",
                    list(s0.columns) == ["t_s","I_A","V_V","T_degC","SOC_bms"])
                chk("Charging sessions: mean I_A > 0", float(s0["I_A"].mean()) > 0,
                    f"mean={s0['I_A'].mean():.2f}A")
                chk("SOC increasing (charging)", float(s0["SOC_bms"].iloc[-1]) >
                    float(s0["SOC_bms"].iloc[0]),
                    f"SOC {s0['SOC_bms'].iloc[0]:.3f}→{s0['SOC_bms'].iloc[-1]:.3f}")
                chk("SOC in [0,1]", s0["SOC_bms"].between(0, 1).all())
                dt_med = float(np.median(np.diff(s0["t_s"].values)))
                chk("dt ~8 s", 5.0 < dt_med < 15.0, f"dt_med={dt_med:.1f}s")

                # Sign invariants on first few qualifying sessions
                import numpy as _np
                qualifying = [(s, m) for s, m in zip(segs, metas)
                              if abs(s["SOC_bms"].iloc[-1] - s["SOC_bms"].iloc[0]) > 0.01]
                n_q_pass = sum(1 for s, _ in qualifying[:10]
                               if _np.trapezoid(s["I_A"].values, s["t_s"].values) > 0 and
                               s["SOC_bms"].iloc[-1] > s["SOC_bms"].iloc[0])
                n_q = min(len(qualifying), 10)
                chk(f"Sign invariant (Q_int>0 & ΔSOC>0) on {n_q} sessions",
                    n_q == 0 or n_q_pass == n_q,
                    f"{n_q_pass}/{n_q} pass")

                # Unique vehicles and session counts
                veh_ids = sorted({m.vehicle_id for m in metas})
                sess_per_veh = {v: sum(1 for m in metas if m.vehicle_id == v) for v in veh_ids}
                dates = [m.segment_id.split("_", 1)[1] for m in metas
                         if "_" in m.segment_id and len(m.segment_id.split("_", 1)[1]) == 10]
                date_span = (f"{min(dates)} → {max(dates)}" if dates else "unknown")

                print(f"\n  Segment sample ({metas[0].vehicle_id}/{metas[0].segment_id}):")
                print(f"    vehicles_loaded={len(veh_ids)}  total_segments={len(segs)}")
                print(f"    sessions_per_veh={sess_per_veh}")
                print(f"    date_span={date_span}")
                print(f"    n_rows={len(s0)}  dt_med={dt_med:.1f}s  "
                      f"I_mean={s0['I_A'].mean():.1f}A  V_mean={s0['V_V'].mean():.1f}V  "
                      f"T_mean={s0['T_degC'].mean():.1f}°C")
                print(s0.head(5).to_string())
        except Exception as exc:
            chk("Deng: load without exception", False, str(exc))

    print(f"\nResult: {'ALL PASS' if ok else 'SOME FAILED'}")
    print("=" * 60)
    return ok


if __name__ == "__main__":
    validate()
