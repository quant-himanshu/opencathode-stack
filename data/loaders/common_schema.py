"""
Common output schema for all automotive field-dataset loaders.

Output schema  (all column names are canonical; no aliases after load)
─────────────────────────────────────────────────────────────────────
  t_s      float64  elapsed time [s] from segment start (starts at 0)
  I_A      float64  pack current [A],  discharge < 0 / charge > 0
  V_V      float64  pack terminal voltage [V]
  T_degC   float64  pack/cell temperature [°C]  — NaN when unavailable
  SOC_bms  float64  BMS state of charge [0..1]  — normalized from %

Design choices
─────────────────────────────────────────────────────────────────────
• "discharge negative" matches the Quartz sign convention used throughout
  this repo (validate_quartz.py: I_SIGN = -1.0, I_dfn = −I_raw × scale).
• All segments are gap-free: a monotonically increasing t_s column with
  uniform or near-uniform dt.  Large gaps (> GAP_THRESH_S) split segments.
• Segment-level metadata is returned as a separate dict, not stuffed into
  the DataFrame so that numpy/scipy operations on the DataFrame are clean.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Schema constants
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_COLS = ["t_s", "I_A", "V_V", "T_degC", "SOC_bms"]
REQUIRED_COLS = ["t_s", "I_A", "V_V", "SOC_bms"]   # T_degC may be NaN

MIN_SEGMENT_ROWS: int = 30       # discard segments shorter than this
GAP_THRESH_S: float = 300.0      # > 5 min gap → new segment
SOC_VALID_RANGE: Tuple[float, float] = (0.0, 1.0)
VOLTAGE_SANITY: Tuple[float, float] = (10.0, 900.0)   # pack-level [V]
CURRENT_SANITY: Tuple[float, float] = (-3000.0, 3000.0)


# ─────────────────────────────────────────────────────────────────────────────
# Segment metadata
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SegmentMeta:
    """Per-segment provenance and statistics."""
    dataset: str
    vehicle_id: str
    segment_id: str              # e.g., "trip_001", "session_2023-04-12"
    n_rows: int
    dt_s_median: float           # median timestep [s]
    duration_s: float
    soc_start: float             # SOC_bms at first row [0..1]
    soc_end: float               # SOC_bms at last row [0..1]
    I_mean_A: float              # mean current (neg = discharge) [A]
    V_mean_V: float
    T_mean_degC: float           # NaN if no temperature
    has_temperature: bool
    notes: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"SegmentMeta({self.dataset}/{self.vehicle_id}/{self.segment_id}"
                f"  n={self.n_rows}  dt={self.dt_s_median:.1f}s"
                f"  SOC {self.soc_start:.2f}→{self.soc_end:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame, source: str = "") -> List[str]:
    """Check schema compliance. Returns list of warning strings (empty = OK)."""
    warnings_out: List[str] = []
    for col in REQUIRED_COLS:
        if col not in df.columns:
            warnings_out.append(f"{source}: missing required column '{col}'")
    if "SOC_bms" in df.columns:
        bad = ((df["SOC_bms"] < -0.01) | (df["SOC_bms"] > 1.01)).sum()
        if bad > 0:
            warnings_out.append(f"{source}: {bad} rows with SOC_bms outside [0,1]")
    if "V_V" in df.columns:
        lo, hi = VOLTAGE_SANITY
        bad = ((df["V_V"] < lo) | (df["V_V"] > hi)).sum()
        if bad > 0:
            warnings_out.append(f"{source}: {bad} rows with V_V outside [{lo},{hi}]V")
    return warnings_out


def _build_segment_meta(
    df: pd.DataFrame,
    dataset: str,
    vehicle_id: str,
    segment_id: str,
    notes: Optional[List[str]] = None,
) -> SegmentMeta:
    """Compute SegmentMeta from a validated segment DataFrame."""
    dt_arr = np.diff(df["t_s"].values)
    return SegmentMeta(
        dataset=dataset,
        vehicle_id=vehicle_id,
        segment_id=segment_id,
        n_rows=len(df),
        dt_s_median=float(np.median(dt_arr)) if len(dt_arr) > 0 else float("nan"),
        duration_s=float(df["t_s"].iloc[-1] - df["t_s"].iloc[0]),
        soc_start=float(df["SOC_bms"].iloc[0]),
        soc_end=float(df["SOC_bms"].iloc[-1]),
        I_mean_A=float(df["I_A"].mean()),
        V_mean_V=float(df["V_V"].mean()),
        T_mean_degC=float(df["T_degC"].mean()) if "T_degC" in df.columns else float("nan"),
        has_temperature="T_degC" in df.columns and df["T_degC"].notna().any(),
        notes=notes or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Segmenter
# ─────────────────────────────────────────────────────────────────────────────

def split_segments(
    df: pd.DataFrame,
    dataset: str,
    vehicle_id: str,
    gap_thresh_s: float = GAP_THRESH_S,
    min_rows: int = MIN_SEGMENT_ROWS,
) -> Iterator[Tuple[pd.DataFrame, SegmentMeta]]:
    """
    Split a DataFrame into gap-free segments and reset t_s to 0.

    Yields (segment_df, meta) pairs.  Segments shorter than min_rows
    are silently dropped.
    """
    if df.empty:
        return

    t = df["t_s"].values.copy()
    gaps = np.where(np.diff(t) > gap_thresh_s)[0]
    split_points = np.concatenate([[0], gaps + 1, [len(df)]])

    seg_num = 0
    for i in range(len(split_points) - 1):
        sl = slice(split_points[i], split_points[i + 1])
        seg = df.iloc[sl].copy().reset_index(drop=True)
        if len(seg) < min_rows:
            continue

        # Re-zero t_s from segment start
        seg["t_s"] = seg["t_s"] - seg["t_s"].iloc[0]

        seg_id = f"seg_{seg_num:04d}"
        meta = _build_segment_meta(seg, dataset, vehicle_id, seg_id)
        seg_num += 1
        yield seg, meta


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise_soc(soc_raw: pd.Series) -> pd.Series:
    """Convert SOC from percent (0–100) to fraction (0–1) if needed."""
    if soc_raw.dropna().max() > 1.1:
        return soc_raw / 100.0
    return soc_raw.astype(float)


def enforce_discharge_negative(I: pd.Series, hint_discharge_positive: bool) -> pd.Series:
    """
    Flip current sign if the source dataset uses discharge-positive convention.
    hint_discharge_positive: True means source has I>0 during discharge.
    """
    if hint_discharge_positive:
        return -I.astype(float)
    return I.astype(float)


def resample_to_uniform_dt(
    df: pd.DataFrame,
    dt_target_s: float,
    method: str = "linear",
) -> pd.DataFrame:
    """
    Resample an irregular-sampled segment to uniform dt_target_s spacing.
    Uses linear interpolation on all numeric columns.
    """
    t_old = df["t_s"].values
    t_new = np.arange(t_old[0], t_old[-1], dt_target_s)
    out: Dict[str, np.ndarray] = {"t_s": t_new}
    for col in df.columns:
        if col == "t_s":
            continue
        vals = df[col].values.astype(float)
        out[col] = np.interp(t_new, t_old, vals)
    return pd.DataFrame(out)


def assert_discharge_negative_consistency(
    t_s: np.ndarray,
    I_A: np.ndarray,
    SOC_bms: np.ndarray,
    source: str = "",
    min_dsoc: float = 0.05,
    min_ah: float = 0.1,
) -> None:
    """
    Permanent sign-convention assertion (2026-07-20 sign-bug postmortem —
    see docs/SIGN_BUG_POSTMORTEM.md).

    The OpenCATHODE schema is DISCHARGE-NEGATIVE (I_A < 0 while the battery
    discharges). Under that convention the net integral of current must
    OPPOSE the net SOC drop: SOC falls ⇒ ∫I dt < 0, SOC rises ⇒ ∫I dt > 0.
    A segment with a material net SOC change (≥ min_dsoc) and material net
    charge throughput (≥ min_ah Ah) where sign(∫I dt) EQUALS
    sign(SOC_start − SOC_end) is discharge-positive data — the exact defect
    that silently inverted the CALCE/UMich benchmark loaders. Fail loudly.

    Thresholds are deliberately conservative so measurement noise, regen
    braking, and BMS quantization can never trip a correctly-signed loader:
    a ≥5 pp net SOC drop with ≥0.1 Ah net throughput cannot have the wrong
    integral sign unless the current sign itself is wrong.
    """
    t = np.asarray(t_s, dtype=np.float64)
    I = np.asarray(I_A, dtype=np.float64)
    soc = np.asarray(SOC_bms, dtype=np.float64)
    if len(t) < 2:
        return
    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0
    ah = float(np.sum(I * dt)) / 3600.0
    dsoc = float(soc[0] - soc[-1])   # > 0 means net discharge
    if abs(dsoc) >= min_dsoc and abs(ah) >= min_ah \
            and np.sign(ah) == np.sign(dsoc):
        raise ValueError(
            f"SIGN-CONVENTION VIOLATION ({source or 'unknown source'}): "
            f"net ΔSOC drop {dsoc*100:+.1f} pp with ∫I dt = {ah:+.2f} Ah — "
            f"sign(∫I dt) must OPPOSE sign(ΔSOC drop) in the "
            f"discharge-negative schema. This loader is emitting "
            f"discharge-positive current; flip it with "
            f"enforce_discharge_negative(). See docs/SIGN_BUG_POSTMORTEM.md."
        )


def make_schema_df(
    t_s: np.ndarray,
    I_A: np.ndarray,
    V_V: np.ndarray,
    T_degC: Optional[np.ndarray],
    SOC_bms: np.ndarray,
    source: str = "",
) -> pd.DataFrame:
    """Construct a schema-compliant DataFrame from arrays.

    Runs the permanent discharge-negative sign assertion on the arrays —
    every dataset load in this project passes through here."""
    assert_discharge_negative_consistency(t_s, I_A, SOC_bms, source=source)
    n = len(t_s)
    df = pd.DataFrame({
        "t_s":     t_s.astype(np.float64),
        "I_A":     I_A.astype(np.float64),
        "V_V":     V_V.astype(np.float64),
        "T_degC":  T_degC.astype(np.float64) if T_degC is not None
                   else np.full(n, np.nan),
        "SOC_bms": SOC_bms.astype(np.float64),
    })
    return df[SCHEMA_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic segment generator — LOADER SELF-TESTS ONLY
# ─────────────────────────────────────────────────────────────────────────────
# Private: call only from validate() functions inside loader files.
# MUST NOT be imported or called from validate_generic.py.
# Results derived from this fixture must never appear in any report.

def _loader_selftest_fixture(
    n_rows: int = 1800,
    dt_s: float = 1.0,
    V_nom: float = 355.0,
    I_discharge_A: float = -80.0,
    soc_init: float = 0.80,
    Q_pack_Ah: float = 60.0,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic pack-level driving segment that obeys Coulomb counting
    and a simple NMC OCV curve.  Used for loader unit tests.
    """
    rng = np.random.default_rng(rng_seed)
    t = np.arange(n_rows, dtype=np.float64) * dt_s

    # Approximate current: mix of constant discharge + noise
    I = np.full(n_rows, I_discharge_A) + rng.normal(0, 5, n_rows)
    # Occasional brief regen pulses
    regen_mask = rng.random(n_rows) < 0.08
    I[regen_mask] = abs(I_discharge_A) * 0.3

    # Coulomb counting SOC
    soc = np.empty(n_rows)
    soc[0] = soc_init
    for i in range(1, n_rows):
        soc[i] = soc[i-1] - I[i-1] * dt_s / (3600.0 * Q_pack_Ah)
    soc = np.clip(soc, 0.0, 1.0)

    # Simple NMC OCV: V_oc ≈ V_nom * (0.8 + 0.2*SOC)
    V_oc = V_nom * (0.80 + 0.20 * soc)
    R_pack = 0.05  # Ω rough pack resistance
    V = V_oc + I * R_pack + rng.normal(0, 0.5, n_rows)

    T = 25.0 + 5.0 * np.sin(2 * np.pi * t / 1800.0) + rng.normal(0, 0.5, n_rows)

    return make_schema_df(t, I, V, T, soc)
