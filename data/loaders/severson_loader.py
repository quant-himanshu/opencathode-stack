#!/usr/bin/env python3
"""
data/loaders/severson_loader.py  —  Severson et al. (2019) LFP dataset loader

Loads 124 LFP/graphite cells (A123 APR18650M1A, 1.1 Ah nominal) from three
MATLAB v7.3 HDF5 batch files.  Returns per-cell degradation dicts compatible
with the Module 4 cross-cell predictor.

Dataset: Severson et al. (2019) Nature Energy 4:383–391
Source:  https://data.matr.io/1/  (requires registration)
Files:   data/severson/2017-05-12_batchdata_updated_struct_errorcorrect.mat  (Batch 1, 46 cells)
         data/severson/2017-06-30_batchdata_updated_struct_errorcorrect.mat  (Batch 2, 48 cells)
         data/severson/2018-04-12_batchdata_updated_struct_errorcorrect.mat  (Batch 3, 46 cells)

Output per cell:
  cell_id     : str      e.g. "b1c0"
  batch        : int      1, 2, or 3
  cycle_life   : int      cycles to EOL (80% capacity) — from dataset metadata
  soh          : ndarray  shape (n_valid,) — SOH[k] = QD[k] / QD[0] (first non-zero)
  D            : ndarray  shape (n_valid,) — D[k] = k (unit-cycle damage, 1-indexed)
  dqv_feature  : float    log10(mean_V[var_k(Qdlin_k − Qdlin_ref)]) for k=ref+1..n_early
                          Severson-style early-cycle variance on the pre-computed
                          1000-pt fixed V-grid [2.0–3.5 V].  NaN if insufficient data.

Key schema facts (verified):
  - summary['QDischarge']: direct float64 array, shape (1, n_cycles) — NOT HDF5 refs
  - cycles['Qdlin'][k, 0]: HDF5 object ref → flatten() → shape (1000,)
  - Batch 1 only: QD[0]=0.0 and Qdlin[0] len=2 (corrupt init cycle) — auto-skipped
  - cross-batch cells (b2c7,8,9,15,16) excluded; their Batch 1 counterparts run to
    shorter cycle counts — stitching not implemented (4% of cells, noted in report)

Honesty note:
  The dqv_feature here uses pre-computed Qdlin on a fixed 1000-pt LFP grid [2.0–3.5 V].
  This is NOT the same V-grid as the NASA feature (V_GRID=[2.75–4.15 V] NMC).
  The two features are not directly comparable.  When using Severson cells alongside
  NASA cells in Module 4, a unified V-grid or separate feature maps are required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    raise ImportError("h5py is required: pip install h5py")

ROOT        = Path(__file__).resolve().parent.parent.parent
SEVERSON_DIR = ROOT / "data" / "severson"

BATCH_FILES = [
    (1, "2017-05-12_batchdata_updated_struct_errorcorrect.mat"),
    (2, "2017-06-30_batchdata_updated_struct_errorcorrect.mat"),
    (3, "2018-04-12_batchdata_updated_struct_errorcorrect.mat"),
]

# Cells to exclude (corrupt, outlier, or incomplete — from original paper code)
EXCLUDE_CELLS: Dict[int, set] = {
    1: {"b1c8",  "b1c10", "b1c12", "b1c13", "b1c22"},
    2: {"b2c7",  "b2c8",  "b2c9",  "b2c15", "b2c16"},  # cross-batch: see module docstring
    3: {"b3c2",  "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"},
}

# ΔQ(V) feature window: use cycles with index ref_cycle+1 .. n_early (0-indexed)
N_FEATURE_CYCLES = 20
FEATURE_REF_IDX  = 1   # 0-indexed; cycle 2 in 1-indexed terms


# ── HDF5 access helpers ───────────────────────────────────────────────────────

def _deref(f: "h5py.File", ref) -> "h5py.Group | np.ndarray":
    """Dereference an h5py.Reference object."""
    return f[ref]


def _get_qd(f: "h5py.File", batch: "h5py.Group", i: int) -> np.ndarray:
    """
    Return the QDischarge array for cell i.
    summary['QDischarge'] is a direct float64 array of shape (1, n_cycles) — no refs.
    """
    summ = _deref(f, batch["summary"][i, 0])
    return summ["QDischarge"][0, :].astype(np.float64)


def _get_qdlin(f: "h5py.File", batch: "h5py.Group", i: int,
               cycle_idx: int) -> Optional[np.ndarray]:
    """
    Return Qdlin for cell i, cycle cycle_idx (0-indexed).
    Returns None if the array is not 1000 points (corrupt/init cycle).
    """
    cyc = _deref(f, batch["cycles"][i, 0])
    arr = _deref(f, cyc["Qdlin"][cycle_idx, 0])[()].flatten()
    if len(arr) != 1000:
        return None
    return arr.astype(np.float64)


def _get_cycle_life(f: "h5py.File", batch: "h5py.Group", i: int) -> int:
    return int(_deref(f, batch["cycle_life"][i, 0])[()].flat[0])


def _get_n_stored(f: "h5py.File", batch: "h5py.Group", i: int) -> int:
    """Number of cycle entries stored in the cycles struct."""
    cyc = _deref(f, batch["cycles"][i, 0])
    return cyc["Qdlin"].shape[0]


# ── SOH computation ───────────────────────────────────────────────────────────

def _compute_soh(qd_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop entries where QD==0 (init/corrupt cycles), normalise by first non-zero.
    Returns (soh, valid_mask) where soh is relative to the first non-zero QD.

    The Batch 1 anomaly: QD[0]=0.0 (the cycle labelled '1' in the dataset is a
    charge/formation cycle with no measured discharge capacity).  We skip all
    zero entries regardless of batch, using a tiny ε to avoid floating-point 0.
    """
    valid = qd_raw > 1e-6
    qd_valid = qd_raw[valid]
    if len(qd_valid) == 0:
        return np.array([]), np.zeros(len(qd_raw), dtype=bool)
    soh = qd_valid / qd_valid[0]
    return soh, valid


# ── ΔQ(V) feature (Severson-native, using pre-computed Qdlin) ─────────────────

def _dqv_feature_severson(
    f: "h5py.File",
    batch: "h5py.Group",
    cell_i: int,
    n_stored: int,
    n_early: int = N_FEATURE_CYCLES,
    ref_idx: int = FEATURE_REF_IDX,
) -> float:
    """
    Compute log10(mean_V[var_k(Qdlin_k − Qdlin_ref)]) for k=ref_idx+1..n_early.

    Uses the pre-computed 1000-point Qdlin arrays on the fixed LFP V-grid [2.0–3.5 V].
    This avoids interpolation noise and the NMC V-range mismatch in the NASA pipeline.

    The Severson feature captures how much the discharge capacity curve shifts in
    the first ~20 cycles relative to cycle 2 (ref_idx=1 → 0-indexed cycle 2 = 1-indexed
    cycle 2 from the dataset).  High variance → aggressive early degradation → high β.

    Returns NaN if fewer than 3 valid Qdlin curves are available.
    """
    # Load reference Qdlin
    ref_qdlin = _get_qdlin(f, batch, cell_i, ref_idx)
    if ref_qdlin is None:
        return float("nan")

    # Collect Qdlin differences for cycles ref_idx+1 .. min(n_early-1, n_stored-1)
    max_cyc = min(n_early, n_stored)
    deltas: List[np.ndarray] = []
    for k in range(ref_idx + 1, max_cyc):
        qdl = _get_qdlin(f, batch, cell_i, k)
        if qdl is not None:
            deltas.append(qdl - ref_qdlin)

    if len(deltas) < 2:
        return float("nan")

    mat = np.stack(deltas, axis=0)           # shape (n_deltas, 1000)
    var_per_v  = np.var(mat, axis=0)         # shape (1000,)
    mean_var   = float(np.mean(var_per_v))
    return float(np.log10(mean_var + 1e-9))


# ── main loader ───────────────────────────────────────────────────────────────

def load_severson(
    data_dir: Path = SEVERSON_DIR,
    n_feature_cycles: int = N_FEATURE_CYCLES,
    verbose: bool = False,
) -> List[Dict]:
    """
    Load all 124 Severson cells and return per-cell degradation dicts.

    Parameters
    ----------
    data_dir        : path containing the three .mat batch files
    n_feature_cycles: number of early cycles for ΔQ(V) feature (default 20)
    verbose         : print per-cell summary lines

    Returns
    -------
    List of dicts, one per cell, each containing:
      cell_id, batch, cycle_life, soh (ndarray), D (ndarray), dqv_feature (float)

    Raises
    ------
    FileNotFoundError if any of the three batch files are missing.
    """
    data_dir = Path(data_dir)
    cells: List[Dict] = []

    for batch_num, fname in BATCH_FILES:
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Severson batch file: {path}\n"
                f"Download from https://data.matr.io/1/"
            )
        prefix = f"b{batch_num}c"
        excl   = EXCLUDE_CELLS[batch_num]

        if verbose:
            print(f"\nBatch {batch_num}: {fname}")

        with h5py.File(path, "r") as f:
            batch  = f["batch"]
            n_raw  = batch["summary"].shape[0]

            for i in range(n_raw):
                cell_id = f"{prefix}{i}"
                if cell_id in excl:
                    if verbose:
                        print(f"  {cell_id}: EXCLUDED")
                    continue

                # ── QD summary → SOH ─────────────────────────────────────
                qd_raw = _get_qd(f, batch, i)
                soh, valid_mask = _compute_soh(qd_raw)
                if len(soh) < 5:
                    if verbose:
                        print(f"  {cell_id}: SKIPPED (only {len(soh)} valid QD values)")
                    continue

                # D_k = k (unit-cycle damage; 1-indexed)
                D = np.arange(1, len(soh) + 1, dtype=np.float64)

                # ── cycle metadata ────────────────────────────────────────
                cycle_life = _get_cycle_life(f, batch, i)
                n_stored   = _get_n_stored(f, batch, i)

                # ── ΔQ(V) feature from Qdlin ─────────────────────────────
                feat = _dqv_feature_severson(
                    f, batch, i, n_stored,
                    n_early=n_feature_cycles,
                    ref_idx=FEATURE_REF_IDX,
                )

                cells.append({
                    "cell_id":     cell_id,
                    "batch":       batch_num,
                    "cycle_life":  cycle_life,
                    "soh":         soh,
                    "D":           D,
                    "dqv_feature": feat,
                })

                if verbose:
                    print(f"  {cell_id}: cycles={len(soh)}  "
                          f"SOH_last={soh[-1]:.3f}  "
                          f"dqv={feat:.3f}")

    return cells


# ── summary stats ─────────────────────────────────────────────────────────────

def summarise(cells: List[Dict]) -> None:
    """Print a short summary of the loaded dataset."""
    if not cells:
        print("No cells loaded.")
        return

    n = len(cells)
    cycle_lives = [c["cycle_life"] for c in cells]
    soh_lasts   = [c["soh"][-1] for c in cells]
    feats       = [c["dqv_feature"] for c in cells
                   if not np.isnan(c["dqv_feature"])]

    by_batch = {}
    for c in cells:
        by_batch.setdefault(c["batch"], 0)
        by_batch[c["batch"]] += 1

    print(f"\nSeverson dataset — loaded {n} cells  "
          f"(expected 124, excluded {140-n} raw)")
    for b in sorted(by_batch):
        print(f"  Batch {b}: {by_batch[b]} cells")
    print(f"  cycle_life : min={min(cycle_lives)}  "
          f"median={int(np.median(cycle_lives))}  "
          f"max={max(cycle_lives)}")
    print(f"  SOH_last   : min={min(soh_lasts):.3f}  "
          f"median={np.median(soh_lasts):.3f}  "
          f"max={max(soh_lasts):.3f}")
    print(f"  dqv_feature: {len(feats)}/{n} valid  "
          f"range=[{min(feats):.3f}, {max(feats):.3f}]")
    nan_feat = n - len(feats)
    if nan_feat > 0:
        print(f"  WARNING: {nan_feat} cells have NaN dqv_feature "
              f"(fewer than 3 valid early Qdlin curves)")


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    print("Loading Severson dataset (this may take 30–60 s for 3 × ~2 GB files)…")
    t0 = time.time()
    cells = load_severson(verbose=verbose)
    elapsed = time.time() - t0

    summarise(cells)
    print(f"\nLoad time: {elapsed:.1f} s")

    # Spot-check b1c0
    b1c0 = next((c for c in cells if c["cell_id"] == "b1c0"), None)
    if b1c0:
        print(f"\nSpot-check b1c0:")
        print(f"  batch={b1c0['batch']}  cycle_life={b1c0['cycle_life']}")
        print(f"  soh: len={len(b1c0['soh'])}  "
              f"first={b1c0['soh'][0]:.4f}  last={b1c0['soh'][-1]:.4f}")
        print(f"  dqv_feature = {b1c0['dqv_feature']:.4f}")
