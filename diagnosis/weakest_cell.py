"""
Negative Selection Algorithm (NSA) for Battery Anomaly Detection.
2D effective subspace implementation: V_norm × T_norm.

WHY 2D NOT 6D
The original design called for a 6D feature vector:
  [SOC, SOH, T_norm, SEI_norm, V_norm, plating_risk]
Of these, 4 carry zero inter-cell information on a healthy pack:
  - SOC:          pack-level scalar — identical for all cells at any timestep
  - SOH:          no per-cell sensor; imputed as 1.0 for all cells
  - SEI_norm:     no per-cell sensor; imputed as 0.05 nominal for all cells
  - plating_risk: no per-cell sensor; imputed as 0.01 nominal for all cells
Only V_norm (per-cell voltage sensor) and T_norm (per-cell temperature sensor)
vary across cells. Operating in full 6D with 4/6 constant features collapses
the self-cloud to a 2D slice embedded at [SOC(t), 1.0, *, 0.05, *, 0.01].
Hand-crafted fault centroids placed outside this slice (SEI_norm=3.0,
SOH=0.60–0.85, plating_risk=0.8) are unreachable by any healthy or
unhealthy real observation — producing the all-zero activation failure
originally observed (min observed distance to any centroid = 0.70,
detector radius = 0.25, a 2.8× gap).

SCOPE LIMITATION (persists after recalibration)
The recalibrated NSA detects anomalies in the V_norm × T_norm plane only.
It cannot detect SOH-based faults, SEI growth, or plating risk because
those features are not available as real per-cell sensors. This limitation
does not disappear with recalibration — it is a hardware/data constraint.
Any non-zero anomaly scores reflect deviations in voltage or temperature
relative to the pack, not SOH or chemistry-level degradation signals.

EXPANSION PATH TO 6D
When real per-cell SOH estimates (EKF output), per-cell SEI estimates
(DFN model output), or plating-risk sensors become available:
  1. Restore _extract_features() to return the full 6D vector.
  2. Update FEATURE_DIM = 6.
  3. Rebuild the self-set from per-cell observations in all 6 dimensions.
The negative-selection training algorithm (Phase 2) is unchanged —
only FEATURE_DIM and the bounding-box computation change.

ALGORITHM
Phase 1 (observe_normal): Accumulate (V_norm, T_norm) from real healthy cells.
Phase 2 (train): Compute self-centroid and p95 radius from self-set.
  Sample N_CANDIDATE random points in (centroid ± 4×p95) bounding box.
  Delete candidates within r = R_MULTIPLIER × p95 of any self-set point.
  Surviving candidates are the detector set.
Phase 3 (detect/score): For an observation x, activation fraction =
  fraction of detectors within r of x. Primary score (chronic health) =
  mean activation fraction over a time window. Secondary score (transient)
  = max activation fraction.

Reference: Forrest et al. (1994) IEEE Symposium on Security and Privacy.

V-DETECTOR UPGRADE (VDetector class below)
Standard NSA uses a fixed radius r, which leaves a dead-zone of width r
around the entire self-cloud boundary — no detector can survive closer than
r to any self-point. On the Quartz pack (Null Type A) this meant 97.7% of
survivors sat in T/V extremes no cell ever reached, because the cycling
envelope (p95≈0.311) consumed all would-be near-boundary detectors.

V-detector (Ji & Dasgupta 2004, "Revisiting negative selection algorithms")
assigns each detector a radius equal to the distance to its nearest self-point,
capped at r_max. Detectors right at the self-boundary get near-zero radii and
fill the immediate periphery; detectors far from self get large radii. The
fixed dead-zone disappears.

Note: V-detector cannot manufacture signal that is absent. If the self-cloud
fills the operating envelope (Null Type A condition), V-detectors at the
boundary still cover nothing inside the envelope — the cycling amplitude
dominates inter-cell differences regardless. The VDetector class is validated
via a synthetic positive control (scripts/vdetector_positive_control.py)
before being applied to Quartz data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# NSA CONSTANTS — 2D EFFECTIVE SUBSPACE
# =============================================================================
FEATURE_DIM: int = 2          # V_norm, T_norm only (see module docstring)
N_CANDIDATE: int = 50_000     # random candidates before self-deletion
R_MULTIPLIER: float = 0.4     # r = R_MULTIPLIER × p95(self-cloud radius)
BOX_MULTIPLIER: float = 4.0   # sampling box = centroid ± BOX_MULTIPLIER × p95
ANOMALY_THRESHOLD: float = 0.01   # activation fraction above which cell is flagged

# Original 6D fault-signature constants retained as reference for the
# 6D expansion path described in the module docstring. Not used in the
# current 2D implementation.
_FAULT_SIGNATURES_6D = {
    "SEI_growth":    {"center": [0.5, 0.75, 0.3, 3.0, 0.5, 0.1],  "spread": 0.2},
    "plating_risk":  {"center": [0.9, 0.85, 0.0, 0.5, 0.9, 0.8],  "spread": 0.15},
    "thermal_stress":{"center": [0.5, 0.80, 1.0, 1.0, 0.5, 0.2],  "spread": 0.2},
    "capacity_fade": {"center": [0.3, 0.60, 0.5, 2.5, 0.3, 0.3],  "spread": 0.2},
}


class NegativeSelectionDetector:
    """
    Negative Selection Algorithm operating in the 2D (V_norm, T_norm) subspace.

    The 4 imputed features (SOC, SOH, SEI_norm, plating_risk) from the original
    6D design carry zero inter-cell information and are excluded. See module
    docstring for rationale and expansion path.

    Usage:
        nsa = NegativeSelectionDetector()
        for state in healthy_observations:
            nsa.observe_normal(state)
        nsa.train()
        frac = nsa.activation_fraction(new_state)  # per-timestep score
    """

    def __init__(self,
                 n_candidate: int = N_CANDIDATE,
                 r_multiplier: float = R_MULTIPLIER,
                 box_multiplier: float = BOX_MULTIPLIER,
                 rng_seed: int = 0) -> None:
        self.n_candidate    = n_candidate
        self.r_multiplier   = r_multiplier
        self.box_multiplier = box_multiplier
        self._rng           = np.random.default_rng(rng_seed)
        self._self_set: List[np.ndarray] = []   # (V_norm, T_norm) per observation
        self._self_trained: bool = False
        self._detectors: Optional[np.ndarray] = None  # (N_surv, 2)
        self._r: float = 0.0
        self._activation_history: Dict[int, List[float]] = {}

    def __repr__(self) -> str:
        n = len(self._detectors) if self._detectors is not None else 0
        return (f"NegativeSelectionDetector(2D, "
                f"n_self={len(self._self_set)}, n_det={n}, "
                f"trained={self._self_trained}, r={self._r:.5f})")

    # ------------------------------------------------------------------
    # Feature extraction — 2D only
    # ------------------------------------------------------------------
    def _extract_features(self, cell_state: dict) -> np.ndarray:
        """
        Extract 2D feature vector [V_norm, T_norm] from a cell-state dict.

        V_norm = clip((V[V] - 3.0) / 1.5, 0, 1)
        T_norm = (T[K] - 298.15) / 50

        The 4 excluded dimensions (SOC, SOH, SEI_norm, plating_risk) are
        imputed constants on a healthy pack and carry zero inter-cell signal.
        See module docstring for the 6D expansion path.
        """
        v = float(cell_state.get("V", 3.7))
        t = float(cell_state.get("T", 298.15))
        v_norm = float(np.clip((v - 3.0) / 1.5, 0.0, 1.0))
        t_norm = float((t - 298.15) / 50.0)
        return np.array([v_norm, t_norm], dtype=np.float64)

    # ------------------------------------------------------------------
    # Phase 1: observe normal operation
    # ------------------------------------------------------------------
    def observe_normal(self, cell_state: dict) -> None:
        """Add one healthy observation (V_norm, T_norm) to the self-set."""
        self._self_set.append(self._extract_features(cell_state).copy())

    # ------------------------------------------------------------------
    # Phase 2: genuine negative-selection training
    # ------------------------------------------------------------------
    def train(self) -> int:
        """
        Generate non-self detectors by random sampling + self-deletion.

        Algorithm:
          1. Compute self-centroid and p95 radius from self-set points.
          2. Set r = R_MULTIPLIER × p95  (single pre-chosen multiplier).
          3. Sample N_CANDIDATE uniform random points in bounding box
             (centroid ± BOX_MULTIPLIER × p95 per dimension).
          4. Delete candidates within r of any self-set point.
          5. Keep survivors as the detector array.

        Returns number of surviving detectors.
        """
        if not self._self_set:
            raise RuntimeError("call observe_normal() before train()")

        self_arr  = np.stack(self._self_set)           # (N_self, 2)
        centroid  = self_arr.mean(axis=0)              # (2,)
        dists     = np.linalg.norm(self_arr - centroid, axis=1)
        p95       = float(np.percentile(dists, 95)) if len(dists) >= 2 else 0.05

        self._r   = self.r_multiplier * p95
        box_lo    = centroid - self.box_multiplier * p95
        box_hi    = centroid + self.box_multiplier * p95

        cands = self._rng.uniform(low=box_lo, high=box_hi,
                                  size=(self.n_candidate, FEATURE_DIM))
        # Vectorised self-deletion in batches to bound memory
        BATCH = 5_000
        r2 = self._r ** 2
        keep = np.ones(self.n_candidate, dtype=bool)
        for start in range(0, self.n_candidate, BATCH):
            end = min(start + BATCH, self.n_candidate)
            diffs = cands[start:end, None, :] - self_arr[None, :, :]  # (B,N,2)
            min_d2 = (diffs ** 2).sum(axis=2).min(axis=1)              # (B,)
            keep[start:end] = min_d2 > r2

        self._detectors  = cands[keep].astype(np.float64)
        self._self_trained = True
        return int(keep.sum())

    # ------------------------------------------------------------------
    # Phase 3: per-timestep scoring
    # ------------------------------------------------------------------
    def activation_fraction(self, cell_state: dict) -> float:
        """
        Fraction of detectors within r of this observation.
        Returns 0.0 if untrained or no detectors.
        """
        if not self._self_trained or self._detectors is None or len(self._detectors) == 0:
            return 0.0
        feat = self._extract_features(cell_state)
        d2   = ((self._detectors - feat) ** 2).sum(axis=1)
        return float((d2 <= self._r ** 2).mean())

    def detect(self, cell_id: int, cell_state: dict) -> Tuple[bool, str, float]:
        """
        Single-timestep anomaly detection (backward-compatible interface).

        Returns:
            is_anomaly (bool): True if activation_fraction > ANOMALY_THRESHOLD
            fault_class (str): 'voltage_temperature_anomaly' or 'normal'
            confidence (float): activation_fraction in [0, 1]

        Note: with a healthy pack and r calibrated to the healthy distribution,
        this will typically return is_anomaly=False and confidence≈0 for all
        healthy cells. The meaningful signal is the relative ranking of
        activation fractions across cells, not the binary is_anomaly flag.
        See module docstring for scope limitations.
        """
        frac = self.activation_fraction(cell_state)
        is_anomaly = frac > ANOMALY_THRESHOLD
        fault_class = "voltage_temperature_anomaly" if is_anomaly else "normal"

        if cell_id not in self._activation_history:
            self._activation_history[cell_id] = []
        self._activation_history[cell_id].append(frac)

        return is_anomaly, fault_class, float(np.clip(frac, 0.0, 1.0))

    def primary_score(self, cell_id: int) -> float:
        """Mean activation fraction (chronic health signal) over observation history."""
        hist = self._activation_history.get(cell_id, [])
        return float(np.mean(hist)) if hist else 0.0

    def secondary_score(self, cell_id: int) -> float:
        """Max activation fraction (transient/peak signal) over observation history."""
        hist = self._activation_history.get(cell_id, [])
        return float(np.max(hist)) if hist else 0.0

    def detect_all_cells(self, cell_states: List[dict]) -> List[Dict]:
        """Run detection on a list of cell states (one per cell at a single timestep)."""
        return [
            {
                "cell_id": i,
                **dict(zip(["anomaly", "fault_class", "confidence"],
                           self.detect(i, state))),
                "features": self._extract_features(state).tolist(),
            }
            for i, state in enumerate(cell_states)
        ]


def validate() -> bool:
    """Self-test suite for diagnosis/weakest_cell.py (2D NSA)."""
    print("=" * 60)
    print("VALIDATING: diagnosis/weakest_cell.py (2D NSA)")
    print("=" * 60)
    all_pass = True

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f"  | {detail}" if detail else ""))
        if not condition:
            all_pass = False

    rng = np.random.default_rng(42)

    # --- Feature extraction ---
    nsa = NegativeSelectionDetector(n_candidate=5_000, rng_seed=0)
    feat = nsa._extract_features({"V": 3.7, "T": 300.0})
    check("Feature shape (2,)", feat.shape == (FEATURE_DIM,))
    check("Feature finite", bool(np.all(np.isfinite(feat))))
    expected_v = np.clip((3.7 - 3.0) / 1.5, 0, 1)
    expected_t = (300.0 - 298.15) / 50.0
    check("V_norm correct", abs(feat[0] - expected_v) < 1e-9,
          f"got={feat[0]:.5f} want={expected_v:.5f}")
    check("T_norm correct", abs(feat[1] - expected_t) < 1e-9,
          f"got={feat[1]:.5f} want={expected_t:.5f}")

    # --- Self-set and training ---
    for _ in range(100):
        nsa.observe_normal({
            "V": 3.6 + 0.1 * float(rng.random()),
            "T": 298.0 + 5.0 * float(rng.random()),
        })
    check("Self-set size", len(nsa._self_set) == 100)
    n_surv = nsa.train()
    check("Trained flag", nsa._self_trained)
    check("Detectors generated", n_surv > 0, f"n_survivors={n_surv}")
    check("r > 0", nsa._r > 0, f"r={nsa._r:.5f}")
    check("r < 0.5 (not degenerate)", nsa._r < 0.5, f"r={nsa._r:.5f}")

    # --- Self-deletion correctness: no detector within r of any self-point ---
    self_arr = np.stack(nsa._self_set)
    diffs = nsa._detectors[:, None, :] - self_arr[None, :, :]  # (N_surv, 100, 2)
    min_dists = np.sqrt((diffs ** 2).sum(axis=2).min(axis=1))
    check("Self-deletion: all detectors > r from self-set",
          bool((min_dists > nsa._r - 1e-9).all()),
          f"min_dist_to_self={min_dists.min():.6f} r={nsa._r:.6f}")

    # --- activation_fraction in [0,1] ---
    state_normal = {"V": 3.68, "T": 299.5}   # near self-set center
    state_far    = {"V": 2.5,  "T": 350.0}   # far from self-set
    frac_normal = nsa.activation_fraction(state_normal)
    frac_far    = nsa.activation_fraction(state_far)
    check("Activation fraction in [0,1] (normal)", 0.0 <= frac_normal <= 1.0,
          f"frac={frac_normal:.5f}")
    check("Activation fraction in [0,1] (far)", 0.0 <= frac_far <= 1.0,
          f"frac={frac_far:.5f}")

    # --- detect() interface backward-compat ---
    is_a, fclass, conf = nsa.detect(0, state_normal)
    check("detect() returns 3-tuple", True, f"({is_a}, {fclass}, {conf:.3f})")
    check("confidence in [0,1]", 0.0 <= conf <= 1.0)

    # --- primary / secondary scores ---
    for _ in range(20):
        nsa.detect(99, {"V": 3.65 + 0.02 * float(rng.random()),
                        "T": 299.0 + 1.0 * float(rng.random())})
    nsa.detect(99, state_far)  # inject one far point
    p = nsa.primary_score(99)
    s = nsa.secondary_score(99)
    check("primary_score in [0,1]", 0.0 <= p <= 1.0, f"p={p:.5f}")
    check("secondary >= primary", s >= p - 1e-9, f"s={s:.5f} p={p:.5f}")

    # --- untrained guard ---
    nsa2 = NegativeSelectionDetector()
    check("Untrained: activation_fraction=0", nsa2.activation_fraction({"V": 3.7, "T": 300.0}) == 0.0)
    is_a2, _, conf2 = nsa2.detect(0, {"V": 3.7, "T": 300.0})
    check("Untrained detect: not anomaly", not is_a2)

    # --- batch interface ---
    batch = nsa.detect_all_cells([
        {"V": 3.7, "T": 299.0},
        {"V": 3.5, "T": 310.0},
    ])
    check("Batch: 2 results", len(batch) == 2)
    check("Batch has features", all(len(r["features"]) == FEATURE_DIM for r in batch))

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    validate()


# =============================================================================
# V-DETECTOR — VARIABLE-RADIUS NSA
# =============================================================================

# Pre-registered hyperparameters (locked before any positive-control run)
V_R_MIN: float = 0.005          # minimum viable radius — ~1.25× inter-cell spread
V_R_MAX: float = 0.15           # cap at ~p95/2; prevents one huge detector dominating
V_N_CANDIDATE: int = 50_000     # initial candidates before filtering
V_BOX_MULTIPLIER: float = 4.0   # sampling box = centroid ± 4×p95
V_SATURATION_THRESHOLD: float = 0.50  # flag if any detector activates >50% of queries


class VDetector:
    """
    Variable-radius Negative Selection Algorithm (V-detector).

    Each retained detector has radius r_i = min(dist_to_nearest_self, r_max).
    Detectors near the self-boundary have small radii; detectors far from self
    have large radii. The fixed dead-zone of standard NSA (width = r, around the
    entire boundary) is replaced by the exact self-cloud surface.

    A detector activates query q when dist(q, center_i) <= r_i.
    PRIMARY score = mean activation fraction over cell's observation history.
    SECONDARY score = max activation fraction (transient/peak signal).

    Pre-registered hyperparameters (V_R_MIN, V_R_MAX, V_N_CANDIDATE,
    V_BOX_MULTIPLIER, V_SATURATION_THRESHOLD) are module-level constants.

    References:
        Ji & Dasgupta (2004) Revisiting Negative Selection Algorithms.
        Stibor et al. (2005) On the appropriateness of negative selection
            for anomaly detection and network intrusion detection (boundary-fixed
            variant discussion).
    """

    def __init__(
        self,
        r_min: float = V_R_MIN,
        r_max: float = V_R_MAX,
        n_candidate: int = V_N_CANDIDATE,
        box_multiplier: float = V_BOX_MULTIPLIER,
        saturation_threshold: float = V_SATURATION_THRESHOLD,
        rng_seed: int = 0,
    ) -> None:
        self.r_min = r_min
        self.r_max = r_max
        self.n_candidate = n_candidate
        self.box_multiplier = box_multiplier
        self.saturation_threshold = saturation_threshold
        self._rng = np.random.default_rng(rng_seed)
        self._self_set: List[np.ndarray] = []
        self._trained: bool = False
        self._centers: Optional[np.ndarray] = None   # (N_surv, 2)
        self._radii: Optional[np.ndarray] = None     # (N_surv,)
        self._activation_history: Dict[int, List[float]] = {}
        self.saturation_flag: bool = False

    def __repr__(self) -> str:
        n = len(self._centers) if self._centers is not None else 0
        return (
            f"VDetector(2D, n_self={len(self._self_set)}, n_det={n}, "
            f"trained={self._trained}, "
            f"r_min={self.r_min}, r_max={self.r_max})"
        )

    # ------------------------------------------------------------------
    # Feature extraction (identical to NegativeSelectionDetector)
    # ------------------------------------------------------------------
    def _extract_features(self, cell_state: dict) -> np.ndarray:
        v = float(cell_state.get("V", 3.7))
        t = float(cell_state.get("T", 298.15))
        v_norm = float(np.clip((v - 3.0) / 1.5, 0.0, 1.0))
        t_norm = float((t - 298.15) / 50.0)
        return np.array([v_norm, t_norm], dtype=np.float64)

    # ------------------------------------------------------------------
    # Phase 1: observe normal operation
    # ------------------------------------------------------------------
    def observe_normal(self, cell_state: dict) -> None:
        self._self_set.append(self._extract_features(cell_state).copy())

    def observe_normal_array(self, features: np.ndarray) -> None:
        """Bulk-load self-set from (N, 2) pre-extracted feature array."""
        for row in features:
            self._self_set.append(np.asarray(row, dtype=np.float64).copy())

    # ------------------------------------------------------------------
    # Phase 2: variable-radius training
    # ------------------------------------------------------------------
    def train(self) -> dict:
        """
        Generate variable-radius non-self detectors.

        Algorithm:
          1. Compute self-centroid and p95 radius.
          2. Sample N_CANDIDATE points uniformly in bounding box.
          3. Build KDTree on self-set; query each candidate for its
             distance to nearest self-point.
          4. r_i = min(nearest_dist, r_max).
          5. Retain candidate iff r_i >= r_min.
          6. Run saturation check: flag if any detector activates
             >V_SATURATION_THRESHOLD of a 500-point self-sample.

        Returns summary dict (n_survivors, r stats, saturation_flag).
        """
        from scipy.spatial import KDTree

        if not self._self_set:
            raise RuntimeError("call observe_normal() or observe_normal_array() before train()")

        self_arr = np.stack(self._self_set)            # (N_self, 2)
        centroid = self_arr.mean(axis=0)
        dists_to_centroid = np.linalg.norm(self_arr - centroid, axis=1)
        p95 = float(np.percentile(dists_to_centroid, 95)) if len(dists_to_centroid) >= 2 else 0.05

        box_lo = centroid - self.box_multiplier * p95
        box_hi = centroid + self.box_multiplier * p95
        cands = self._rng.uniform(low=box_lo, high=box_hi,
                                  size=(self.n_candidate, FEATURE_DIM))

        # Distance from each candidate to its nearest self-point
        tree = KDTree(self_arr)
        nearest_dists, _ = tree.query(cands, k=1, workers=-1)   # (N_cand,)

        # Variable radii
        raw_radii = np.minimum(nearest_dists, self.r_max)
        keep = raw_radii >= self.r_min

        self._centers = cands[keep].astype(np.float64)
        self._radii = raw_radii[keep].astype(np.float64)
        self._trained = True

        # Saturation check (pre-registered: flag if any detector activates
        # >50% of a 500-point sample from the self-set)
        if len(self._centers) > 0:
            self.saturation_flag = self._check_saturation(self_arr)

        n_surv = int(keep.sum())
        return {
            "n_candidates": self.n_candidate,
            "n_survivors": n_surv,
            "n_filtered_r_min": int((~keep).sum()),
            "p95_self_cloud": float(p95),
            "centroid": centroid.tolist(),
            "r_min_actual": float(self._radii.min()) if n_surv else 0.0,
            "r_max_actual": float(self._radii.max()) if n_surv else 0.0,
            "r_mean": float(self._radii.mean()) if n_surv else 0.0,
            "r_median": float(np.median(self._radii)) if n_surv else 0.0,
            "r_p5": float(np.percentile(self._radii, 5)) if n_surv else 0.0,
            "r_p95": float(np.percentile(self._radii, 95)) if n_surv else 0.0,
            "saturation_flag": self.saturation_flag,
        }

    def _check_saturation(self, self_arr: np.ndarray) -> bool:
        """Return True if any single detector activates >saturation_threshold of self-sample."""
        idx = self._rng.choice(len(self_arr), min(500, len(self_arr)), replace=False)
        sample = self_arr[idx]           # (n_sample, 2)
        n_sample = len(sample)

        BATCH = 2_000
        max_frac = 0.0
        for start in range(0, len(self._centers), BATCH):
            end = min(start + BATCH, len(self._centers))
            centers_b = self._centers[start:end]    # (B, 2)
            radii_b = self._radii[start:end]        # (B,)
            # dists: (n_sample, B)
            diffs = sample[:, None, :] - centers_b[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))
            fracs = (dists <= radii_b[None, :]).mean(axis=0)   # (B,)
            max_frac = max(max_frac, float(fracs.max()))
            if max_frac > self.saturation_threshold:
                return True
        return False

    # ------------------------------------------------------------------
    # Phase 3: scoring
    # ------------------------------------------------------------------
    def activation_fraction(self, cell_state: dict) -> float:
        """Fraction of detectors that activate for a single observation."""
        if not self._trained or self._centers is None or len(self._centers) == 0:
            return 0.0
        feat = self._extract_features(cell_state)
        dists = np.sqrt(((self._centers - feat) ** 2).sum(axis=1))
        return float((dists <= self._radii).mean())

    def activation_fraction_array(self, features: np.ndarray) -> np.ndarray:
        """
        Vectorized scoring for a pre-extracted (N, 2) feature array.
        Returns (N,) array of per-timestep activation fractions.
        Batches over detectors to bound peak memory to ~32 MB.
        """
        if not self._trained or self._centers is None or len(self._centers) == 0:
            return np.zeros(len(features), dtype=np.float64)

        N_q = len(features)
        N_d = len(self._centers)
        activations = np.zeros(N_q, dtype=np.float64)

        BATCH_D = 1_000   # detectors per batch; (N_q × BATCH_D × 2) × 8B ≤ 38 MB
        for start in range(0, N_d, BATCH_D):
            end = min(start + BATCH_D, N_d)
            centers_b = self._centers[start:end]    # (B, 2)
            radii_b = self._radii[start:end]        # (B,)
            # diffs: (N_q, B, 2)
            diffs = features[:, None, :] - centers_b[None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=2))   # (N_q, B)
            activations += (dists <= radii_b[None, :]).sum(axis=1).astype(np.float64)

        return activations / N_d

    def record_timestep(self, cell_id: int, activation: float) -> None:
        if cell_id not in self._activation_history:
            self._activation_history[cell_id] = []
        self._activation_history[cell_id].append(activation)

    def primary_score(self, cell_id: int) -> float:
        hist = self._activation_history.get(cell_id, [])
        return float(np.mean(hist)) if hist else 0.0

    def secondary_score(self, cell_id: int) -> float:
        hist = self._activation_history.get(cell_id, [])
        return float(np.max(hist)) if hist else 0.0


def validate_vdetector() -> bool:
    """Unit tests for VDetector."""
    print("=" * 60)
    print("VALIDATING: VDetector (variable-radius NSA)")
    print("=" * 60)
    all_pass = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f"  | {detail}" if detail else ""))
        if not cond:
            all_pass = False

    rng = np.random.default_rng(7)

    # --- Basic training ---
    vd = VDetector(n_candidate=5_000, rng_seed=1)
    for _ in range(200):
        vd.observe_normal({
            "V": 3.6 + 0.1 * float(rng.random()),
            "T": 298.0 + 5.0 * float(rng.random()),
        })
    stats = vd.train()
    check("trained flag", vd._trained)
    check("n_survivors > 0", stats["n_survivors"] > 0, f"n={stats['n_survivors']}")
    check("p95 > 0", stats["p95_self_cloud"] > 0, f"p95={stats['p95_self_cloud']:.5f}")

    # --- Variable radii: all in [r_min, r_max] ---
    check("radii >= r_min",
          bool((vd._radii >= vd.r_min - 1e-9).all()),
          f"min_r={vd._radii.min():.5f}")
    check("radii <= r_max",
          bool((vd._radii <= vd.r_max + 1e-9).all()),
          f"max_r={vd._radii.max():.5f}")

    # --- Self-deletion: no center within its own radius of any self-point ---
    self_arr = np.stack(vd._self_set)
    from scipy.spatial import KDTree
    tree = KDTree(self_arr)
    nearest_dists, _ = tree.query(vd._centers, k=1)
    # By construction r_i = min(nearest_dist, r_max), so r_i <= nearest_dist
    check("self-deletion: r_i <= nearest_dist for all detectors",
          bool((vd._radii <= nearest_dists + 1e-9).all()),
          f"max_violation={float(np.maximum(vd._radii - nearest_dists, 0).max()):.2e}")

    # --- Activation fraction in [0, 1] ---
    frac_near = vd.activation_fraction({"V": 3.65, "T": 299.5})
    frac_far  = vd.activation_fraction({"V": 2.0,  "T": 400.0})
    check("activation in [0,1] (near)", 0.0 <= frac_near <= 1.0,
          f"frac={frac_near:.5f}")
    check("activation in [0,1] (far)", 0.0 <= frac_far <= 1.0,
          f"frac={frac_far:.5f}")

    # --- Array interface matches scalar ---
    pts = np.array([[np.clip((3.65 - 3.0) / 1.5, 0, 1), (299.5 - 298.15) / 50.0],
                    [np.clip((2.0  - 3.0) / 1.5, 0, 1), (400.0 - 298.15) / 50.0]])
    arr_fracs = vd.activation_fraction_array(pts)
    check("array matches scalar (near)",
          abs(arr_fracs[0] - frac_near) < 1e-9,
          f"arr={arr_fracs[0]:.6f} scalar={frac_near:.6f}")
    check("array matches scalar (far)",
          abs(arr_fracs[1] - frac_far) < 1e-9,
          f"arr={arr_fracs[1]:.6f} scalar={frac_far:.6f}")

    # --- observe_normal_array ---
    vd2 = VDetector(n_candidate=3_000, rng_seed=2)
    feats = np.column_stack([
        3.6 + 0.1 * rng.random(100),
        298.0 + 5.0 * rng.random(100),
    ])
    # feats is in raw (V, T) space — convert to (V_norm, T_norm) manually for bulk
    v_norm = np.clip((feats[:, 0] - 3.0) / 1.5, 0, 1)
    t_norm = (feats[:, 1] - 298.15) / 50.0
    bulk = np.column_stack([v_norm, t_norm])
    vd2.observe_normal_array(bulk)
    check("observe_normal_array: self-set size", len(vd2._self_set) == 100,
          f"got {len(vd2._self_set)}")

    # --- primary <= secondary ---
    for _ in range(30):
        vd.record_timestep(5, float(rng.random() * 0.01))
    vd.record_timestep(5, 0.5)   # inject one high activation
    p = vd.primary_score(5)
    s = vd.secondary_score(5)
    check("secondary >= primary", s >= p - 1e-9, f"s={s:.5f} p={p:.5f}")

    # --- untrained guard ---
    vd3 = VDetector()
    check("untrained activation = 0.0", vd3.activation_fraction({"V": 3.7, "T": 300.0}) == 0.0)
    arr3 = vd3.activation_fraction_array(np.zeros((5, 2)))
    check("untrained array = zeros", float(arr3.sum()) == 0.0)

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_pass
