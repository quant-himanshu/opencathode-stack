"""
Negative Selection Algorithm (NSA) for Battery Anomaly Detection.

Reference: Forrest et al. (1994) IEEE Symposium on Security and Privacy.
Battery adaptation: internal methodology.
Feature space: 6D normalized sensor features per cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.dfn_cell import EPS

# =============================================================================
# NSA CONSTANTS
# =============================================================================
N_SELF_CYCLES: int = 100
N_DETECTORS: int = 200
DETECTOR_RADIUS: float = 0.25
FEATURE_DIM: int = 6
ANOMALY_THRESHOLD: float = 0.02

FAULT_SEI: str = "SEI_growth"
FAULT_PLATING: str = "plating_risk"
FAULT_THERMAL: str = "thermal_stress"
FAULT_CAPACITY: str = "capacity_fade"
FAULT_NORMAL: str = "normal"


@dataclass
class Detector:
    """
    Single hypersphere detector in 6D feature space.
    Activates when input is within radius of center.
    Reference: Forrest et al. (1994) Eq. 1.
    """
    center: np.ndarray   # shape (6,) normalized
    radius: float        # activation radius [dimensionless]
    fault_class: str     # associated fault label
    activation_count: int = 0

    def __repr__(self) -> str:
        return f"Detector(class={self.fault_class}, r={self.radius:.3f}, act={self.activation_count})"

    def activates(self, feature_vec: np.ndarray) -> bool:
        """Euclidean distance from center < radius => activated."""
        return float(np.linalg.norm(feature_vec - self.center)) < self.radius


class NegativeSelectionDetector:
    """
    Negative Selection Algorithm for per-cell anomaly detection.
    Phase 1: observe normal operation to define self-set.
    Phase 2: generate detectors that do NOT match self-set.
    Phase 3: classify new samples against detectors.
    """

    def __init__(self, n_detectors: int = N_DETECTORS,
                 detector_radius: float = DETECTOR_RADIUS,
                 rng_seed: int = 0) -> None:
        self.n_detectors = n_detectors
        self.detector_radius = detector_radius
        self._rng = np.random.default_rng(rng_seed)
        self._self_set: List[np.ndarray] = []
        self._self_trained: bool = False
        self.detectors: List[Detector] = []
        self._anomaly_history: Dict[int, List[bool]] = {}
        self._n_self_observations: int = 0

    def __repr__(self) -> str:
        return (f"NegativeSelectionDetector(n_det={len(self.detectors)}, "
                f"trained={self._self_trained}, obs={self._n_self_observations})")

    def _extract_features(self, cell_state: dict) -> np.ndarray:
        """
        Extract 6D normalized feature vector.
        [SOC, SOH, T_norm, SEI_norm, V_norm, plating_risk]
        """
        return np.array([
            float(np.clip(cell_state.get("SOC", 0.5), 0.0, 1.0)),
            float(np.clip(cell_state.get("SOH", 1.0), 0.0, 1.0)),
            float(np.clip((cell_state.get("T", 298.15) - 298.15) / 50.0, -1.0, 2.0)),
            float(np.clip(cell_state.get("delta_SEI_m", 5e-9) / 1e-7, 0.0, 5.0)),
            float(np.clip((cell_state.get("V", 3.7) - 3.0) / 1.5, 0.0, 1.0)),
            float(np.clip(cell_state.get("plating_risk", 0.0), 0.0, 1.0)),
        ], dtype=np.float64)

    def observe_normal(self, cell_state: dict) -> None:
        """Add one normal observation to self-set (Phase 1)."""
        self._self_set.append(self._extract_features(cell_state).copy())
        self._n_self_observations += 1

    def _in_self_set(self, candidate: np.ndarray, radius: float) -> bool:
        """True if candidate overlaps with any self-set sample."""
        if not self._self_set:
            return False
        self_arr = np.stack(self._self_set)
        dists = np.linalg.norm(self_arr - candidate[np.newaxis, :], axis=1)
        return bool(np.any(dists < radius))

    def train(self) -> int:
        """
        Generate non-self detectors (Phase 2).
        Candidates overlapping self-set are rejected.
        Reference: Forrest et al. (1994) Algorithm 1.
        Returns number of valid detectors kept.
        """
        if len(self._self_set) < 10:
            for _ in range(20):
                self.observe_normal({
                    "SOC": 0.7, "SOH": 0.95, "T": 298.15,
                    "delta_SEI_m": 5e-9, "V": 3.7, "plating_risk": 0.01
                })

        self.detectors = []

        # Fault-signature regions in 6D feature space
        fault_regions = {
            FAULT_SEI: {"center": np.array([0.5, 0.75, 0.3, 3.0, 0.5, 0.1]), "spread": 0.2},
            FAULT_PLATING: {"center": np.array([0.9, 0.85, 0.0, 0.5, 0.9, 0.8]), "spread": 0.15},
            FAULT_THERMAL: {"center": np.array([0.5, 0.80, 1.0, 1.0, 0.5, 0.2]), "spread": 0.2},
            FAULT_CAPACITY: {"center": np.array([0.3, 0.60, 0.5, 2.5, 0.3, 0.3]), "spread": 0.2},
        }

        n_per_fault = self.n_detectors // len(fault_regions)
        for fault, region in fault_regions.items():
            generated = 0
            attempts = 0
            while generated < n_per_fault and attempts < n_per_fault * 20:
                attempts += 1
                candidate = region["center"] + self._rng.normal(0, region["spread"], FEATURE_DIM)
                candidate = np.clip(candidate, -0.5, 5.5)
                if not self._in_self_set(candidate, self.detector_radius):
                    self.detectors.append(Detector(
                        center=candidate, radius=self.detector_radius, fault_class=fault))
                    generated += 1

        self._self_trained = True
        return len(self.detectors)

    def detect(self, cell_id: int, cell_state: dict) -> Tuple[bool, str, float]:
        """
        Real-time anomaly detection for one cell using minimum-distance criterion.
        Per original NSA (Forrest 1994): a sample is anomalous if it falls within
        the radius of ANY non-self detector. In continuous 6D space, counting
        activations is unreliable (sparse coverage), so we use the nearest
        fault-class detector distance as the anomaly score.

        Args:
            cell_id: Cell identifier.
            cell_state: Current cell state dict.
        Returns:
            (is_anomaly, fault_class, confidence [0,1])
        """
        if not self._self_trained:
            return False, FAULT_NORMAL, 0.0

        features = self._extract_features(cell_state)

        # Vectorized: compute distance from features to ALL detector centers
        if not self.detectors:
            return False, FAULT_NORMAL, 0.0

        centers = np.stack([d.center for d in self.detectors])  # (N_det, 6)
        dists = np.linalg.norm(centers - features[np.newaxis, :], axis=1)  # (N_det,)

        # Per original NSA: anomaly if ANY detector activates (dist < radius)
        activated_mask = dists < self.detector_radius
        is_anomaly = bool(np.any(activated_mask))

        # Per-fault-class minimum distance for classification
        fault_min_dist: Dict[str, float] = {f: np.inf for f in fault_regions_list()}
        for i, det in enumerate(self.detectors):
            if dists[i] < fault_min_dist.get(det.fault_class, np.inf):
                fault_min_dist[det.fault_class] = float(dists[i])
                if activated_mask[i]:
                    det.activation_count += 1

        # Fault classification: nearest fault class (min distance)
        min_fault = min(fault_min_dist, key=lambda k: fault_min_dist[k])
        min_dist_overall = float(np.min(dists))

        if is_anomaly:
            dominant = min_fault
            # Confidence: how far inside the detector sphere (1=center, 0=edge)
            confidence = float(max(0.0, 1.0 - fault_min_dist[min_fault] / self.detector_radius))
        else:
            dominant = FAULT_NORMAL
            # Confidence of "normal": how far outside the nearest detector
            confidence = float(min(1.0, (min_dist_overall - self.detector_radius) / self.detector_radius))

        if cell_id not in self._anomaly_history:
            self._anomaly_history[cell_id] = []
        self._anomaly_history[cell_id].append(is_anomaly)

        return is_anomaly, dominant, float(np.clip(confidence, 0.0, 1.0))

    def detect_all_cells(self, cell_states: List[dict]) -> List[Dict]:
        """Run detection on list of cell states."""
        return [
            {
                "cell_id": i,
                **dict(zip(["anomaly", "fault_class", "confidence"],
                           self.detect(i, state))),
                "features": self._extract_features(state).tolist(),
            }
            for i, state in enumerate(cell_states)
        ]

    def anomaly_rate(self, cell_id: int) -> float:
        """Fraction of recent steps flagged anomalous (last 50)."""
        hist = self._anomaly_history.get(cell_id, [])
        if not hist:
            return 0.0
        window = hist[-50:]
        return float(sum(window) / len(window))


def fault_regions_list() -> List[str]:
    return [FAULT_SEI, FAULT_PLATING, FAULT_THERMAL, FAULT_CAPACITY]


def validate() -> bool:
    """Self-test suite for diagnosis/weakest_cell.py."""
    print("=" * 60)
    print("VALIDATING: diagnosis/weakest_cell.py")
    print("=" * 60)
    all_pass = True

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f"  | {detail}" if detail else ""))
        if not condition:
            all_pass = False

    rng = np.random.default_rng(123)
    det = NegativeSelectionDetector(n_detectors=100, rng_seed=42)
    check("Detector created", True, str(det))

    # Feature extraction
    feat = det._extract_features({"SOC": 0.7, "SOH": 0.95, "T": 300.0,
                                   "delta_SEI_m": 5e-9, "V": 3.7, "plating_risk": 0.01})
    check("Features shape (6,)", feat.shape == (FEATURE_DIM,))
    check("Features finite", bool(np.all(np.isfinite(feat))))

    # Self-set
    for _ in range(50):
        det.observe_normal({
            "SOC": 0.5 + 0.3 * rng.random(), "SOH": 0.90 + 0.1 * rng.random(),
            "T": 298.15 + 5 * rng.random(), "delta_SEI_m": 5e-9 + 1e-9 * rng.random(),
            "V": 3.6 + 0.2 * rng.random(), "plating_risk": 0.01 * rng.random(),
        })
    check("Self-set size", len(det._self_set) == 50)

    # Training
    n_det = det.train()
    check("Detectors generated", n_det > 0, f"n={n_det}")
    check("Trained flag set", det._self_trained)

    # Normal cell: should not be anomaly
    is_a, fault, conf = det.detect(0, {"SOC": 0.65, "SOH": 0.93, "T": 299.0,
                                        "delta_SEI_m": 6e-9, "V": 3.7, "plating_risk": 0.005})
    check("Normal state: not anomaly", not is_a, f"fault={fault}, conf={conf:.3f}")

    # Hot cell: should trigger anomaly
    is_a2, fault2, conf2 = det.detect(1, {"SOC": 0.5, "SOH": 0.80, "T": 350.0,
                                           "delta_SEI_m": 1e-7, "V": 3.5, "plating_risk": 0.2})
    check("Hot cell: anomaly detected", is_a2, f"fault={fault2}, conf={conf2:.3f}")

    # Plating risk
    is_a3, fault3, _ = det.detect(2, {"SOC": 0.95, "SOH": 0.85, "T": 295.0,
                                       "delta_SEI_m": 5e-8, "V": 4.1, "plating_risk": 0.9})
    check("Plating state: fault detected", is_a3, f"fault={fault3}")

    # Batch
    results = det.detect_all_cells([
        {"SOC": 0.7, "SOH": 0.95, "T": 300.0, "delta_SEI_m": 5e-9, "V": 3.7, "plating_risk": 0.01},
        {"SOC": 0.5, "SOH": 0.70, "T": 345.0, "delta_SEI_m": 2e-7, "V": 3.5, "plating_risk": 0.3},
    ])
    check("Batch: 2 results", len(results) == 2)
    check("Batch has fault_class", all("fault_class" in r for r in results))

    # Anomaly rate
    rate = det.anomaly_rate(0)
    check("Anomaly rate in [0,1]", 0.0 <= rate <= 1.0, f"rate={rate:.3f}")

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    validate()
