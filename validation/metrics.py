"""
validation/metrics.py — standardized SOC-estimation metrics (Phase 1).

One place for every per-trip metric and every cross-trip aggregate used in
the paper, so a number can never mean two different things in two tables.

Conventions
-----------
- SOC arrays are FRACTIONS in [0, 1]; all reported errors are in
  PERCENTAGE POINTS (×100).
- "trip" = one held-out evaluation segment (a BMW/VED trip, a Deng charging
  session, or a CALCE/UMich lab window).
- t_s is seconds from segment start (loaders emit t_s[0] == 0 after their
  own windowing); non-uniform dt is allowed everywhere.

Convergence definitions (threshold = 5 percentage points, 0.05)
---------------------------------------------------------------
STRICT   : first time t_c such that |SOC_est − SOC_true| < threshold at t_c
           and at EVERY later sample of the trip. This is the primary
           definition reported in the paper.
HOLD-600 : first time t_c such that the error stays below threshold
           continuously for ≥ 600 s from t_c, or — when fewer than 600 s of
           trip remain after t_c — until the end of the trip ("whichever the
           trip length supports"). Kept as a sensitivity variant: unlike
           STRICT it forgives a late re-divergence after 10 min of holding.
LEGACY   : the project's pre-Phase-1 definition (validate_generic.py
           `_ekf_convergence_time`): first sample index i such that the next
           30 CONSECUTIVE SAMPLES are below threshold. Sample-count based,
           therefore dt-dependent (30 samples ≈ 30 s on 1 s BMW data but
           240 s on 8 s Deng data), and it does NOT require holding to the
           end of the trip. Reported alongside the new definitions so the
           shift between old and new numbers is fully visible; ported
           verbatim, including its two edge quirks (a trip with ≤ 30 samples
           can never converge; a window may not START within the last 30
           samples).

The post-convergence window for RMSE/MAE/max-error is defined by STRICT t_c.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

CONV_THRESHOLD = 0.05      # 5 percentage points, SOC fraction units
HOLD_WINDOW_S = 600.0      # HOLD-600 sensitivity variant
LEGACY_N_CONSECUTIVE = 30  # samples, pre-Phase-1 definition
RECOVERED_END_ERR_PP = 10.0  # "recovered" tier: |err| at trip end ≤ 10 pp

# Standard per-trip outcome tiers (2026-07-19 review):
#   converged — STRICT convergence reached (t_conv_strict_s not None)
#   recovered — not strictly converged, but |err| at the LAST sample ≤ 10 pp:
#               the filter pulled the +20 pp wrong init back most of the way
#               even if it never entered-and-held the 5 pp band
#   diverged  — neither: trip ends ≥ 10 pp wrong
# Mutually exclusive; precedence converged > recovered > diverged.
OUTCOME_CONVERGED = "converged"
OUTCOME_RECOVERED = "recovered"
OUTCOME_DIVERGED = "diverged"


# ─────────────────────────────────────────────────────────────────────────────
# Per-trip error metrics
# ─────────────────────────────────────────────────────────────────────────────

def _check(soc_est: np.ndarray, soc_true: np.ndarray) -> None:
    if len(soc_est) != len(soc_true):
        raise ValueError(f"length mismatch: est {len(soc_est)} vs true {len(soc_true)}")
    if len(soc_est) == 0:
        raise ValueError("empty trajectory")


def rmse_pct(soc_est: np.ndarray, soc_true: np.ndarray) -> float:
    """SOC RMSE in percentage points: sqrt(mean((est − true)²)) × 100."""
    _check(soc_est, soc_true)
    return float(np.sqrt(np.mean((np.asarray(soc_est, dtype=np.float64)
                                  - np.asarray(soc_true, dtype=np.float64)) ** 2))) * 100.0


def mae_pct(soc_est: np.ndarray, soc_true: np.ndarray) -> float:
    """SOC MAE in percentage points."""
    _check(soc_est, soc_true)
    return float(np.mean(np.abs(np.asarray(soc_est, dtype=np.float64)
                                - np.asarray(soc_true, dtype=np.float64)))) * 100.0


def max_abs_err_pct(soc_est: np.ndarray, soc_true: np.ndarray) -> float:
    """Maximum absolute SOC error in percentage points."""
    _check(soc_est, soc_true)
    return float(np.max(np.abs(np.asarray(soc_est, dtype=np.float64)
                               - np.asarray(soc_true, dtype=np.float64)))) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Convergence times
# ─────────────────────────────────────────────────────────────────────────────

def convergence_time_strict(
    t_s: np.ndarray,
    soc_est: np.ndarray,
    soc_true: np.ndarray,
    threshold: float = CONV_THRESHOLD,
) -> Optional[float]:
    """First t_c with |err| < threshold from t_c to the END of the trip.

    Returns None if the final sample is not below threshold (never converged).
    """
    _check(soc_est, soc_true)
    err = np.abs(np.asarray(soc_est, dtype=np.float64)
                 - np.asarray(soc_true, dtype=np.float64))
    bad = np.flatnonzero(err >= threshold)
    if len(bad) == 0:
        return float(t_s[0])
    last_bad = bad[-1]
    if last_bad == len(err) - 1:
        return None
    return float(t_s[last_bad + 1])


def convergence_time_hold(
    t_s: np.ndarray,
    soc_est: np.ndarray,
    soc_true: np.ndarray,
    threshold: float = CONV_THRESHOLD,
    hold_s: float = HOLD_WINDOW_S,
) -> Optional[float]:
    """First t_c whose below-threshold run lasts ≥ hold_s, or reaches the end
    of the trip (for trips/tails shorter than hold_s). Sensitivity variant."""
    _check(soc_est, soc_true)
    t = np.asarray(t_s, dtype=np.float64)
    err = np.abs(np.asarray(soc_est, dtype=np.float64)
                 - np.asarray(soc_true, dtype=np.float64))
    below = err < threshold
    n = len(below)
    i = 0
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and below[j + 1]:
            j += 1
        # run of consecutive below-threshold samples: indices [i, j]
        if (t[j] - t[i]) >= hold_s or j == n - 1:
            return float(t[i])
        i = j + 1
    return None


def convergence_time_legacy(
    t_s: np.ndarray,
    soc_est: np.ndarray,
    soc_true: np.ndarray,
    threshold: float = CONV_THRESHOLD,
    n_consecutive: int = LEGACY_N_CONSECUTIVE,
) -> Optional[float]:
    """Verbatim port of validate_generic._ekf_convergence_time (pre-Phase-1):
    first index i whose next `n_consecutive` samples are all below threshold.
    dt-dependent; does not require holding to end of trip; quirks preserved."""
    _check(soc_est, soc_true)
    diff = np.abs(np.asarray(soc_est, dtype=np.float64)
                  - np.asarray(soc_true, dtype=np.float64))
    for i in range(len(diff) - n_consecutive):
        if np.all(diff[i: i + n_consecutive] < threshold):
            return float(t_s[i])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-trip bundle
# ─────────────────────────────────────────────────────────────────────────────

def trip_metrics(
    t_s: np.ndarray,
    soc_est: np.ndarray,
    soc_true: np.ndarray,
    threshold: float = CONV_THRESHOLD,
    hold_s: float = HOLD_WINDOW_S,
) -> Dict[str, Optional[float]]:
    """All standardized per-trip metrics for one method on one trip.

    Keys:
      rmse_full_pct, mae_full_pct, maxerr_full_pct          — full trip
      rmse_postconv_pct, mae_postconv_pct, maxerr_postconv_pct
          — over t ≥ t_conv_strict_s; None when never strictly converged
      t_conv_strict_s, t_conv_hold_s, t_conv_legacy_s       — None = not converged
      err_end_pct       — |err| at the trip's LAST sample (recovered tier)
      min_abs_err_pct   — smallest |err| reached anywhere in the trip
                          (< 5 pp means the estimate at least touched the band)
      outcome           — converged / recovered / diverged (see module consts)
      duration_s, n_samples
    """
    t = np.asarray(t_s, dtype=np.float64)
    est = np.asarray(soc_est, dtype=np.float64)
    tru = np.asarray(soc_true, dtype=np.float64)
    _check(est, tru)

    abs_err = np.abs(est - tru)
    t_strict = convergence_time_strict(t, est, tru, threshold)
    err_end_pp = float(abs_err[-1]) * 100.0
    if t_strict is not None:
        outcome = OUTCOME_CONVERGED
    elif err_end_pp <= RECOVERED_END_ERR_PP:
        outcome = OUTCOME_RECOVERED
    else:
        outcome = OUTCOME_DIVERGED

    out: Dict[str, Optional[float]] = {
        "rmse_full_pct":   rmse_pct(est, tru),
        "mae_full_pct":    mae_pct(est, tru),
        "maxerr_full_pct": max_abs_err_pct(est, tru),
        "t_conv_strict_s": t_strict,
        "t_conv_hold_s":   convergence_time_hold(t, est, tru, threshold, hold_s),
        "t_conv_legacy_s": convergence_time_legacy(t, est, tru, threshold),
        "err_end_pct":     err_end_pp,
        "min_abs_err_pct": float(np.min(abs_err)) * 100.0,
        "outcome":         outcome,
        "duration_s":      float(t[-1] - t[0]),
        "n_samples":       float(len(t)),
    }
    if t_strict is not None:
        m = t >= t_strict
        out["rmse_postconv_pct"]   = rmse_pct(est[m], tru[m])
        out["mae_postconv_pct"]    = mae_pct(est[m], tru[m])
        out["maxerr_postconv_pct"] = max_abs_err_pct(est[m], tru[m])
    else:
        out["rmse_postconv_pct"] = None
        out["mae_postconv_pct"] = None
        out["maxerr_postconv_pct"] = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-trip aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _dist(vals: List[float]) -> Dict[str, Optional[float]]:
    """median / q25 / q75 / mean / n of a list (Nones already removed)."""
    if not vals:
        return {"median": None, "q25": None, "q75": None, "mean": None, "n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "q25":    float(np.percentile(a, 25)),
        "q75":    float(np.percentile(a, 75)),
        "mean":   float(np.mean(a)),
        "n":      int(len(a)),
    }


def aggregate_trips(
    per_trip: List[Dict[str, Optional[float]]],
    censor_t_s: Optional[float] = None,
) -> Dict:
    """Aggregate trip_metrics() dicts across trips.

    Distribution metrics (median/IQR primary, mean kept for comparability
    with the pre-Phase-1 mean-based headline numbers) for each error metric;
    convergence RATE (fraction of trips converged) and median
    time-to-converge among converged trips, for each of the three
    convergence definitions; outcome-tier rates (converged/recovered/
    diverged, see module constants).

    censor_t_s — optional censoring-aware view (2026-07-19 review): a trip
    that did NOT strictly converge AND is shorter than censor_t_s (the
    dataset's characteristic convergence timescale, typically the EKF's
    median strict t_conv) is CENSORED — too short to have shown convergence
    — rather than counted as a failure. Adds:
      n_censored, conv_rate_strict_censaware = n_converged / (n − n_censored)
    (None when every trip is censored). Raw rates are always kept alongside.
    """
    def _vals(key):
        return [m[key] for m in per_trip if m.get(key) is not None
                and np.isfinite(m[key])]

    out: Dict = {"n_trips": len(per_trip)}
    for key in ("rmse_full_pct", "mae_full_pct", "maxerr_full_pct",
                "rmse_postconv_pct", "mae_postconv_pct", "maxerr_postconv_pct",
                "err_end_pct", "min_abs_err_pct"):
        out[key] = _dist(_vals(key))

    n = max(len(per_trip), 1)
    for name, key in (("strict", "t_conv_strict_s"),
                      ("hold", "t_conv_hold_s"),
                      ("legacy", "t_conv_legacy_s")):
        conv = _vals(key)
        out[f"conv_rate_{name}"] = len(conv) / n
        out[f"t_conv_{name}_median_s"] = float(np.median(conv)) if conv else None

    # Outcome tiers (trips missing 'outcome' — pre-extension dicts — ignored)
    outcomes = [m.get("outcome") for m in per_trip if m.get("outcome")]
    n_out = max(len(outcomes), 1)
    for tier in (OUTCOME_CONVERGED, OUTCOME_RECOVERED, OUTCOME_DIVERGED):
        out[f"rate_{tier}"] = outcomes.count(tier) / n_out

    # Censoring-aware strict convergence rate
    if censor_t_s is not None:
        n_conv = sum(1 for m in per_trip if m.get("t_conv_strict_s") is not None)
        n_censored = sum(
            1 for m in per_trip
            if m.get("t_conv_strict_s") is None
            and m.get("duration_s") is not None
            and m["duration_s"] < censor_t_s
        )
        out["censor_t_s"] = float(censor_t_s)
        out["n_censored"] = n_censored
        denom = len(per_trip) - n_censored
        out["conv_rate_strict_censaware"] = (n_conv / denom) if denom > 0 else None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Standing table footnotes (decision (c), 2026-07-19 review)
# ─────────────────────────────────────────────────────────────────────────────

TABLE_FOOTNOTES = [
    "Calibration split: 10% of segments per vehicle for the fleet datasets "
    "(BMW i3, Deng, VED); 40% per cell for CALCE and 30% per module for "
    "UMich/Ford — data-size-driven exceptions, decided before evaluation "
    "(see data/soc_baseline_benchmark_calce.py / _module.py).",
    "All SOC errors in percentage points; every estimator (including "
    "coulomb counting) starts from the same deliberately wrong initial SOC.",
    "Initial SOC is clipped to [2%, 98%], so trips starting near a rail "
    "receive LESS than the nominal offset; this materially lowers the "
    "coulomb baseline's aggregate stress-test RMSE in 23 of 45 "
    "dataset×offset cells (e.g. +20 pp: 43% of BMW and 51% of VED trips "
    "clipped; clipped-trip coulomb RMSE ~2–12 pp vs ~20 pp unclipped) — a "
    "protocol artifact, not estimator skill; medians are far less affected "
    "than means. Full grid: results/coulomb_clipping_diagnostic.csv.",
]


def footnote_lines(prefix: str = "# ") -> List[str]:
    """TABLE_FOOTNOTES formatted for embedding in CSV (# comments) or
    markdown (prefix='> ')."""
    return [f"{prefix}{note}" for note in TABLE_FOOTNOTES]
