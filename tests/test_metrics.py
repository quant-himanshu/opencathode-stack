"""
tests/test_metrics.py — unit tests for validation/metrics.py (Phase 1).

Synthetic trajectories with hand-computable answers. Run with
`pytest tests/test_metrics.py` or directly `python tests/test_metrics.py`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.metrics import (
    CONV_THRESHOLD, OUTCOME_CONVERGED, OUTCOME_DIVERGED, OUTCOME_RECOVERED,
    aggregate_trips, convergence_time_hold, convergence_time_legacy,
    convergence_time_strict, mae_pct, max_abs_err_pct, rmse_pct, trip_metrics,
)


# ── error metrics ────────────────────────────────────────────────────────────

def test_constant_error():
    true = np.full(100, 0.60)
    est = true + 0.10
    assert abs(rmse_pct(est, true) - 10.0) < 1e-12
    assert abs(mae_pct(est, true) - 10.0) < 1e-12
    assert abs(max_abs_err_pct(est, true) - 10.0) < 1e-12


def test_rmse_vs_mae_mixed_error():
    # errors: half +0.10, half −0.02 → MAE = 6%, RMSE = sqrt(0.0052) ≈ 7.2111%
    true = np.full(10, 0.5)
    est = true + np.array([0.10] * 5 + [-0.02] * 5)
    assert abs(mae_pct(est, true) - 6.0) < 1e-12
    assert abs(rmse_pct(est, true) - 100 * np.sqrt(0.0052)) < 1e-12
    assert abs(max_abs_err_pct(est, true) - 10.0) < 1e-12


def test_length_mismatch_raises():
    try:
        rmse_pct(np.zeros(3), np.zeros(4))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ── strict convergence ───────────────────────────────────────────────────────

def test_strict_simple_crossing():
    # error 0.20 for t<50, 0.01 afterwards → t_c = 50
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    err = np.where(t < 50, 0.20, 0.01)
    tc = convergence_time_strict(t, true + err, true)
    assert tc == 50.0


def test_strict_ignores_transient_dip():
    # dips below threshold at t=10..19, re-diverges, settles from t=60
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    err = np.full(100, 0.20)
    err[10:20] = 0.01
    err[60:] = 0.01
    tc = convergence_time_strict(t, true + err, true)
    assert tc == 60.0  # the dip must NOT count


def test_strict_never_converged():
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    est = true + 0.20
    assert convergence_time_strict(t, est, true) is None


def test_strict_diverges_at_last_sample():
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    err = np.full(100, 0.01)
    err[-1] = 0.20
    assert convergence_time_strict(t, true + err, true) is None


def test_strict_converged_from_start():
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    est = true + 0.01
    assert convergence_time_strict(t, est, true) == 0.0


def test_strict_boundary_exact_threshold_not_converged():
    # |err| exactly == threshold does not count as converged (strict <)
    t = np.arange(0, 10, 1.0)
    true = np.full(10, 0.5)
    est = true + CONV_THRESHOLD
    assert convergence_time_strict(t, est, true) is None


# ── hold-600 convergence ─────────────────────────────────────────────────────

def test_hold_forgives_late_redivergence():
    # 1 s samples: below threshold t=100..800 (700 s ≥ 600), re-diverges after.
    # HOLD-600 → 100; STRICT → 1500 (the final settle).
    t = np.arange(0, 2000, 1.0)
    true = np.full(2000, 0.5)
    err = np.full(2000, 0.20)
    err[100:801] = 0.01
    err[1500:] = 0.01
    est = true + err
    assert convergence_time_hold(t, est, true) == 100.0
    assert convergence_time_strict(t, est, true) == 1500.0


def test_hold_short_run_not_counted():
    # below threshold only 300 s mid-trip (not trip-end) → no hold conv from it
    t = np.arange(0, 2000, 1.0)
    true = np.full(2000, 0.5)
    err = np.full(2000, 0.20)
    err[100:401] = 0.01
    est = true + err
    assert convergence_time_hold(t, est, true) is None


def test_hold_trip_end_shorter_than_window():
    # trip is 400 s total, converges at 200 and holds to end (200 s < 600 s):
    # "whichever the trip length supports" → counts
    t = np.arange(0, 400, 1.0)
    true = np.full(400, 0.5)
    err = np.where(t < 200, 0.20, 0.01)
    assert convergence_time_hold(t, true + err, true) == 200.0


# ── legacy convergence (verbatim port) ───────────────────────────────────────

def test_legacy_matches_original_implementation():
    # original: first i with next 30 consecutive samples below threshold
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    err = np.full(100, 0.20)
    err[40:] = 0.01           # below from index 40 on
    est = true + err
    assert convergence_time_legacy(t, est, true) == 40.0


def test_legacy_counts_transient_30_sample_dip_strict_does_not():
    # 35-sample dip mid-trip then re-divergence to the end:
    # legacy converges at the dip; strict never converges.
    t = np.arange(0, 200, 1.0)
    true = np.full(200, 0.5)
    err = np.full(200, 0.20)
    err[50:85] = 0.01
    est = true + err
    assert convergence_time_legacy(t, est, true) == 50.0
    assert convergence_time_strict(t, est, true) is None


def test_legacy_short_trip_never_converges():
    # ≤30 samples → range(len-30) empty → None even if perfectly converged
    t = np.arange(0, 30, 1.0)
    true = np.full(30, 0.5)
    est = true.copy()
    assert convergence_time_legacy(t, est, true) is None
    assert convergence_time_strict(t, est, true) == 0.0


def test_legacy_is_dt_dependent_hold_is_not():
    # same 240 s of real converged time; 1 s sampling → legacy fires,
    # 8 s sampling (30 samples = 240 s exactly at the tail edge) → legacy
    # cannot START a window in the last 30 samples → None. Hold-600/strict
    # treat both identically (time-based).
    true_frac = 0.5
    for dt, legacy_expected in ((1.0, 760.0), (8.0, None)):
        t = np.arange(0.0, 1000.0, dt)
        n = len(t)
        true = np.full(n, true_frac)
        err = np.where(t < 760.0, 0.20, 0.01)
        est = true + err
        got = convergence_time_legacy(t, est, true)
        assert got == legacy_expected, (dt, got)
        assert convergence_time_strict(t, est, true) == 760.0


# ── trip_metrics bundle ──────────────────────────────────────────────────────

def test_trip_metrics_postconv_window():
    # err 0.20 for t<50, 0.01 after → post-conv RMSE/MAE/max = 1%
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    err = np.where(t < 50, 0.20, 0.01)
    m = trip_metrics(t, true + err, true)
    assert m["t_conv_strict_s"] == 50.0
    assert abs(m["rmse_postconv_pct"] - 1.0) < 1e-9
    assert abs(m["mae_postconv_pct"] - 1.0) < 1e-9
    assert abs(m["maxerr_postconv_pct"] - 1.0) < 1e-9
    # full-trip RMSE = sqrt(0.5·0.2² + 0.5·0.01²)·100
    assert abs(m["rmse_full_pct"] - 100 * np.sqrt(0.5 * 0.04 + 0.5 * 1e-4)) < 1e-9
    assert m["duration_s"] == 99.0
    assert m["n_samples"] == 100


def test_trip_metrics_never_converged_postconv_none():
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    m = trip_metrics(t, true + 0.20, true)
    assert m["t_conv_strict_s"] is None
    assert m["rmse_postconv_pct"] is None


# ── aggregation ──────────────────────────────────────────────────────────────

def test_aggregate_known_distribution():
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    trips = []
    # five trips, constant errors 1..5% → rmse list [1,2,3,4,5]
    for e in (0.01, 0.02, 0.03, 0.04, 0.05):
        # add tiny margin below threshold for the 5% one? 0.05 == threshold →
        # that trip never converges strictly. Deliberate: tests rate 4/5.
        trips.append(trip_metrics(t, true + e, true))
    agg = aggregate_trips(trips)
    assert agg["n_trips"] == 5
    d = agg["rmse_full_pct"]
    assert abs(d["median"] - 3.0) < 1e-9
    assert abs(d["mean"] - 3.0) < 1e-9
    assert abs(d["q25"] - 2.0) < 1e-9
    assert abs(d["q75"] - 4.0) < 1e-9
    assert d["n"] == 5
    # convergence: errors 1–4% converge at t=0; the 5%-error trip does not
    assert abs(agg["conv_rate_strict"] - 0.8) < 1e-12
    assert agg["t_conv_strict_median_s"] == 0.0
    # one diverged trip must not destroy the median (it WOULD destroy a mean
    # of, e.g., time-to-converge if Nones were coded as inf — they are not)


def test_aggregate_empty_and_all_none():
    agg = aggregate_trips([])
    assert agg["n_trips"] == 0
    assert agg["rmse_full_pct"]["median"] is None
    assert agg["conv_rate_strict"] == 0.0
    assert agg["t_conv_strict_median_s"] is None


def test_outcome_converged():
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    err = np.where(t < 50, 0.20, 0.01)
    m = trip_metrics(t, true + err, true)
    assert m["outcome"] == OUTCOME_CONVERGED
    assert abs(m["err_end_pct"] - 1.0) < 1e-9
    assert abs(m["min_abs_err_pct"] - 1.0) < 1e-9


def test_outcome_recovered():
    # never below 5 pp (min err 7 pp) but ends at 7 pp ≤ 10 pp → recovered
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    err = np.linspace(0.20, 0.07, 100)
    m = trip_metrics(t, true + err, true)
    assert m["t_conv_strict_s"] is None
    assert m["outcome"] == OUTCOME_RECOVERED
    assert abs(m["err_end_pct"] - 7.0) < 1e-9
    assert abs(m["min_abs_err_pct"] - 7.0) < 1e-9


def test_outcome_recovered_boundary_10pp_inclusive():
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    est = true + 0.10          # constant exactly 10 pp
    m = trip_metrics(t, est, true)
    assert m["outcome"] == OUTCOME_RECOVERED  # ≤ 10 pp is inclusive


def test_outcome_diverged():
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    m = trip_metrics(t, true + 0.20, true)
    assert m["outcome"] == OUTCOME_DIVERGED
    assert abs(m["err_end_pct"] - 20.0) < 1e-9


def test_outcome_touched_band_but_rediverged_to_end_is_diverged():
    # dips into the 5 pp band mid-trip, ends 20 pp wrong → diverged
    # (min_abs_err records the touch)
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    err = np.full(100, 0.20)
    err[40:60] = 0.01
    m = trip_metrics(t, true + err, true)
    assert m["outcome"] == OUTCOME_DIVERGED
    assert abs(m["min_abs_err_pct"] - 1.0) < 1e-9


def test_aggregate_outcome_rates():
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    trips = [
        trip_metrics(t, true + 0.01, true),                       # converged
        trip_metrics(t, true + np.linspace(0.2, 0.07, 100), true),  # recovered
        trip_metrics(t, true + 0.20, true),                       # diverged
        trip_metrics(t, true + 0.20, true),                       # diverged
    ]
    agg = aggregate_trips(trips)
    assert abs(agg["rate_converged"] - 0.25) < 1e-12
    assert abs(agg["rate_recovered"] - 0.25) < 1e-12
    assert abs(agg["rate_diverged"] - 0.50) < 1e-12


def test_censoring_aware_rate():
    true_long = np.full(1000, 0.5)
    t_long = np.arange(0, 1000, 1.0)
    true_short = np.full(100, 0.5)
    t_short = np.arange(0, 100, 1.0)
    trips = [
        # converges at t=400 (duration 999)
        trip_metrics(t_long, true_long + np.where(t_long < 400, 0.2, 0.01), true_long),
        # long, never converges (duration 999) → genuine failure
        trip_metrics(t_long, true_long + 0.2, true_long),
        # short, never converges (duration 99 < censor 400) → censored
        trip_metrics(t_short, true_short + 0.2, true_short),
    ]
    agg = aggregate_trips(trips, censor_t_s=400.0)
    assert agg["n_censored"] == 1
    # raw: 1/3; censoring-aware: 1 converged / (3 − 1 censored) = 0.5
    assert abs(agg["conv_rate_strict"] - 1 / 3) < 1e-12
    assert abs(agg["conv_rate_strict_censaware"] - 0.5) < 1e-12
    # converged trips are never censored regardless of duration
    agg2 = aggregate_trips(trips[:1], censor_t_s=1e9)
    assert agg2["n_censored"] == 0
    assert agg2["conv_rate_strict_censaware"] == 1.0


def test_censoring_all_censored_gives_none():
    t = np.arange(0, 100, 1.0)
    true = np.full(100, 0.5)
    agg = aggregate_trips([trip_metrics(t, true + 0.2, true)], censor_t_s=1e9)
    assert agg["n_censored"] == 1
    assert agg["conv_rate_strict_censaware"] is None


def _run_all():
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {name}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
