#!/usr/bin/env python3
"""
data/retune_theta_wide.py — post-sweep θ re-tuning on an ADAPTIVELY WIDENED
grid (2026-07-19 review), scalar-bias row re-runs, table regeneration, and
the bias-identifiability check.

Why: the first θ grid (Q_θ ∈ {1e-10…1e-6} V²/s, R_θ ∈ {1e-6…1e-4} V²)
produced edge selections on 4 of 5 fleets — the baseline might not have
been at its honest best. This script:

  1. Re-tunes (Q_θ, R_θ) per fleet on the CALIBRATION split only, +20 pp
     protocol, starting from a wide grid
     (Q_θ 1e-14…1e-2 V²/s × R_θ 1e-10…1e-2 V²) and EXTENDING any edge the
     argmin lands on by two decades, repeatedly, until the argmin is
     interior or a hard bound (Q_θ, R_θ ∈ [1e-18, 1]) is reached (bound
     hits are logged). Ties keep the smallest Q_θ, then smallest R_θ.
     Full grid + selections → results/theta_tuning_wide_<stamp>.json.
  2. For every fleet whose selection changed vs the sweep's values
     (read from the sweep meta JSON), re-runs rbc_dekf (all 9 offsets) and
     rbc_coupled (+20 pp) on the held-out segments.
  3. Regenerates offset_sweep.csv / baseline_comparison.csv /
     nominal_accuracy.csv+.md / outcome_tiers.csv by merging the re-run
     rows into the sweep's per-trip dump (my_ekf / coulomb / ocv_lookup
     rows and their PASSED abort-on-mismatch cross-checks are untouched —
     θ does not enter those methods).
  4. Identifiability pass (final tuned params, +20 pp): per held-out
     segment, records final bias θ_end, SIGNED trip-end SOC error, and the
     segment-mean |∂OCV/∂SOC|; per dataset, the correlation of θ_end with
     −slope·err_end (the "θ absorbs SOC error" signature) →
     results/identifiability_check.csv.

One command: venv/bin/python -u data/retune_theta_wide.py
"""
from __future__ import annotations

import csv
import glob
import json
import multiprocessing as mp
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data.run_offset_sweep as ros
from validation.metrics import aggregate_trips, footnote_lines

RESULTS_DIR = ROOT / "results"
HEADLINE = ros.HEADLINE_OFFSET

Q_GRID0 = [1e-14, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
R_GRID0 = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
HARD_MIN, HARD_MAX = 1e-18, 1.0
EXTEND_DECADES = 2

N_WORKERS = max(1, (os.cpu_count() or 4) - 2)
CTX = mp.get_context("fork")


# ─────────────────────────────────────────────────────────────────────────────
# Wide adaptive tuning
# ─────────────────────────────────────────────────────────────────────────────

def _tune_combo(args) -> Tuple[float, float, Optional[float], int]:
    fleet, q, r = args
    fl = ros._FLEETS[fleet]
    rmses = []
    for seg_df, vid, cfg in fl["cal_items"][:ros.THETA_TUNE_MAX_SEGS]:
        try:
            filt = ros._make_filter("rbc_dekf", cfg, fl["cal"], fl["ocv_fn"],
                                    HEADLINE, (q, r))
            t, est, tru, _ = ros.run_lean_traj(seg_df, cfg, filt, HEADLINE)
            rmses.append(float(np.sqrt(np.mean((est - tru) ** 2))) * 100.0)
        except Exception:
            pass
    return q, r, (float(np.mean(rmses)) if rmses else None), len(rmses)


def _extend(grid: List[float], direction: str) -> Tuple[List[float], bool]:
    """Extend grid by EXTEND_DECADES decades below/above; True if bounded."""
    g = sorted(grid)
    added, bounded = [], False
    for k in range(1, EXTEND_DECADES + 1):
        v = g[0] * 10.0 ** (-k) if direction == "down" else g[-1] * 10.0 ** k
        if v < HARD_MIN or v > HARD_MAX:
            bounded = True
            break
        added.append(v)
    return sorted(set(g + added)), bounded


def tune_wide(fleet: str) -> Dict:
    qgrid, rgrid = list(Q_GRID0), list(R_GRID0)
    evaluated: Dict[Tuple[float, float], Tuple[Optional[float], int]] = {}
    bound_hits: List[str] = []
    rounds = 0
    while True:
        rounds += 1
        todo = [(fleet, q, r) for q in qgrid for r in rgrid
                if (q, r) not in evaluated]
        if todo:
            with CTX.Pool(N_WORKERS) as pool:
                for q, r, m, n in pool.imap_unordered(_tune_combo, todo,
                                                      chunksize=1):
                    evaluated[(q, r)] = (m, n)
        valid = {k: v for k, v in evaluated.items() if v[0] is not None}
        if not valid:
            return {"fleet": fleet, "error": "no valid combos"}
        # argmin; ties → smallest Q, then smallest R
        best_qr = min(valid, key=lambda k: (valid[k][0], k[0], k[1]))
        bq, br = best_qr
        grew = False
        for name, grid, val in (("Q", qgrid, bq), ("R", rgrid, br)):
            g = sorted(grid)
            for direction, edge in (("down", g[0]), ("up", g[-1])):
                if val == edge:
                    new_grid, bounded = _extend(g, direction)
                    if bounded:
                        bound_hits.append(f"{name} {direction} bound at {edge:.0e}")
                    if len(new_grid) > len(g):
                        grew = True
                        if name == "Q":
                            qgrid = new_grid
                        else:
                            rgrid = new_grid
        if not grew:
            break
        if rounds > 8:
            bound_hits.append("stopped after 8 extension rounds")
            break
    return {
        "fleet": fleet,
        "chosen_Q_theta_V2_per_s": bq, "chosen_R_theta_V2": br,
        "chosen_mean_cal_rmse_pct": valid[best_qr][0],
        "n_cal_segments_used": min(len(ros._FLEETS[fleet]["cal_items"]),
                                   ros.THETA_TUNE_MAX_SEGS),
        "offset_used": HEADLINE, "rounds": rounds,
        "bound_hits": bound_hits,
        "interior": not any("bound" in b for b in bound_hits),
        "grid": [{"Q_theta_V2_per_s": q, "R_theta_V2": r,
                  "mean_cal_rmse_pct": m, "n_segs": n}
                 for (q, r), (m, n) in sorted(evaluated.items())],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load the sweep's per-trip dump
# ─────────────────────────────────────────────────────────────────────────────

_FLOATS = {"rmse_full_pct", "mae_full_pct", "maxerr_full_pct",
           "rmse_postconv_pct", "t_conv_strict_s", "t_conv_hold_s",
           "t_conv_legacy_s", "err_end_pct", "min_abs_err_pct",
           "applied_offset_pp", "duration_s", "n_samples"}


def load_dump() -> Tuple[Dict, str]:
    paths = sorted(glob.glob(str(RESULTS_DIR / "offset_sweep_per_trip_*.csv")))
    if not paths:
        sys.exit("no offset_sweep_per_trip_*.csv found — run the sweep first")
    path = paths[-1]
    results: Dict[str, Dict[Tuple[str, float], Dict[int, Dict]]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            m: Dict = {}
            for k, v in row.items():
                if k in ("dataset", "method", "offset_pp", "seg_index"):
                    continue
                if v == "":
                    m[k] = None
                elif k in _FLOATS:
                    m[k] = float(v)
                else:
                    m[k] = v
            if m.get("error"):
                m = {"error": m["error"], "vehicle_id": m.get("vehicle_id")}
            else:
                # a None/blank error must not leave an 'error' KEY behind —
                # downstream filters use key membership ("error" in t)
                m.pop("error", None)
            fleet = row["dataset"]
            key = (row["method"], float(row["offset_pp"]) / 100.0)
            results.setdefault(fleet, {}).setdefault(key, {})[int(row["seg_index"])] = m
    # ocv_lookup was dumped at the headline offset only — replicate
    for fleet in results:
        base = results[fleet].get(("ocv_lookup", HEADLINE))
        if base:
            for off in ros.OFFSETS:
                results[fleet][("ocv_lookup", off)] = base
    return results, path


# ─────────────────────────────────────────────────────────────────────────────
# Re-run + regenerate
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, cwd=ROOT,
                                check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    # previous (narrow-grid) selections, from the sweep meta
    metas = sorted(glob.glob(str(RESULTS_DIR / "offset_sweep_meta_*.json")))
    if not metas:
        sys.exit("no offset_sweep_meta_*.json found — run the sweep first")
    sweep_meta = json.loads(Path(metas[-1]).read_text())
    old_theta = {k: (v["chosen_Q_theta_V2_per_s"], v["chosen_R_theta_V2"])
                 for k, v in sweep_meta["theta_tuning"].items()}
    print("Previous (narrow-grid) selections:", old_theta)

    # ── prep fleets ─────────────────────────────────────────────────────────
    for fleet, prep in ros.FLEET_PREPS.items():
        print(f"\n[{fleet}] loading + calibration…")
        eval_items, cal_items, cal = prep()
        ros._FLEETS[fleet] = {"eval": eval_items, "cal_items": cal_items,
                              "cal": cal, "ocv_fn": (cal.ocv_fn if cal else None),
                              "theta_qr": old_theta.get(fleet, (1e-8, 1e-5))}

    # ── wide adaptive tuning ────────────────────────────────────────────────
    tuning, changed = {}, []
    for fleet in ros.FLEET_PREPS:
        print(f"\n[{fleet}] wide adaptive θ tuning…")
        tr = tune_wide(fleet)
        tuning[fleet] = tr
        new_qr = (tr["chosen_Q_theta_V2_per_s"], tr["chosen_R_theta_V2"])
        moved = new_qr != old_theta.get(fleet)
        if moved:
            changed.append(fleet)
        ros._FLEETS[fleet]["theta_qr"] = new_qr
        print(f"[{fleet}] chosen Q_θ={new_qr[0]:.0e} V²/s R_θ={new_qr[1]:.0e} V² "
              f"(cal RMSE {tr['chosen_mean_cal_rmse_pct']:.2f}%, "
              f"rounds={tr['rounds']}, interior={tr['interior']}) "
              f"{'CHANGED vs narrow grid' if moved else 'unchanged'}")
    (RESULTS_DIR / f"theta_tuning_wide_{stamp}.json").write_text(
        json.dumps(tuning, indent=2))
    print(f"\nChanged fleets: {changed or 'none'}")

    # ── load sweep dump, re-run scalar-bias rows for changed fleets ────────
    results, dump_path = load_dump()
    print(f"Loaded sweep per-trip dump: {dump_path}")

    if changed:
        tasks = []
        for fleet in changed:
            n = len(ros._FLEETS[fleet]["eval"])
            for i in range(n):
                for off in ros.OFFSETS:
                    tasks.append((fleet, i, "rbc_dekf", off))
                tasks.append((fleet, i, "rbc_coupled", HEADLINE))
        print(f"Re-running {len(tasks)} scalar-bias evaluations "
              f"on {N_WORKERS} workers…")
        done = 0
        with CTX.Pool(N_WORKERS) as pool:
            for fleet, idx, method, off, m in pool.imap_unordered(
                    ros._worker, tasks, chunksize=16):
                results[fleet].setdefault((method, off), {})[idx] = m
                done += 1
                if done % 2000 == 0:
                    print(f"  {done}/{len(tasks)}")

    # ── regenerate tables (same machinery as the sweep) ────────────────────
    censor_t: Dict[str, Optional[float]] = {}
    for fleet in ros.FLEET_PREPS:
        ek = [t for t in results[fleet][("my_ekf", HEADLINE)].values()
              if t is not None and "error" not in t
              and t.get("t_conv_strict_s") is not None]
        censor_t[fleet] = (float(np.median([t["t_conv_strict_s"] for t in ek]))
                           if ek else None)

    def _trips(fleet, method, off):
        d = results[fleet].get((method, off), {})
        return [d[i] for i in sorted(d)]

    hdr = [f"# regenerated by data/retune_theta_wide.py {stamp} (git {commit}) "
           f"after WIDE-GRID θ re-tune; changed fleets: {changed or 'none'}; "
           f"my_ekf/coulomb/ocv_lookup rows unchanged from the cross-checked sweep",
           "# ocv_lookup is init-independent: identical values replicated "
           "across offset rows by construction",
           "# censoring threshold per dataset = EKF median strict t_conv at +20pp"]

    rows = []
    for fleet in ros.FLEET_PREPS:
        for method in ("my_ekf", "rbc_dekf", "coulomb", "ocv_lookup"):
            for off in ros.OFFSETS:
                rows.append(ros.sweep_row(fleet, method, off,
                                          _trips(fleet, method, off),
                                          censor_t_s=censor_t[fleet]))
    ros._write_csv(RESULTS_DIR / "offset_sweep.csv", hdr, ros.SWEEP_COLS, rows)

    rows20 = [ros.sweep_row(fleet, method, HEADLINE,
                            _trips(fleet, method, HEADLINE),
                            censor_t_s=censor_t[fleet])
              for fleet in ros.FLEET_PREPS for method in ros.ALL_METHODS]
    ros._write_csv(RESULTS_DIR / "baseline_comparison.csv", hdr,
                   ros.SWEEP_COLS, rows20)

    rows0 = [ros.sweep_row(fleet, method, 0.0, _trips(fleet, method, 0.0),
                           censor_t_s=censor_t[fleet])
             for fleet in ros.FLEET_PREPS
             for method in ("my_ekf", "rbc_dekf", "coulomb", "ocv_lookup")]
    ros._write_csv(RESULTS_DIR / "nominal_accuracy.csv",
                   hdr + ["# NOMINAL protocol: correct initial SOC (offset 0), "
                          "P0_soc=(2pp)^2 floor — unlike every other table, "
                          "which uses the adversarial wrong-init protocol"],
                   ros.SWEEP_COLS, rows0)
    md = ["# Nominal accuracy (offset = 0, correct initialization)", "",
          f"Regenerated {stamp} after wide-grid θ re-tune. Median (IQR) "
          f"primary, mean secondary — all pp.", "",
          "| Dataset | Method | n | RMSE med (IQR) | RMSE mean | MAE med "
          "| Converged% | Recovered% | Diverged% |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows0:
        md.append(
            f"| {r['dataset']} | {r['method']} | {r['n_trips']} "
            f"| {ros._f(r['rmse_median'], 2)} ({ros._f(r['rmse_q25'], 2)}–"
            f"{ros._f(r['rmse_q75'], 2)}) | {ros._f(r['rmse_mean'], 2)} "
            f"| {ros._f(r['mae_median'], 2)} "
            f"| {ros._f((r['conv_rate_strict'] or 0) * 100, 1)} "
            f"| {ros._f((r['rate_recovered'] or 0) * 100, 1)} "
            f"| {ros._f((r['rate_diverged'] or 0) * 100, 1)} |")
    md += ["", "> Nominal protocol: correct initial SOC; all other project "
           "tables use the +20 pp (or swept) adversarial wrong-init protocol.",
           *footnote_lines("> ")]
    (RESULTS_DIR / "nominal_accuracy.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {RESULTS_DIR / 'nominal_accuracy.md'}")

    ros._write_csv(
        RESULTS_DIR / "outcome_tiers.csv", hdr,
        ["dataset", "method", "n_trips", "conv_rate_strict", "conv_rate_hold",
         "conv_rate_legacy", "rate_converged", "rate_recovered",
         "rate_diverged", "n_censored", "conv_rate_strict_censaware",
         "censor_t_s"],
        [{**ros.sweep_row(fleet, method, HEADLINE,
                          _trips(fleet, method, HEADLINE),
                          censor_t_s=censor_t[fleet]),
          "rate_converged": aggregate_trips(
              [t for t in _trips(fleet, method, HEADLINE)
               if t is not None and "error" not in t])["rate_converged"],
          "censor_t_s": censor_t[fleet]}
         for fleet in ros.FLEET_PREPS for method in ros.ALL_METHODS])

    # updated per-trip dump for the changed rows
    if changed:
        dump_cols = ["dataset", "method", "offset_pp", "seg_index",
                     "vehicle_id", "rmse_full_pct", "mae_full_pct",
                     "maxerr_full_pct", "rmse_postconv_pct", "t_conv_strict_s",
                     "t_conv_hold_s", "t_conv_legacy_s", "err_end_pct",
                     "min_abs_err_pct", "outcome", "applied_offset_pp",
                     "duration_s", "n_samples", "error"]
        p = RESULTS_DIR / f"offset_sweep_per_trip_retuned_{stamp}.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=dump_cols)
            w.writeheader()
            for fleet in ros.FLEET_PREPS:
                for (method, off), d in sorted(results[fleet].items()):
                    if method == "ocv_lookup" and off != HEADLINE:
                        continue
                    for i in sorted(d):
                        m = d[i] or {}
                        row = {"dataset": fleet, "method": method,
                               "offset_pp": round(off * 100), "seg_index": i}
                        row.update({k: m.get(k) for k in dump_cols if k in m})
                        w.writerow({k: ("" if row.get(k) is None else row.get(k))
                                    for k in dump_cols})
        print(f"Wrote {p}")

    # ── identifiability pass (final tuned params, +20 pp) ──────────────────
    print("\nIdentifiability pass (rbc_dekf @ +20, θ_end vs signed SOC error)…")

    def _ident_worker(task):
        fleet, idx = task
        fl = ros._FLEETS[fleet]
        seg_df, vid, cfg = fl["eval"][idx]
        try:
            filt = ros._make_filter("rbc_dekf", cfg, fl["cal"], fl["ocv_fn"],
                                    HEADLINE, fl["theta_qr"])
            t, est, tru, _ = ros.run_lean_traj(seg_df, cfg, filt, HEADLINE)
            h = 0.005
            ocv = fl["ocv_fn"] if fl["ocv_fn"] is not None else filt._ocv
            s = np.clip(tru, 0.01, 0.99)
            slopes = np.array([(float(ocv(x + h)) - float(ocv(x - h))) / (2 * h)
                               for x in s[:: max(1, len(s) // 200)]])
            return (fleet, idx, {
                "theta_end_mV": float(filt.theta) * 1000.0,
                "soc_err_end_signed_pp": float(est[-1] - tru[-1]) * 100.0,
                "slope_mean_V_per_soc": float(np.mean(np.abs(slopes))),
                "rmse_full_pct": float(np.sqrt(np.mean((est - tru) ** 2))) * 100.0,
            })
        except Exception as exc:
            return (fleet, idx, {"error": str(exc)})

    ident_tasks = [(fleet, i) for fleet in ros.FLEET_PREPS
                   for i in range(len(ros._FLEETS[fleet]["eval"]))]
    ident: Dict[str, List[Dict]] = {fleet: [] for fleet in ros.FLEET_PREPS}
    with CTX.Pool(N_WORKERS) as pool:
        for fleet, idx, m in pool.imap_unordered(_ident_worker, ident_tasks,
                                                 chunksize=16):
            ident[fleet].append(m)

    ident_rows = []
    for fleet in ros.FLEET_PREPS:
        ok = [m for m in ident[fleet] if "error" not in m]
        th = np.array([m["theta_end_mV"] for m in ok])
        er = np.array([m["soc_err_end_signed_pp"] for m in ok])
        sl = np.array([m["slope_mean_V_per_soc"] for m in ok])
        # θ-absorbs-SOC-error signature: θ_end ≈ −slope·err_end
        pred = -sl * (er / 100.0) * 1000.0   # mV
        r = (float(np.corrcoef(th, pred)[0, 1])
             if len(ok) > 2 and np.std(th) > 0 and np.std(pred) > 0 else None)
        ident_rows.append({
            "dataset": fleet, "n": len(ok),
            "mean_abs_docv_dsoc_V": float(np.mean(sl)),
            "chosen_Q_theta_V2_per_s": ros._FLEETS[fleet]["theta_qr"][0],
            "chosen_R_theta_V2": ros._FLEETS[fleet]["theta_qr"][1],
            "median_abs_theta_end_mV": float(np.median(np.abs(th))),
            "corr_theta_vs_minus_slope_err": r,
            "median_signed_err_end_pp": float(np.median(er)),
        })
    ros._write_csv(RESULTS_DIR / "identifiability_check.csv",
                   [f"# generated by data/retune_theta_wide.py {stamp}; "
                    f"rbc_dekf @ +20pp with final tuned (Q_θ,R_θ); "
                    f"corr column = Pearson r of θ_end vs −slope·err_end "
                    f"(the 'θ absorbs SOC error' signature)"],
                   ["dataset", "n", "mean_abs_docv_dsoc_V",
                    "chosen_Q_theta_V2_per_s", "chosen_R_theta_V2",
                    "median_abs_theta_end_mV", "corr_theta_vs_minus_slope_err",
                    "median_signed_err_end_pp"], ident_rows)

    meta = {"script": "data/retune_theta_wide.py", "utc": stamp,
            "git_commit": commit, "changed_fleets": changed,
            "old_theta": old_theta,
            "new_theta": {k: list(ros._FLEETS[k]["theta_qr"])
                          for k in ros.FLEET_PREPS},
            "censor_t_s": censor_t,
            "source_dump": dump_path}
    (RESULTS_DIR / f"retune_meta_{stamp}.json").write_text(
        json.dumps(meta, indent=2))
    print(f"Wrote {RESULTS_DIR / f'retune_meta_{stamp}.json'}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
