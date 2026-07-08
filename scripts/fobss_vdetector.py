"""
FOBSS V-detector second-dataset validation.

Runs NSA variable-radius LOO V-detector on the FOBSS dataset (44 series cells,
KIT test bench). Compares V-detector weak-cell identification against the trivial
baseline (argmin mean V_norm) across all discharge experiments.

Pre-registration: data/fobss_vdetector_preregistration.json (commit 9e1598e)
Dataset:         doi:10.35097/1174, CC BY 4.0, RADAR4KIT
"""

import os
import sys
import json
import argparse
import zipfile
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
from collections import Counter

# ── constants matching pre-registration ──────────────────────────────────────
HC_THRESHOLD_A = 5.0      # |I| > 5A = high-current
HC_CAP         = 20_000   # max HC rows (same as BattGP)
MIN_HC_ROWS    = 200       # minimum HC rows to include experiment
POWER_GATE_RATIO = 2.0     # gap_median / gap_p95_pairwise >= 2.0 to pass
N_CELLS        = 44
N_SLAVES       = 4
CELLS_PER_SLAVE = 11
SEEDS          = [42, 43, 44, 45]

# Exclude these experiment type prefixes
EXCLUDED_PREFIXES = ["Ri ", "osc", "voc_soc", "current_ramp"]

# ── NSA variable-radius LOO (same algorithm as battgp_vdetector.py) ─────────

def _train_nsa(X_train, seed):
    """Variable-radius NSA: each detector covers the gap to nearest non-self point."""
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    idx = rng.permutation(n)
    X = X_train[idx]
    detectors = []
    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        dists[i] = np.inf
        r = dists.min()
        detectors.append((X[i].copy(), r))
    return detectors


def _novelty_score(x, detectors):
    """Fraction of detectors that do NOT cover x (higher = more novel)."""
    covered = sum(1 for c, r in detectors if np.linalg.norm(x - c) <= r)
    return 1.0 - covered / len(detectors)


def loo_novelty(X, seed):
    """LOO: train on all-but-i, score x_i. Return array of novelty scores."""
    scores = np.zeros(len(X))
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False
        dets = _train_nsa(X[mask], seed)
        scores[i] = _novelty_score(X[i], dets)
    return scores


# ── data loading ─────────────────────────────────────────────────────────────

def load_slave_voltages(zf, exp_dir):
    """Load all 4 slave voltage CSVs; return DataFrame[time, cell_0..cell_43]."""
    frames = []
    for s in range(N_SLAVES):
        path = f"{exp_dir}/cells/Slave_{s}_Cell_Voltages.csv"
        try:
            raw = zf.read(path).decode("utf-8", errors="replace")
        except KeyError:
            return None
        lines = [l for l in raw.splitlines() if not l.startswith("#")]
        if not lines:
            return None
        df = pd.read_csv(StringIO("\n".join(lines)), sep=";", header=None)
        # col 0 = time, cols 1..11 = cell voltages
        df.columns = ["time"] + [f"cell_{s * CELLS_PER_SLAVE + c}" for c in range(CELLS_PER_SLAVE)]
        frames.append(df.set_index("time"))
    if len(frames) != N_SLAVES:
        return None
    merged = pd.concat(frames, axis=1).reset_index()
    merged = merged.sort_values("time").reset_index(drop=True)
    return merged


def load_current(zf, exp_dir):
    """Load Battery_Current.csv; return Series indexed by time."""
    path = f"{exp_dir}/battery/Battery_Current.csv"
    try:
        raw = zf.read(path).decode("utf-8", errors="replace")
    except KeyError:
        return None
    lines = [l for l in raw.splitlines() if not l.startswith("#")]
    if not lines:
        return None
    df = pd.read_csv(StringIO("\n".join(lines)), sep=";", header=None,
                     names=["time", "current"])
    return df.set_index("time")["current"]


def merge_current_voltage(v_df, i_series):
    """Merge on nearest time; tolerance = 0.5s (data is ~0.1s resolution)."""
    v_df = v_df.copy()
    v_df["current"] = np.interp(v_df["time"], i_series.index, i_series.values)
    return v_df


# ── per-experiment analysis ───────────────────────────────────────────────────

def run_experiment(zf, exp_dir, exp_name):
    v_df = load_slave_voltages(zf, exp_dir)
    if v_df is None:
        return None, "no_voltage_data"

    i_series = load_current(zf, exp_dir)
    if i_series is None:
        return None, "no_current_data"

    df = merge_current_voltage(v_df, i_series)

    # High-current rows: discharge = negative current in FOBSS convention (I < -HC_THRESHOLD)
    hc_mask = df["current"] < -HC_THRESHOLD_A
    df_hc = df[hc_mask].copy()

    if len(df_hc) < MIN_HC_ROWS:
        return None, f"insufficient_hc_rows({len(df_hc)})"

    # Subsample if needed
    if len(df_hc) > HC_CAP:
        stride = len(df_hc) // HC_CAP
        df_hc = df_hc.iloc[::stride].copy()

    cell_cols = [f"cell_{i}" for i in range(N_CELLS)]
    V = df_hc[cell_cols].values.astype(float)

    # Per-cell min-max normalisation
    vmin = V.min(axis=0, keepdims=True)
    vmax = V.max(axis=0, keepdims=True)
    denom = np.where(vmax - vmin < 1e-9, 1.0, vmax - vmin)
    V_norm = (V - vmin) / denom

    # Trivial baseline
    mean_v = V_norm.mean(axis=0)
    trivial_weak = int(np.argmin(mean_v))

    # Power gate: compare LOO gap of argmin vs spread of pairwise gaps
    gaps = {}
    for seed in SEEDS:
        # Feature: mean V_norm per cell (scalar), shape (N_CELLS,)
        X = mean_v.reshape(-1, 1)
        scores = loo_novelty(X, seed)
        gaps[seed] = scores

    primary_scores = gaps[SEEDS[0]]
    median_gap = np.median(primary_scores)
    # pairwise deltas between all pairs of cells
    pairwise = []
    for i in range(N_CELLS):
        for j in range(i + 1, N_CELLS):
            pairwise.append(abs(primary_scores[i] - primary_scores[j]))
    noise_ceil = np.percentile(pairwise, 95)
    if noise_ceil < 1e-9:
        return None, "degenerate_scores"
    ratio = median_gap / noise_ceil

    power_pass = ratio >= POWER_GATE_RATIO
    vdet_weak = int(np.argmax(primary_scores))

    result = {
        "exp_name": exp_name,
        "n_hc_rows": int(len(df_hc)),
        "trivial_weak_cell": trivial_weak,
        "trivial_mean_v": float(mean_v[trivial_weak]),
        "vdet_weak_cell": vdet_weak,
        "vdet_primary_score": float(primary_scores[vdet_weak]),
        "power_gate_ratio": float(ratio),
        "power_gate_pass": bool(power_pass),
        "agree": bool(trivial_weak == vdet_weak),
        "per_cell_mean_v": mean_v.tolist(),
        "per_cell_score": primary_scores.tolist(),
        "skip_reason": None
    }
    return result, None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="/tmp/fobss_data/data",
                        help="Path to extracted FOBSS data/ directory")
    parser.add_argument("--out", default="data/fobss_vdetector_results.json")
    args = parser.parse_args()

    data_dir = Path(args.zip)
    if not data_dir.exists():
        sys.exit(f"Data dir not found: {data_dir}")

    # Collect all experiment directories (must have Battery_Current.csv)
    exp_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "battery" / "Battery_Current.csv").exists()
    ])

    print(f"Found {len(exp_dirs)} experiment directories.")

    results = []
    skipped = []

    for exp_path in exp_dirs:
        exp_name = exp_path.name

        # Exclusion filter
        skip = any(exp_name.startswith(p) for p in EXCLUDED_PREFIXES)
        if skip:
            skipped.append({"exp_name": exp_name, "reason": "excluded_type"})
            print(f"  SKIP (excluded type): {exp_name}")
            continue

        # Also skip experiments with no discharge (positive-only current)
        # We'll detect this after loading
        print(f"  Processing: {exp_name}")

        # Load directly from directory (already extracted)
        v_df = None
        frames = []
        for s in range(N_SLAVES):
            vpath = exp_path / "cells" / f"Slave_{s}_Cell_Voltages.csv"
            if not vpath.exists():
                frames = []
                break
            raw = vpath.read_text(errors="replace")
            lines = [l for l in raw.splitlines() if not l.startswith("#")]
            if not lines:
                frames = []
                break
            df_s = pd.read_csv(StringIO("\n".join(lines)), sep=";", header=None)
            df_s.columns = ["time"] + [f"cell_{s * CELLS_PER_SLAVE + c}" for c in range(CELLS_PER_SLAVE)]
            frames.append(df_s.set_index("time"))
        if len(frames) != N_SLAVES:
            skipped.append({"exp_name": exp_name, "reason": "missing_slave_files"})
            print(f"    -> SKIP: missing slave files")
            continue

        # Slaves have nearly-synchronous but not identical timestamps; align by position
        # (all slaves sample the same battery at the same ~1Hz rate)
        n_rows = min(len(f) for f in frames)
        aligned = []
        ref_time = frames[0].reset_index()["time"].values[:n_rows]
        for f in frames:
            sub = f.reset_index().iloc[:n_rows].copy()
            sub["time"] = ref_time
            aligned.append(sub.set_index("time"))
        v_df = pd.concat(aligned, axis=1).reset_index().sort_values("time").reset_index(drop=True)

        ipath = exp_path / "battery" / "Battery_Current.csv"
        raw_i = ipath.read_text(errors="replace")
        lines_i = [l for l in raw_i.splitlines() if not l.startswith("#")]
        df_i = pd.read_csv(StringIO("\n".join(lines_i)), sep=";", header=None,
                            names=["time", "current"])
        i_series = df_i.set_index("time")["current"]

        df = v_df.copy()
        df["current"] = np.interp(df["time"], i_series.index, i_series.values)

        hc_mask = df["current"] > HC_THRESHOLD_A
        df_hc = df[hc_mask].copy()

        if len(df_hc) < MIN_HC_ROWS:
            reason = f"insufficient_hc_rows({len(df_hc)})"
            skipped.append({"exp_name": exp_name, "reason": reason})
            print(f"    -> SKIP: {reason}")
            continue

        if len(df_hc) > HC_CAP:
            stride = len(df_hc) // HC_CAP
            df_hc = df_hc.iloc[::stride].copy()

        # Per-slave within-module analysis
        # FOBSS is modular: inter-slave voltage diffs (~70 mV) dominate over intra-slave
        # cell variation (~10-20 mV). Analyze each 11-cell slave independently.
        for slave in range(N_SLAVES):
            slave_cells = [f"cell_{slave * CELLS_PER_SLAVE + c}" for c in range(CELLS_PER_SLAVE)]
            V_s = df_hc[slave_cells].values.astype(float)

            # Within-slave row centering
            row_mean_s = V_s.mean(axis=1, keepdims=True)
            V_dev_s = V_s - row_mean_s
            mean_dev_s = V_dev_s.mean(axis=0)   # shape (11,)

            trivial_weak_local = int(np.argmin(mean_dev_s))
            trivial_weak_global = slave * CELLS_PER_SLAVE + trivial_weak_local

            # NSA LOO within this slave (11 cells, scalar feature)
            X_s = mean_dev_s.reshape(-1, 1)
            primary_scores_s = loo_novelty(X_s, SEEDS[0])

            n_local = CELLS_PER_SLAVE
            pairwise_s = [abs(primary_scores_s[i] - primary_scores_s[j])
                          for i in range(n_local) for j in range(i + 1, n_local)]
            noise_ceil_s = np.percentile(pairwise_s, 95) if pairwise_s else 1.0
            if noise_ceil_s < 1e-9:
                continue

            ratio_s = float(np.median(primary_scores_s) / noise_ceil_s)
            power_pass_s = ratio_s >= POWER_GATE_RATIO

            vdet_weak_local = int(np.argmax(primary_scores_s))
            vdet_weak_global = slave * CELLS_PER_SLAVE + vdet_weak_local

            agree_s = trivial_weak_local == vdet_weak_local

            result = {
                "exp_name": exp_name,
                "slave": slave,
                "n_hc_rows": int(len(df_hc)),
                "trivial_weak_local": trivial_weak_local,
                "trivial_weak_global": trivial_weak_global,
                "vdet_weak_local": vdet_weak_local,
                "vdet_weak_global": vdet_weak_global,
                "power_gate_ratio": round(ratio_s, 3),
                "power_gate_pass": power_pass_s,
                "agree": agree_s,
                "slave_mean_dev_mV": [round(float(x)*1000, 3) for x in mean_dev_s],
                "slave_scores": [round(float(x), 6) for x in primary_scores_s]
            }
            results.append(result)

        # Print one-line summary per experiment (show all 4 slaves)
        exp_results = [r for r in results if r["exp_name"] == exp_name]
        summary_parts = []
        for r in exp_results:
            s = r["slave"]
            status = "A" if r["agree"] else "D"
            gate = "P" if r["power_gate_pass"] else "F"
            summary_parts.append(f"s{s}:{status}{gate}(t={r['trivial_weak_local']},v={r['vdet_weak_local']})")
        print(f"    -> {' | '.join(summary_parts)}")

    # Summary statistics — per-slave analysis
    # Each entry in results is one (experiment, slave) pair
    passing = [r for r in results if r["power_gate_pass"]]
    agree_among_passing = [r for r in passing if r["agree"]]

    # Per-slave breakdown
    per_slave = {}
    for s in range(N_SLAVES):
        s_results = [r for r in results if r["slave"] == s]
        s_passing = [r for r in s_results if r["power_gate_pass"]]
        s_agree = [r for r in s_passing if r["agree"]]
        vdet_dist = Counter(r["vdet_weak_local"] for r in s_passing)
        trivial_dist = Counter(r["trivial_weak_local"] for r in s_passing)
        most_common_vdet = vdet_dist.most_common(1)[0] if vdet_dist else (None, 0)
        per_slave[s] = {
            "n_trials": len(s_results),
            "n_pass": len(s_passing),
            "n_agree": len(s_agree),
            "pct_agree": round(len(s_agree)/len(s_passing), 3) if s_passing else None,
            "most_common_vdet_local": most_common_vdet[0],
            "most_common_vdet_count": most_common_vdet[1],
            "consistency_pct": round(most_common_vdet[1]/len(s_passing), 3) if s_passing else 0.0,
            "vdet_dist": dict(vdet_dist),
            "trivial_dist": dict(trivial_dist),
        }

    overall_pct_agree = round(len(agree_among_passing)/len(passing), 3) if passing else None
    vdet_global_dist = Counter(r["vdet_weak_global"] for r in passing)
    most_common_vdet_global = vdet_global_dist.most_common(1)[0] if vdet_global_dist else (None, 0)
    consistency_pct = most_common_vdet_global[1]/len(passing) if passing else 0.0

    print("\n" + "="*60)
    print("FOBSS V-DETECTOR RESULTS (within-slave analysis)")
    print("="*60)
    print(f"  Total (exp×slave) trials:  {len(results)}")
    print(f"  Power gate pass:           {len(passing)}")
    print(f"  Agree (among passing):     {len(agree_among_passing)} / {len(passing)} = {overall_pct_agree}")

    for s in range(N_SLAVES):
        sd = per_slave[s]
        print(f"\n  Slave {s} (cells {s*CELLS_PER_SLAVE}-{s*CELLS_PER_SLAVE+10}):")
        print(f"    pass={sd['n_pass']} agree={sd['n_agree']} pct={sd['pct_agree']} "
              f"consistency={sd['consistency_pct']:.0%}→local_{sd['most_common_vdet_local']}")
        print(f"    vdet dist: {sd['vdet_dist']}")
        print(f"    trivial dist: {sd['trivial_dist']}")

    # H1 / H2 / H3
    H1 = overall_pct_agree >= 0.85 if overall_pct_agree is not None else False
    # H2: consistency within the slave that has highest agreement (Slave 1 expected)
    H2 = any(per_slave[s]["consistency_pct"] >= 0.80 for s in range(N_SLAVES))
    H3 = (len(passing) / len(results)) >= 0.40 if results else False

    print(f"\n  H1 (agree >= 85%): {'PASS' if H1 else 'FAIL'} ({overall_pct_agree})")
    print(f"  H2 (any slave consistency >= 80%): {'PASS' if H2 else 'FAIL'}")
    print(f"  H3 (gate pass >= 40%): {'PASS' if H3 else 'FAIL'} ({len(passing)}/{len(results)})")

    summary = {
        "n_exp_dirs": len(exp_dirs),
        "n_excluded": len(skipped),
        "n_trials": len(results),
        "n_pass": len(passing),
        "n_agree": len(agree_among_passing),
        "pct_agree": overall_pct_agree,
        "per_slave": per_slave
    }

    output = {
        "preregistration_commit": "9e1598e",
        "structural_amendment": "within-slave analysis due to modular battery structure",
        "summary": summary,
        "hypothesis_results": {
            "H1_agree_passing_gte_85pct": H1,
            "H2_any_slave_consistency_gte_80pct": H2,
            "H3_power_gate_pass_gte_40pct": H3
        },
        "skipped": skipped,
        "results": results
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
