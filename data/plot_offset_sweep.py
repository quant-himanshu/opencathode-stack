#!/usr/bin/env python3
"""
data/plot_offset_sweep.py — Phase 3 figure: figures/offset_sweep.pdf.

Reads results/offset_sweep.csv (data/run_offset_sweep.py). Layout: 2 rows ×
5 columns — top row median full-trip SOC RMSE vs initial-offset, bottom row
strict convergence rate vs offset; one column per dataset, one line per
method. Vector PDF, colorblind-safe palette (validated: CVD ΔE ≥ 16.2,
normal ΔE ≥ 29; the low-contrast yellow gets marker-shape relief — identity
is never color-alone). No figure title — the caption lives in the paper.

One command: venv/bin/python data/plot_offset_sweep.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "offset_sweep.csv"
OUT = ROOT / "figures" / "offset_sweep.pdf"

DATASETS = ["BMW_i3", "Deng_BAIC_EU500", "VED", "CALCE_A123", "Parallel_Module"]
DATASET_LABEL = {"BMW_i3": "BMW i3", "Deng_BAIC_EU500": "Deng BAIC EU500",
                 "VED": "VED", "CALCE_A123": "CALCE A123",
                 "Parallel_Module": "UMich/Ford module"}
# fixed method order, fixed colors (never cycled), marker = secondary encoding
METHODS = [
    ("my_ekf",     "Dual EKF (ours)",   "#2a78d6", "o", "-"),
    ("rbc_dekf",   "Scalar-bias DEKF",  "#008300", "s", "-"),
    ("coulomb",    "Coulomb counting",  "#eda100", "^", "--"),
    ("ocv_lookup", "OCV lookup",        "#4a3aa7", "D", ":"),
]

TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"


def load_rows():
    rows = []
    with CSV.open() as f:
        for line in f:
            if not line.startswith("#"):
                break
        f.seek(0)
        rdr = csv.DictReader(l for l in f if not l.startswith("#"))
        for r in rdr:
            rows.append(r)
    return rows


def main():
    if not CSV.exists():
        sys.exit(f"missing {CSV} — run data/run_offset_sweep.py first")
    rows = load_rows()
    OUT.parent.mkdir(exist_ok=True)

    def cell(ds, method, col):
        out = {}
        for r in rows:
            if r["dataset"] == ds and r["method"] == method and r[col] != "":
                out[int(r["offset_pp"])] = float(r[col])
        xs = sorted(out)
        return xs, [out[x] for x in xs]

    plt.rcParams.update({
        "font.size": 8, "font.family": "sans-serif", "text.color": TEXT,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 5, figsize=(10.6, 4.2), sharex=True)

    for j, ds in enumerate(DATASETS):
        ax_r, ax_c = axes[0][j], axes[1][j]
        for key, label, color, marker, ls in METHODS:
            xs, ys = cell(ds, key, "rmse_median")
            ax_r.plot(xs, ys, ls, color=color, marker=marker, ms=3.5,
                      lw=1.4, label=label, clip_on=False, zorder=3)
            xs, ys = cell(ds, key, "conv_rate_strict")
            ax_c.plot(xs, [y * 100 for y in ys], ls, color=color,
                      marker=marker, ms=3.5, lw=1.4, clip_on=False, zorder=3)

        ax_r.set_title(DATASET_LABEL[ds], fontsize=8.5, color=TEXT, pad=4)
        for ax in (ax_r, ax_c):
            ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
            ax.tick_params(labelsize=7)
        ax_r.set_ylim(bottom=0)
        ax_c.set_ylim(0, 100)
        ax_c.set_xlabel("initial SOC offset (pp)", fontsize=7.5)
        if j == 0:
            ax_r.set_ylabel("median SOC RMSE (pp)", fontsize=8)
            ax_c.set_ylabel("strict conv. rate (%)", fontsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
