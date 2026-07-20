#!/usr/bin/env python3
"""
analysis/plot_ablation.py — figures/ablation.pdf from results/ablation.csv.

One panel per dataset; horizontal bars = median full-trip SOC RMSE (+20 pp)
per variant, IQR whiskers; the production configuration ('full') in blue,
ablated variants in neutral gray. Identity is position-encoded (variant
labels on the shared y-axis), so no legend is needed; values are labeled
directly. Vector PDF, no baked title.

One command: venv/bin/python analysis/plot_ablation.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "ablation.csv"
OUT = ROOT / "figures" / "ablation.pdf"

DATASETS = ["BMW_i3", "Deng_BAIC_EU500", "VED", "CALCE_A123", "Parallel_Module"]
LABEL = {"BMW_i3": "BMW i3", "Deng_BAIC_EU500": "Deng BAIC EU500",
         "VED": "VED", "CALCE_A123": "CALCE A123",
         "Parallel_Module": "UMich/Ford module"}
VARIANTS = [  # display order, top→bottom
    ("full", "full method"),
    ("joseph_form", "Joseph-form update"),
    ("coupled_dcal_H", "∂δV/∂SOC in H (Round-2)"),
    ("no_corrections", "no δV, no δR0"),
    ("dR0_only", "δR0 only"),
    ("dV_only", "δV only"),
    ("const_Q", "constant Q (no slope adapt)"),
    ("slow_loops_on", "sanity gate disabled"),
    ("slow_loops_off", "slow loops off (x2 frozen)"),
]
BLUE, GRAY, TEXT, MUTED, GRID = "#2a78d6", "#b3b2ac", "#0b0b0b", "#52514e", "#d9d8d4"


def main():
    if not CSV.exists():
        sys.exit(f"missing {CSV}")
    rows = {}
    with CSV.open() as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            rows[(r["dataset"], r["variant"])] = r

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif",
                         "text.color": TEXT, "axes.edgecolor": MUTED,
                         "axes.labelcolor": TEXT, "xtick.color": MUTED,
                         "ytick.color": TEXT, "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 3.4), sharey=True)
    ypos = range(len(VARIANTS) - 1, -1, -1)

    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        for y, (key, label) in zip(ypos, VARIANTS):
            r = rows.get((ds, key))
            if r is None or r["rmse_median"] == "":
                continue
            med = float(r["rmse_median"])
            q25, q75 = float(r["rmse_q25"]), float(r["rmse_q75"])
            color = BLUE if key == "full" else GRAY
            ax.barh(y, med, height=0.62, color=color, zorder=3)
            ax.plot([q25, q75], [y, y], color=MUTED, lw=0.9, zorder=4)
            ax.text(med + (q75 - q25) * 0.02 + 0.8, y, f"{med:.1f}",
                    va="center", fontsize=6.5, color=TEXT, zorder=5)
        ax.set_title(LABEL[ds], fontsize=8.5, pad=4)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("median SOC RMSE (pp)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, None)
    axes[0].set_yticks(list(ypos))
    axes[0].set_yticklabels([label for _, label in VARIANTS], fontsize=7.5)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
