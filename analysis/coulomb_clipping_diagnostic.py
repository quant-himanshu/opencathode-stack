#!/usr/bin/env python3
"""
analysis/coulomb_clipping_diagnostic.py — protocol-artifact check
(2026-07-20 review item): does initial-SOC clipping hand the coulomb
baseline an artificially small applied offset?

Protocol rule (validate_generic.py:291, shared by every init-based method):
initial SOC = clip(SOC_true[0] + offset, 0.02, 0.98). Trips that start near
the rails therefore receive LESS than the nominal offset — e.g. a Deng
session starting at 85% under +20 pp initializes at 98%, an applied offset
of only +13 pp. A drift-free integrator (coulomb counting) keeps whatever
initial error it was given, so clipping directly lowers its stress-test
RMSE on those trips. This script quantifies that per dataset × offset from
the sweep's per-trip dump (coulomb rows; the same fractions apply to the
EKF family by construction, but a converging filter's RMSE is far less
sensitive to the applied offset):

  n, n_clipped (|applied − nominal| > 0.01 pp), clip fraction,
  median RMSE overall / clipped-only / unclipped-only,
  material_artifact flag = clip fraction ≥ 10% AND clipped median RMSE
  at least 2 pp lower than unclipped median.

Outputs: results/coulomb_clipping_diagnostic.csv + .md (flagged cells).
One command: venv/bin/python analysis/coulomb_clipping_diagnostic.py
"""
from __future__ import annotations

import csv
import glob
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def newest_dump() -> Path:
    cands = [Path(p) for pat in ("offset_sweep_per_trip_*.csv",
                                 "offset_sweep_per_trip_retuned_*.csv")
             for p in glob.glob(str(RES / pat))]
    if not cands:
        sys.exit("no per-trip dump in results/ — run the sweep first")
    return max(cands, key=lambda p: p.stat().st_mtime)


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dump = newest_dump()
    df = pd.read_csv(dump)
    cc = df[(df.method == "coulomb") & df.applied_offset_pp.notna()].copy()
    cc["clipped"] = (cc.applied_offset_pp - cc.offset_pp).abs() > 0.01

    rows = []
    for (ds, off), g in cc.groupby(["dataset", "offset_pp"]):
        cl, un = g[g.clipped], g[~g.clipped]
        med = lambda x: float(x.rmse_full_pct.median()) if len(x) else None
        frac = len(cl) / len(g)
        m_cl, m_un = med(cl), med(un)
        material = bool(frac >= 0.10 and m_cl is not None and m_un is not None
                        and (m_un - m_cl) > 2.0)
        rows.append({
            "dataset": ds, "offset_pp": int(off), "n": len(g),
            "n_clipped": int(len(cl)), "clip_frac": round(frac, 4),
            "mean_applied_offset_pp": round(float(g.applied_offset_pp.mean()), 2),
            "rmse_median_all": round(float(g.rmse_full_pct.median()), 3),
            "rmse_median_clipped": None if m_cl is None else round(m_cl, 3),
            "rmse_median_unclipped": None if m_un is None else round(m_un, 3),
            "material_artifact": material,
        })

    cols = list(rows[0].keys())
    p = RES / "coulomb_clipping_diagnostic.csv"
    with p.open("w", newline="") as f:
        f.write(f"# generated {stamp} from {dump.name}; clip bounds are "
                f"[2%, 98%] (the protocol's rule), not [0%, 100%]; clipped = "
                f"|applied − nominal| > 0.01 pp; material_artifact = clip "
                f"fraction ≥ 10% AND clipped median RMSE > 2 pp below "
                f"unclipped\n")
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in cols})
    print(f"Wrote {p}")

    flagged = [r for r in rows if r["material_artifact"]]
    md = ["# Coulomb-baseline initial-SOC clipping diagnostic", "",
          f"Generated {stamp} from `{dump.name}`. Clip bounds [2%, 98%]. A "
          f"flagged cell means the coulomb baseline's aggregate RMSE at that "
          f"offset is materially lowered because ≥10% of trips started near "
          f"a rail and received a smaller applied offset than nominal — a "
          f"protocol artifact, not estimator skill.", "",
          "| Dataset | Offset (pp) | n | clipped | frac | mean applied (pp) "
          "| RMSE med all | clipped | unclipped |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in (flagged or rows):
        md.append(f"| {r['dataset']} | {r['offset_pp']:+d} | {r['n']} "
                  f"| {r['n_clipped']} | {r['clip_frac']*100:.1f}% "
                  f"| {r['mean_applied_offset_pp']:+.1f} "
                  f"| {r['rmse_median_all']} | {r['rmse_median_clipped']} "
                  f"| {r['rmse_median_unclipped']} |")
    if flagged:
        md.insert(6, f"**{len(flagged)} flagged (material) cells — table "
                     f"shows flagged cells only; full grid in the CSV.**")
    else:
        md.insert(6, "**No cell met the materiality threshold — full grid "
                     "shown.**")
    (RES / "coulomb_clipping_diagnostic.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {RES / 'coulomb_clipping_diagnostic.md'}")
    print(f"{len(flagged)} material cells / {len(rows)} total")


if __name__ == "__main__":
    main()
