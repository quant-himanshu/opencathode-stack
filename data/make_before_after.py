#!/usr/bin/env python3
"""
data/make_before_after.py — single before/after table for the 2026-07-20
sign fix (docs/SIGN_BUG_POSTMORTEM.md).

Compares results/pre_sign_fix_snapshot/ (taken immediately before the fix)
against the regenerated artifacts, for every reported number that could
move: the headline benchmark aggregates, the Phase-1 main table, the
Phase-2 baseline comparison (+20 pp), and the nominal-accuracy (offset-0)
table. Emits only rows where |Δ| > TOL alongside a full CSV of all cells.

Outputs: results/sign_fix_before_after.csv (all cells) and
results/sign_fix_before_after.md (changed cells, grouped).

One command: venv/bin/python data/make_before_after.py
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SNAP = RES / "pre_sign_fix_snapshot"
TOL = 0.005

# validity map from the postmortem: which (dataset, method) cells were
# computed under an inverted sign BEFORE the fix
INVERTED_BEFORE = {
    ("BMW_i3", "coulomb"), ("Deng_BAIC_EU500", "coulomb"), ("VED", "coulomb"),
    ("CALCE_A123", "my_ekf"), ("CALCE_A123", "rbc_dekf"),
    ("CALCE_A123", "rbc_coupled"), ("CALCE_A123", "ekf"),
    ("Parallel_Module", "my_ekf"), ("Parallel_Module", "rbc_dekf"),
    ("Parallel_Module", "rbc_coupled"), ("Parallel_Module", "ekf"),
}


def _read_csv(path: Path) -> List[Dict]:
    with path.open() as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def _num(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compare_csv(name: str, key_cols: List[str], val_cols: List[str],
                out: List[Dict]) -> None:
    old_p, new_p = SNAP / name, RES / name
    if not old_p.exists() or not new_p.exists():
        print(f"  [skip] {name} (missing)")
        return
    old = {tuple(r[k] for k in key_cols): r for r in _read_csv(old_p)}
    new = {tuple(r[k] for k in key_cols): r for r in _read_csv(new_p)}
    for key in sorted(set(old) | set(new)):
        o, n = old.get(key), new.get(key)
        for col in val_cols:
            ov = _num(o.get(col)) if o else None
            nv = _num(n.get(col)) if n else None
            if ov is None and nv is None:
                continue
            delta = (nv - ov) if (ov is not None and nv is not None) else None
            ds = key[0]
            meth = key[1] if len(key) > 1 else ""
            out.append({
                "table": name, "dataset": ds, "method": meth,
                "key": "|".join(key), "metric": col,
                "before": ov, "after": nv, "delta": delta,
                "was_inverted_before": (ds, meth) in INVERTED_BEFORE,
                "changed": (delta is not None and abs(delta) > TOL)
                           or (ov is None) != (nv is None),
            })


def compare_headline(out: List[Dict]) -> None:
    """The Phase-0 benchmark report aggregates (mean RMSE columns)."""
    specs = [
        ("soc_baseline_benchmark_report.json",
         ["BMW_i3", "Deng_BAIC_EU500", "VED"], True),
        ("soc_baseline_benchmark_calce_report.json", ["CALCE_A123"], False),
        ("soc_baseline_benchmark_module_report.json", ["Parallel_Module"], False),
    ]
    for fname, fleets, has_subkeys in specs:
        old_p, new_p = SNAP / fname, ROOT / "data" / fname
        if not old_p.exists() or not new_p.exists():
            continue
        old_doc = json.loads(old_p.read_text())
        new_doc = json.loads(new_p.read_text())
        for fleet in fleets:
            o = (old_doc[fleet]["aggregate"] if has_subkeys
                 else old_doc["aggregate"])
            n = (new_doc[fleet]["aggregate"] if has_subkeys
                 else new_doc["aggregate"])
            for method, key in (("ekf", "ekf_soc_rmse_pct_mean"),
                                ("coulomb", "coulomb_only_soc_rmse_pct_mean"),
                                ("ocv_lookup", "ocv_lookup_only_soc_rmse_pct_mean")):
                ov, nv = _num(o.get(key)), _num(n.get(key))
                delta = (nv - ov) if (ov is not None and nv is not None) else None
                out.append({
                    "table": "headline(" + fname + ")", "dataset": fleet,
                    "method": method, "key": fleet, "metric": key,
                    "before": ov, "after": nv, "delta": delta,
                    "was_inverted_before": (fleet, method) in INVERTED_BEFORE,
                    "changed": delta is not None and abs(delta) > TOL,
                })


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: List[Dict] = []

    compare_headline(rows)
    compare_csv("main_table.csv", ["dataset", "method"],
                ["rmse_full_median", "rmse_full_mean", "mae_full_median",
                 "conv_rate_strict", "conv_rate_legacy"], rows)
    compare_csv("baseline_comparison.csv", ["dataset", "method"],
                ["rmse_median", "rmse_mean", "conv_rate_strict",
                 "rate_recovered", "rate_diverged"], rows)
    compare_csv("nominal_accuracy.csv", ["dataset", "method"],
                ["rmse_median", "rmse_mean", "conv_rate_strict"], rows)

    cols = ["table", "dataset", "method", "key", "metric", "before", "after",
            "delta", "was_inverted_before", "changed"]
    p_csv = RES / "sign_fix_before_after.csv"
    with p_csv.open("w", newline="") as f:
        f.write(f"# generated {stamp} by data/make_before_after.py; "
                f"before = results/pre_sign_fix_snapshot (pre-fix), "
                f"after = regenerated post-fix; TOL={TOL}\n")
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})

    changed = [r for r in rows if r["changed"]]
    md = ["# Sign fix — before/after (only cells with |Δ| > 0.005)", "",
          f"Generated {stamp}. `inv` marks cells computed under an inverted "
          f"sign BEFORE the fix (expected to change); unmarked changed cells "
          f"changed only through re-fit calibrations or downstream coupling.",
          "",
          "| Table | Dataset | Method | Metric | Before | After | Δ | inv |",
          "|---|---|---|---|---|---|---|---|"]
    for r in changed:
        b = "" if r["before"] is None else f"{r['before']:.3f}"
        a = "" if r["after"] is None else f"{r['after']:.3f}"
        d = "" if r["delta"] is None else f"{r['delta']:+.3f}"
        md.append(f"| {r['table']} | {r['dataset']} | {r['method']} "
                  f"| {r['metric']} | {b} | {a} | {d} "
                  f"| {'⚠' if r['was_inverted_before'] else ''} |")
    md += ["", f"{len(changed)} changed cells of {len(rows)} compared."]
    (RES / "sign_fix_before_after.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {p_csv}")
    print(f"Wrote {RES / 'sign_fix_before_after.md'}")
    print(f"{len(changed)} changed / {len(rows)} compared")


if __name__ == "__main__":
    main()
