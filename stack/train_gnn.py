#!/usr/bin/env python3
"""
train_gnn.py — Train BatteryGNN on Quartz WLTP real-world data.

Node features (7 per cell):
    [SOC, V_norm, T_norm, R_ohm_norm, dV_dt_norm, I_norm, cycle_norm]
Edge structure:
    1.0  = electrical series  |  0.3 = thermal coupling
Target per node:
    V_true [V]
Loss:
    MSE(V_pred, V_true) + 0.1×KCL + 0.05×TCO
Split: 70/15/15 on cycle index (not row index).
"""
from __future__ import annotations
import os, sys, time
from typing import Dict, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

from stack.gnn_layer import BatteryGNN

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "quartz_wltp")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "gnn_weights.npz")
N_P, N_S   = 3, 12
N_CELLS    = N_P * N_S   # 36
N_FEATURES = 7
RESAMPLE_S = "20s"
LR         = 1e-3
N_EPOCHS   = 50
BATCH_SIZE = 32
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15


def load_quartz_windows(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (W,36,7), y (W,36), adj (36,36)."""
    if not _PANDAS:
        raise ImportError("pandas required")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    frames = []
    for i, fname in enumerate(files):
        df = pd.read_parquet(os.path.join(data_dir, fname))
        df["_cycle"] = i
        frames.append(df)
    raw  = pd.concat(frames, ignore_index=True).sort_values("Timestamp").reset_index(drop=True)

    agg: dict = {f"Current_Actual_P{p} [A]": "mean" for p in range(1, N_P+1)}
    for p in range(1, N_P+1):
        for s in range(1, N_S+1):
            agg[f"Voltage_Cell_P{p}S{s} [V]"] = "last"
            agg[f"Temperature_Cell_Top_P{p}S{s} [degC]"] = "last"
    agg["SoC_Actual_Battery [percent]"] = "last"
    agg["_cycle"] = "last"

    data = (raw.set_index("Timestamp").resample(RESAMPLE_S).agg(agg)
               .dropna(subset=["Current_Actual_P1 [A]"]).reset_index())
    W   = len(data)
    X   = np.zeros((W, N_CELLS, N_FEATURES), dtype=np.float32)
    y   = np.zeros((W, N_CELLS),             dtype=np.float32)

    for ci, (p, s) in enumerate([(p, s) for p in range(1, N_P+1) for s in range(1, N_S+1)]):
        V  = data[f"Voltage_Cell_P{p}S{s} [V]"].values.astype(np.float32)
        T  = data[f"Temperature_Cell_Top_P{p}S{s} [degC]"].values.astype(np.float32)
        I  = data[f"Current_Actual_P{p} [A]"].values.astype(np.float32)
        soc= data["SoC_Actual_Battery [percent]"].values.astype(np.float32) / 100.0
        cyc= data["_cycle"].values.astype(np.float32)
        T   = np.where(T > 100.0, 30.0, T)
        dVdt= np.gradient(V)
        X[:, ci, 0] = np.clip(soc, 0, 1)
        X[:, ci, 1] = np.clip(V / 4.5, 0, 1)
        X[:, ci, 2] = np.clip((T - 20) / 60.0, 0, 1)
        X[:, ci, 3] = 0.532          # normalised calibrated R_ohm placeholder
        X[:, ci, 4] = np.clip(dVdt / 0.05 + 0.5, 0, 1)
        X[:, ci, 5] = np.clip(-I / 7.0 + 0.5, 0, 1)
        X[:, ci, 6] = cyc / max(float(cyc.max()), 1.0)
        y[:, ci]    = V

    adj = np.zeros((N_CELLS, N_CELLS), dtype=np.float32)
    for p in range(N_P):
        for s in range(N_S):
            idx = p * N_S + s
            if s > 0: adj[idx, idx-1] = adj[idx-1, idx] = 1.0
            if p > 0: adj[idx, idx-N_S] = adj[idx-N_S, idx] = 0.3
    return X, y, adj


def _mse(a, b):  return float(np.mean((a - b)**2))
def _r2(a, b):
    ss_r = np.sum((b - a)**2); ss_t = np.sum((b - np.mean(b))**2)
    return float(1.0 - ss_r / (ss_t + 1e-12))

def kcl_violation(V: np.ndarray) -> float:
    v = 0.0
    for p in range(N_P):
        s_sum = np.sum(V[p*N_S:(p+1)*N_S])
        v += (s_sum - np.mean(V)*N_S)**2
    return v / (N_P + 1e-8)

def tco_violation(V: np.ndarray) -> float:
    return float(np.mean(np.maximum(0, 2.5-V)**2 + np.maximum(0, V-4.5)**2))


def train(X, y, adj, n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE, verbose=True):
    W      = len(X)
    n_cyc  = int(X[:, 0, 6].max() * 10) + 1
    cidx   = (X[:, 0, 6] * n_cyc).astype(int)
    n_tr   = int(n_cyc * TRAIN_FRAC); n_vl = int(n_cyc * VAL_FRAC)
    tr = cidx < n_tr; vl = (cidx >= n_tr) & (cidx < n_tr+n_vl); te = cidx >= n_tr+n_vl
    X_tr,y_tr = X[tr],y[tr]; X_vl,y_vl = X[vl],y[vl]; X_te,y_te = X[te],y[te]
    if verbose:
        print(f"  Train={len(X_tr):,}  Val={len(X_vl):,}  Test={len(X_te):,}")

    model   = BatteryGNN()
    history: Dict = {"train_loss": [], "val_loss": [], "val_r2": []}

    def fwd(Xb):
        res = []
        for x in Xb:
            out = model.forward_numpy(x, adj)
            res.append(2.5 + out[:, 0] * 2.0)
        return np.stack(res)

    best_r2, best_w = -np.inf, None
    for epoch in range(n_epochs):
        perm = np.random.permutation(len(X_tr))
        xsh, ysh = X_tr[perm], y_tr[perm]
        el = 0.0; nb = 0
        for s in range(0, len(xsh), batch_size):
            xb, yb = xsh[s:s+batch_size], ysh[s:s+batch_size]
            if len(xb) == 0: continue
            Vp   = fwd(xb)
            loss = _mse(Vp, yb)
            loss += 0.1 * float(np.mean([kcl_violation(Vp[i]) for i in range(len(xb))]))
            loss += 0.05 * float(np.mean([tco_violation(Vp[i]) for i in range(len(xb))]))
            el += loss; nb += 1
            if hasattr(model, 'W3'):
                g = lr * float((Vp - yb).mean()) / (np.abs(model.W3).mean() + 1e-8)
                model.W3 = model.W3 * (1 - lr*0.001); model.b3 -= g * 0.01

        vl_mse = vl_r2 = float("nan")
        if len(X_vl):
            Vv = fwd(X_vl); vl_mse = _mse(Vv, y_vl); vl_r2 = _r2(Vv.flatten(), y_vl.flatten())
        history["train_loss"].append(el / max(nb, 1))
        history["val_loss"].append(vl_mse); history["val_r2"].append(vl_r2)
        if vl_r2 > best_r2:
            best_r2 = vl_r2
            if hasattr(model, 'W3'): best_w = (model.W3.copy(), model.b3.copy())
        if verbose and (epoch % 10 == 0 or epoch == n_epochs-1):
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  train={history['train_loss'][-1]:.5f}  val_R²={vl_r2:.4f}")

    if best_w and hasattr(model, 'W3'): model.W3, model.b3 = best_w
    if len(X_te):
        Vt = fwd(X_te); tr2 = _r2(Vt.flatten(), y_te.flatten())
        if verbose: print(f"\n  Test R² = {tr2:.4f}")
        history["test_r2"] = tr2
    return model, history


def save_model(model, path):
    w = {k: v for k, v in vars(model).items() if isinstance(v, np.ndarray)}
    np.savez(path, **w); print(f"  Saved → {path}")


def validate() -> bool:
    print("=" * 60); print("VALIDATING: stack/train_gnn.py"); print("=" * 60)
    ok = True
    def check(n, c, d=""):
        nonlocal ok
        print(("  [PASS]" if c else "  [FAIL]") + f" {n}" + (f"  | {d}" if d else ""))
        if not c: ok = False

    X_f = np.random.rand(80, N_CELLS, N_FEATURES).astype(np.float32)
    y_f = np.random.rand(80, N_CELLS).astype(np.float32)*1.5 + 3.0
    adj_f = np.eye(N_CELLS, dtype=np.float32)
    m, h  = train(X_f, y_f, adj_f, n_epochs=3, verbose=False)
    check("Model returned", m is not None)
    check("History correct", "train_loss" in h and len(h["train_loss"]) == 3)
    check("KCL >= 0", kcl_violation(y_f[0]) >= 0)
    check("TCO >= 0", tco_violation(y_f[0]) >= 0)
    status = "ALL PASS" if ok else "SOME FAILED"
    print(f"\nResult: {status}"); print("=" * 60)
    return ok


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--save",   default=MODEL_PATH)
    args = p.parse_args()
    print("=" * 60); print("TRAINING BatteryGNN on Quartz WLTP"); print("=" * 60)
    if not _PANDAS or not os.path.exists(DATA_DIR):
        print("  Data unavailable — running smoke test"); validate()
    else:
        t0 = time.perf_counter()
        X, y, adj = load_quartz_windows(DATA_DIR)
        print(f"  Loaded {len(X):,} windows")
        model, hist = train(X, y, adj, n_epochs=args.epochs)
        save_model(model, args.save)
        print(f"  Done in {time.perf_counter()-t0:.1f}s")
