#!/usr/bin/env python3
"""
Train BatteryGNN on Quartz WLTP data and evaluate against EKF baseline.

The existing training script (stack/train_gnn.py) has three bugs:
  1. forward_numpy() strips gradients — model never learns
  2. Manual weight update targets model.W3 which doesn't exist on PyTorch model
  3. save_model uses vars(model) instead of state_dict()

This script fixes all three with proper PyTorch Adam + MSELoss training.

Architecture:
  - Existing BatteryGNN (3-layer GraphSAGE, 7→64→32→16→4) from stack/gnn_layer.py
  - Additional voltage head: linear(4→1) trained to predict V_cell in Volts
  - The GNN's 4-output is treated as a learned embedding; only voltage head is new

EKF baseline (from data/validate_quartz.py, sensor-update rows only):
  MAE = 18.6 mV, R² = 0.9217

Split: 70 / 15 / 15 on cycle index (not row index) — same as original intent.
"""

from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from stack.gnn_layer import BatteryGNN

# ── constants ──────────────────────────────────────────────────────────────────
N_P, N_S = 3, 12
N_CELLS   = N_P * N_S       # 36
N_FEAT    = 7
RESAMPLE  = "20s"
LR        = 1e-3
N_EPOCHS  = 60
BATCH     = 64
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
EKF_MAE_MV = 18.6           # EKF baseline to beat (mV, sensor-update rows)

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(_ROOT, "data", "quartz_wltp")
MODEL_PATH  = os.path.join(_ROOT, "stack", "gnn_weights.pt")
RESULTS_PATH = os.path.join(_ROOT, "data", "gnn_quartz_results.json")


# ── adjacency (3P × 12S Quartz topology) ─────────────────────────────────────

def build_adj() -> np.ndarray:
    """Build 36×36 adjacency for 3-parallel × 12-series Quartz pack.
    Cell index: p*N_S + s  where p=parallel index, s=series index.
    Series electrical edges: adj=1.0
    Parallel thermal coupling: adj=0.3
    """
    adj = np.zeros((N_CELLS, N_CELLS), dtype=np.float32)
    for p in range(N_P):
        for s in range(N_S):
            idx = p * N_S + s
            if s > 0:                          # series neighbour
                adj[idx, idx-1] = adj[idx-1, idx] = 1.0
            if p > 0:                          # parallel thermal coupling
                adj[idx, idx-N_S] = adj[idx-N_S, idx] = 0.3
    return adj


# ── data loading ──────────────────────────────────────────────────────────────

def load_quartz() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X (W, N_CELLS, N_FEAT), y (W, N_CELLS), adj (N_CELLS, N_CELLS), cycle_idx (W,)."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    frames = []
    for i, fname in enumerate(files):
        df = pd.read_parquet(os.path.join(DATA_DIR, fname))
        df["_cycle"] = i
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True).sort_values("Timestamp").reset_index(drop=True)

    agg: dict = {f"Current_Actual_P{p} [A]": "mean" for p in range(1, N_P+1)}
    for p in range(1, N_P+1):
        for s in range(1, N_S+1):
            agg[f"Voltage_Cell_P{p}S{s} [V]"] = "last"
            agg[f"Temperature_Cell_Top_P{p}S{s} [degC]"] = "last"
    agg["SoC_Actual_Battery [percent]"] = "last"
    agg["_cycle"] = "last"

    data = (raw.set_index("Timestamp").resample(RESAMPLE).agg(agg)
               .dropna(subset=["Current_Actual_P1 [A]"]).reset_index())

    W = len(data)
    X = np.zeros((W, N_CELLS, N_FEAT), dtype=np.float32)
    y = np.zeros((W, N_CELLS), dtype=np.float32)

    soc_global = data["SoC_Actual_Battery [percent]"].values.astype(np.float32) / 100.0
    cycle_arr  = data["_cycle"].values.astype(np.float32)

    for ci, (p, s) in enumerate([(pp, ss) for pp in range(1, N_P+1) for ss in range(1, N_S+1)]):
        V  = data[f"Voltage_Cell_P{p}S{s} [V]"].values.astype(np.float32)
        T  = data[f"Temperature_Cell_Top_P{p}S{s} [degC]"].values.astype(np.float32)
        I  = data[f"Current_Actual_P{p} [A]"].values.astype(np.float32)
        T  = np.where(T > 100.0, 30.0, T)
        dVdt = np.gradient(V)
        X[:, ci, 0] = np.clip(soc_global, 0, 1)
        X[:, ci, 1] = np.clip(V / 4.5, 0, 1)
        X[:, ci, 2] = np.clip((T - 20) / 60.0, 0, 1)
        X[:, ci, 3] = 0.532                                    # R_ohm placeholder (calibrated)
        X[:, ci, 4] = np.clip(dVdt / 0.05 + 0.5, 0, 1)
        X[:, ci, 5] = np.clip(-I / 7.0 + 0.5, 0, 1)
        X[:, ci, 6] = cycle_arr / max(float(cycle_arr.max()), 1.0)
        y[:, ci]    = V

    return X, y, build_adj(), cycle_arr.astype(int)


# ── model with voltage head ───────────────────────────────────────────────────

class GNNVoltagePredictor(nn.Module):
    """
    BatteryGNN backbone + linear voltage head.

    The existing BatteryGNN was designed for [SOC, SOH, T, fault_prob] but
    has no labelled training targets for SOH or fault_prob in Quartz.
    We treat the 4-dim GNN output as a learned representation and add a
    trainable linear layer to regress V_cell in Volts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gnn = BatteryGNN()
        # Voltage regression head: maps 4-dim GNN repr → 1 voltage value
        # Initialised to approximate the sigmoid→[2.5,4.5] V range
        self.v_head = nn.Linear(4, 1)
        nn.init.constant_(self.v_head.weight, 0.5)
        nn.init.constant_(self.v_head.bias, 3.5)   # typical cell resting V

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Return voltage predictions (N_CELLS,)."""
        gnn_out = self.gnn(x, adj)          # (N, 4)
        v_pred  = self.v_head(gnn_out)      # (N, 1)
        return v_pred.squeeze(-1)           # (N,)


# ── training ──────────────────────────────────────────────────────────────────

def train_and_eval():
    print("=" * 60)
    print("GNN VOLTAGE PREDICTION — Quartz WLTP")
    print("=" * 60)

    print("\nLoading Quartz data...")
    X, y, adj_np, cycle_idx = load_quartz()
    W = len(X)
    print(f"  Windows: {W:,}  Cells: {N_CELLS}  Features: {N_FEAT}")

    # 70/15/15 split by cycle
    n_cyc = int(cycle_idx.max()) + 1
    n_tr  = int(n_cyc * TRAIN_FRAC)
    n_vl  = int(n_cyc * VAL_FRAC)
    tr_mask = cycle_idx <  n_tr
    vl_mask = (cycle_idx >= n_tr) & (cycle_idx < n_tr + n_vl)
    te_mask = cycle_idx >= n_tr + n_vl

    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_vl, y_vl = X[vl_mask], y[vl_mask]
    X_te, y_te = X[te_mask], y[te_mask]
    print(f"  Split — train: {len(X_tr):,}  val: {len(X_vl):,}  test: {len(X_te):,}")

    adj_t = torch.from_numpy(adj_np)
    model  = GNNVoltagePredictor()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    loss_fn   = nn.MSELoss()

    def run_batch(Xb, yb, train=True):
        Xb_t = torch.from_numpy(Xb.astype(np.float32))  # (B, N, 7)
        yb_t = torch.from_numpy(yb.astype(np.float32))  # (B, N)
        preds = []
        for i in range(len(Xb_t)):
            v = model(Xb_t[i], adj_t)   # (N,)
            preds.append(v)
        pred_t = torch.stack(preds)      # (B, N)
        loss = loss_fn(pred_t, yb_t)
        return loss, pred_t.detach().numpy()

    best_val_mae = np.inf
    best_state   = None
    history = {"train_mse": [], "val_mae_mV": [], "test_mae_mV": None}

    print(f"\nTraining {N_EPOCHS} epochs, batch={BATCH}, lr={LR}")
    print(f"{'Epoch':>5}  {'train_MSE':>10}  {'val_MAE_mV':>11}  {'vs_EKF':>8}")
    print("-" * 45)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(len(X_tr))
        total_loss = 0.0; n_batches = 0
        for s in range(0, len(X_tr), BATCH):
            idx = perm[s:s+BATCH]
            Xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss, _ = run_batch(Xb, yb, train=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = []
            for s in range(0, len(X_vl), BATCH):
                _, pred = run_batch(X_vl[s:s+BATCH], y_vl[s:s+BATCH], train=False)
                val_preds.append(pred)
        vp = np.concatenate(val_preds, axis=0)
        val_mae_mV = float(np.mean(np.abs(vp - y_vl))) * 1000

        history["train_mse"].append(total_loss / max(n_batches, 1))
        history["val_mae_mV"].append(round(val_mae_mV, 2))

        if val_mae_mV < best_val_mae:
            best_val_mae = val_mae_mV
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == N_EPOCHS:
            vs = f"{'BEATS' if val_mae_mV < EKF_MAE_MV else 'WORSE'}"
            print(f"{epoch:5d}  {total_loss/max(n_batches,1):10.5f}  {val_mae_mV:11.2f}  {vs}")

    # Load best weights → test
    model.load_state_dict(best_state)
    model.eval()

    print(f"\nBest val MAE: {best_val_mae:.2f} mV")

    with torch.no_grad():
        te_preds = []
        for s in range(0, len(X_te), BATCH):
            _, pred = run_batch(X_te[s:s+BATCH], y_te[s:s+BATCH], train=False)
            te_preds.append(pred)
    tp = np.concatenate(te_preds, axis=0)
    test_mae_mV  = float(np.mean(np.abs(tp - y_te))) * 1000
    test_rmse_mV = float(np.sqrt(np.mean((tp - y_te)**2))) * 1000
    ss_res = np.sum((tp - y_te)**2)
    ss_tot = np.sum((y_te - y_te.mean())**2)
    test_r2 = float(1 - ss_res / (ss_tot + 1e-12))

    history["test_mae_mV"]  = round(test_mae_mV, 2)
    history["test_rmse_mV"] = round(test_rmse_mV, 2)
    history["test_r2"]      = round(test_r2, 4)
    history["best_val_mae_mV"] = round(best_val_mae, 2)

    print("\n" + "=" * 60)
    print("TEST RESULTS (held-out 15%)")
    print("=" * 60)
    print(f"  GNN   MAE  : {test_mae_mV:.2f} mV")
    print(f"  GNN   RMSE : {test_rmse_mV:.2f} mV")
    print(f"  GNN   R²   : {test_r2:.4f}")
    print(f"  EKF   MAE  : {EKF_MAE_MV:.1f} mV  (baseline, sensor-update rows)")
    delta = test_mae_mV - EKF_MAE_MV
    verdict = ("GNN BEATS EKF" if delta < 0 else
               f"EKF better by {delta:.1f} mV")
    print(f"  Verdict    : {verdict}")

    # Per-cell MAE
    per_cell_mae = np.abs(tp - y_te).mean(axis=0) * 1000
    worst_cell  = int(np.argmax(per_cell_mae))
    best_cell   = int(np.argmin(per_cell_mae))
    print(f"\n  Per-cell MAE: min={per_cell_mae.min():.1f} mV (cell {best_cell}) "
          f"max={per_cell_mae.max():.1f} mV (cell {worst_cell})")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n  Weights saved → {MODEL_PATH}")

    # Save results
    results = {
        "model": "BatteryGNN (3-layer GraphSAGE) + voltage head",
        "n_params": n_params,
        "dataset": "Quartz WLTP (36-cell NMC811)",
        "split": "70/15/15 by cycle index",
        "n_windows_total": W,
        "n_cells": N_CELLS,
        "epochs": N_EPOCHS,
        "test_mae_mV": round(test_mae_mV, 2),
        "test_rmse_mV": round(test_rmse_mV, 2),
        "test_r2": round(test_r2, 4),
        "best_val_mae_mV": round(best_val_mae, 2),
        "ekf_baseline_mae_mV": EKF_MAE_MV,
        "gnn_vs_ekf_delta_mV": round(delta, 2),
        "verdict": verdict,
        "per_cell_mae_mV": [round(float(x), 2) for x in per_cell_mae],
        "training_history": history
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    t0 = time.perf_counter()
    results = train_and_eval()
    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")
