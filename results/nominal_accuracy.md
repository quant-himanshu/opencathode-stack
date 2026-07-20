# Nominal accuracy (offset = 0, correct initialization)

Regenerated 20260719T195435Z (wide-grid θ). Median (IQR) primary, mean secondary — all pp.

| Dataset | Method | n | RMSE med (IQR) | RMSE mean | MAE med | Converged% | Recovered% | Diverged% |
|---|---|---|---|---|---|---|---|---|
| BMW_i3 | my_ekf | 63 | 13.94 (0.25–21.50) | 15.48 | 11.53 | 33.3 | 7.9 | 58.7 |
| BMW_i3 | rbc_dekf | 63 | 13.39 (0.25–23.72) | 15.07 | 10.15 | 34.9 | 11.1 | 54.0 |
| BMW_i3 | coulomb | 63 | 0.23 (0.15–0.33) | 0.35 | 0.20 | 98.4 | 0 | 1.6 |
| BMW_i3 | ocv_lookup | 63 | 38.43 (25.66–49.99) | 37.75 | 30.41 | 39.7 | 4.8 | 55.6 |
| Deng_BAIC_EU500 | my_ekf | 2000 | 9.21 (5.45–11.44) | 8.76 | 6.80 | 87.2 | 10.8 | 2.1 |
| Deng_BAIC_EU500 | rbc_dekf | 2000 | 6.07 (4.33–7.39) | 5.88 | 4.61 | 86.0 | 13.7 | 0.4 |
| Deng_BAIC_EU500 | coulomb | 2000 | 4.98 (3.28–6.94) | 5.16 | 4.33 | 22.4 | 43.8 | 33.9 |
| Deng_BAIC_EU500 | ocv_lookup | 2000 | 6.35 (4.65–7.33) | 6.05 | 5.30 | 74.8 | 16.8 | 8.5 |
| VED | my_ekf | 408 | 12.87 (5.54–23.59) | 17.62 | 11.26 | 22.3 | 15.0 | 62.7 |
| VED | rbc_dekf | 408 | 10.94 (4.58–21.80) | 15.68 | 9.49 | 26.2 | 18.9 | 54.9 |
| VED | coulomb | 408 | 1.43 (0.63–2.47) | 1.68 | 1.24 | 90.9 | 8.6 | 0.5 |
| VED | ocv_lookup | 408 | 11.40 (7.67–20.80) | 14.83 | 9.25 | 35.3 | 21.8 | 42.9 |
| CALCE_A123 | my_ekf | 26 | 28.45 (16.61–42.20) | 29.38 | 22.32 | 23.1 | 3.8 | 73.1 |
| CALCE_A123 | rbc_dekf | 26 | 26.87 (11.46–39.42) | 27.13 | 20.08 | 34.6 | 0 | 65.4 |
| CALCE_A123 | coulomb | 26 | 0.10 (0.05–0.15) | 0.20 | 0.08 | 100.0 | 0 | 0 |
| CALCE_A123 | ocv_lookup | 26 | 34.68 (23.92–52.97) | 36.81 | 31.30 | 26.9 | 15.4 | 57.7 |
| Parallel_Module | my_ekf | 216 | 2.72 (1.70–6.23) | 8.11 | 2.34 | 62.5 | 16.2 | 21.3 |
| Parallel_Module | rbc_dekf | 216 | 2.66 (1.70–6.27) | 7.09 | 2.30 | 59.7 | 17.1 | 23.1 |
| Parallel_Module | coulomb | 216 | 2.53 (1.86–2.81) | 2.68 | 2.19 | 78.7 | 21.3 | 0 |
| Parallel_Module | ocv_lookup | 216 | 16.32 (9.43–28.30) | 21.07 | 14.36 | 24.1 | 13.9 | 62.0 |

> Nominal protocol: correct initial SOC; all other project tables use the +20 pp (or swept) adversarial wrong-init protocol.
> Calibration split: 10% of segments per vehicle for the fleet datasets (BMW i3, Deng, VED); 40% per cell for CALCE and 30% per module for UMich/Ford — data-size-driven exceptions, decided before evaluation (see data/soc_baseline_benchmark_calce.py / _module.py).
> All SOC errors in percentage points; every estimator (including coulomb counting) starts from the same deliberately wrong initial SOC.
> Initial SOC is clipped to [2%, 98%], so trips starting near a rail receive LESS than the nominal offset; this materially lowers the coulomb baseline's aggregate stress-test RMSE in 23 of 45 dataset×offset cells (e.g. +20 pp: 43% of BMW and 51% of VED trips clipped; clipped-trip coulomb RMSE ~2–12 pp vs ~20 pp unclipped) — a protocol artifact, not estimator skill; medians are far less affected than means. Full grid: results/coulomb_clipping_diagnostic.csv.
