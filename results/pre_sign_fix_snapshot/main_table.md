# Main Table — standardized metrics (Phase 1)

Generated 20260719T130327Z by `data/run_main_table.py` (git `9b749c0`). All numbers in percentage points. Median (IQR) primary; mean kept for comparability with pre-Phase-1 headline numbers (which were means of per-trip full-window RMSE — see docs/METRICS.md).

| Dataset | Method | n | RMSE med (IQR) | RMSE mean | MAE med | MaxErr med | RMSE post-conv med | Conv% strict | t_conv strict med (s) | Conv% hold-600 | Conv% legacy | t_conv legacy med (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BMW_i3 | ekf | 63 | 17.61 (13.09–19.88) | 17.69 | 16.23 | 20.01 | 2.05 | 7.9 | 903 | 9.5 | 14.3 | 982 |
| BMW_i3 | coulomb | 63 | 28.39 (23.74–35.86) | 29.42 | 27.60 | 35.80 |  | 0.0 |  | 0.0 | 1.6 | 1527 |
| BMW_i3 | ocv_lookup | 63 | 38.43 (25.66–49.99) | 37.75 | 30.41 | 73.40 | 2.14 | 39.7 | 1287 | 39.7 | 79.4 | 118 |
| Deng_BAIC_EU500 | ekf | 2000 | 9.28 (5.86–12.16) | 9.69 | 6.91 | 21.06 | 3.32 | 83.9 | 2642 | 93.5 | 91.0 | 912 |
| Deng_BAIC_EU500 | coulomb | 2000 | 43.42 (26.06–56.10) | 40.12 | 35.91 | 78.81 | 2.77 | 3.5 | 526 | 8.6 | 96.0 | 530 |
| Deng_BAIC_EU500 | ocv_lookup | 2000 | 6.35 (4.65–7.33) | 6.05 | 5.30 | 16.00 | 3.65 | 74.8 | 3731 | 92.9 | 95.6 | 10 |
| VED | ekf | 408 | 23.98 (11.49–34.74) | 25.71 | 22.04 | 31.85 | 1.67 | 16.2 | 139 | 19.9 | 37.5 | 0 |
| VED | coulomb | 408 | 20.81 (6.06–22.67) | 16.05 | 20.76 | 22.08 | 1.33 | 15.4 | 0 | 19.1 | 31.1 | 0 |
| VED | ocv_lookup | 408 | 11.40 (7.67–20.80) | 14.83 | 9.25 | 31.20 | 2.70 | 35.3 | 444 | 36.5 | 78.4 | 30 |
| CALCE_A123 | ekf | 26 | 25.29 (16.06–45.12) | 35.18 | 23.45 | 49.10 | 2.67 | 7.7 | 1471 | 7.7 | 42.3 | 533 |
| CALCE_A123 | coulomb | 26 | 19.89 (13.63–19.99) | 15.66 | 19.89 | 20.02 | 1.39 | 23.1 | 0 | 23.1 | 23.1 | 0 |
| CALCE_A123 | ocv_lookup | 26 | 34.68 (23.92–52.97) | 36.81 | 31.30 | 81.68 | 1.94 | 26.9 | 1253 | 38.5 | 61.5 | 578 |
| Parallel_Module | ekf | 216 | 15.57 (6.43–23.52) | 15.49 | 15.30 | 20.00 | 2.37 | 26.9 | 25 | 28.7 | 26.4 | 0 |
| Parallel_Module | coulomb | 216 | 17.75 (10.46–22.22) | 15.33 | 17.70 | 20.00 | 1.62 | 19.0 | 0 | 19.0 | 19.0 | 0 |
| Parallel_Module | ocv_lookup | 216 | 16.32 (9.43–28.30) | 21.07 | 14.36 | 25.01 | 2.71 | 24.1 | 635 | 26.4 | 16.7 | 0 |

> Calibration split: 10% of segments per vehicle for the fleet datasets (BMW i3, Deng, VED); 40% per cell for CALCE and 30% per module for UMich/Ford — data-size-driven exceptions, decided before evaluation (see data/soc_baseline_benchmark_calce.py / _module.py).
> All SOC errors in percentage points; every estimator (including coulomb counting) starts from the same deliberately wrong initial SOC.
