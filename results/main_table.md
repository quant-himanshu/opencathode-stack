# Main Table — standardized metrics (Phase 1)

Generated 20260719T193437Z by `data/run_main_table.py` (git `9b749c0`). All numbers in percentage points. Median (IQR) primary; mean kept for comparability with pre-Phase-1 headline numbers (which were means of per-trip full-window RMSE — see docs/METRICS.md).

| Dataset | Method | n | RMSE med (IQR) | RMSE mean | MAE med | MaxErr med | RMSE post-conv med | Conv% strict | t_conv strict med (s) | Conv% hold-600 | Conv% legacy | t_conv legacy med (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BMW_i3 | ekf | 63 | 17.61 (13.09–19.88) | 17.69 | 16.23 | 20.01 | 2.05 | 7.9 | 903 | 9.5 | 14.3 | 982 |
| BMW_i3 | coulomb | 63 | 19.66 (12.71–19.84) | 16.88 | 19.66 | 20.01 |  | 0.0 |  | 0.0 | 0.0 |  |
| BMW_i3 | ocv_lookup | 63 | 38.43 (25.66–49.99) | 37.75 | 30.41 | 73.40 | 2.14 | 39.7 | 1287 | 39.7 | 79.4 | 118 |
| Deng_BAIC_EU500 | ekf | 2000 | 9.28 (5.86–12.16) | 9.69 | 6.91 | 21.06 | 3.32 | 83.9 | 2642 | 93.5 | 91.0 | 912 |
| Deng_BAIC_EU500 | coulomb | 2000 | 14.83 (13.86–15.79) | 14.83 | 14.02 | 20.12 | 3.66 | 62.0 | 4073 | 62.0 | 39.9 | 4224 |
| Deng_BAIC_EU500 | ocv_lookup | 2000 | 6.35 (4.65–7.33) | 6.05 | 5.30 | 16.00 | 3.65 | 74.8 | 3731 | 92.9 | 95.6 | 10 |
| VED | ekf | 408 | 23.98 (11.49–34.74) | 25.71 | 22.04 | 31.85 | 1.67 | 16.2 | 139 | 19.9 | 37.5 | 0 |
| VED | coulomb | 408 | 19.75 (3.53–20.69) | 14.03 | 19.75 | 20.30 | 2.66 | 26.7 | 0 | 28.2 | 31.1 | 0 |
| VED | ocv_lookup | 408 | 11.40 (7.67–20.80) | 14.83 | 9.25 | 31.20 | 2.70 | 35.3 | 444 | 36.5 | 78.4 | 30 |
| CALCE_A123 | ekf | 26 | 31.70 (18.51–61.40) | 38.28 | 28.06 | 51.05 | 2.85 | 26.9 | 918 | 34.6 | 50.0 | 494 |
| CALCE_A123 | coulomb | 26 | 19.89 (13.63–19.99) | 15.66 | 19.89 | 20.02 | 1.39 | 23.1 | 0 | 23.1 | 23.1 | 0 |
| CALCE_A123 | ocv_lookup | 26 | 34.68 (23.92–52.97) | 36.81 | 31.30 | 81.68 | 1.94 | 26.9 | 1253 | 38.5 | 61.5 | 578 |
| Parallel_Module | ekf | 216 | 16.89 (6.81–21.17) | 15.41 | 16.48 | 20.00 | 1.91 | 26.9 | 0 | 27.3 | 24.5 | 0 |
| Parallel_Module | coulomb | 216 | 17.75 (10.46–22.22) | 15.33 | 17.70 | 20.00 | 1.62 | 19.0 | 0 | 19.0 | 19.0 | 0 |
| Parallel_Module | ocv_lookup | 216 | 16.32 (9.43–28.30) | 21.07 | 14.36 | 25.01 | 2.71 | 24.1 | 635 | 26.4 | 16.7 | 0 |

> Calibration split: 10% of segments per vehicle for the fleet datasets (BMW i3, Deng, VED); 40% per cell for CALCE and 30% per module for UMich/Ford — data-size-driven exceptions, decided before evaluation (see data/soc_baseline_benchmark_calce.py / _module.py).
> All SOC errors in percentage points; every estimator (including coulomb counting) starts from the same deliberately wrong initial SOC.

> Initial-SOC clipping to [2%,98%] materially lowers coulomb stress-test aggregates in 23/45 dataset-offset cells (see results/coulomb_clipping_diagnostic.csv); medians far less affected than means. (Footnote appended post-generation, 2026-07-20.)
