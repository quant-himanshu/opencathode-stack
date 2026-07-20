# Nominal accuracy (offset = 0, correct initialization)

Regenerated 20260719T183042Z (wide-grid θ). Median (IQR) primary, mean secondary — all pp.

| Dataset | Method | n | RMSE med (IQR) | RMSE mean | MAE med | Converged% | Recovered% | Diverged% |
|---|---|---|---|---|---|---|---|---|
| BMW_i3 | my_ekf | 63 | 13.94 (0.25–21.50) | 15.48 | 11.53 | 33.3 | 7.9 | 58.7 |
| BMW_i3 | rbc_dekf | 63 | 13.39 (0.25–23.72) | 15.07 | 10.15 | 34.9 | 11.1 | 54.0 |
| BMW_i3 | coulomb | 63 | 16.75 (10.81–20.74) | 17.28 | 14.66 | 1.6 | 6.3 | 92.1 |
| BMW_i3 | ocv_lookup | 63 | 38.43 (25.66–49.99) | 37.75 | 30.41 | 39.7 | 4.8 | 55.6 |
| Deng_BAIC_EU500 | my_ekf | 2000 | 9.21 (5.45–11.44) | 8.76 | 6.80 | 87.2 | 10.8 | 2.1 |
| Deng_BAIC_EU500 | rbc_dekf | 2000 | 6.07 (4.33–7.39) | 5.88 | 4.61 | 86.0 | 13.7 | 0.4 |
| Deng_BAIC_EU500 | coulomb | 2000 | 55.74 (40.81–62.54) | 49.58 | 49.19 | 0.3 | 1.4 | 98.3 |
| Deng_BAIC_EU500 | ocv_lookup | 2000 | 6.35 (4.65–7.33) | 6.05 | 5.30 | 74.8 | 16.8 | 8.5 |
| VED | my_ekf | 408 | 12.87 (5.54–23.59) | 17.62 | 11.26 | 22.3 | 15.0 | 62.7 |
| VED | rbc_dekf | 408 | 10.94 (4.58–21.80) | 15.68 | 9.49 | 26.2 | 18.9 | 54.9 |
| VED | coulomb | 408 | 2.86 (1.61–4.88) | 3.98 | 2.45 | 51.2 | 29.4 | 19.4 |
| VED | ocv_lookup | 408 | 11.40 (7.67–20.80) | 14.83 | 9.25 | 35.3 | 21.8 | 42.9 |
| CALCE_A123 | my_ekf | 26 | 28.48 (15.11–43.60) | 28.99 | 24.26 | 7.7 | 3.8 | 88.5 |
| CALCE_A123 | rbc_dekf | 26 | 23.98 (15.31–31.18) | 23.95 | 21.25 | 7.7 | 3.8 | 88.5 |
| CALCE_A123 | coulomb | 26 | 0.10 (0.05–0.15) | 0.20 | 0.08 | 100.0 | 0 | 0 |
| CALCE_A123 | ocv_lookup | 26 | 34.68 (23.92–52.97) | 36.81 | 31.30 | 26.9 | 15.4 | 57.7 |
| Parallel_Module | my_ekf | 216 | 4.96 (2.59–8.98) | 9.05 | 4.28 | 37.0 | 34.7 | 28.2 |
| Parallel_Module | rbc_dekf | 216 | 5.06 (2.98–9.66) | 8.69 | 4.33 | 29.2 | 36.6 | 34.3 |
| Parallel_Module | coulomb | 216 | 2.53 (1.86–2.81) | 2.68 | 2.19 | 78.7 | 21.3 | 0 |
| Parallel_Module | ocv_lookup | 216 | 16.32 (9.43–28.30) | 21.07 | 14.36 | 24.1 | 13.9 | 62.0 |

> Nominal protocol: correct initial SOC; all other project tables use the +20 pp (or swept) adversarial wrong-init protocol.
> Calibration split: 10% of segments per vehicle for the fleet datasets (BMW i3, Deng, VED); 40% per cell for CALCE and 30% per module for UMich/Ford — data-size-driven exceptions, decided before evaluation (see data/soc_baseline_benchmark_calce.py / _module.py).
> All SOC errors in percentage points; every estimator (including coulomb counting) starts from the same deliberately wrong initial SOC.
