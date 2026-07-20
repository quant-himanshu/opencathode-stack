# Coulomb-baseline initial-SOC clipping diagnostic

Generated 20260719T195353Z from `offset_sweep_per_trip_20260719T193500Z.csv`. Clip bounds [2%, 98%]. A flagged cell means the coulomb baseline's aggregate RMSE at that offset is materially lowered because ≥10% of trips started near a rail and received a smaller applied offset than nominal — a protocol artifact, not estimator skill.

| Dataset | Offset (pp) | n | clipped | frac | mean applied (pp) | RMSE med all | clipped | unclipped |
|---|---|---|---|---|---|---|---|---|
**23 flagged (material) cells — table shows flagged cells only; full grid in the CSV.**
| BMW_i3 | +20 | 63 | 27 | 42.9% | +17.1 | 19.658 | 12.406 | 19.814 |
| BMW_i3 | +30 | 63 | 45 | 71.4% | +21.3 | 22.954 | 15.176 | 29.79 |
| CALCE_A123 | -30 | 26 | 8 | 30.8% | -23.2 | 29.858 | 4.665 | 29.988 |
| CALCE_A123 | -20 | 26 | 7 | 26.9% | -16.1 | 19.903 | 1.744 | 19.997 |
| CALCE_A123 | -10 | 26 | 4 | 15.4% | -8.4 | 9.988 | 0.804 | 10.014 |
| CALCE_A123 | -5 | 26 | 4 | 15.4% | -4.2 | 4.988 | 0.804 | 5.014 |
| CALCE_A123 | +5 | 26 | 4 | 15.4% | +4.4 | 4.951 | 0.686 | 4.97 |
| CALCE_A123 | +10 | 26 | 5 | 19.2% | +8.7 | 9.91 | 0.88 | 9.952 |
| CALCE_A123 | +20 | 26 | 7 | 26.9% | +16.1 | 19.89 | 1.894 | 19.933 |
| CALCE_A123 | +30 | 26 | 8 | 30.8% | +23.2 | 29.883 | 5.727 | 29.943 |
| Deng_BAIC_EU500 | -30 | 2000 | 810 | 40.5% | -25.0 | 32.254 | 24.879 | 33.466 |
| Deng_BAIC_EU500 | -20 | 2000 | 452 | 22.6% | -18.2 | 23.521 | 19.258 | 23.972 |
| Parallel_Module | -30 | 216 | 55 | 25.5% | -25.9 | 27.843 | 14.628 | 30.436 |
| Parallel_Module | -20 | 216 | 38 | 17.6% | -18.1 | 19.132 | 9.917 | 20.453 |
| Parallel_Module | -10 | 216 | 22 | 10.2% | -9.4 | 10.349 | 3.908 | 10.454 |
| Parallel_Module | +5 | 216 | 38 | 17.6% | +4.3 | 3.493 | 1.381 | 4.012 |
| Parallel_Module | +10 | 216 | 58 | 26.9% | +8.2 | 7.917 | 2.259 | 8.675 |
| Parallel_Module | +20 | 216 | 76 | 35.2% | +15.1 | 17.752 | 4.667 | 19.649 |
| Parallel_Module | +30 | 216 | 98 | 45.4% | +21.0 | 25.997 | 8.242 | 29.647 |
| VED | +5 | 408 | 127 | 31.1% | +3.3 | 5.433 | 2.798 | 5.92 |
| VED | +10 | 408 | 150 | 36.8% | +6.6 | 10.237 | 3.062 | 10.839 |
| VED | +20 | 408 | 209 | 51.2% | +12.1 | 19.748 | 3.664 | 20.652 |
| VED | +30 | 408 | 280 | 68.6% | +16.2 | 20.768 | 9.154 | 30.657 |
