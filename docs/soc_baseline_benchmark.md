# SOC Baseline Benchmark

Compares this project's Dual EKF against two generic, unnamed baselines (pure coulomb counting; pure naive OCV lookup) on the SAME held-out real fleet segments. Does NOT benchmark against any named commercial chip/vendor (TI, ADI, etc.) -- their exact firmware and tuned calibration are not available to this project.

| Fleet | n segments | EKF SOC RMSE | Coulomb-only | OCV-lookup-only | EKF beats both? |
|---|---|---|---|---|---|
| BMW_i3 | 63 | 17.69% | 16.88% | 37.75% | No |
| Deng_BAIC_EU500 | 2000 | 9.69% | 14.83% | 6.05% | No |
| VED | 408 | 25.71% | 14.03% | 14.83% | No |
