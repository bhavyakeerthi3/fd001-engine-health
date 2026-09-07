# Uncertainty reasoning

Take the arithmetic mean and sample standard deviation (ddof=1) across the four bounded model predictions at each window. Display mean +/- 1.96*std, clipped to [0, 125]. Equal model weights are a declared convention, not learned reliability weights. The multiplier borrows a normal-reference scale, but four correlated predictors do NOT yield a calibrated confidence or prediction interval. Shared bias, aleatoric noise, ceiling effects and out-of-distribution uncertainty can all be missed. The dashboard calls this a heuristic spread interval with no nominal coverage claim.

Measured endpoint coverage and mean width against BOTH target definitions are in interval_metrics.csv. These observations are diagnostic only; the multiplier is not tuned to test coverage.

Future uncertainty methods can replace evaluation.ensemble_spread while retaining interval diagnostics and the dashboard contract. No additional uncertainty method is implemented.

