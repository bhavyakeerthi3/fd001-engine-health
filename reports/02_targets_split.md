# Target and split reasoning

Training RUL is engine max cycle minus current cycle. Test RUL adds the supplied terminal RUL to the last observed cycle minus current cycle. Never treat a censored test endpoint as failure.

The fixed split uses seed 42: 80 fitting engines, 20 validation engines, and 100 separate official test engines. The ID lists and validation truncation cycles are in artifacts/split_manifest.json. Equal numeric IDs in train and test refer to different physical simulated engines. No row split is used. Scaler, selection, PCA and Isolation Forest see fitting engines only.

Fit a piecewise target min(raw RUL, 125). The ceiling is a declared project assumption about weak early-life identifiability, not a fitted damage-onset estimate. It prevents large indistinguishable early-life labels from dominating the loss. It also prevents predictions above the ceiling, even when true RUL is higher. Metrics for BOTH targets and unbounded model outputs are saved; raw-label errors reveal the resulting ceiling bias. The ceiling is fixed before model fitting; no ceiling search or test tuning is performed.

Validation selection uses one deterministic synthetic censored endpoint per validation engine, sampled at a fraction in [0.3, 0.9] of its observed run-to-failure life. This uses held-out lifetimes only to construct an evaluation task, never input features. It avoids selecting on terminal RUL zero for every engine. This censoring distribution is an assumption and may differ from NASA's official censoring. Secondary all-cycle metrics weight engines equally. Training endpoint metrics are all failure endpoints and therefore are not directly comparable with censored validation/test endpoint metrics; their R2 is undefined and left blank.

Windows have length 30, end at the prediction cycle and never cross engines. Short histories repeat the earliest available measurement on the left, include every engine, and report observed_window. This padding implies a flat unobserved history, not actual measurements. No future cycle, total lifetime, RUL, ID, or cycle-age feature enters the predictors. The same window is represented as last/mean/population-std/OLS-slope features for classical models and as a sequence for the LSTM. Training losses weight each engine equally, so long trajectories do not dominate.

