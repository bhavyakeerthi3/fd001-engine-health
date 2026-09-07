# Explainability reasoning

Exact tree-path-dependent TreeSHAP explains XGBoost's raw output before clipping. The fitted tree path counts define its background; no test rows are used as a fitted background. Per-feature contributions are summed by sensor across last/mean/std/slope features, preserving local additivity. Positive contributions raise predicted remaining life, negative contributions lower it. The baseline and sum are checked against model.predict, with maximum residual persisted. Attributions may be shared between correlated sensors; they describe model behavior, not causal fault diagnosis.

Global importance is the mean absolute SENSOR-AGGREGATED contribution across official test endpoints, rather than the sum of absolute statistic contributions. This can show cancellation within sensors. Local endpoint attributions are exported for every test engine; arbitrary observed-cycle explanations are calculated on demand in the dashboard. RF SHAP is omitted to keep the requested optional extension from adding compute and clutter.

