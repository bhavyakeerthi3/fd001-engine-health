# Dashboard reasoning

The single-engine view loads fitted artifacts and uses only that engine's prefix through the selected cycle. It displays relative PCA health, all four predictions, equal-weight ensemble RUL, heuristic spread, raw sensor trends, Isolation Forest flag and signed XGBoost SHAP sensor contributions. Official RUL is optional and explicitly evaluation-only.

Status uses the LOWER heuristic RUL bound with strict less-than boundaries: Red below 20, Orange below 40, Yellow below 80, Green otherwise. Equality enters the next higher band. These values live in config.json, are saved with the trained model and displayed in the app. They are illustrative project policy choices, not NASA/GE limits, learned optimums, safety probabilities or maintenance instructions. The score and anomaly flag are not hidden overrides. A weak interval also makes this status weak.

