"""Load fitted state and predict a single engine's observed prefix."""
from pathlib import Path
import joblib
import numpy as np
import torch
from .rul_models import RULLSTM, predict_models
from .feature_engineering import build_windows, WindowData
from .evaluation import ensemble_spread, maintenance_status
from .explainability import explain


def load_artifacts(root):
    root = Path(root)
    bundle = joblib.load(root/"artifacts"/"pipeline.joblib")
    p = bundle["config"]["lstm"]
    torch.set_num_threads(bundle["config"]["threads"])
    lstm = RULLSTM(len(bundle["sensors"]), p["hidden_size"], p["layers"])
    lstm.load_state_dict(torch.load(root/"artifacts"/"lstm.pt", map_location="cpu", weights_only=True))
    lstm.eval(); bundle["models"]["LSTM"] = lstm
    return bundle


def predict_prefix(prefix, bundle):
    if prefix.engine_id.nunique() != 1 or prefix.empty:
        raise ValueError("Select exactly one non-empty engine prefix")
    prefix = prefix.sort_values("cycle").reset_index(drop=True)
    cfg, sensors = bundle["config"], bundle["sensors"]
    windows = build_windows(prefix, sensors, bundle["scaler"], cfg["window"])
    last = WindowData(windows.sequence[-1:], windows.tabular[-1:], windows.metadata.iloc[-1:], windows.feature_names)
    predictions, raw = predict_models(bundle["models"], last, cfg["rul_cap"])
    mean, lower, upper, spread = ensemble_spread(predictions, cfg["rul_cap"], cfg["spread_multiplier"])
    explanation, contributions, error = explain(bundle["models"]["XGBoost"], last.tabular, windows.feature_names)
    health = bundle["hi"].transform(bundle["scaler"].transform(prefix[sensors]))
    decision = bundle["anomaly"].decision_function(windows.tabular)
    return {"predictions": {k: float(v[0]) for k,v in predictions.items()},
        "raw_xgb": float(raw["XGBoost"][0]), "shap_baseline": float(explanation.base_values[0]),
        "shap": contributions.iloc[0], "shap_error": error,
        "mean": float(mean[0]), "lower": float(lower[0]), "upper": float(upper[0]), "spread": float(spread[0]),
        "status": maintenance_status(lower[0], cfg["maintenance_rul_thresholds"]),
        "health_history": health, "anomaly_history": decision, "window_observed": int(last.metadata.observed_window.iloc[0])}
