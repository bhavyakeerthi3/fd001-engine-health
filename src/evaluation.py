"""Metrics, correlated-model spread, and paired engine bootstrap."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, p, weights=None):
    return {"MAE": float(mean_absolute_error(y, p, sample_weight=weights)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p, sample_weight=weights))),
            "R2": float(r2_score(y, p, sample_weight=weights)) if np.ptp(y) > 0 else float("nan")}


def ensemble_spread(predictions, cap, multiplier):
    stack = np.column_stack(list(predictions.values()))
    center, spread = stack.mean(1), stack.std(1, ddof=1)
    return center, np.clip(center-multiplier*spread, 0, cap), np.clip(center+multiplier*spread, 0, cap), spread


def interval_metrics(y, lo, hi):
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi-lo))}


def paired_bootstrap(y, xgb, lstm, config):
    rng = np.random.default_rng(config["seed"])
    indices = rng.integers(0, len(y), (config["bootstrap_repeats"], len(y)))
    delta = np.sqrt(np.mean((y[indices]-xgb[indices])**2, axis=1))-np.sqrt(np.mean((y[indices]-lstm[indices])**2, axis=1))
    return {"rmse_improvement_lstm_over_xgb": float(np.sqrt(np.mean((y-xgb)**2))-np.sqrt(np.mean((y-lstm)**2))),
            "paired_bootstrap_95_low": float(np.quantile(delta,.025)),
            "paired_bootstrap_95_high": float(np.quantile(delta,.975)),
            "repeats": config["bootstrap_repeats"]}


def maintenance_status(lower_rul, thresholds):
    for color, key in [("Red", "red_below"), ("Orange", "orange_below"), ("Yellow", "yellow_below")]:
        if lower_rul < thresholds[key]:
            return color
    return "Green"
