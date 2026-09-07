"""Training-partition diagnostics; no validation/test data influence selection."""
import numpy as np
import pandas as pd
from .preprocessing import SENSORS, SETTINGS


def safe_corr(x, y, method="pearson"):
    if pd.Series(x).nunique() < 2 or pd.Series(y).nunique() < 2:
        return np.nan
    return float(pd.Series(np.asarray(x)).corr(pd.Series(np.asarray(y)), method=method))


def analyze(fit, config):
    records, per_engine = [], []
    for s in SENSORS:
        values = fit[s]
        slopes, ranks, strengths, noise, mono, starts = [], [], [], [], [], []
        for engine, g in fit.groupby("engine_id"):
            x, y = g.cycle.to_numpy(), g[s].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            variance = np.var(y)
            strength = 1 - np.mean((y-(slope*x+intercept))**2) / variance if variance > 1e-16 else np.nan
            smooth = pd.Series(y).rolling(config["window"], min_periods=1).mean().to_numpy()
            delta = np.diff(smooth)
            monotonicity = abs(np.sign(delta).sum()) / len(delta)
            # First differences retain some degradation; this is a noise proxy, not pure measurement noise.
            noise_ratio = np.std(np.diff(y)) / np.sqrt(2*variance) if variance > 1e-16 else np.nan
            rank = safe_corr(x, y, "spearman")
            start = y[:config["hi_anchor_cycles"]].mean()
            slopes.append(slope); ranks.append(abs(rank)); strengths.append(strength)
            noise.append(noise_ratio); mono.append(monotonicity); starts.append(start)
            per_engine.append({"engine_id": engine, "sensor": s, "cycles_survived": len(g),
                "slope_per_cycle": slope, "start_mean": start, "end_mean": y[-config["hi_terminal_cycles"]:].mean(),
                "trend_r2": strength, "spearman_cycle": rank})
        policy = config["sensor_policy"]
        rank_med = float(np.nanmedian(ranks)) if np.isfinite(ranks).any() else np.nan
        strength_med = float(np.nanmedian(strengths)) if np.isfinite(strengths).any() else np.nan
        unique = int(values.nunique())
        if unique < policy["minimum_unique"]:
            verdict = "Remove"
            reason = f"Only {unique} unique values; effectively constant or binary near-constant signal under this regime."
        elif rank_med < policy["minimum_median_abs_spearman"] or strength_med < policy["minimum_trend_r2"]:
            verdict = "Investigate"
            reason = f"Weak within-engine trend: median |rho|={rank_med:.3f}, linear trend R2={strength_med:.3f}; excluded pending evidence."
        else:
            verdict = "Keep"
            reason = f"Consistent within-engine trend: median |rho|={rank_med:.3f}, trend R2={strength_med:.3f}; retain and standardize on training engines."
        records.append({"sensor": s, "unique_values": unique, "variance": values.var(),
            "pearson_cycle": safe_corr(values, fit.cycle), "pearson_rul_raw": safe_corr(values, fit.rul_raw),
            "pearson_rul_clipped": safe_corr(values, fit.rul_target),
            "median_abs_spearman_cycle": rank_med, "median_monotonicity_smoothed": np.median(mono),
            "median_trend_r2": strength_med, "median_noise_ratio": np.nanmedian(noise) if np.isfinite(noise).any() else np.nan,
            "slope_median": np.median(slopes), "slope_iqr": np.quantile(slopes,.75)-np.quantile(slopes,.25),
            "starting_mean_std_between_engines": np.std(starts, ddof=1),
            "classification": verdict, "preprocessing": "Normalize" if verdict == "Keep" else "None",
            "reason": reason})
    return pd.DataFrame(records), pd.DataFrame(per_engine), fit[SETTINGS].describe().T
