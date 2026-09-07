"""Run in the repository root: python -m src.pipeline."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from .preprocessing import load_fd001, split_engines
from .feature_engineering import build_windows, endpoint_mask, engine_weights
from .eda import analyze
from .health_indicator import PCAHealthIndicator
from .anomaly_detection import fit_anomaly_detector
from .rul_models import fit_models, predict_models
from .evaluation import metrics, ensemble_spread, interval_metrics, paired_bootstrap
from .explainability import explain


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); root = args.root
    config = json.loads((root/"config.json").read_text())
    artifacts, reports = root/"artifacts", root/"reports"
    for folder in [artifacts, reports, reports/"figures", root/"notebooks"]:
        folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    train, test = load_fd001(root/"data"/"raw", config["rul_cap"])
    if train.engine_id.nunique() != 100 or test.engine_id.nunique() != 100:
        raise ValueError("This build is strictly the standard FD001 subset")
    fit, val, split = split_engines(train, config)
    save_json(artifacts/"split_manifest.json", split)
    save_json(artifacts/"config.json", config)
    sensors, engine_sensors, settings = analyze(fit, config)
    sensors.to_csv(reports/"sensor_analysis.csv", index=False)
    engine_sensors.to_csv(reports/"engine_sensor_analysis.csv", index=False)
    settings.to_csv(reports/"operating_settings.csv")
    lifetime = train.groupby("engine_id").cycle.max().rename("cycles_survived").to_frame()
    lifetime["partition"] = np.where(lifetime.index.isin(split["train_engine_ids"]), "train", "validation")
    lifetime.to_csv(reports/"engine_lifetimes.csv")
    kept = sensors.loc[sensors.classification == "Keep", "sensor"].tolist()
    if not kept:
        raise ValueError("No sensors satisfy training-only retention rules")
    print(f"Retained sensors: {kept}", flush=True)
    scaler = StandardScaler().fit(fit[kept])
    scaled = scaler.transform(fit[kept])
    hi = PCAHealthIndicator().fit(scaled, fit, config)
    hi_details = {"sensors": kept, "loadings": dict(zip(kept, hi.pca.components_[0].tolist())),
        "explained_variance_ratio": float(hi.pca.explained_variance_ratio_[0]),
        "healthy_anchor": hi.healthy_anchor, "failed_anchor": hi.failed_anchor,
        "score_meaning": "100=training early-life median, 0=training terminal median; clipped, not failure probability"}
    save_json(reports/"health_indicator.json", hi_details)
    tw = build_windows(fit, kept, scaler, config["window"])
    vw = build_windows(val, kept, scaler, config["window"])
    vm = endpoint_mask(vw.metadata, split["validation_cutoffs"])
    anomaly = fit_anomaly_detector(tw, config)
    train_decision = anomaly.decision_function(tw.tabular)
    reference = tw.metadata.cycle <= config["healthy_cycles"]
    anomaly_details = {"reference_windows": int(reference.sum()),
        "reference_flag_rate": float(np.mean(train_decision[reference] < 0)),
        "all_training_flag_rate": float(np.mean(train_decision < 0)),
        "offset": float(anomaly.offset_), "flag_threshold_decision_function": 0.0,
        "label_warning": "No point anomaly labels; no anomaly precision/recall claim. Early life is a proxy for nominal state."}
    save_json(reports/"anomaly_summary.json", anomaly_details)
    # Test arrays are deliberately constructed only after all model selection completes.
    models, trials, history, timings = fit_models(tw, vw, vm, config)
    trials.to_csv(reports/"model_selection.csv", index=False)
    history.to_csv(reports/"lstm_history.csv", index=False)
    save_json(reports/"training_cost.json", timings)
    joblib.dump({"sensors": kept, "scaler": scaler, "hi": hi, "anomaly": anomaly,
        "models": {k:v for k,v in models.items() if k != "LSTM"}, "feature_names": tw.feature_names,
        "config": config}, artifacts/"pipeline.joblib", compress=3)
    torch.save(models["LSTM"].state_dict(), artifacts/"lstm.pt")
    models["XGBoost"].save_model(artifacts/"xgboost.json")
    print("Model selection complete. Opening final test evaluation.", flush=True)
    sw = build_windows(test, kept, scaler, config["window"])
    metric_rows, interval_rows, prediction_frames = [], [], {}
    for partition, windows, mask in [("train", tw, endpoint_mask(tw.metadata)), ("validation", vw, vm), ("test", sw, endpoint_mask(sw.metadata))]:
        t0 = time.perf_counter()
        predictions, raw_predictions = predict_models(models, windows, config["rul_cap"])
        frame = windows.metadata.copy()
        for name, p in predictions.items():
            frame[name] = p
            frame[f"{name}__unbounded"] = raw_predictions[name]
        center, lower, upper, spread = ensemble_spread(predictions, config["rul_cap"], config["spread_multiplier"])
        frame["ensemble_mean"], frame["interval_lower"], frame["interval_upper"], frame["ensemble_std"] = center, lower, upper, spread
        frame["is_endpoint"] = mask
        raw_df = {"train": fit, "validation": val, "test": test}[partition]
        frame["health_score"] = hi.transform(scaler.transform(raw_df[kept]))
        frame["anomaly_decision"] = anomaly.decision_function(windows.tabular)
        frame["anomaly_flag"] = frame.anomaly_decision < 0
        frame.to_csv(reports/f"{partition}_predictions.csv", index=False)
        prediction_frames[partition] = frame
        for scope, selected, weights in [("endpoint", mask, None), ("all_cycles_engine_weighted", np.ones(len(frame), dtype=bool), engine_weights(frame))]:
            for target in ["rul_target", "rul_raw"]:
                for name, p in {**predictions, "Ensemble mean": center}.items():
                    metric_rows.append({"partition": partition, "scope": scope, "target": target, "model": name,
                        "n_predictions": int(selected.sum()), "n_engines": frame.engine_id.nunique(),
                        **metrics(frame[target].to_numpy()[selected], p[selected], weights)})
            if scope == "endpoint":
                for target in ["rul_target", "rul_raw"]:
                    interval_rows.append({"partition": partition, "target": target,
                        **interval_metrics(frame[target].to_numpy()[selected], lower[selected], upper[selected])})
        timings.setdefault("inference", {})[partition] = {"rows": len(frame), "all_models_seconds": time.perf_counter()-t0}
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(reports/"metrics.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(reports/"interval_metrics.csv", index=False)
    endpoint = prediction_frames["test"].query("is_endpoint")
    comparison = paired_bootstrap(endpoint.rul_target.to_numpy(), endpoint.XGBoost.to_numpy(), endpoint.LSTM.to_numpy(), config)
    save_json(reports/"lstm_xgb_comparison.json", comparison)
    # Exact local and global XGBoost explanations at every official test endpoint.
    em = endpoint_mask(sw.metadata)
    explanation, sensor_shap, error = explain(models["XGBoost"], sw.tabular[em], sw.feature_names)
    sensor_shap.insert(0, "engine_id", endpoint.engine_id.to_numpy())
    sensor_shap.to_csv(reports/"shap_sensor_contributions.csv", index=False)
    np.savez_compressed(artifacts/"test_shap.npz", values=explanation.values, base_values=explanation.base_values,
        features=sw.tabular[em], feature_names=np.array(sw.feature_names))
    pd.DataFrame({"sensor": kept, "mean_abs_shap": sensor_shap[kept].abs().mean().to_numpy()}).sort_values("mean_abs_shap", ascending=False).to_csv(reports/"shap_global.csv", index=False)
    save_json(reports/"shap_validation.json", {"max_additivity_error_cycles": error,
        "explained_output": "Raw XGBoost output before [0, cap] clipping; signed statistics summed per sensor",
        "baseline": float(np.asarray(explanation.base_values).ravel()[0])})
    run = {"completed_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "processor": platform.processor(), "torch_device": "cpu",
        "train_rows": len(fit), "validation_rows": len(val), "test_rows": len(test),
        "train_engines": fit.engine_id.nunique(), "validation_engines": val.engine_id.nunique(), "test_engines": test.engine_id.nunique(),
        "kept_sensors": kept, "total_pipeline_seconds": time.perf_counter()-start,
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost", "torch", "shap", "streamlit", "matplotlib", "plotly"]},
        "source_hashes": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.joinpath("src").glob("*.py"))},
        "config_sha256": hashlib.sha256((root/"config.json").read_bytes()).hexdigest()}
    save_json(reports/"run_manifest.json", run)
    save_json(reports/"training_cost.json", timings)
    from .reporting import build_reports
    build_reports(root)
    print(metric_frame.query("partition == 'test' and scope == 'endpoint'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
