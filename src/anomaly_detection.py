"""Isolation Forest on window features with a training early-life reference."""
from sklearn.ensemble import IsolationForest


def fit_anomaly_detector(windows, config):
    reference = windows.metadata.cycle <= config["healthy_cycles"]
    model = IsolationForest(n_estimators=config["isolation_trees"],
        contamination=config["isolation_contamination"], random_state=config["seed"], n_jobs=config["threads"])
    model.fit(windows.tabular[reference])
    return model
