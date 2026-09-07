"""Strict NASA parsing, engine partitioning, and target construction."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
COLUMNS = ["engine_id", "cycle", *SETTINGS, *SENSORS]


def load_trajectories(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected 26 columns, got {df.shape[1]} in {path}")
    df.columns = COLUMNS
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Non-finite measurement")
    for col in ["engine_id", "cycle"]:
        if not (df[col] == df[col].astype(int)).all() or (df[col] < 1).any():
            raise ValueError(f"Invalid {col}")
        df[col] = df[col].astype(int)
    if df.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle")
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    for _, g in df.groupby("engine_id"):
        if not np.array_equal(g.cycle, np.arange(1, len(g) + 1)):
            raise ValueError("Cycles must start at one and be contiguous")
    return df


def add_rul(df, cap, terminal_rul=None):
    out = df.copy()
    remaining = out.groupby("engine_id").cycle.transform("max") - out.cycle
    if terminal_rul is not None:
        if set(terminal_rul.index) != set(out.engine_id.unique()):
            raise ValueError("RUL labels do not match engine IDs")
        remaining = remaining + out.engine_id.map(terminal_rul)
    out["rul_raw"] = remaining.astype(float)
    out["rul_target"] = np.minimum(remaining, cap).astype(float)
    return out


def load_fd001(data_dir, cap):
    data_dir = Path(data_dir)
    train = load_trajectories(data_dir / "train_FD001.txt")
    test = load_trajectories(data_dir / "test_FD001.txt")
    rul = pd.read_csv(data_dir / "RUL_FD001.txt", sep=r"\s+", header=None)
    if rul.shape != (test.engine_id.nunique(), 1) or (rul[0] < 0).any() or not np.isfinite(rul).all().all():
        raise ValueError("Invalid terminal RUL file")
    if not np.array_equal(np.sort(test.engine_id.unique()), np.arange(1, len(rul) + 1)):
        raise ValueError("NASA RUL lines map to sequential test engine IDs")
    labels = pd.Series(rul[0].to_numpy(), index=np.arange(1, len(rul) + 1))
    return add_rul(train, cap), add_rul(test, cap, labels)


def split_engines(train, config):
    fit_ids, val_ids = train_test_split(np.sort(train.engine_id.unique()),
        test_size=config["validation_fraction"], random_state=config["seed"])
    fit = train[train.engine_id.isin(fit_ids)].reset_index(drop=True)
    val = train[train.engine_id.isin(val_ids)].reset_index(drop=True)
    rng = np.random.default_rng(config["seed"])
    low, high = config["validation_cutoff_fraction"]
    cutoffs = {int(e): max(1, int(n * rng.uniform(low, high)))
               for e, n in val.groupby("engine_id").cycle.max().items()}
    manifest = {"train_engine_ids": sorted(map(int, fit_ids)),
                "validation_engine_ids": sorted(map(int, val_ids)),
                "validation_cutoffs": cutoffs,
                "test_namespace": "NASA test IDs are different engines from equally numbered train IDs"}
    return fit, val, manifest
