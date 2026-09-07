"""Causal windows shared by classical models, LSTM, and anomaly detector."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WindowData:
    sequence: np.ndarray
    tabular: np.ndarray
    metadata: pd.DataFrame
    feature_names: list


def build_windows(df, sensors, scaler, window):
    sequences, metadata = [], []
    for engine, g in df.groupby("engine_id", sort=True):
        g = g.sort_values("cycle")
        values = scaler.transform(g[sensors]).astype(np.float32)
        for j, row in enumerate(g.itertuples(index=False)):
            history = values[max(0, j-window+1):j+1]
            history = np.pad(history, ((window-len(history), 0), (0, 0)), mode="edge")
            sequences.append(history)
            metadata.append({"engine_id": int(engine), "cycle": int(row.cycle),
                "rul_raw": getattr(row, "rul_raw", np.nan),
                "rul_target": getattr(row, "rul_target", np.nan),
                "observed_window": min(j+1, window)})
    seq = np.stack(sequences)
    time = np.arange(window, dtype=np.float32)
    time -= time.mean()
    slopes = np.einsum("ntd,t->nd", seq, time) / np.sum(time**2)
    tabular = np.concatenate([seq[:, -1], seq.mean(1), seq.std(1), slopes], axis=1)
    names = [f"{s}__{stat}" for stat in ["last", "mean", "std", "slope"] for s in sensors]
    return WindowData(seq, tabular, pd.DataFrame(metadata), names)


def endpoint_mask(meta, cutoffs=None):
    if cutoffs is None:
        return (meta.cycle == meta.groupby("engine_id").cycle.transform("max")).to_numpy()
    return (meta.cycle == meta.engine_id.map(cutoffs)).to_numpy()


def engine_weights(meta):
    w = 1.0 / meta.engine_id.map(meta.engine_id.value_counts()).to_numpy()
    return w / w.mean()
