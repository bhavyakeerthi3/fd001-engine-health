"""Bounded model comparison, equal-engine loss weighting, CPU reproducibility."""
import copy
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .feature_engineering import engine_weights
from .evaluation import metrics


class RULLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=48, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1]).squeeze(-1)


def predict_lstm(model, sequence, cap, batch_size=512):
    model.eval()
    out = []
    with torch.inference_mode():
        for batch in range(0, len(sequence), batch_size):
            out.append(model(torch.from_numpy(sequence[batch:batch+batch_size])).numpy()*cap)
    return np.concatenate(out)


def fit_models(train, val, val_mask, config, log=print):
    seed, cap = config["seed"], config["rul_cap"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    x, y = train.tabular, train.metadata.rul_target.to_numpy()
    vx, vy = val.tabular[val_mask], val.metadata.rul_target.to_numpy()[val_mask]
    weights = engine_weights(train.metadata)
    models, trials, timings = {}, [], {}
    candidates = {
        "Linear Regression": [LinearRegression()],
        "Random Forest": [RandomForestRegressor(**p, n_jobs=config["threads"], random_state=seed) for p in config["rf_candidates"]],
        "XGBoost": [XGBRegressor(**p, objective="reg:squarederror", tree_method="hist", subsample=.9,
                   colsample_bytree=.9, reg_lambda=1, n_jobs=config["threads"], random_state=seed) for p in config["xgb_candidates"]]}
    for name, options in candidates.items():
        best = np.inf
        total = time.perf_counter()
        for index, model in enumerate(options):
            start = time.perf_counter(); model.fit(x, y, sample_weight=weights)
            elapsed = time.perf_counter()-start
            result = metrics(vy, np.clip(model.predict(vx), 0, cap))
            trials.append({"model": name, "candidate": index, "fit_seconds": elapsed, **result})
            log(f"{name} candidate {index}: validation RMSE {result['RMSE']:.3f}", flush=True)
            if result["RMSE"] < best:
                best = result["RMSE"]; models[name] = model
                timings[name] = {"selected_fit_seconds": elapsed, "selected_candidate": index}
        timings[name]["search_seconds"] = time.perf_counter()-total
    p = config["lstm"]
    model = RULLSTM(train.sequence.shape[-1], p["hidden_size"], p["layers"])
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    data = TensorDataset(torch.from_numpy(train.sequence), torch.tensor(y/cap, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32))
    loader = DataLoader(data, batch_size=p["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0)
    history, best, stale, best_state = [], np.inf, 0, None
    start = time.perf_counter()
    for epoch in range(1, p["max_epochs"]+1):
        model.train(); total_loss = 0
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            loss = ((model(xb)-yb).square()*wb).mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total_loss += loss.item()*len(xb)
        pred = np.clip(predict_lstm(model, val.sequence[val_mask], cap), 0, cap)
        score = metrics(vy, pred)["RMSE"]
        history.append({"epoch": epoch, "train_weighted_mse_normalized": total_loss/len(data), "validation_endpoint_RMSE": score})
        log(f"LSTM epoch {epoch}: validation RMSE {score:.3f}", flush=True)
        if score < best:
            best = score; stale = 0; best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch
        else:
            stale += 1
        if stale >= p["patience"]:
            break
    model.load_state_dict(best_state); models["LSTM"] = model
    timings["LSTM"] = {"selected_fit_seconds": time.perf_counter()-start, "best_epoch": best_epoch,
        "epochs_run": epoch, "parameters": sum(v.numel() for v in model.parameters())}
    return models, pd.DataFrame(trials), pd.DataFrame(history), timings


def predict_models(models, windows, cap):
    raw = {name: predict_lstm(model, windows.sequence, cap) if name == "LSTM" else model.predict(windows.tabular)
           for name, model in models.items()}
    return {name: np.clip(p, 0, cap) for name, p in raw.items()}, raw
