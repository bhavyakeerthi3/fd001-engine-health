"""Recompute metrics from saved predictions without fitting or model selection."""
import sys
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.evaluation import metrics
from src.feature_engineering import engine_weights
from src.reporting import build_reports

rows = []
for partition in ["train","validation","test"]:
    frame = pd.read_csv(ROOT/"reports"/f"{partition}_predictions.csv")
    for scope, selected, weights in [("endpoint", frame.is_endpoint.to_numpy(), None),
            ("all_cycles_engine_weighted", np.ones(len(frame), dtype=bool), engine_weights(frame))]:
        for target in ["rul_target","rul_raw"]:
            for name in ["Linear Regression","Random Forest","XGBoost","LSTM","Ensemble mean"]:
                column = "ensemble_mean" if name == "Ensemble mean" else name
                rows.append({"partition":partition,"scope":scope,"target":target,"model":name,
                    "n_predictions":int(selected.sum()),"n_engines":frame.engine_id.nunique(),
                    **metrics(frame[target].to_numpy()[selected],frame[column].to_numpy()[selected],weights)})
pd.DataFrame(rows).to_csv(ROOT/"reports"/"metrics.csv",index=False)
audit = {"refreshed_utc":datetime.now(timezone.utc).isoformat(),
    "reason":"Recompute from immutable saved predictions; undefined constant-target R2 is NaN, not sklearn's force-finite zero. No retraining or selection.",
    "source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"src").glob("*.py"))},
    "prediction_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"reports").glob("*_predictions.csv")}}
(ROOT/"reports"/"evaluation_refresh.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
build_reports(ROOT)
print("Recomputed metrics and reports from saved predictions.")
