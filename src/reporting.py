"""Generate interview notes, figures, notebook, and README from recorded outputs."""
import base64
import json
import subprocess
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from .schema import SENSOR_DESCRIPTIONS


def markdown_table(df):
    def cell(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x).replace("|", "/").replace("\n", " ")
    return "| " + " | ".join(map(str, df.columns)) + " |\n| " + " | ".join(["---"]*len(df.columns)) + " |\n" + "\n".join("| " + " | ".join(cell(x) for x in row) + " |" for row in df.itertuples(index=False, name=None))


def build_reports(root):
    reports = root/"reports"
    read = lambda name: json.loads((reports/name).read_text())
    cfg = json.loads((root/"artifacts"/"config.json").read_text())
    run, hi, comparison = read("run_manifest.json"), read("health_indicator.json"), read("lstm_xgb_comparison.json")
    sensors = pd.read_csv(reports/"sensor_analysis.csv")
    metrics = pd.read_csv(reports/"metrics.csv")
    intervals = pd.read_csv(reports/"interval_metrics.csv")
    cost = read("training_cost.json")
    lifetimes = pd.read_csv(reports/"engine_lifetimes.csv")
    per_engine = pd.read_csv(reports/"engine_sensor_analysis.csv")
    test = pd.read_csv(reports/"test_predictions.csv")
    endpoint = test[test.is_endpoint]
    kept = run["kept_sensors"]
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9,4))
    for partition, g in lifetimes.groupby("partition"):
        ax.hist(g.cycles_survived, bins=18, alpha=.6, label=partition)
    ax.set(xlabel="Cycles to failure", ylabel="Engines", title="FD001 lifetime variation"); ax.legend()
    fig.tight_layout(); fig.savefig(reports/"figures"/"engine_lifetimes.png"); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    per_engine[per_engine.sensor.isin(kept)].boxplot(column="slope_per_cycle", by="sensor", ax=axes[0], rot=90)
    # Standardize differences per sensor for a comparable plot, CSV retains raw units.
    starts = per_engine[per_engine.sensor.isin(kept)].copy()
    starts["standardized_start"] = starts.groupby("sensor").start_mean.transform(lambda x:(x-x.mean())/x.std())
    starts.boxplot(column="standardized_start", by="sensor", ax=axes[1], rot=90)
    axes[0].set_title("Degradation slopes (sensor units/cycle)"); axes[1].set_title("Starting-condition variation")
    fig.suptitle("Training engines only"); fig.tight_layout(); fig.savefig(reports/"figures"/"engine_variation.png"); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(10,9))
    for ax, name in zip(axes.flat, ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]):
        ax.scatter(endpoint.rul_raw, endpoint[name], s=18, alpha=.7)
        maximum = max(endpoint.rul_raw.max(), cfg["rul_cap"])
        ax.plot([0,maximum],[0,maximum], "--", color="gray")
        ax.set(title=name, xlabel="Original RUL (cycles)", ylabel="Predicted RUL (cycles)")
    fig.tight_layout(); fig.savefig(reports/"figures"/"test_predictions.png"); plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(11,7), sharex=True)
    example_id = int(endpoint.engine_id.iloc[0]); sample = test[test.engine_id == example_id]
    axes[0].plot(sample.cycle, sample.health_score, label="PCA health score", color="#087f8c")
    flagged = sample[sample.anomaly_flag]
    axes[0].scatter(flagged.cycle, flagged.health_score, s=9, color="#de684a", label="IF flag")
    axes[0].set(ylabel="Health score", title=f"Test engine {example_id}: causal history"); axes[0].legend()
    axes[1].fill_between(sample.cycle, sample.interval_lower, sample.interval_upper, alpha=.2, label="Heuristic model spread")
    axes[1].plot(sample.cycle, sample.ensemble_mean, label="Ensemble mean")
    axes[1].plot(sample.cycle, sample.rul_raw, "--", label="Original RUL (evaluation only)")
    axes[1].set(xlabel="Cycle", ylabel="RUL (cycles)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(reports/"figures"/"single_engine.png"); plt.close(fig)
    global_shap = pd.read_csv(reports/"shap_global.csv")
    fig, ax = plt.subplots(figsize=(8,5))
    ordered = global_shap.sort_values("mean_abs_shap")
    ax.barh(ordered.sensor, ordered.mean_abs_shap, color="#087f8c")
    ax.set(xlabel="Mean absolute aggregated SHAP (RUL cycles)", title="XGBoost: official test endpoints")
    fig.tight_layout(); fig.savefig(reports/"figures"/"shap_global.png"); plt.close(fig)
    history = pd.read_csv(reports/"lstm_history.csv")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(history.epoch, history.validation_endpoint_RMSE)
    ax.axvline(cost["LSTM"]["best_epoch"], linestyle="--", color="gray")
    ax.set(xlabel="Epoch", ylabel="Validation endpoint RMSE", title="LSTM checkpoint selection")
    fig.tight_layout(); fig.savefig(reports/"figures"/"lstm_learning_curve.png"); plt.close(fig)
    # Column identities are exactly those documented by the NASA archive. Do not invent engineering units.
    columns = [("engine_id", "Integer", "Engine identifier within train or test; namespaces are separate. Never a model feature."),
               ("cycle", "Cycles", "Observed operational age, starts at one; used for ordering and targets, excluded from RUL features.")]
    columns += [(f"setting_{i}", "Unspecified", f"NASA operational setting {i}; raw archive does not provide a named quantity/unit. Retained in raw data; excluded under FD001's single regime.") for i in range(1,4)]
    columns += [(f"sensor_{i}", unit, f"{symbol}: {description}. Source column {i+5}; mapping follows Table 2 in the bundled NASA modeling paper.") for i, (symbol,description,unit) in enumerate(SENSOR_DESCRIPTIONS,1)]
    dictionary = pd.DataFrame(columns, columns=["column", "unit", "description"])
    dictionary.to_csv(root/"data"/"column_dictionary.csv", index=False)
    (root/"data"/"README.md").write_text("# FD001 source data\n\nOfficial NASA archive: https://data.nasa.gov/docs/legacy/CMAPSSData.zip\n\nSee provenance.json for retrieval time, byte counts and locally calculated SHA-256 identities. These hashes are reproducibility checks, not an independent authenticity guarantee. Only FD001 and the source documentation are extracted. Raw files are immutable pipeline inputs.\n\n"+markdown_table(dictionary)+"\n\nRUL_FD001.txt: one nonnegative number per test engine, in ascending engine-ID order. It is remaining life AFTER the final observed test cycle. Training trajectories reach failure; test trajectories are censored.\n", encoding="utf-8")
    notes = {
        "01_data_eda": f"""# Data and EDA reasoning

The official NASA archive is retained by reference and identity hashes. Parsing rejects wrong column counts, missing/non-finite values, duplicate engine/cycle pairs and noncontiguous cycles. Every raw column is documented in data/column_dictionary.csv. Sensor symbols and units follow Table 2 in the bundled NASA modeling paper; operational setting names/units remain unspecified where the numbered text schema does not identify their order. HPC/LPC mean high/low pressure compressor; HPT/LPT mean high/low pressure turbine. The source labels PCNfR_dmd in rpm; preserve that attribution rather than silently interpreting the stored scale.

Partition training engines before learning any sensor policy. Lifetimes for the supplied train file are descriptive; sensor decisions, slopes and starting-condition statistics use only the fitting partition. See engine_lifetimes.csv, engine_sensor_analysis.csv, operating_settings.csv and sensor_analysis.csv.

Each sensor's pooled variance and Pearson correlations with cycle, raw RUL and clipped RUL are reported. Pooled correlations can reflect engine offsets, so retention uses median within-engine absolute Spearman correlation and linear-fit R2. Monotonicity is the absolute mean sign of first differences after a causal rolling mean. Trend strength is within-engine linear R2. Noise proxy is std(first difference)/sqrt(2*variance); it includes some degradation signal and is not a measured sensor-noise calibration.

The recorded selection rules are {cfg['sensor_policy']}. These are project analysis rules, not NASA limits or optimized physical constants. Remove low-cardinality channels, Investigate weak trends, Keep supported trends. Normalize is recorded as a separate preprocessing action on every Keep sensor: physical scales should not determine PCA loadings or LSTM optimization. No sensor must receive each verdict just to fill the categories. Starting-condition variation is the between-engine standard deviation of the first {cfg['hi_anchor_cycles']} cycle means. Degradation rates are descriptive whole-life slopes, not constant-rate physical models.

"""+markdown_table(sensors[["sensor","classification","preprocessing","reason"]]),
        "02_targets_split": f"""# Target and split reasoning

Training RUL is engine max cycle minus current cycle. Test RUL adds the supplied terminal RUL to the last observed cycle minus current cycle. Never treat a censored test endpoint as failure.

The fixed split uses seed {cfg['seed']}: {run['train_engines']} fitting engines, {run['validation_engines']} validation engines, and {run['test_engines']} separate official test engines. The ID lists and validation truncation cycles are in artifacts/split_manifest.json. Equal numeric IDs in train and test refer to different physical simulated engines. No row split is used. Scaler, selection, PCA and Isolation Forest see fitting engines only.

Fit a piecewise target min(raw RUL, {cfg['rul_cap']}). The ceiling is a declared project assumption about weak early-life identifiability, not a fitted damage-onset estimate. It prevents large indistinguishable early-life labels from dominating the loss. It also prevents predictions above the ceiling, even when true RUL is higher. Metrics for BOTH targets and unbounded model outputs are saved; raw-label errors reveal the resulting ceiling bias. The ceiling is fixed before model fitting; no ceiling search or test tuning is performed.

Validation selection uses one deterministic synthetic censored endpoint per validation engine, sampled at a fraction in {cfg['validation_cutoff_fraction']} of its observed run-to-failure life. This uses held-out lifetimes only to construct an evaluation task, never input features. It avoids selecting on terminal RUL zero for every engine. This censoring distribution is an assumption and may differ from NASA's official censoring. Secondary all-cycle metrics weight engines equally. Training endpoint metrics are all failure endpoints and therefore are not directly comparable with censored validation/test endpoint metrics; their R2 is undefined and left blank.

Windows have length {cfg['window']}, end at the prediction cycle and never cross engines. Short histories repeat the earliest available measurement on the left, include every engine, and report observed_window. This padding implies a flat unobserved history, not actual measurements. No future cycle, total lifetime, RUL, ID, or cycle-age feature enters the predictors. The same window is represented as last/mean/population-std/OLS-slope features for classical models and as a sequence for the LSTM. Training losses weight each engine equally, so long trajectories do not dominate.
""",
        "03_health_indicator": f"""# Health indicator reasoning

One PCA component is used because it gives a compact sensor-only coordinate with auditable loadings and no learned nonlinear HI. PCA is fitted to standardized Keep sensors on fitting engines only; it optimizes variance, not prognostic accuracy. Explained variance ratio from the run is {hi['explained_variance_ratio']:.6f}. Exact loadings are in health_indicator.json, not hand-assigned weights.

Healthy anchor: median PC score during the first {cfg['hi_anchor_cycles']} cycles of fitting engines. Failed anchor: median PC score in the last {cfg['hi_terminal_cycles']} cycles. These training lifecycle labels orient/calibrate the score; it is not fully unsupervised calibration. Health = clip(100*(PC-failed)/(healthy-failed), 0, 100). This definition resolves PCA's arbitrary sign. It is a relative project score, not probability of failure; raw cycle-level health can fluctuate and is not forced monotonic. HI is shown independently rather than fed into the RUL models.
""",
        "04_anomaly_detection": f"""# Anomaly detection reasoning

Isolation Forest fits last/mean/std/slope sensor-window features from the first {cfg['healthy_cycles']} cycles of fitting engines, as a proxy for nominal operation. The forest has {cfg['isolation_trees']} trees and contamination {cfg['isolation_contamination']}; these are declared project settings. Including all terminal degradation in nominal training would weaken the intended early-life reference. Initial wear still varies, so early cycles are not certified healthy.

A negative decision_function marks unusual windows relative to this reference. It can flag expected degradation as well as other deviations. No labeled anomaly truth is supplied by FD001; anomaly precision, recall and fault-onset detection accuracy are therefore not reported. Flag rates and the learned offset appear in anomaly_summary.json. Anomaly output does not alter RUL or the maintenance-status rule.
""",
        "05_models": f"""# Model comparison reasoning

Linear Regression is a transparent unregularized baseline. Random Forest and XGBoost each select between the candidates recorded in config.json using capped validation endpoint RMSE. Each candidate uses the same equal-engine training weights and causal history. One LSTM architecture is trained on CPU with deterministic algorithms and a fixed seed; hidden size, optimizer, gradient clipping and target scaling are in source/config. Best epoch is selected by the same validation endpoint metric. Candidate results, LSTM learning curve and measured wall times are persisted. Models remain fitted on the fitting partition after selection; validation is not silently folded into training.

The comparison is a bounded experiment, not an exhaustive or equal-compute hyperparameter search. Repeated seeds are not run; the bootstrap reflects sampling uncertainty across test engines, not training-seed uncertainty. All predictions are clipped to the target domain for every model. Both unclipped model predictions and original target labels are retained in prediction CSVs for auditing.

"""+comparison_text(comparison, cost),
        "06_uncertainty": f"""# Uncertainty reasoning

Take the arithmetic mean and sample standard deviation (ddof=1) across the four bounded model predictions at each window. Display mean +/- {cfg['spread_multiplier']}*std, clipped to [0, {cfg['rul_cap']}]. Equal model weights are a declared convention, not learned reliability weights. The multiplier borrows a normal-reference scale, but four correlated predictors do NOT yield a calibrated confidence or prediction interval. Shared bias, aleatoric noise, ceiling effects and out-of-distribution uncertainty can all be missed. The dashboard calls this a heuristic spread interval with no nominal coverage claim.

Measured endpoint coverage and mean width against BOTH target definitions are in interval_metrics.csv. These observations are diagnostic only; the multiplier is not tuned to test coverage.

Future uncertainty methods can replace evaluation.ensemble_spread while retaining interval diagnostics and the dashboard contract. No additional uncertainty method is implemented.
""",
        "07_explainability": """# Explainability reasoning

Exact tree-path-dependent TreeSHAP explains XGBoost's raw output before clipping. The fitted tree path counts define its background; no test rows are used as a fitted background. Per-feature contributions are summed by sensor across last/mean/std/slope features, preserving local additivity. Positive contributions raise predicted remaining life, negative contributions lower it. The baseline and sum are checked against model.predict, with maximum residual persisted. Attributions may be shared between correlated sensors; they describe model behavior, not causal fault diagnosis.

Global importance is the mean absolute SENSOR-AGGREGATED contribution across official test endpoints, rather than the sum of absolute statistic contributions. This can show cancellation within sensors. Local endpoint attributions are exported for every test engine; arbitrary observed-cycle explanations are calculated on demand in the dashboard. RF SHAP is omitted to keep the requested optional extension from adding compute and clutter.
""",
        "08_dashboard": f"""# Dashboard reasoning

The single-engine view loads fitted artifacts and uses only that engine's prefix through the selected cycle. It displays relative PCA health, all four predictions, equal-weight ensemble RUL, heuristic spread, raw sensor trends, Isolation Forest flag and signed XGBoost SHAP sensor contributions. Official RUL is optional and explicitly evaluation-only.

Status uses the LOWER heuristic RUL bound with strict less-than boundaries: Red below {cfg['maintenance_rul_thresholds']['red_below']}, Orange below {cfg['maintenance_rul_thresholds']['orange_below']}, Yellow below {cfg['maintenance_rul_thresholds']['yellow_below']}, Green otherwise. Equality enters the next higher band. These values live in config.json, are saved with the trained model and displayed in the app. They are illustrative project policy choices, not NASA/GE limits, learned optimums, safety probabilities or maintenance instructions. The score and anomaly flag are not hidden overrides. A weak interval also makes this status weak.
"""}
    for name, content in notes.items():
        (reports/f"{name}.md").write_text(content+"\n", encoding="utf-8")
    lifetime_summary = lifetimes.groupby("partition").cycles_survived.agg(["count","min","median","mean","max","std"]).reset_index()
    slope_summary = per_engine[per_engine.sensor.isin(kept)].groupby("sensor").agg(
        minimum_slope=("slope_per_cycle","min"), median_slope=("slope_per_cycle","median"),
        maximum_slope=("slope_per_cycle","max"), starting_mean_std=("start_mean","std")).reset_index()
    lifetime_summary.to_csv(reports/"lifetime_summary.csv",index=False)
    slope_summary.to_csv(reports/"degradation_summary.csv",index=False)
    notes["01_data_eda"] += "\n\n## Recorded lifetime distribution\n\n" + markdown_table(lifetime_summary)
    notes["01_data_eda"] += "\n\n## Recorded rate and initial-state variation\n\n" + markdown_table(slope_summary)
    notes["01_data_eda"] += "\n\nSlopes are in each sensor's own units, so compare engines within a sensor rather than slope magnitude across sensors. Nonzero starting-mean spread shows why one nominal reference trajectory cannot represent every engine. Whole-life slopes summarize different rates but also compress nonlinear degradation and measurement noise.\n"
    (reports/"01_data_eda.md").write_text(notes["01_data_eda"],encoding="utf-8")
    capped = metrics.query("partition == 'test' and scope == 'endpoint' and target == 'rul_target'")[["model","MAE","RMSE","R2"]]
    raw = metrics.query("partition == 'test' and scope == 'endpoint' and target == 'rul_raw'")[["model","MAE","RMSE","R2"]]
    validation = metrics.query("partition == 'validation' and scope == 'endpoint' and target == 'rul_target'")[["model","MAE","RMSE","R2"]]
    validation_models = validation[validation.model != "Ensemble mean"]
    test_models = capped[capped.model != "Ensemble mean"]
    val_winner = validation_models.loc[validation_models.RMSE.idxmin(),"model"]
    test_winner = test_models.loc[test_models.RMSE.idxmin(),"model"]
    ranking_note = f"The lowest validation RMSE belongs to {val_winner}; the lowest official test RMSE belongs to {test_winner}. Rankings on a small, synthetically censored validation partition need not transfer to test engines. The test comparison is descriptive, not a new round of model selection."
    t = cfg["maintenance_rul_thresholds"]
    readme = f"""# FD001 · Aircraft engine health and remaining life

A reproducible portfolio research project for a GE Aerospace Data Science internship application. Built on NASA's simulated turbofan degradation data. This is an independent project, not a GE product or an operational maintenance system.

## Recorded experiment

This README is generated by `src/reporting.py` from recorded artifacts, not manually entered metrics. Run completed {run['completed_utc']}. Partition sizes: {run['train_engines']} fitting engines / {run['validation_engines']} validation engines / {run['test_engines']} official test engines. Seed: {cfg['seed']}. Window: {cfg['window']} cycles. Target ceiling: {cfg['rul_cap']} cycles. Device: {run['torch_device']}. Total pipeline time before report rendering: {run['total_pipeline_seconds']:.2f} seconds. Full settings, package versions, source hashes and split IDs are saved in `config.json`, `reports/run_manifest.json`, and `artifacts/split_manifest.json`.

FD001 is the single-operating-condition, single-HPC-degradation subset. NASA supplies separate run-to-failure training trajectories and censored test trajectories. Data sources: [NASA dataset record](https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data), [official archive](https://data.nasa.gov/docs/legacy/CMAPSSData.zip). Citation: A. Saxena and K. Goebel, Turbofan Engine Degradation Simulation Data Set, NASA Ames Prognostics Data Repository (2008). The official archive's README and modeling paper are included in `data/raw/`.

## Run locally

Use Python {run['python']}. From this repository directory:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.lock.txt
# Data and trained artifacts are included. To fetch a fresh official copy:
.venv/Scripts/python.exe scripts/download_data.py
.venv/Scripts/python.exe -m src.pipeline
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

On macOS/Linux use `.venv/bin/python` instead. `requirements.txt` pins direct dependencies; `requirements.lock.txt` records the complete Windows environment used for this result. The platform-specific lock may need re-resolution on other operating systems. The pipeline is CPU-bounded and intentionally runs one LSTM. Re-running overwrites generated artifacts; preserve a copy to compare experiments. Model files are trusted local Python artifacts and should only be loaded from trusted sources.

## Data and EDA

Every source column is documented in [the data dictionary](data/README.md). Sensor numbers are the authoritative source identities; sensor symbols and units are transcribed from Table 2 of NASA's bundled modeling paper. [Sensor evidence table](reports/sensor_analysis.csv) includes variance, cycle/RUL correlations, within-engine rank correlation, monotonicity, trend strength, noise proxy, slope variation, initial variation, decisions and reasons. Retained sensors: {', '.join(kept)}. The Normalize action applies to each Keep sensor and is fitted on training engines only.

![Lifetimes](reports/figures/engine_lifetimes.png)

See [EDA reasoning](reports/01_data_eda.md) and [per-engine degradation and starting-condition evidence](reports/engine_sensor_analysis.csv). HI uses one standardized PCA component with recorded explained variance ratio {hi['explained_variance_ratio']:.6f}; see [exact learned loadings and anchors](reports/health_indicator.json).

## Honest evaluation

No engine crosses fitting and validation partitions. Official test IDs occupy a separate namespace. Selection uses synthetic censored validation endpoints; all test engines are evaluated at their final observed cycle. Classical models receive last/mean/std/slope summaries of the same sensor history consumed by the LSTM. Scaler, sensor policy, PCA and anomaly detector never fit on validation/test measurements. Training losses weight engines equally. See [target, split and window rationale](reports/02_targets_split.md).

Validation endpoints, capped target:

{markdown_table(validation)}

Official test endpoints, capped target:

{markdown_table(capped)}

Official test endpoints, ORIGINAL NASA RUL (same bounded predictions):

{markdown_table(raw)}

The capped task reflects limited early-life identifiability but underestimates life above the ceiling. Both tables are necessary; they are different target definitions and should not be mixed when comparing papers. MAE/RMSE units are cycles; R2 is dimensionless. [Full metrics](reports/metrics.csv) include train/validation/test and engine-weighted all-cycle results. Training all-cycle errors are in-sample diagnostics. Adjacent-cycle errors are dependent; the primary test has one prediction per engine.

{comparison_text(comparison, cost)}

{ranking_note}

![Test predictions](reports/figures/test_predictions.png)

## Health, anomalies and uncertainty

PCA health is anchored to the early/terminal training medians and clipped to the score domain. It is a relative score and can fluctuate. Isolation Forest compares sensor-window statistics with an early-life fitting reference. No anomaly-label accuracy is claimed. The uncertainty display is equal-model mean +/- {cfg['spread_multiplier']} sample standard deviations, bounded by the target ceiling. It is a heuristic spread interval, not a calibrated interval or a coverage guarantee.

Measured official test endpoint interval diagnostics:

{markdown_table(intervals.query("partition == 'test'")[["target","coverage","mean_width"]])}

Coverage is an observed fraction, not a nominal confidence level. Correlated models can agree and still be wrong. [Uncertainty reasoning](reports/06_uncertainty.md) details the limitations.

## Explainability and dashboard

The UI uses a navy control sidebar and three tabs: Overview, Sensor trends, and Explainability. It includes causal RUL history, model comparison bars, readable sensor names, a training-standardized display toggle, and observation/SHAP CSV downloads. See [UI design notes](reports/09_ui_design.md).

Run the Streamlit command above, choose a test engine and observed cycle, and inspect health, RUL/spread, sensor trends, anomaly status and top signed SHAP contributors. XGBoost SHAP explains its raw output before clipping; the app shows its baseline and reconstructed prediction. [Global sensor importance](reports/shap_global.csv) and [local endpoint explanations](reports/shap_sensor_contributions.csv) are exported. These are model attributions, not causal fault diagnoses.

Project-defined status uses the lower heuristic RUL bound: Red below {t['red_below']}, Orange below {t['orange_below']}, Yellow below {t['yellow_below']}, Green otherwise. Thresholds are illustrative values in `config.json`, not real aerospace limits. They are shown in the UI and never represented as learned safety thresholds.

![Single engine](reports/figures/single_engine.png)

## Interview walkthrough and repository

`notebooks/01_fd001_walkthrough.ipynb` contains executed artifact inspection with figures and decision notes. `reports/01_*.md` through `08_*.md` explain each major step. `data/` holds source files, provenance and column documentation; `src/` holds the modular pipeline; `dashboard/app.py` is the single Streamlit app; `artifacts/` contains fitted state; `reports/` contains auditable predictions, metrics and figures; `tests/` checks targets, causal windows, split isolation, artifact consistency and dashboard interactions.

Reproducibility limits: one seed and one held-out validation partition; synthetic validation censoring; bounded, unequal hyperparameter-search budgets; correlated sliding windows; simulated data under a single regime; heuristic interval; no labeled anomaly truth. Bootstrap intervals describe held-out-engine sampling variation only. No test-driven retuning was performed. `scripts/refresh_evaluation.py` can recompute metrics and reports from saved predictions without fitting; `reports/evaluation_refresh.json`, when present, records that audit and its source hashes. Constant-target R2 is undefined and left blank rather than forced to zero.

## Roadmap / Phase 2 — not implemented

- FD002/FD004: extend the loader/split namespace and add operating-condition normalization in `preprocessing.py`.
- Real-flight-condition NASA dataset: add a separate data adapter and condition-aware evaluation; do not reuse FD001 assumptions.
- Temporal CNN/Transformer: extend the model registry in `rul_models.py` with the same window/evaluation contract.
- Quantile regression, MC dropout or conformal prediction: replace `evaluation.ensemble_spread`, add appropriate training/calibration partitions and retain interval diagnostics.
- FastAPI: expose the existing prefix-inference contract as a service.
- Docker: package the pinned runtime and trained artifacts.
- Live-replay simulation: feed observed prefixes into the existing inference path.
"""
    (root/"README.md").write_text(readme, encoding="utf-8")
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name":"Python 3", "language":"python", "name":"python3"}
    notebook.cells = [nbformat.v4.new_markdown_cell("# FD001 interview walkthrough\n\nExecuted artifact inspection from the recorded pipeline. Re-run the pipeline from the repository root before refreshing these cells. No fitting occurs inside the notebook.")]
    setup = nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd() if (Path.cwd() / 'reports').exists() else Path.cwd().parent\nR = ROOT / 'reports'\nprint(json.loads((R / 'run_manifest.json').read_text())['completed_utc'])")
    setup.execution_count = 1
    setup.outputs = [nbformat.v4.new_output("stream", name="stdout", text=run["completed_utc"]+"\n")]
    notebook.cells.append(setup)
    tables = [("01_data_eda", "sensor_analysis.csv"), ("02_targets_split", "engine_lifetimes.csv"), ("03_health_indicator", None),
              ("04_anomaly_detection", None), ("05_models", "metrics.csv"), ("06_uncertainty", "interval_metrics.csv"),
              ("07_explainability", "shap_global.csv"), ("08_dashboard", None)]
    count = 1
    for name, table in tables:
        notebook.cells.append(nbformat.v4.new_markdown_cell(notes[name]))
        if table:
            count += 1
            code = nbformat.v4.new_code_cell(f"display(pd.read_csv(R / '{table}'))")
            code.execution_count = count
            df = pd.read_csv(reports/table)
            code.outputs = [nbformat.v4.new_output("display_data", data={"text/html":df.to_html(index=False), "text/plain":df.to_string(index=False)})]
            notebook.cells.append(code)
    for name in ["engine_lifetimes", "engine_variation", "lstm_learning_curve", "test_predictions", "shap_global", "single_engine"]:
        count += 1
        code = nbformat.v4.new_code_cell(f"display(Image(filename=str(R / 'figures' / '{name}.png')))")
        code.execution_count = count
        code.outputs = [nbformat.v4.new_output("display_data", data={"image/png":base64.b64encode((reports/"figures"/f"{name}.png").read_bytes()).decode(), "text/plain":f"{name} figure"})]
        notebook.cells.append(code)
    nbformat.write(notebook, root/"notebooks"/"01_fd001_walkthrough.ipynb")
    # Execute the actual code cells before delivering; generated previews are not execution evidence.
    subprocess.run([sys.executable, str(root/"scripts"/"execute_notebook.py")], check=True)


def comparison_text(comparison, cost):
    delta = comparison["rmse_improvement_lstm_over_xgb"]
    low, high = comparison["paired_bootstrap_95_low"], comparison["paired_bootstrap_95_high"]
    if low > 0:
        verdict = "The LSTM improves capped test RMSE with a paired interval above zero in this run. Whether that gain is worth deployment complexity remains a project decision; repeat seeds before claiming a robust architectural advantage."
    elif delta > 0:
        verdict = "The LSTM has lower capped test RMSE in this run, but the paired interval includes zero. This experiment does not establish enough evidence to justify its added complexity over XGBoost."
    else:
        verdict = "The LSTM does not beat XGBoost on capped test RMSE in this run. This experiment does not justify choosing the LSTM for its added complexity."
    return (f"{verdict} Measured XGBoost-minus-LSTM RMSE difference: {delta:.4f} cycles; paired engine bootstrap interval ({comparison['repeats']} resamples): [{low:.4f}, {high:.4f}]. "
            f"Selected XGBoost fit: {cost['XGBoost']['selected_fit_seconds']:.2f} seconds (candidate search {cost['XGBoost']['search_seconds']:.2f} seconds); "
            f"LSTM training including validation: {cost['LSTM']['selected_fit_seconds']:.2f} seconds, {cost['LSTM']['parameters']} parameters, "
            f"best epoch {cost['LSTM']['best_epoch']} of {cost['LSTM']['epochs_run']} run. Timing scopes differ and are stated rather than presented as a controlled benchmark. "
            "See `reports/lstm_xgb_comparison.json`, `reports/training_cost.json` and `reports/model_selection.csv`.")


if __name__ == "__main__":
    build_reports(Path(__file__).resolve().parents[1])
