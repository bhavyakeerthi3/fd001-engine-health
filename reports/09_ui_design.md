# Dashboard UI refinement

The interface is organized around the interview and inspection flow: observe an engine, compare predictions, inspect sensors, then explain the estimate.

- **Overview:** a navy control sidebar and a light workspace separate inputs from evidence. Four compact metrics summarize health, RUL, heuristic spread, and anomaly reference. Status bands show their exact project thresholds alongside the selected band. Text labels accompany every status color.
- **History and comparison:** the RUL line and spread band come from saved causal prediction rows, filtered to the selected engine and cycle. Model bars compare the current four predictions on a common axis. Health and Isolation Forest flags remain visible as separate evidence. No model was retrained for this UI change.
- **Sensor trends:** NASA sensor symbols and descriptions replace opaque numbers where practical. Multiple selected channels appear in two columns, with raw values and causal rolling means. The optional standardized display uses the saved training scaler and changes charts only.
- **Explainability:** signed bars separate positive and negative contributions, with a baseline-to-prediction equation and all contributions available below. Correlation and causal-interpretation limitations remain explicit.
- **Details on demand:** explanatory tooltips, expandable model details, and CSV downloads support deeper inspection without pushing the primary plots below a long block of methodological text. The truth overlay is explicitly evaluation-only and remains off by default.

Presentation code is in dashboard/app.py and dashboard/charts.py. Layout/color rules are in dashboard/theme.css and .streamlit/config.toml. Numerical prediction and policy definitions remain in the original trained artifacts.
