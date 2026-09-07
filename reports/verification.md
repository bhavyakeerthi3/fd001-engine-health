# Verification evidence

Story: choose an FD001 test engine and observed cycle in Streamlit, build only its available sensor prefix, run saved preprocessing/models, and display health, RUL/spread and sensor explanations.

| Boundary | Result | Evidence |
| --- | --- | --- |
| Source data | Passed | NASA ZIP CRC read succeeds; FD001 schema and source identities in data/provenance.json |
| Targets and partitions | Passed | Tests cover per-engine train labels, censored test labels, disjoint fit/validation IDs and equal-engine weights |
| Causal preprocessing | Passed | Tests perturb future data and confirm earlier windows/features unchanged; no crossing between engine windows |
| Saved model inference | Passed | Label-free prefixes for early, partial and complete histories match exported predictions; scaler matches fitting partition |
| Explainability | Passed | Exact XGBoost SHAP reconstruction checked at every test endpoint and in prefix tests |
| Metrics | Passed | Official test endpoint metrics recomputed from saved prediction rows; constant-target R2 left undefined |
| Dashboard state changes | Passed | Streamlit AppTest switches engine, selects first cycle, clears sensors and enables evaluation truth; no exceptions |
| Browser rendering | Passed | agent-browser opened the live app, switched Engine 001 to Engine 002, rendered metrics/trends/SHAP; browser errors command returned no errors |
| Visual inspection | Passed | Metric card truncation corrected and verified in reports/figures/dashboard.png |
| Notebook | Passed | All 12 code cells executed with the project Python interpreter; outputs embedded |

Automated suite: 7 passed. Three SHAP/Matplotlib pending-deprecation warnings are retained in verification_tests.txt; no test failures. Tests intentionally do not assert universal accuracy thresholds or anomaly-label metrics.

The browser uses Streamlit's local server and saved models. No external prediction API, database or deployment is part of this build. The local app is available while its server process remains running; use the README command to restart it.
