"""Exact TreeSHAP for XGBoost; aggregate window statistics back to sensors."""
import numpy as np
import pandas as pd
import shap


def explain(model, tabular, feature_names):
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    explanation = explainer(tabular, check_additivity=True)
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    error = float(np.max(np.abs(reconstructed-model.predict(tabular))))
    if not np.allclose(reconstructed, model.predict(tabular), atol=1e-3, rtol=1e-4):
        raise ValueError(f"SHAP additivity failed: {error}")
    frame = pd.DataFrame(explanation.values, columns=feature_names)
    sensor_values = frame.T.groupby(lambda key: key.split("__")[0], sort=False).sum().T
    return explanation, sensor_values, error
