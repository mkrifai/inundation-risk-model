"""
Model explainability engine using SHAP (SHapley Additive exPlanations).
Computes Shapley values and feature importance rankings for hazard models.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def compute_shap_explanations(
    model: Any,
    X_sample: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values and global feature importance ranking.

    Parameters:
    -----------
    model : fitted tree estimator (RandomForest, XGBoost, or CatBoost)
    X_sample : pd.DataFrame
        Sample of feature observations used for explanation (e.g. 500-1000 points).

    Returns:
    --------
    shap_values : np.ndarray, shape (n_samples, n_features)
    importance_df : pd.DataFrame (columns: ['feature', 'mean_abs_shap', 'rank'])
    """
    if not HAS_SHAP:
        # Fallback to feature_importances_ if shap is not installed
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.ones(X_sample.shape[1]) / X_sample.shape[1]
        df_rank = pd.DataFrame({
            "feature": X_sample.columns,
            "mean_abs_shap": importances
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        df_rank["rank"] = np.arange(1, len(df_rank) + 1)
        return np.zeros(X_sample.shape), df_rank

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    # In binary classification, some models return list of 2 arrays [class_0, class_1]
    if isinstance(shap_vals, list):
        shap_vals_class1 = shap_vals[1]
    elif len(shap_vals.shape) == 3:
        shap_vals_class1 = shap_vals[:, :, 1]
    else:
        shap_vals_class1 = shap_vals

    mean_abs = np.mean(np.abs(shap_vals_class1), axis=0)

    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    return shap_vals_class1, importance_df
