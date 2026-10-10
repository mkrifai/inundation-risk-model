"""
Model trainer and spatial predictor for flood hazard susceptibility.
Integrates CatBoost, XGBoost, and Random Forest with spatial CV and calibration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from inundation_risk.hazard_ml.spatial_cv import SpatialBlockKFold

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class HazardModelTrainer:
    """
    Unified training engine for flood hazard susceptibility models.
    """

    def __init__(
        self,
        algorithm: str = "random_forest",
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        self.algorithm = algorithm.lower()
        self.params = params or {}
        self.random_state = random_state
        self.model = self._create_model()
        self.cv_results_: Dict[str, Any] = {}

    def _create_model(self):
        if self.algorithm == "random_forest":
            base_params = {
                "n_estimators": 200,
                "max_depth": 10,
                "random_state": self.random_state,
                "n_jobs": -1
            }
            base_params.update(self.params)
            return RandomForestClassifier(**base_params)

        elif self.algorithm == "xgboost":
            if not HAS_XGB:
                raise ImportError("xgboost is not installed. Install via pip install xgboost.")
            base_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "eval_metric": "logloss",
                "random_state": self.random_state,
                "n_jobs": -1
            }
            base_params.update(self.params)
            return xgb.XGBClassifier(**base_params)

        elif self.algorithm == "catboost":
            if not HAS_CATBOOST:
                raise ImportError("catboost is not installed. Install via pip install catboost.")
            base_params = {
                "iterations": 300,
                "depth": 6,
                "learning_rate": 0.05,
                "verbose": False,
                "random_seed": self.random_state,
                "thread_count": -1
            }
            base_params.update(self.params)
            return cb.CatBoostClassifier(**base_params)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}. Choose 'random_forest', 'xgboost', or 'catboost'.")

    def evaluate_spatial_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        coordinates: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate model performance using Spatial Block K-Fold Cross-Validation.
        """
        cv = SpatialBlockKFold(n_splits=n_splits, random_state=self.random_state)
        roc_scores = []
        pr_scores = []
        f1_scores = []
        acc_scores = []
        brier_scores = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, coordinates)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fold_model = self._create_model()
            fold_model.fit(X_train, y_train)

            # Probabilities for class 1
            y_prob = fold_model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            roc_scores.append(roc_auc_score(y_test, y_prob))
            pr_scores.append(average_precision_score(y_test, y_prob))
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
            acc_scores.append(accuracy_score(y_test, y_pred))
            brier_scores.append(brier_score_loss(y_test, y_prob))

        self.cv_results_ = {
            "roc_auc_mean": float(np.mean(roc_scores)),
            "roc_auc_std": float(np.std(roc_scores)),
            "pr_auc_mean": float(np.mean(pr_scores)),
            "f1_mean": float(np.mean(f1_scores)),
            "accuracy_mean": float(np.mean(acc_scores)),
            "brier_score_mean": float(np.mean(brier_scores)),
            "n_splits": n_splits,
        }
        return self.cv_results_

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit final model on full dataset."""
        self.model.fit(X, y)
        return self

    def predict_spatial_raster(
        self,
        factors: Dict[str, np.ndarray],
        feature_names: List[str]
    ) -> np.ndarray:
        """
        Generate full continuous spatial hazard susceptibility probability map [0, 1].
        """
        first_arr = factors[feature_names[0]]
        rows, cols = first_arr.shape

        valid_mask = np.ones((rows, cols), dtype=bool)
        for name in feature_names:
            valid_mask &= ~np.isnan(factors[name])

        prob_raster = np.full((rows, cols), np.nan, dtype=np.float32)
        if not np.any(valid_mask):
            return prob_raster

        # Stack valid cells into matrix
        valid_coords = np.argwhere(valid_mask)
        r_idx = valid_coords[:, 0]
        c_idx = valid_coords[:, 1]

        data_dict = {name: factors[name][r_idx, c_idx] for name in feature_names}
        df_grid = pd.DataFrame(data_dict)

        # Batch prediction
        probs = self.model.predict_proba(df_grid)[:, 1]
        prob_raster[r_idx, c_idx] = probs.astype(np.float32)

        return prob_raster
