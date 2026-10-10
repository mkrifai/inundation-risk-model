"""
Unit tests for machine learning hazard modeling, dataset builder, and spatial CV.
"""

import pytest
import numpy as np
import pandas as pd
from inundation_risk.hazard_ml.dataset import build_ml_dataset
from inundation_risk.hazard_ml.spatial_cv import SpatialBlockKFold
from inundation_risk.hazard_ml.trainers import HazardModelTrainer


def test_build_ml_dataset(synthetic_dem_grid, synthetic_flow_accum_grid):
    factors = {
        "dem": synthetic_dem_grid,
        "accum": synthetic_flow_accum_grid
    }
    # Create synthetic presence mask
    presence = np.zeros_like(synthetic_dem_grid, dtype=bool)
    presence[10:20, 10:20] = True

    X, y, coords = build_ml_dataset(factors, presence, ratio_negative=1.0, random_state=42)
    assert len(X) == len(y)
    assert len(coords) == len(y)
    assert "dem" in X.columns
    assert "accum" in X.columns
    assert set(y.unique()) == {0, 1}
    assert np.sum(y == 1) == 100


def test_spatial_block_cv():
    # 200 samples across a 2D space
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 1000, size=(200, 2))
    X = pd.DataFrame({"feat1": rng.normal(size=200)})
    y = pd.Series(rng.integers(0, 2, size=200))

    cv = SpatialBlockKFold(n_splits=5, random_state=42)
    splits = list(cv.split(X, y, coords))
    assert len(splits) == 5

    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Disjoint sets
        assert len(set(train_idx).intersection(set(test_idx))) == 0


def test_hazard_model_trainer(synthetic_dem_grid):
    rng = np.random.default_rng(42)
    n_samples = 150
    X = pd.DataFrame({
        "slope": rng.uniform(0, 30, size=n_samples),
        "twi": rng.uniform(5, 20, size=n_samples)
    })
    # Synthetic target correlated with high TWI and low slope
    prob = 1.0 / (1.0 + np.exp(-(X["twi"] * 0.3 - X["slope"] * 0.2)))
    y = pd.Series((prob > 0.5).astype(int))
    coords = rng.uniform(0, 500, size=(n_samples, 2))

    trainer = HazardModelTrainer(algorithm="random_forest", random_state=42)
    metrics = trainer.evaluate_spatial_cv(X, y, coords, n_splits=3)

    assert "roc_auc_mean" in metrics
    assert metrics["roc_auc_mean"] >= 0.70  # Should learn the relationship

    trainer.fit(X, y)
    factors = {
        "slope": np.full((10, 10), 5.0, dtype=np.float32),
        "twi": np.full((10, 10), 15.0, dtype=np.float32)
    }
    raster_prob = trainer.predict_spatial_raster(factors, ["slope", "twi"])
    assert raster_prob.shape == (10, 10)
    assert np.all(raster_prob >= 0.0)
    assert np.all(raster_prob <= 1.0)
