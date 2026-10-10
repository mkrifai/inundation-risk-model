"""
Pytest fixtures: in-memory synthetic rasters and hydroclimatic series.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def synthetic_rainfall_series():
    """Returns 40 years of synthetic annual maximum rainfall."""
    rng = np.random.default_rng(42)
    return rng.gumbel(loc=90.0, scale=25.0, size=40)


@pytest.fixture
def synthetic_daily_rainfall():
    """Returns 365 days of synthetic daily rainfall."""
    rng = np.random.default_rng(42)
    daily = rng.exponential(scale=5.0, size=365)
    # Zero out 60% dry days
    dry_mask = rng.uniform(0, 1, size=365) > 0.4
    daily[dry_mask] = 0.0
    # Inject heavy storms
    daily[120] = 85.0
    daily[121] = 45.0
    daily[122] = 60.0
    return pd.Series(daily)


@pytest.fixture
def synthetic_dem_grid():
    """Returns a 50x50 synthetic DEM (meters)."""
    rows, cols = 50, 50
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)
    # Smooth slope from 600m to 200m
    dem = (600.0 - 400.0 * (X + Y) / 2.0).astype(np.float32)
    return dem


@pytest.fixture
def synthetic_flow_accum_grid():
    """Returns a 50x50 synthetic flow accumulation raster."""
    rows, cols = 50, 50
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)
    X, Y = np.meshgrid(x, y)
    accum = ((X + Y)**2 * 1000.0).astype(np.float32)
    return accum
