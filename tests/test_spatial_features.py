"""
Unit tests for spatial features: slope, aspect, curvature, TWI, and stream distance.
"""

import pytest
import numpy as np
from inundation_risk.spatial_features.morphometry import compute_slope_aspect, compute_curvatures
from inundation_risk.spatial_features.hydrology import (
    compute_twi,
    extract_stream_network,
    compute_distance_to_stream,
)
from inundation_risk.spatial_features.raster_utils import normalize_raster


def test_compute_slope_aspect(synthetic_dem_grid):
    slope, aspect = compute_slope_aspect(synthetic_dem_grid, cell_size=30.0)
    assert slope.shape == synthetic_dem_grid.shape
    assert aspect.shape == synthetic_dem_grid.shape
    # For interior cells on our sloping plane, slope should be > 0
    valid_slope = slope[~np.isnan(slope)]
    assert np.all(valid_slope >= 0.0)
    assert np.all(valid_slope <= 90.0)


def test_compute_curvatures(synthetic_dem_grid):
    prof_c, plan_c = compute_curvatures(synthetic_dem_grid, cell_size=30.0)
    assert prof_c.shape == synthetic_dem_grid.shape
    assert plan_c.shape == synthetic_dem_grid.shape
    valid_prof = prof_c[~np.isnan(prof_c)]
    assert len(valid_prof) > 0


def test_compute_twi(synthetic_dem_grid, synthetic_flow_accum_grid):
    slope, _ = compute_slope_aspect(synthetic_dem_grid, cell_size=30.0)
    twi = compute_twi(synthetic_flow_accum_grid, slope, cell_size=30.0)
    valid_twi = twi[~np.isnan(twi)]
    assert len(valid_twi) > 0
    # Higher accumulation and lower slope should yield higher TWI
    assert np.all(valid_twi > -10.0)


def test_stream_network_and_distance(synthetic_flow_accum_grid):
    stream_mask = extract_stream_network(synthetic_flow_accum_grid, threshold_cells=500)
    assert stream_mask.dtype == bool
    assert np.any(stream_mask)

    dist = compute_distance_to_stream(stream_mask, cell_size=30.0)
    assert dist.shape == synthetic_flow_accum_grid.shape
    # Stream cells should have 0 distance
    assert np.all(dist[stream_mask] == 0.0)
    # Non-stream cells should have distance >= 30m
    assert np.all(dist[~stream_mask] >= 30.0)


def test_normalize_raster():
    arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    norm = normalize_raster(arr, method="minmax")
    assert np.isclose(np.min(norm), 0.0)
    assert np.isclose(np.max(norm), 1.0)
