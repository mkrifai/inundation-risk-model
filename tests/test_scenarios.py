"""
Unit tests for scenario simulation and policy overlay.
"""

import pytest
import numpy as np
from inundation_risk.scenarios.simulation import simulate_adaptation_scenario
from inundation_risk.scenarios.policy_overlay import evaluate_spatial_plan_conflicts


def test_simulate_adaptation_scenario():
    h = np.full((10, 10), 0.5, dtype=np.float32)
    r = np.full((10, 10), 0.4, dtype=np.float32)

    scen_h, scen_r, metrics = simulate_adaptation_scenario(
        h, r, rainfall_multiplier=1.20, mitigation_efficiency=0.30
    )
    assert scen_h.shape == (10, 10)
    assert scen_r.shape == (10, 10)
    # +20% rain should increase hazard
    assert np.all(scen_h >= h)
    # With 30% mitigation, risk reduction is registered
    assert "percent_risk_change" in metrics


def test_evaluate_spatial_plan_conflicts():
    h = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    zoning = np.array([[1, 2], [3, 1]], dtype=np.int32)
    pop = np.array([[500.0, 100.0], [200.0, 800.0]], dtype=np.float32)

    # Planned urban code is 1 (present at [0,0] and [1,1])
    # Both [0,0] (h=0.8) and [1,1] (h=0.9) have h >= 0.60
    metrics = evaluate_spatial_plan_conflicts(
        hazard_raster=h,
        zoning_raster=zoning,
        planned_urban_codes=[1],
        population_density=pop,
        hazard_threshold=0.60,
        pixel_area_km2=0.01
    )
    assert metrics["conflict_area_km2"] == 0.02
    assert metrics["percent_planned_area_in_conflict"] == 100.0
    assert metrics["exposed_population_in_conflict"] == 1300.0
