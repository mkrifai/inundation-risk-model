"""
Unit tests for IPCC Risk integration and AHP multi-criteria calculations.
"""

import pytest
import numpy as np
from inundation_risk.risk_ipcc.ahp import compute_ahp_weights, verify_ahp_consistency
from inundation_risk.risk_ipcc.exposure import compute_exposure_index
from inundation_risk.risk_ipcc.vulnerability import compute_vulnerability_index
from inundation_risk.risk_ipcc.integration import (
    compute_ipcc_risk,
    classify_risk_levels,
    rank_administrative_hotspots,
)


def test_ahp_weights_and_consistency():
    # Consistent 3x3 pairwise comparison matrix
    matrix = np.array([
        [1.0, 2.0, 4.0],
        [0.5, 1.0, 3.0],
        [0.25, 0.3333, 1.0]
    ])
    weights, lmax, cr = compute_ahp_weights(matrix, ["Hazard", "Exposure", "Vulnerability"])

    # Sum of weights must equal 1.0
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-3)
    # Order of importance: Hazard > Exposure > Vulnerability
    assert weights["Hazard"] > weights["Exposure"] > weights["Vulnerability"]
    # Consistency Ratio must be < 0.10
    assert cr < 0.10
    assert verify_ahp_consistency(cr) is True


def test_exposure_and_vulnerability_indices():
    pop = np.array([[100.0, 200.0], [50.0, 400.0]], dtype=np.float32)
    built = np.array([[0.2, 0.8], [0.1, 0.9]], dtype=np.float32)

    exp = compute_exposure_index(pop, built)
    assert exp.shape == (2, 2)
    assert np.all(exp >= 0.0)
    assert np.all(exp <= 1.0)

    vuln = compute_vulnerability_index({"factor1": pop, "factor2": built})
    assert vuln.shape == (2, 2)
    assert np.all(vuln >= 0.0)
    assert np.all(vuln <= 1.0)


def test_compute_ipcc_risk():
    h = np.array([[0.8, 0.2], [0.5, 0.9]], dtype=np.float32)
    e = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    v = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)

    risk = compute_ipcc_risk(h, e, v)
    assert risk.shape == (2, 2)
    # Risk = 0.8 * 0.5 * 0.5 = 0.20
    assert np.isclose(risk[0, 0], 0.20)
    assert np.isclose(risk[0, 1], 0.05)


def test_classify_and_rank_hotspots():
    risk = np.linspace(0.01, 1.0, 100).reshape((10, 10))
    classified, thresholds = classify_risk_levels(risk, n_classes=5)
    assert len(thresholds) == 6
    assert np.min(classified) == 1
    assert np.max(classified) == 5

    # Admin zones
    zones = np.zeros((10, 10), dtype=np.int32)
    zones[:5, :] = 1  # Zone 1
    zones[5:, :] = 2  # Zone 2 (higher risk)
    names = {1: "North_District", 2: "South_District"}

    df_rank = rank_administrative_hotspots(risk, zones, names)
    assert len(df_rank) == 2
    # South district has higher values on linspace so must rank 1
    assert df_rank.iloc[0]["zone_name"] == "South_District"
