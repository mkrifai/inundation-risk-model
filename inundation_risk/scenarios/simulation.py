"""
Future adaptation and climate change scenario simulation.
Simulates projected hazard probability shifts under rainfall scaling
and Nature-based Solutions (NbS) adaptation interventions.

Note:
-----
Scenario outputs represent forward simulations under user-defined scaling
and mitigation efficiencies, useful for comparative policy testing rather than
deterministic predictions of future disaster outcomes.
"""

from typing import Dict, Optional, Tuple
import numpy as np


def simulate_adaptation_scenario(
    baseline_hazard: np.ndarray,
    baseline_risk: np.ndarray,
    rainfall_multiplier: float = 1.15,
    mitigation_efficiency: float = 0.20,
    mitigation_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Simulate scenario impact on hazard and risk.

    Parameters:
    -----------
    baseline_hazard : 2D np.ndarray [0, 1]
    baseline_risk : 2D np.ndarray [0, 1]
    rainfall_multiplier : float
        Rainfall intensification factor (e.g., 1.15 for +15% extreme rainfall).
    mitigation_efficiency : float
        Fractional risk reduction achieved by adaptation measures (e.g. 0.20 for 20% reduction).
    mitigation_mask : 2D np.ndarray (bool), optional
        Spatial zone where adaptation interventions are applied. If None, applied domain-wide.

    Returns:
    --------
    scenario_hazard : 2D np.ndarray
    scenario_risk : 2D np.ndarray
    metrics : dict (mean_baseline_risk, mean_scenario_risk, percent_risk_change)
    """
    valid = (~np.isnan(baseline_hazard)) & (~np.isnan(baseline_risk))

    # Climate change hazard shift
    hazard_scenario = np.full_like(baseline_hazard, np.nan, dtype=np.float32)
    hazard_scenario[valid] = np.clip(baseline_hazard[valid] * (rainfall_multiplier ** 0.5), 0.0, 1.0)

    # Risk scenario
    risk_scenario = np.full_like(baseline_risk, np.nan, dtype=np.float32)
    risk_scenario[valid] = baseline_risk[valid] * (hazard_scenario[valid] / np.maximum(baseline_hazard[valid], 1e-4))

    # Apply mitigation interventions where specified
    if mitigation_mask is not None:
        target = valid & mitigation_mask
        risk_scenario[target] *= (1.0 - mitigation_efficiency)
    else:
        risk_scenario[valid] *= (1.0 - mitigation_efficiency)

    risk_scenario = np.clip(risk_scenario, 0.0, 1.0)

    base_mean = float(np.mean(baseline_risk[valid]))
    scen_mean = float(np.mean(risk_scenario[valid]))
    pct_change = float(((scen_mean - base_mean) / base_mean) * 100.0) if base_mean > 0 else 0.0

    return hazard_scenario, risk_scenario, {
        "mean_baseline_risk": round(base_mean, 4),
        "mean_scenario_risk": round(scen_mean, 4),
        "percent_risk_change": round(pct_change, 2),
    }
