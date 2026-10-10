"""
Scenario simulation and policy planning overlay module.
Evaluates climate change adaptation, spatial plan conflicts, and avoided risk metrics.
"""

from inundation_risk.scenarios.simulation import simulate_adaptation_scenario
from inundation_risk.scenarios.policy_overlay import evaluate_spatial_plan_conflicts

__all__ = [
    "simulate_adaptation_scenario",
    "evaluate_spatial_plan_conflicts",
]
