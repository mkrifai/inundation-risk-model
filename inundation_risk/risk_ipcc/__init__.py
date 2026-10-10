"""
IPCC Risk Framework integration module:
Risk = Hazard × Exposure × Vulnerability
Includes AHP multi-criteria weighting, exposure & vulnerability layers, and hotspot analysis.
"""

from inundation_risk.risk_ipcc.ahp import compute_ahp_weights, verify_ahp_consistency
from inundation_risk.risk_ipcc.exposure import compute_exposure_index
from inundation_risk.risk_ipcc.vulnerability import compute_vulnerability_index
from inundation_risk.risk_ipcc.integration import (
    compute_ipcc_risk,
    classify_risk_levels,
    rank_administrative_hotspots,
)

__all__ = [
    "compute_ahp_weights",
    "verify_ahp_consistency",
    "compute_exposure_index",
    "compute_vulnerability_index",
    "compute_ipcc_risk",
    "classify_risk_levels",
    "rank_administrative_hotspots",
]
