"""
InundationRisk: A Generalized Open-Source Framework for Regional Inundation
and Multi-Hazard Risk Modeling.

Author:
- Mochamad Khoirul Rifai
Affiliation:
- Geophysics Research Group, Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Negeri Malang (UM), Indonesia
"""

__version__ = "1.0.0"
__author__ = "Mochamad Khoirul Rifai"
__affiliation__ = "Geophysics Research Group, Department of Physics, Faculty of Mathematics and Natural Sciences, Universitas Negeri Malang (UM), Indonesia"
__license__ = "BSD-3-Clause"

from inundation_risk.hydroclimate.extreme_distributions import fit_gumbel, fit_gev, compute_return_periods
from inundation_risk.hydroclimate.etccdi import compute_etccdi_indices
from inundation_risk.spatial_features.morphometry import compute_slope_aspect, compute_curvatures
from inundation_risk.spatial_features.hydrology import compute_twi, compute_distance_to_stream
from inundation_risk.hazard_ml.trainers import HazardModelTrainer
from inundation_risk.hazard_ml.spatial_cv import SpatialBlockKFold
from inundation_risk.risk_ipcc.ahp import compute_ahp_weights
from inundation_risk.risk_ipcc.integration import compute_ipcc_risk

__all__ = [
    "fit_gumbel",
    "fit_gev",
    "compute_return_periods",
    "compute_etccdi_indices",
    "compute_slope_aspect",
    "compute_curvatures",
    "compute_twi",
    "compute_distance_to_stream",
    "HazardModelTrainer",
    "SpatialBlockKFold",
    "compute_ahp_weights",
    "compute_ipcc_risk",
]
