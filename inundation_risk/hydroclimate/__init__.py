"""
Hydroclimatic extreme analysis module: GEV/Gumbel fitting and ETCCDI climate indices.
"""

from inundation_risk.hydroclimate.extreme_distributions import (
    fit_gumbel,
    fit_gev,
    compute_return_periods,
    gumbel_reduced_variate,
)
from inundation_risk.hydroclimate.etccdi import compute_etccdi_indices

__all__ = [
    "fit_gumbel",
    "fit_gev",
    "compute_return_periods",
    "gumbel_reduced_variate",
    "compute_etccdi_indices",
]
