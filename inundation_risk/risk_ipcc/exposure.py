"""
Exposure layer synthesis for IPCC Risk Framework.
Combines human population, building footprints, and infrastructure exposure.
"""

from typing import Dict, Optional
import numpy as np
from inundation_risk.spatial_features.raster_utils import normalize_raster


def compute_exposure_index(
    population_density: np.ndarray,
    builtup_density: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Synthesize normalized Exposure Index [0, 1].

    Parameters:
    -----------
    population_density : 2D np.ndarray
        Population per grid cell (e.g. WorldPop).
    builtup_density : 2D np.ndarray, optional
        Built-up or impervious surface fraction.
    weights : dict, optional
        Weights for components, e.g. {'pop': 0.6, 'built': 0.4}.

    Returns:
    --------
    exposure_index : 2D np.ndarray [0, 1]
    """
    norm_pop = normalize_raster(population_density, method="minmax")

    if builtup_density is None:
        return norm_pop

    norm_built = normalize_raster(builtup_density, method="minmax")

    w_pop = weights.get("pop", 0.6) if weights else 0.6
    w_built = weights.get("built", 0.4) if weights else 0.4
    total_w = w_pop + w_built

    valid = (~np.isnan(norm_pop)) & (~np.isnan(norm_built))
    exposure = np.full_like(norm_pop, np.nan, dtype=np.float32)
    exposure[valid] = (w_pop * norm_pop[valid] + w_built * norm_built[valid]) / total_w

    return exposure
