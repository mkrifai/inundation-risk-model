"""
Spatial planning overlay and disaster mitigation policy analysis.
Identifies spatial development conflicts between municipal plans and hazard zones.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def evaluate_spatial_plan_conflicts(
    hazard_raster: np.ndarray,
    zoning_raster: np.ndarray,
    planned_urban_codes: List[int],
    population_density: np.ndarray,
    hazard_threshold: float = 0.60,
    pixel_area_km2: float = 0.0009
) -> Dict[str, float]:
    """
    Evaluate spatial conflicts where planned settlement/urban expansion overlaps
    with high or very high flood hazard areas.

    Parameters:
    -----------
    hazard_raster : 2D np.ndarray [0, 1]
    zoning_raster : 2D np.ndarray (integer land-use / spatial plan codes)
    planned_urban_codes : list of int
        Categorical codes representing planned residential/commercial zones.
    population_density : 2D np.ndarray (inhabitants per cell)
    hazard_threshold : float
        Threshold for high hazard (e.g. 0.60).
    pixel_area_km2 : float
        Area of one grid pixel in square kilometers.

    Returns:
    --------
    dict containing conflict metrics:
        - conflict_area_km2
        - total_planned_area_km2
        - percent_planned_area_in_conflict
        - exposed_population_in_conflict
    """
    valid = (~np.isnan(hazard_raster)) & (~np.isnan(zoning_raster))

    planned_mask = np.isin(zoning_raster, planned_urban_codes) & valid
    high_hazard_mask = (hazard_raster >= hazard_threshold) & valid

    conflict_mask = planned_mask & high_hazard_mask

    total_planned_cells = int(np.sum(planned_mask))
    conflict_cells = int(np.sum(conflict_mask))

    total_planned_km2 = float(total_planned_cells * pixel_area_km2)
    conflict_km2 = float(conflict_cells * pixel_area_km2)

    pct_conflict = float((conflict_km2 / total_planned_km2) * 100.0) if total_planned_km2 > 0 else 0.0

    exposed_pop = 0.0
    if population_density is not None:
        pop_valid = population_density.copy()
        pop_valid[np.isnan(pop_valid)] = 0.0
        exposed_pop = float(np.sum(pop_valid[conflict_mask]))

    return {
        "conflict_area_km2": round(conflict_km2, 3),
        "total_planned_area_km2": round(total_planned_km2, 3),
        "percent_planned_area_in_conflict": round(pct_conflict, 2),
        "exposed_population_in_conflict": round(exposed_pop, 0),
    }
