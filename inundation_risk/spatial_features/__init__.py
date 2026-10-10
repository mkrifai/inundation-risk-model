"""
Spatial conditioning features extraction: morphometry, hydrology, and raster utilities.
"""

from inundation_risk.spatial_features.raster_utils import (
    read_geotiff,
    write_geotiff,
    align_and_match_rasters,
    normalize_raster,
)
from inundation_risk.spatial_features.morphometry import (
    compute_slope_aspect,
    compute_curvatures,
)
from inundation_risk.spatial_features.hydrology import (
    compute_twi,
    extract_stream_network,
    compute_distance_to_stream,
)

__all__ = [
    "read_geotiff",
    "write_geotiff",
    "align_and_match_rasters",
    "normalize_raster",
    "compute_slope_aspect",
    "compute_curvatures",
    "compute_twi",
    "extract_stream_network",
    "compute_distance_to_stream",
]
