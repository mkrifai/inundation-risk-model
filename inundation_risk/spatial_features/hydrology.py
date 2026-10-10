"""
Hydrological conditioning indices: Topographic Wetness Index (TWI),
stream extraction, and Euclidean distance to stream networks.
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_twi(
    flow_accumulation: np.ndarray,
    slope_deg: np.ndarray,
    cell_size: float = 30.0,
    min_slope_deg: float = 0.1
) -> np.ndarray:
    """
    Compute Topographic Wetness Index (TWI):
    TWI = ln( (A * cell_size) / tan(beta) )
    where A is upstream contributing flow accumulation cells,
    and beta is slope in radians.

    Parameters:
    -----------
    flow_accumulation : 2D np.ndarray
        Raster of flow accumulation (upstream contributing cell counts).
    slope_deg : 2D np.ndarray
        Raster of terrain slope in degrees.
    cell_size : float
        Grid resolution in meters (default: 30.0).
    min_slope_deg : float
        Lower threshold for slope to prevent division by zero in flat terrain.

    Returns:
    --------
    twi : 2D np.ndarray
    """
    valid_mask = (~np.isnan(flow_accumulation)) & (~np.isnan(slope_deg))
    twi = np.full_like(flow_accumulation, np.nan, dtype=np.float32)

    # Specific catchment area per unit contour width (a = (flow_accum + 1) * cell_size)
    sca = (np.maximum(flow_accumulation[valid_mask], 0.0) + 1.0) * cell_size

    # Clamped slope in radians
    clamped_slope = np.maximum(slope_deg[valid_mask], min_slope_deg)
    slope_rad = np.radians(clamped_slope)
    tan_slope = np.tan(slope_rad)
    tan_slope = np.maximum(tan_slope, 1e-4)

    twi[valid_mask] = np.log(sca / tan_slope)
    return twi


def extract_stream_network(
    flow_accumulation: np.ndarray,
    threshold_cells: int = 1000
) -> np.ndarray:
    """
    Extract boolean binary stream mask where upstream accumulation >= threshold_cells.
    """
    valid_mask = ~np.isnan(flow_accumulation)
    stream_mask = np.zeros_like(flow_accumulation, dtype=bool)
    stream_mask[valid_mask] = flow_accumulation[valid_mask] >= threshold_cells
    return stream_mask


def compute_distance_to_stream(
    stream_mask: np.ndarray,
    cell_size: float = 30.0
) -> np.ndarray:
    """
    Compute Euclidean distance (in meters) from each cell to the nearest stream channel.
    Uses Exact Euclidean Distance Transform (EDT).
    """
    # EDT measures distance to nearest zero-pixel, so we invert stream_mask
    dist_cells = distance_transform_edt(~stream_mask)
    dist_meters = dist_cells.astype(np.float32) * cell_size
    return dist_meters
