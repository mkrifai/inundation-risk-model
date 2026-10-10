"""
Morphometric terrain analysis: Slope, Aspect, and Curvatures from DEM.
Uses Horn's finite-difference method and Zevenbergen & Thorne polynomial surface fitting.

References:
-----------
- Horn, B. K. P. (1981). Hill shading and the reflectance map. Proceedings of the IEEE, 69(1), 14-47.
- Zevenbergen, L. W., & Thorne, C. R. (1987). Quantitative analysis of land surface topography.
  Earth Surface Processes and Landforms, 12(1), 47-56.
"""

from typing import Tuple
import numpy as np


def compute_slope_aspect(
    dem: np.ndarray,
    cell_size: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (in degrees) and aspect (in degrees from North) from a DEM.

    Parameters:
    -----------
    dem : 2D np.ndarray
        Digital Elevation Model in projected coordinates (meters).
    cell_size : float
        Spatial resolution in meters (default: 30.0).

    Returns:
    --------
    slope_deg : 2D np.ndarray (degrees, 0 to 90)
    aspect_deg : 2D np.ndarray (degrees, 0 to 360, where 0 is North)
    """
    rows, cols = dem.shape
    slope = np.full_like(dem, np.nan, dtype=np.float32)
    aspect = np.full_like(dem, np.nan, dtype=np.float32)

    # 3x3 window gradients using Horn's algorithm
    # dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cell_size)
    # dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cell_size)
    dz_dx = np.zeros_like(dem, dtype=np.float32)
    dz_dy = np.zeros_like(dem, dtype=np.float32)

    # Interior cells
    dz_dx[1:-1, 1:-1] = (
        (dem[0:-2, 2:] + 2.0 * dem[1:-1, 2:] + dem[2:, 2:]) -
        (dem[0:-2, 0:-2] + 2.0 * dem[1:-1, 0:-2] + dem[2:, 0:-2])
    ) / (8.0 * cell_size)

    dz_dy[1:-1, 1:-1] = (
        (dem[2:, 0:-2] + 2.0 * dem[2:, 1:-1] + dem[2:, 2:]) -
        (dem[0:-2, 0:-2] + 2.0 * dem[0:-2, 1:-1] + dem[0:-2, 2:])
    ) / (8.0 * cell_size)

    gradient = np.sqrt(dz_dx**2 + dz_dy**2)
    slope = np.degrees(np.arctan(gradient))

    # Aspect calculation
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = 90.0 - np.degrees(aspect_rad)
    aspect_deg[aspect_deg < 0] += 360.0
    aspect_deg[gradient == 0] = -1.0  # Flat terrain
    aspect = aspect_deg.copy()

    # Mask boundary cells
    slope[0, :] = slope[-1, :] = slope[:, 0] = slope[:, -1] = np.nan
    aspect[0, :] = aspect[-1, :] = aspect[:, 0] = aspect[:, -1] = np.nan

    return slope, aspect


def compute_curvatures(
    dem: np.ndarray,
    cell_size: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Profile and Planform Curvatures from DEM using Zevenbergen & Thorne (1987).

    Returns:
    --------
    profile_curvature : 2D np.ndarray (flow acceleration / deceleration)
    planform_curvature : 2D np.ndarray (flow convergence / divergence)
    """
    rows, cols = dem.shape
    prof_curv = np.full_like(dem, np.nan, dtype=np.float32)
    plan_curv = np.full_like(dem, np.nan, dtype=np.float32)

    L = cell_size
    L2 = cell_size ** 2

    # 3x3 neighborhood coefficients:
    # z1 z2 z3
    # z4 z5 z6
    # z7 z8 z9
    z1 = dem[0:-2, 0:-2]
    z2 = dem[0:-2, 1:-1]
    z3 = dem[0:-2, 2:]
    z4 = dem[1:-1, 0:-2]
    z5 = dem[1:-1, 1:-1]
    z6 = dem[1:-1, 2:]
    z7 = dem[2:, 0:-2]
    z8 = dem[2:, 1:-1]
    z9 = dem[2:, 2:]

    D = ((z4 + z6) / 2.0 - z5) / L2
    E = ((z2 + z8) / 2.0 - z5) / L2
    F = (-z1 + z3 + z7 - z9) / (4.0 * L2)
    G = (-z4 + z6) / (2.0 * L)
    H = (z2 - z8) / (2.0 * L)

    denom = G**2 + H**2
    valid = denom > 1e-8

    prof = np.zeros_like(z5, dtype=np.float32)
    plan = np.zeros_like(z5, dtype=np.float32)

    # Zevenbergen & Thorne formulas:
    prof[valid] = -2.0 * (D[valid] * G[valid]**2 + E[valid] * H[valid]**2 + F[valid] * G[valid] * H[valid]) / denom[valid]
    plan[valid] = 2.0 * (D[valid] * H[valid]**2 + E[valid] * G[valid]**2 - F[valid] * G[valid] * H[valid]) / denom[valid]

    prof_curv[1:-1, 1:-1] = prof
    plan_curv[1:-1, 1:-1] = plan

    return prof_curv, plan_curv
