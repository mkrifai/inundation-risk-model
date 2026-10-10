"""
IPCC Risk synthesis engine:
Risk = Hazard × Exposure × Vulnerability
Classification and administrative hotspot ranking.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_ipcc_risk(
    hazard: np.ndarray,
    exposure: np.ndarray,
    vulnerability: np.ndarray
) -> np.ndarray:
    """
    Compute IPCC Multi-Hazard Risk Index:
    Risk = Hazard * Exposure * Vulnerability

    All input rasters must be on the same grid and normalized to [0, 1].
    
    Note:
    -----
    The output is a dimensionless relative Risk Index in the range [0, 1],
    useful for spatial comparison and zoning. It does not represent absolute
    monetary damage or casualty estimates.
    """
    valid_mask = (~np.isnan(hazard)) & (~np.isnan(exposure)) & (~np.isnan(vulnerability))
    risk = np.full_like(hazard, np.nan, dtype=np.float32)

    risk[valid_mask] = hazard[valid_mask] * exposure[valid_mask] * vulnerability[valid_mask]
    return risk


def classify_risk_levels(
    risk: np.ndarray,
    n_classes: int = 5,
    method: str = "quantile"
) -> Tuple[np.ndarray, List[float]]:
    """
    Classify continuous risk values into discrete ordinal hazard/risk classes (1 to n_classes).
    1: Very Low, 2: Low, 3: Moderate, 4: High, 5: Very High.
    """
    valid_mask = ~np.isnan(risk)
    classified = np.zeros_like(risk, dtype=np.int32)
    if not np.any(valid_mask):
        return classified, []

    valid_vals = risk[valid_mask]

    if method == "quantile":
        percentiles = np.linspace(0, 100, n_classes + 1)
        thresholds = [float(np.percentile(valid_vals, p)) for p in percentiles]
    elif method == "equal":
        vmin, vmax = np.min(valid_vals), np.max(valid_vals)
        thresholds = [float(v) for v in np.linspace(vmin, vmax, n_classes + 1)]
    else:
        raise ValueError(f"Unknown classification method: {method}. Use 'quantile' or 'equal'.")

    # Digitize into classes 1 .. n_classes
    # np.digitize returns 1-based bins
    bins = thresholds[1:-1]
    classes = np.digitize(valid_vals, bins) + 1
    classified[valid_mask] = classes

    return classified, thresholds


def rank_administrative_hotspots(
    risk_raster: np.ndarray,
    admin_zone_raster: np.ndarray,
    zone_names: Dict[int, str],
    pixel_area_km2: float = 0.0009  # 30m x 30m = 900 m^2 = 0.0009 km^2
) -> pd.DataFrame:
    """
    Aggregate risk metrics per administrative zone to rank disaster risk priority.

    Returns:
    --------
    pd.DataFrame containing zone name, mean risk, max risk, and high-risk area (km²).
    """
    records = []
    unique_zones = np.unique(admin_zone_raster[~np.isnan(admin_zone_raster)])

    for zone_id in unique_zones:
        zone_id = int(zone_id)
        name = zone_names.get(zone_id, f"Zone_{zone_id}")
        zone_mask = (admin_zone_raster == zone_id) & (~np.isnan(risk_raster))

        if not np.any(zone_mask):
            continue

        zone_risk = risk_raster[zone_mask]
        mean_risk = float(np.mean(zone_risk))
        max_risk = float(np.max(zone_risk))
        # Area in top 20% risk
        high_risk_threshold = np.percentile(risk_raster[~np.isnan(risk_raster)], 80.0)
        high_risk_cells = np.sum(zone_risk >= high_risk_threshold)
        high_risk_area_km2 = float(high_risk_cells * pixel_area_km2)

        records.append({
            "zone_id": zone_id,
            "zone_name": name,
            "mean_risk": round(mean_risk, 4),
            "max_risk": round(max_risk, 4),
            "high_risk_area_km2": round(high_risk_area_km2, 3),
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="mean_risk", ascending=False).reset_index(drop=True)
        df["priority_rank"] = np.arange(1, len(df) + 1)
    return df
