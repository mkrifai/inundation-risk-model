"""
Geospatial raster utilities: I/O, reprojection, grid alignment, and normalization.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def read_geotiff(filepath: str) -> Tuple[np.ndarray, Dict]:
    """
    Read a GeoTIFF raster file and return the 2D array and metadata profile.
    """
    with rasterio.open(filepath) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return arr, profile


def write_geotiff(
    filepath: str,
    data: np.ndarray,
    profile: Dict,
    nodata: float = -9999.0,
    dtype: str = "float32"
) -> None:
    """
    Write a 2D numpy array to a GeoTIFF file preserving geospatial metadata.
    """
    out_profile = profile.copy()
    out_profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": dtype,
        "nodata": nodata,
        "compress": "lzw",
    })

    out_arr = data.copy()
    if np.isnan(out_arr).any():
        out_arr = np.nan_to_num(out_arr, nan=nodata)

    with rasterio.open(filepath, "w", **out_profile) as dst:
        dst.write(out_arr.astype(dtype), 1)


def align_and_match_rasters(
    source_path: str,
    reference_path: str,
    resampling_method: Resampling = Resampling.bilinear
) -> np.ndarray:
    """
    Reproject and align source raster onto the grid and CRS of a reference raster.
    """
    with rasterio.open(reference_path) as ref_src:
        dst_crs = ref_src.crs
        dst_transform = ref_src.transform
        dst_width = ref_src.width
        dst_height = ref_src.height
        dst_nodata = ref_src.nodata or -9999.0

    destination = np.full((dst_height, dst_width), dst_nodata, dtype=np.float32)

    with rasterio.open(source_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling_method,
            src_nodata=src.nodata,
            dst_nodata=dst_nodata,
        )

    destination[destination == dst_nodata] = np.nan
    return destination


def normalize_raster(arr: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize raster values to [0, 1] range ignoring NaNs.
    """
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return arr.copy()

    valid_vals = arr[valid_mask]
    norm_arr = np.full_like(arr, np.nan, dtype=np.float32)

    if method == "minmax":
        vmin = np.min(valid_vals)
        vmax = np.max(valid_vals)
        if vmax > vmin:
            norm_arr[valid_mask] = (valid_vals - vmin) / (vmax - vmin)
        else:
            norm_arr[valid_mask] = 0.0
    elif method == "zscore":
        mean = np.mean(valid_vals)
        std = np.std(valid_vals)
        if std > 0:
            norm_arr[valid_mask] = (valid_vals - mean) / std
        else:
            norm_arr[valid_mask] = 0.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return norm_arr
