"""
ETCCDI (Expert Team on Climate Change Detection and Indices) extreme rainfall indices.
Calculates standardized indices for hydroclimatic risk characterization.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def compute_etccdi_indices(
    series: Union[pd.Series, np.ndarray],
    dates: Union[pd.DatetimeIndex, None] = None,
    wet_threshold_mm: float = 1.0,
    baseline_quantile_p95: float = 95.0,
    baseline_quantile_p99: float = 99.0
) -> Dict[str, float]:
    """
    Compute core ETCCDI extreme precipitation indices for a daily rainfall series.

    Parameters:
    -----------
    series : pd.Series or 1D np.ndarray
        Daily rainfall series in millimeters.
    dates : pd.DatetimeIndex, optional
        Dates corresponding to each rainfall measurement. If series is a pd.Series
        with a DatetimeIndex, this is inferred automatically.
    wet_threshold_mm : float
        Threshold above which a day is considered a 'wet day' (standard: 1.0 mm).
    baseline_quantile_p95 : float
        Percentile for very wet days (standard: 95.0).
    baseline_quantile_p99 : float
        Percentile for extremely wet days (standard: 99.0).

    Returns:
    --------
    dict of ETCCDI indices:
        - rx1day (mm)
        - rx5day (mm)
        - prcptot (mm)
        - sdii (mm/day)
        - r95p (mm)
        - r99p (mm)
        - r95p_days (count)
        - r99p_days (count)
        - max_cwd (days)
        - max_cdd (days)
    """
    if isinstance(series, pd.Series):
        vals = series.values.astype(np.float64)
    else:
        vals = np.asarray(series, dtype=np.float64)

    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        raise ValueError("Rainfall series is empty or contains only NaNs.")

    # 1. RX1day: Maximum 1-day precipitation
    rx1day = float(np.max(vals))

    # 2. RX5day: Maximum 5-day rolling sum
    if len(vals) >= 5:
        # Use convolution to compute 5-day moving sum
        rolling_5day = np.convolve(vals, np.ones(5), mode="valid")
        rx5day = float(np.max(rolling_5day))
    else:
        rx5day = float(np.sum(vals))

    # 3. Wet days filter
    wet_days = vals[vals >= wet_threshold_mm]
    wet_count = len(wet_days)

    if wet_count > 0:
        prcptot = float(np.sum(wet_days))
        sdii = float(prcptot / wet_count)

        q95 = np.percentile(wet_days, baseline_quantile_p95)
        q99 = np.percentile(wet_days, baseline_quantile_p99)

        r95p_mask = wet_days > q95
        r99p_mask = wet_days > q99

        r95p = float(np.sum(wet_days[r95p_mask]))
        r99p = float(np.sum(wet_days[r99p_mask]))
        r95p_days = int(np.sum(r95p_mask))
        r99p_days = int(np.sum(r99p_mask))
    else:
        prcptot = 0.0
        sdii = 0.0
        r95p = 0.0
        r99p = 0.0
        r95p_days = 0
        r99p_days = 0

    # 4. Consecutive wet & dry days
    is_wet = vals >= wet_threshold_mm
    max_cwd = 0
    max_cdd = 0
    cur_cwd = 0
    cur_cdd = 0

    for w in is_wet:
        if w:
            cur_cwd += 1
            if cur_cwd > max_cwd:
                max_cwd = cur_cwd
            cur_cdd = 0
        else:
            cur_cdd += 1
            if cur_cdd > max_cdd:
                max_cdd = cur_cdd
            cur_cwd = 0

    return {
        "rx1day": round(rx1day, 2),
        "rx5day": round(rx5day, 2),
        "prcptot": round(prcptot, 2),
        "sdii": round(sdii, 2),
        "r95p": round(r95p, 2),
        "r99p": round(r99p, 2),
        "r95p_days": r95p_days,
        "r99p_days": r99p_days,
        "max_cwd": max_cwd,
        "max_cdd": max_cdd,
        "total_days": len(vals),
        "wet_days": wet_count,
    }
