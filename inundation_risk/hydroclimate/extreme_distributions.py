"""
Extreme value distribution modeling for hydroclimatic precipitation series.
Supports Gumbel (GEV Type I) and generalized extreme value (GEV) distribution fitting,
return period design rainfall estimation, and goodness-of-fit assessment.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from scipy import stats


def gumbel_reduced_variate(return_period: float) -> float:
    """
    Compute the Gumbel reduced variate y_T for a given return period T in years.
    y_T = -ln(-ln(1 - 1/T))
    """
    if return_period <= 1.0:
        raise ValueError("Return period T must be strictly greater than 1.0 year.")
    prob_exceedance = 1.0 / return_period
    prob_non_exceedance = 1.0 - prob_exceedance
    return -np.log(-np.log(prob_non_exceedance))


def fit_gumbel(
    data: Union[np.ndarray, List[float]],
    method: str = "mle"
) -> Dict[str, float]:
    """
    Fit Gumbel distribution (GEV Type I) to annual maximum precipitation series.

    Parameters:
    -----------
    data : array-like
        Annual maximum precipitation series (mm).
    method : str
        'mle' for Maximum Likelihood Estimation, or 'moments' for Method of Moments.

    Returns:
    --------
    dict containing 'loc' (mu), 'scale' (beta), 'mean', 'std', and KS test 'p_value'.
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        raise ValueError("At least 5 annual maximum observations required for distribution fitting.")

    if method.lower() == "mle":
        # scipy.stats.gumbel_r represents standard Gumbel for maxima
        loc, scale = stats.gumbel_r.fit(arr)
    elif method.lower() == "moments":
        sample_mean = np.mean(arr)
        sample_std = np.std(arr, ddof=1)
        scale = (sample_std * np.sqrt(6)) / np.pi
        euler_mascheroni = 0.5772156649015329
        loc = sample_mean - (euler_mascheroni * scale)
    else:
        raise ValueError(f"Unknown estimation method: {method}. Use 'mle' or 'moments'.")

    # Goodness-of-fit: Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(arr, "gumbel_r", args=(loc, scale))

    return {
        "loc": float(loc),
        "scale": float(scale),
        "sample_mean": float(np.mean(arr)),
        "sample_std": float(np.std(arr, ddof=1)),
        "ks_stat": float(ks_stat),
        "ks_p_value": float(p_value),
        "n_samples": len(arr),
    }


def fit_gev(data: Union[np.ndarray, List[float]]) -> Dict[str, float]:
    """
    Fit Generalized Extreme Value (GEV) distribution using MLE.

    Returns:
    --------
    dict containing 'shape' (c / xi), 'loc', 'scale', and KS test statistics.
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        raise ValueError("At least 5 observations required for GEV fitting.")

    c, loc, scale = stats.genextreme.fit(arr)
    ks_stat, p_value = stats.kstest(arr, "genextreme", args=(c, loc, scale))

    return {
        "shape": float(c),
        "loc": float(loc),
        "scale": float(scale),
        "ks_stat": float(ks_stat),
        "ks_p_value": float(p_value),
        "n_samples": len(arr),
    }


def compute_return_periods(
    fit_params: Dict[str, float],
    return_periods: List[float] = [2, 5, 10, 25, 50, 100],
    dist_type: str = "gumbel"
) -> Dict[str, float]:
    """
    Compute design rainfall quantiles for specified return periods.

    Parameters:
    -----------
    fit_params : dict
        Parameters output by fit_gumbel or fit_gev.
    return_periods : list of float
        Return periods in years (e.g. [2, 5, 10, 25, 50, 100]).
    dist_type : str
        'gumbel' or 'gev'.

    Returns:
    --------
    dict mapping return period strings (e.g., 'T2', 'T100') to rainfall amounts (mm).
    """
    results = {}
    for T in return_periods:
        prob = 1.0 - (1.0 / float(T))
        key = f"T{int(T) if float(T).is_integer() else T}"
        if dist_type.lower() == "gumbel":
            loc = fit_params["loc"]
            scale = fit_params["scale"]
            quantile = stats.gumbel_r.ppf(prob, loc=loc, scale=scale)
        elif dist_type.lower() == "gev":
            c = fit_params["shape"]
            loc = fit_params["loc"]
            scale = fit_params["scale"]
            quantile = stats.genextreme.ppf(prob, c=c, loc=loc, scale=scale)
        else:
            raise ValueError(f"Unknown dist_type: {dist_type}. Use 'gumbel' or 'gev'.")
        results[key] = float(np.round(quantile, 2))
    return results
