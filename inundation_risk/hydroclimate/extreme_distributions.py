"""
Extreme value distribution modeling for hydroclimatic precipitation series.
Supports Gumbel (GEV Type I) and generalized extreme value (GEV) distribution fitting,
return period design rainfall estimation, asymptotic confidence intervals, and goodness-of-fit assessment.

References:
-----------
- Gumbel, E. J. (1958). Statistics of Extremes. Columbia University Press.
- Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer.
- Kite, G. W. (1977). Frequency and Risk Analyses in Hydrology. Water Resources Publications.
- Chow, V. T., Maidment, D. R., & Mays, L. W. (1988). Applied Hydrology. McGraw-Hill.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from scipy import stats


def gumbel_reduced_variate(return_period: float) -> float:
    """
    Compute the Gumbel reduced variate y_T for a given return period T in years.
    y_T = -ln(-ln(1 - 1/T))

    Parameters:
    -----------
    return_period : float
        Design recurrence interval T (years), must be > 1.0.

    Returns:
    --------
    float : Reduced variate y_T.
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
        'mle' for Maximum Likelihood Estimation (via scipy.stats.gumbel_r),
        or 'moments' for Method of Moments.

    Returns:
    --------
    dict containing:
        'loc' : location parameter (mu)
        'scale' : scale parameter (beta)
        'sample_mean' : empirical sample mean
        'sample_std' : empirical sample standard deviation (ddof=1)
        'ks_stat' : Kolmogorov-Smirnov test statistic
        'ks_p_value' : KS test p-value (goodness-of-fit)
        'n_samples' : sample size (N)
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
    Fit Generalized Extreme Value (GEV) distribution using Maximum Likelihood Estimation.

    Returns:
    --------
    dict containing:
        'shape' : shape parameter (c / xi)
        'loc' : location parameter
        'scale' : scale parameter
        'ks_stat' : Kolmogorov-Smirnov test statistic
        'ks_p_value' : KS test p-value
        'n_samples' : sample size
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
    dist_type: str = "gumbel",
    return_ci: bool = False,
    ci_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute design rainfall quantiles and optional asymptotic confidence intervals
    for specified return periods.

    Asymptotic standard error for Gumbel design rainfall is computed using the Kite/Chow
    delta-method formulation:
        SE(x_T) = (beta / sqrt(N)) * sqrt(1.11 + 0.52 * y_T + 1.10 * y_T^2)
        CI = x_T +/- z * SE(x_T)

    Parameters:
    -----------
    fit_params : dict
        Parameters output by fit_gumbel or fit_gev.
    return_periods : list of float
        Return periods in years (e.g. [2, 5, 10, 25, 50, 100]).
    dist_type : str
        'gumbel' or 'gev'.
    return_ci : bool, default False
        Whether to include standard error and (lower, upper) confidence intervals.
    ci_level : float, default 0.95
        Confidence level for uncertainty bounds (e.g., 0.95 for 95% CI).

    Returns:
    --------
    dict mapping return period keys (e.g., 'T100') to rainfall amounts (mm).
    If return_ci=True, also includes '{key}_se', '{key}_ci_lower', and '{key}_ci_upper'.
    """
    results = {}
    z_score = stats.norm.ppf(0.5 + (ci_level / 2.0))
    n_samples = fit_params.get("n_samples", 30)

    for T in return_periods:
        prob = 1.0 - (1.0 / float(T))
        key = f"T{int(T) if float(T).is_integer() else T}"

        if dist_type.lower() == "gumbel":
            loc = fit_params["loc"]
            scale = fit_params["scale"]
            quantile = stats.gumbel_r.ppf(prob, loc=loc, scale=scale)

            # Asymptotic Standard Error (Kite 1977 / Chow et al. 1988)
            y_T = gumbel_reduced_variate(float(T))
            variance_factor = 1.11 + (0.52 * y_T) + (1.10 * (y_T ** 2))
            se = (scale / np.sqrt(n_samples)) * np.sqrt(variance_factor)
        elif dist_type.lower() == "gev":
            c = fit_params["shape"]
            loc = fit_params["loc"]
            scale = fit_params["scale"]
            quantile = stats.genextreme.ppf(prob, c=c, loc=loc, scale=scale)
            # First-order approximation of SE for GEV
            y_T = gumbel_reduced_variate(float(T))
            se = (scale / np.sqrt(n_samples)) * np.sqrt(1.11 + 0.52 * y_T + 1.10 * (y_T ** 2))
        else:
            raise ValueError(f"Unknown dist_type: {dist_type}. Use 'gumbel' or 'gev'.")

        results[key] = float(np.round(quantile, 2))

        if return_ci:
            results[f"{key}_se"] = float(np.round(se, 2))
            results[f"{key}_ci_lower"] = float(np.round(quantile - (z_score * se), 2))
            results[f"{key}_ci_upper"] = float(np.round(quantile + (z_score * se), 2))

    return results
