"""
Unit tests for hydroclimatic extreme distribution fitting and ETCCDI calculations.
"""

import pytest
import numpy as np
from inundation_risk.hydroclimate.extreme_distributions import (
    fit_gumbel,
    fit_gev,
    compute_return_periods,
    gumbel_reduced_variate,
)
from inundation_risk.hydroclimate.etccdi import compute_etccdi_indices


def test_gumbel_reduced_variate():
    # T=2 corresponds to y_T = -ln(-ln(0.5)) ~ 0.3665
    y2 = gumbel_reduced_variate(2.0)
    assert np.isclose(y2, 0.3665129, atol=1e-4)

    # T=100 corresponds to y_T = -ln(-ln(0.99)) ~ 4.6001
    y100 = gumbel_reduced_variate(100.0)
    assert np.isclose(y100, 4.600149, atol=1e-4)

    with pytest.raises(ValueError):
        gumbel_reduced_variate(0.5)


def test_fit_gumbel(synthetic_rainfall_series):
    params_mle = fit_gumbel(synthetic_rainfall_series, method="mle")
    assert "loc" in params_mle
    assert "scale" in params_mle
    assert params_mle["scale"] > 0
    assert params_mle["ks_p_value"] > 0.05  # Should fit well

    params_mom = fit_gumbel(synthetic_rainfall_series, method="moments")
    assert "loc" in params_mom
    assert np.isclose(params_mle["loc"], params_mom["loc"], rtol=0.2)


def test_compute_return_periods(synthetic_rainfall_series):
    params = fit_gumbel(synthetic_rainfall_series)
    rp = compute_return_periods(params, [2, 10, 50, 100])
    assert "T2" in rp
    assert "T100" in rp
    # Quantiles must be strictly monotonically increasing with return period
    assert rp["T2"] < rp["T10"] < rp["T50"] < rp["T100"]


def test_compute_return_periods_with_ci(synthetic_rainfall_series):
    params = fit_gumbel(synthetic_rainfall_series)
    rp_ci = compute_return_periods(params, [2, 10, 50, 100], return_ci=True)
    assert "T100" in rp_ci
    assert "T100_se" in rp_ci
    assert "T100_ci_lower" in rp_ci
    assert "T100_ci_upper" in rp_ci
    assert rp_ci["T100_se"] > 0
    assert rp_ci["T100_ci_lower"] < rp_ci["T100"] < rp_ci["T100_ci_upper"]


def test_etccdi_indices(synthetic_daily_rainfall):
    indices = compute_etccdi_indices(synthetic_daily_rainfall)
    assert indices["rx1day"] >= 85.0  # Injected peak was 85.0
    assert indices["rx5day"] >= indices["rx1day"]
    assert indices["prcptot"] > 0
    assert indices["sdii"] > 0
    assert indices["max_cwd"] > 0
    assert indices["max_cdd"] > 0
