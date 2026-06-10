"""Reusable building blocks for the time-series analysis notebooks.

Collects the modelling code that was originally inline in the notebooks so it can
be imported, tested, and reused across the three workflows: stock-price
forecasting, unemployment-rate forecasting, and volatility modelling.
"""

from .data import (
    adf_test,
    compute_returns,
    fetch_fred,
    fetch_stock,
    kpss_test,
    prepare_prices,
    train_test_split_ts,
)
from .evaluate import forecast_metrics, variance_error
from .forecasting import (
    auto_arima_select,
    auto_ets,
    fit_arima,
    forecast_arima,
    holt,
    holt_winters,
    prophet_forecast,
    seasonal_decompose_ts,
    simple_exp_smoothing,
    stl_decompose,
)
from .volatility import (
    arma_garch,
    ccc_garch,
    fit_garch,
    hurst,
    rolling_volatility_forecast,
    simulate_volatility,
)

__all__ = [
    "fetch_stock",
    "fetch_fred",
    "prepare_prices",
    "compute_returns",
    "train_test_split_ts",
    "adf_test",
    "kpss_test",
    "seasonal_decompose_ts",
    "stl_decompose",
    "simple_exp_smoothing",
    "holt",
    "holt_winters",
    "auto_ets",
    "fit_arima",
    "forecast_arima",
    "auto_arima_select",
    "prophet_forecast",
    "hurst",
    "fit_garch",
    "arma_garch",
    "rolling_volatility_forecast",
    "simulate_volatility",
    "ccc_garch",
    "forecast_metrics",
    "variance_error",
]
