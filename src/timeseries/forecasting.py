"""Point forecasting: decomposition, exponential smoothing, ARIMA/SARIMA, Prophet.

Extracted from the stock-price and unemployment-rate notebooks. Decomposition and
the stationarity work motivate the model choices; the forecasters return either
fitted statsmodels/pmdarima result objects or forecast series, so the notebooks
keep their plotting while the modelling logic is reusable and testable.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL, seasonal_decompose

# --- Decomposition --------------------------------------------------------

def seasonal_decompose_ts(series: pd.Series, model: str = "additive", period: int | None = None):
    """Classical seasonal decomposition (trend / seasonal / residual)."""
    return seasonal_decompose(series, model=model, period=period)


def stl_decompose(series: pd.Series, **kwargs):
    """STL (Seasonal-Trend decomposition using LOESS)."""
    return STL(series, **kwargs).fit()


# --- Exponential smoothing -------------------------------------------------

def simple_exp_smoothing(train: pd.Series, horizon: int, smoothing_level: float | None = None):
    """Simple exponential smoothing forecast. ``smoothing_level=None`` auto-fits alpha."""
    fit = SimpleExpSmoothing(train).fit(smoothing_level=smoothing_level)
    return fit.forecast(horizon)


def holt(train: pd.Series, horizon: int, exponential: bool = False, damped_trend: bool = False):
    """Holt's linear/exponential trend forecast, optionally damped."""
    fit = Holt(train, exponential=exponential, damped_trend=damped_trend).fit()
    return fit.forecast(horizon)


def holt_winters(
    train: pd.Series,
    horizon: int,
    trend: str = "mul",
    seasonal: str = "add",
    seasonal_periods: int = 12,
    damped_trend: bool = False,
):
    """Holt-Winters triple exponential smoothing (trend + seasonality)."""
    fit = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
    ).fit()
    return fit.forecast(horizon)


def auto_ets(train: pd.Series, horizon: int, seasonal_periods: int = 12):
    """Automatic ETS model selection via sktime's AutoETS (lazy import)."""
    from sktime.forecasting.ets import AutoETS

    model = AutoETS(auto=True, n_jobs=-1, sp=seasonal_periods)
    model.fit(train.to_period())
    return model.predict(fh=list(range(1, horizon + 1)))


# --- ARIMA / SARIMA --------------------------------------------------------

def fit_arima(train: pd.Series, order: tuple[int, int, int]):
    """Fit a statsmodels ARIMA of the given ``(p, d, q)`` order."""
    return ARIMA(train, order=order).fit()


def forecast_arima(model, horizon: int) -> pd.DataFrame:
    """Forecast ``horizon`` steps; returns the statsmodels summary frame.

    Columns: ``mean``, ``mean_se``, ``mean_ci_lower``, ``mean_ci_upper``.
    """
    return model.get_forecast(horizon).summary_frame()


def auto_arima_select(train: pd.Series, seasonal: bool = False, **kwargs):
    """Order selection with ``pmdarima.auto_arima`` (lazy import).

    ``seasonal=True`` searches SARIMA orders. Defaults mirror the notebooks'
    stepwise ADF-based search; any keyword overrides them.
    """
    import pmdarima as pm

    params = dict(
        start_p=0,
        start_q=0,
        test="adf",
        max_p=12,
        max_q=12,
        d=1,
        seasonal=seasonal,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    params.update(kwargs)
    return pm.auto_arima(train, **params)


def prophet_forecast(train: pd.Series, horizon: int, freq: str = "D"):
    """Fit Prophet on a price series and forecast ``horizon`` periods (lazy import).

    Returns Prophet's forecast frame (``ds``, ``yhat``, ``yhat_lower``,
    ``yhat_upper``, ...).
    """
    from prophet import Prophet

    df = train.reset_index()
    df.columns = ["ds", "y"]
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq=freq)
    return model.predict(future)
