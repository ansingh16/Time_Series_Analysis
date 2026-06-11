"""Walk-forward (expanding-window) backtesting for the forecasting models.

A single in-sample fit flatters a model; what matters for forecasting is how it
holds up out-of-sample as the horizon grows. These routines re-fit the model at
every origin on an expanding window, forecast several steps ahead, and record the
per-step error so accuracy degradation over the horizon can be measured and plotted.

``walk_forward_arima`` backtests a level forecast (e.g. price / unemployment rate);
``walk_forward_garch`` backtests a volatility forecast against realised volatility
(proxied by the absolute return). Both return a tidy frame that
``metrics_by_horizon`` collapses into per-step MAE / RMSE / MAPE / directional
accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_arima(
    series: pd.Series,
    order: tuple[int, int, int],
    initial_train_size: int,
    horizon: int = 5,
    step: int = 1,
) -> pd.DataFrame:
    """Expanding-window walk-forward backtest of an ARIMA(``order``) level forecast.

    Starting at ``initial_train_size`` and advancing by ``step``, the model is
    re-fit on ``series[:t]`` and forecasts ``horizon`` steps. Each forecast is
    paired with the realised value at that step. Returns a tidy frame with one
    row per (origin, step): ``origin``, ``step`` (1..horizon), ``anchor`` (last
    training value, used for directional accuracy), ``y_true``, ``y_pred``.
    """
    from statsmodels.tsa.arima.model import ARIMA

    series = pd.Series(series).astype(float)
    n = len(series)
    if initial_train_size >= n:
        raise ValueError("initial_train_size must be smaller than the series length")

    rows = []
    for t in range(initial_train_size, n, step):
        h = min(horizon, n - t)
        if h <= 0:
            break
        train = series.iloc[:t]
        anchor = float(train.iloc[-1])
        forecast = np.asarray(ARIMA(train, order=order).fit().forecast(h), dtype=float)
        actual = series.iloc[t : t + h].to_numpy(dtype=float)
        for i in range(h):
            rows.append(
                {
                    "origin": series.index[t - 1],
                    "step": i + 1,
                    "anchor": anchor,
                    "y_true": actual[i],
                    "y_pred": forecast[i],
                }
            )
    return pd.DataFrame(rows)


def walk_forward_garch(
    returns: pd.Series,
    initial_train_size: int,
    horizon: int = 5,
    step: int = 1,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    mean: str = "Constant",
    vol: str = "GARCH",
    dist: str = "normal",
) -> pd.DataFrame:
    """Expanding-window walk-forward backtest of a GARCH-family volatility forecast.

    At each origin the model is re-fit on ``returns[:t]`` and produces an
    ``horizon``-step variance forecast; the forecast volatility (sqrt of the
    variance) is compared against realised volatility, proxied by the absolute
    return at each future step. Returns a tidy frame with ``origin``, ``step``,
    ``y_true`` (realised |return|), ``y_pred`` (forecast volatility).
    """
    from arch import arch_model

    returns = pd.Series(returns).astype(float)
    n = len(returns)
    if initial_train_size >= n:
        raise ValueError("initial_train_size must be smaller than the series length")

    rows = []
    for t in range(initial_train_size, n, step):
        h = min(horizon, n - t)
        if h <= 0:
            break
        train = returns.iloc[:t]
        result = arch_model(train, p=p, q=q, o=o, mean=mean, vol=vol, dist=dist).fit(disp="off")
        variance = result.forecast(horizon=h, reindex=False).variance.iloc[-1].to_numpy(dtype=float)
        forecast_vol = np.sqrt(variance)
        realised = returns.iloc[t : t + h].abs().to_numpy(dtype=float)
        for i in range(h):
            rows.append(
                {
                    "origin": returns.index[t - 1],
                    "step": i + 1,
                    "y_true": realised[i],
                    "y_pred": forecast_vol[i],
                }
            )
    return pd.DataFrame(rows)


def metrics_by_horizon(backtest: pd.DataFrame) -> pd.DataFrame:
    """Collapse a walk-forward backtest frame into per-step (horizon) metrics.

    Expects ``step``, ``y_true``, ``y_pred`` columns (and optionally ``anchor``
    for directional accuracy). Returns one row per step with the fold count and
    MAE, RMSE, MAPE (percent), and directional accuracy at that horizon.
    """
    out = []
    for step, grp in backtest.groupby("step"):
        y_true = grp["y_true"].to_numpy(dtype=float)
        y_pred = grp["y_pred"].to_numpy(dtype=float)
        errors = y_pred - y_true

        nonzero = y_true != 0
        mape = (
            float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100)
            if nonzero.any()
            else np.nan
        )

        if "anchor" in grp.columns:
            anchor = grp["anchor"].to_numpy(dtype=float)
            directional = float(np.mean(np.sign(y_true - anchor) == np.sign(y_pred - anchor)))
        else:
            directional = np.nan

        out.append(
            {
                "step": int(step),
                "n": int(len(grp)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mape": mape,
                "directional_accuracy": directional,
            }
        )
    return pd.DataFrame(out).sort_values("step").reset_index(drop=True)


def plot_degradation(metrics: pd.DataFrame, metric: str = "rmse", ax=None, **plot_kwargs):
    """Plot a metric against the forecast horizon to show accuracy degradation.

    ``metrics`` is the output of :func:`metrics_by_horizon`. Returns the matplotlib
    axis so the caller can overlay several models on the same figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    plot_kwargs.setdefault("marker", "o")
    ax.plot(metrics["step"], metrics[metric], **plot_kwargs)
    ax.set_xlabel("Forecast horizon (steps ahead)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Forecast {metric.upper()} by horizon")
    ax.grid(True, alpha=0.3)
    return ax
