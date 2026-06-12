"""Compare all forecasting models on the same series and the same split.

Pulls the GOOGL daily close (2017-2019, the series the price notebook uses), holds
out the final 10% as a chronological test set, and runs two comparisons:

- Level forecasters on the price -- a naive last-value baseline, ARIMA (auto order),
  and Prophet -- scored by MAE / RMSE / MAPE / directional accuracy.
- Volatility models on the log returns -- GARCH(1,1), GJR-GARCH, EGARCH -- scored by
  variance error against realised variance, plus AIC / BIC.

Writes the two metric tables (``results/comparison_level.csv``,
``results/comparison_volatility.csv``) and two plots
(``results/forecast_comparison.png``, ``results/volatility_models.png``).

    python scripts/run_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from timeseries import (  # noqa: E402
    auto_arima_select,
    compute_returns,
    fetch_stock,
    fit_arima,
    fit_garch,
    forecast_arima,
    forecast_metrics,
    prepare_prices,
    prophet_forecast,
    train_test_split_ts,
    variance_error,
)

TICKER, START, END = "GOOGL", "2017-01-01", "2019-12-31"
RESULTS = ROOT / "results"


def compare_level_models(prices: pd.Series, test_size: float = 0.1):
    """Naive / ARIMA / Prophet point-forecast comparison on the price series.

    Returns ``(metrics, train, test, forecasts)`` where ``forecasts`` maps each
    model name to its test-period forecast array.
    """
    train, test = train_test_split_ts(prices, test_size=test_size)
    horizon = len(test)
    forecasts: dict[str, np.ndarray] = {}

    forecasts["Naive (last value)"] = np.full(horizon, float(train.iloc[-1]))

    order = auto_arima_select(train).order
    forecasts[f"ARIMA{order}"] = forecast_arima(fit_arima(train, order), horizon)[
        "mean"
    ].to_numpy(float)

    try:
        forecasts["Prophet"] = prophet_forecast(train, horizon)["yhat"].to_numpy(float)[-horizon:]
    except Exception as exc:  # Prophet is an optional dependency
        print(f"  Prophet skipped: {exc}")

    y_true = test.to_numpy(float)
    metrics = pd.DataFrame({name: forecast_metrics(y_true, fc) for name, fc in forecasts.items()}).T
    return metrics, train, test, forecasts


def compare_volatility_models(returns: pd.Series):
    """GARCH / GJR-GARCH / EGARCH comparison on the return series.

    Returns ``(metrics, fits)``; ``fits`` maps each name to its fitted arch result.
    """
    specs = {
        "GARCH(1,1)": dict(vol="GARCH", o=0),
        "GJR-GARCH": dict(vol="GARCH", o=1),
        "EGARCH": dict(vol="EGARCH", o=1),
    }
    rows, fits = {}, {}
    for name, kwargs in specs.items():
        result = fit_garch(returns, p=1, q=1, **kwargs)
        err = variance_error(returns, result.conditional_volatility)
        rows[name] = {
            "var_mae": err["mae"],
            "var_mse": err["mse"],
            "aic": result.aic,
            "bic": result.bic,
        }
        fits[name] = result
    return pd.DataFrame(rows).T, fits


def _plot_forecasts(train, test, forecasts, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index[-120:], train.iloc[-120:], color="black", lw=1, label="train")
    ax.plot(test.index, test, color="black", lw=2, label="actual")
    for name, fc in forecasts.items():
        ax.plot(test.index, fc, lw=1.5, ls="--", label=name)
    ax.set_title(f"{TICKER} price forecast comparison (last {len(test)} days held out)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_volatility(returns, fits, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(returns.index, returns.abs(), color="lightgrey", lw=0.8, label="|returns| (realised)")
    for name, result in fits.items():
        ax.plot(returns.index, result.conditional_volatility, lw=1.2, label=name)
    ax.set_title(f"{TICKER} conditional volatility by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    print(f"Fetching {TICKER} {START}..{END}")
    close = fetch_stock(TICKER, START, END)["Close"]
    prices = prepare_prices(close, freq="D")
    returns = compute_returns(prices, kind="log", scale=100)

    print("\nLevel model comparison (price):")
    level, train, test, forecasts = compare_level_models(prices)
    print(level.round(4).to_string())
    level.to_csv(RESULTS / "comparison_level.csv")
    _plot_forecasts(train, test, forecasts, RESULTS / "forecast_comparison.png")

    print("\nVolatility model comparison (returns):")
    vol, fits = compare_volatility_models(returns)
    print(vol.round(4).to_string())
    vol.to_csv(RESULTS / "comparison_volatility.csv")
    _plot_volatility(returns, fits, RESULTS / "volatility_models.png")

    print(f"\nSaved tables and plots to {RESULTS}/")


if __name__ == "__main__":
    main()
