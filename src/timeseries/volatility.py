"""Volatility modelling: Hurst exponent and the GARCH family.

Extracted from the volatility notebook. ``fit_garch`` is a single entry point that
covers plain GARCH, GJR-GARCH (``o>0``), and EGARCH (``vol="EGARCH"``). The
rolling/expanding forecasters and the CCC-GARCH portfolio routine are written as
self-contained functions, which also fixes the notebook cells that referenced an
undefined ``stocks`` frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def hurst(ts, max_lag: int = 100) -> float:
    """Hurst exponent. H < 0.5 mean-reverting, ~0.5 random walk, > 0.5 trending."""
    ts = np.asarray(ts, dtype=float)
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    mean: str = "Constant",
    vol: str = "GARCH",
    dist: str = "normal",
    **fit_kwargs,
):
    """Fit a volatility model with the ``arch`` package.

    - plain GARCH(1,1):  ``p=1, q=1, o=0, vol="GARCH"``
    - GJR-GARCH:         ``o=1, vol="GARCH"`` (asymmetric leverage term)
    - EGARCH:            ``vol="EGARCH"``

    ``dist`` accepts e.g. ``"normal"``, ``"t"``, ``"skewt"``. Optimiser output is
    suppressed unless overridden via ``fit_kwargs``.
    """
    model = arch_model(returns, p=p, q=q, o=o, mean=mean, vol=vol, dist=dist)
    fit_kwargs.setdefault("disp", "off")
    return model.fit(**fit_kwargs)


def arma_garch(series: pd.Series, order: tuple[int, int, int] = (0, 1, 0), p: int = 1, q: int = 1):
    """Two-stage ARMA(SARIMAX) mean model, then GARCH on its residuals.

    Returns ``(mean_result, garch_result)``.
    """
    import statsmodels.api as sm

    mean_result = sm.tsa.SARIMAX(endog=series, order=order).fit(disp=False)
    garch_result = arch_model(mean_result.resid, mean="Zero", p=p, q=q).fit(disp="off")
    return mean_result, garch_result


def rolling_volatility_forecast(
    log_returns: pd.Series, window_size: int = 50, mode: str = "fixed"
) -> pd.DataFrame:
    """One-step-ahead GARCH(1,1) volatility forecast over a moving window.

    ``mode="fixed"`` refits on the trailing ``window_size`` observations;
    ``mode="expanding"`` refits on all data up to each point. Returns a frame of
    ``Date`` and forecast ``Volatility`` (sqrt of the one-step variance forecast).
    """
    if mode not in ("fixed", "expanding"):
        raise ValueError("mode must be 'fixed' or 'expanding'")
    vols = []
    for i in range(window_size, len(log_returns)):
        window = log_returns.iloc[i - window_size : i] if mode == "fixed" else log_returns.iloc[:i]
        result = arch_model(window, mean="Zero", p=1, q=1).fit(disp="off")
        variance = result.forecast(horizon=1).variance.iloc[-1].values[0]
        vols.append(variance**0.5)
    return pd.DataFrame({"Date": log_returns.index[window_size:], "Volatility": vols})


def simulate_volatility(result, horizon: int = 5, method: str = "simulation"):
    """Variance-path forecast via ``simulation`` or ``bootstrap`` from a fitted model."""
    forecast = result.forecast(horizon=horizon, method=method)
    return forecast.simulations.residual_variances[-1].T


def ccc_garch(portfolio_returns: pd.DataFrame) -> dict:
    """Constant Conditional Correlation GARCH for a portfolio.

    Fits a univariate GARCH(1,1) per asset, builds the constant correlation matrix
    R from standardised residuals, and returns the one-step covariance forecast
    Sigma = D R D (D = diagonal of one-step volatility forecasts). Returns the
    per-asset params, conditional volatilities, residual correlation R, and Sigma.
    """
    assets = list(portfolio_returns.columns)
    params, cond_vol, std_res, models = [], [], [], []
    for asset in assets:
        result = arch_model(
            portfolio_returns[asset], mean="Constant", vol="GARCH", p=1, q=1
        ).fit(update_freq=0, disp="off")
        params.append(result.params)
        cond_vol.append(result.conditional_volatility)
        std_res.append(result.std_resid)
        models.append(result)

    params = pd.DataFrame(params, index=assets)
    std_res = pd.DataFrame(np.array(std_res).T, columns=assets, index=portfolio_returns.index)

    R = std_res.T @ std_res / portfolio_returns.shape[0]

    diag = np.sqrt([m.forecast(horizon=1).variance.iloc[-1, 0] for m in models])
    D = np.zeros((len(assets), len(assets)))
    np.fill_diagonal(D, diag)
    sigma = D @ R.values @ D

    return {"params": params, "R": R, "sigma": sigma, "models": models}
