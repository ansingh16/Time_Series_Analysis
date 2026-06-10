"""Data loading, preparation, and stationarity testing.

Two data sources are used across the notebooks: stock prices from Yahoo Finance
(``yfinance``) and the US unemployment rate from FRED via Nasdaq Data Link. Both
fetchers cache to parquet so the rest of the pipeline runs offline once the data
has been pulled once. Stationarity helpers (ADF, KPSS) back the model selection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def fetch_stock(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: str | Path | None = "data",
) -> pd.DataFrame:
    """Download OHLCV data for one ticker from Yahoo Finance, with parquet caching.

    Replaces the notebooks' deprecated ``pandas_datareader`` /
    ``yfin.pdr_override()`` path with a direct ``yfinance.download`` call.
    """
    import yfinance as yf

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_path = cache_dir / f"{ticker}_{start}_{end}_{interval}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    df = yf.download(
        ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=True
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({start} to {end}, {interval}).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    return df


def fetch_fred(
    series_id: str = "FRED/UNRATENSA",
    start: str = "2010-01-01",
    end: str = "2019-12-31",
    api_key: str | None = None,
    column: str = "unemp_rate",
    cache_dir: str | Path | None = "data",
) -> pd.DataFrame:
    """Fetch a FRED series through Nasdaq Data Link, with parquet caching.

    The API key is read from the ``api_key`` argument; if omitted it falls back
    to ``config.NASDAQ`` (a gitignored module). Never hard-code the key here.
    """
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        safe = series_id.replace("/", "_")
        cache_path = cache_dir / f"{safe}_{start}_{end}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    import nasdaqdatalink

    if api_key is None:
        import config

        api_key = config.NASDAQ
    nasdaqdatalink.ApiConfig.api_key = api_key

    df = nasdaqdatalink.get(dataset=series_id, start_date=start, end_date=end).rename(
        columns={"Value": column}
    )
    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    return df


def prepare_prices(close: pd.Series, freq: str = "D") -> pd.Series:
    """Reindex a price series to a regular frequency and forward-fill gaps."""
    close = close.copy()
    close.index = pd.DatetimeIndex(close.index)
    close = close.asfreq(freq)
    if close.isna().any():
        close = close.ffill()
    return close


def compute_returns(close: pd.Series, kind: str = "pct", scale: float = 1.0) -> pd.Series:
    """Return a stationary returns series.

    ``kind="pct"`` gives simple percentage returns, ``kind="log"`` log returns.
    The volatility notebook scales returns by 100 to help the GARCH optimiser.
    """
    if kind == "pct":
        returns = close.pct_change()
    elif kind == "log":
        returns = np.log(close).diff()
    else:
        raise ValueError("kind must be 'pct' or 'log'")
    return (returns * scale).dropna()


def train_test_split_ts(series: pd.Series, test_size: float | int = 0.1):
    """Chronological split. ``test_size`` < 1 is a fraction; >= 1 a number of rows."""
    n = len(series)
    n_test = int(round(n * test_size)) if test_size < 1 else int(test_size)
    if not 0 < n_test < n:
        raise ValueError("test_size must leave a non-empty train and test set")
    return series.iloc[:-n_test], series.iloc[-n_test:]


def adf_test(x: pd.Series, alpha: float = 0.05) -> dict:
    """Augmented Dickey-Fuller test (null: unit root). ``stationary`` if p < alpha."""
    stat, pvalue, nlags, nobs = adfuller(x.dropna(), autolag="AIC")[:4]
    return {
        "statistic": stat,
        "pvalue": pvalue,
        "n_lags": nlags,
        "n_obs": nobs,
        "stationary": pvalue < alpha,
    }


def kpss_test(x: pd.Series, regression: str = "c", alpha: float = 0.05) -> dict:
    """KPSS test (null: stationary). ``stationary`` if p > alpha — the reverse of ADF."""
    stat, pvalue, nlags = kpss(x.dropna(), regression=regression, nlags="auto")[:3]
    return {
        "statistic": stat,
        "pvalue": pvalue,
        "n_lags": nlags,
        "stationary": pvalue > alpha,
    }
