# Time Series Analysis

Time-series forecasting and volatility modelling across financial and
macroeconomic data. Stock prices come from Yahoo Finance (`yfinance`) and the
unemployment series from the FRED database via Nasdaq Data Link. The reusable
code lives in the `timeseries` package under `src/`; the notebooks are the
analysis narrative.

## Notebooks

- **`01_stock_price_forecasting.ipynb`** — stock price forecasting
  - Moving Average (MA), Autoregressive (AR), ARIMA
  - Grid search and Auto ARIMA for order selection
  - Prophet

- **`02_unemployment_forecasting.ipynb`** — US unemployment-rate forecasting (FRED/UNRATENSA)
  - Seasonal and STL decomposition
  - Stationarity testing (ADF, KPSS)
  - Exponential smoothing: Simple, Holt, Holt-Winters, Auto-ETS
  - ARIMA and Auto-SARIMA

- **`03_volatility_modelling.ipynb`** — stock-return volatility
  - Hurst exponent, GARCH(1,1)
  - Information asymmetry: GJR-GARCH, EGARCH
  - Volatility forecasting: fixed and expanding rolling windows
  - Simulation- and bootstrap-based forecasts
  - CCC-GARCH for portfolio volatility
  - Model evaluation

- **`04_walk_forward_validation.ipynb`** — out-of-sample evaluation
  - Expanding-window walk-forward backtest for ARIMA and GARCH
  - Per-horizon MAE / RMSE / MAPE and directional accuracy
  - Accuracy-degradation plots: how far ahead each model stays useful

## The `timeseries` package

The modelling code from the notebooks is factored into an importable package so it
can be reused and tested:

| Module | Contents |
|--------|----------|
| `data` | Yahoo Finance / FRED fetching (parquet-cached), return construction, ADF & KPSS stationarity tests, chronological split |
| `forecasting` | Seasonal & STL decomposition, exponential smoothing (SES / Holt / Holt-Winters / Auto-ETS), ARIMA, Auto-ARIMA/SARIMA, Prophet |
| `volatility` | Hurst exponent, GARCH / GJR-GARCH / EGARCH, ARMA+GARCH, rolling & expanding volatility forecasts, simulation forecasts, CCC-GARCH |
| `backtest` | Expanding-window walk-forward backtesting and per-horizon metrics |
| `evaluate` | Point-forecast metrics and GARCH variance-error scoring |

```python
from timeseries import fetch_stock, prepare_prices, walk_forward_arima, metrics_by_horizon

prices = prepare_prices(fetch_stock("GOOGL", "2017-01-01", "2019-12-31")["Close"])
backtest = walk_forward_arima(prices, order=(2, 1, 2), initial_train_size=800, horizon=5)
print(metrics_by_horizon(backtest))
```

## Model comparison

`scripts/run_comparison.py` runs every model on the **same series and the same
chronological split** so the numbers are comparable:

```
python scripts/run_comparison.py
```

It holds out the final 10% of the GOOGL 2017–2019 series and produces:

- **Level forecast** (price): a naive last-value baseline vs ARIMA (auto order) vs
  Prophet, scored by MAE / RMSE / MAPE and directional accuracy.
- **Volatility** (returns): GARCH(1,1) vs GJR-GARCH vs EGARCH, scored by variance
  error against realised variance, with AIC / BIC.

Outputs are written to `results/`: `comparison_level.csv`,
`comparison_volatility.csv`, `forecast_comparison.png`, `volatility_models.png`.

### Key findings

The clearest result comes from the walk-forward validation in notebook 04, run on
GOOGL 2017–2019 (22 forecast origins, expanding window, 5-day horizon).

- **The price is a random walk — the model says so itself.** `auto_arima` selected
  **ARIMA(0, 1, 0)** on the price: a pure random walk whose best estimate of every
  future day is the last observed price. Walk-forward error grows steadily with the
  horizon and directional accuracy decays to nothing:

  | Horizon (days) | RMSE | MAPE | Directional accuracy |
  |:--:|:--:|:--:|:--:|
  | 1 | 0.61 | 0.68% | 0.32 |
  | 2 | 1.37 | 1.27% | 0.18 |
  | 3 | 1.42 | 1.59% | 0.00 |
  | 4 | 1.47 | 1.73% | 0.00 |
  | 5 | 1.49 | 1.83% | 0.00 |

  A multi-day price forecast carries essentially no directional edge — the
  validation makes that honest rather than hiding it behind one flattering split.

- **Volatility is where the structure is.** Returns are near-unforecastable in the
  mean, but their variance is persistent, so the GARCH(1,1) volatility forecast MAE
  *stays flat* across the horizon (≈0.014 at one day, ≈0.007 by five) instead of
  compounding like the price error. The asymmetric models (GJR-GARCH, EGARCH) add a
  leverage term for the "bad news raises volatility more than good news" effect;
  whether that extra parameter pays off is decided on AIC / BIC and variance error
  (see `scripts/run_comparison.py`), not on in-sample fit alone.

## Install

```
pip install -e .            # core
pip install -e ".[extras]"  # + Prophet and sktime (Auto-ETS)
pip install -e ".[dev]"     # + pytest, ruff
```

## Configuration

`02_unemployment_forecasting.ipynb` (and `fetch_fred`) read a Nasdaq Data Link API
key from a local `config.py` (gitignored):

```python
NASDAQ = "your-api-key"
```
