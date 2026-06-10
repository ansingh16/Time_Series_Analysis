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

## Configuration

`02_unemployment_forecasting.ipynb` reads a Nasdaq Data Link API key from a
local `config.py` (gitignored):

```python
NASDAQ = "your-api-key"
```
