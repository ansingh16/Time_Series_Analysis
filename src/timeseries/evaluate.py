"""Forecast and volatility evaluation metrics.

Point-forecast error metrics for the price/macro models, plus the variance-error
metric used to score the GARCH fits against realised (squared, demeaned) returns.
"""

from __future__ import annotations

import numpy as np


def forecast_metrics(y_true, y_pred) -> dict:
    """Point-forecast metrics: MAE, RMSE, MAPE (percent), directional accuracy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    nonzero = y_true != 0
    if nonzero.any():
        mape = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100)
    else:
        mape = np.nan

    if len(y_true) > 1:
        directional = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))
    else:
        directional = np.nan

    return {"mae": mae, "rmse": rmse, "mape": mape, "directional_accuracy": directional}


def variance_error(returns, conditional_volatility) -> dict:
    """Score a GARCH fit: MAE/MSE of conditional variance vs realised variance.

    Realised variance is the squared demeaned return, the proxy the volatility
    notebook compares the model's conditional variance against.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    returns = np.asarray(returns, dtype=float)
    realised_var = (returns - returns.mean()) ** 2
    model_var = np.asarray(conditional_volatility, dtype=float) ** 2
    return {
        "mae": float(mean_absolute_error(realised_var, model_var)),
        "mse": float(mean_squared_error(realised_var, model_var)),
    }
