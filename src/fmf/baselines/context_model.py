"""Context models u_hat(z) for the fault-free control effort.

Gradient-boosted trees map the operating context z (temperatures, airflow,
setpoints, time of day) to the actuator command. Residuals on the training
period are computed out of fold, grouped by day, so that calibration
residuals are out-of-sample like the residuals of later test days.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


def _model(params: dict, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**params, random_state=seed, verbose=-1)


def cross_fit_predict(df: pd.DataFrame, features: list[str], target: str, groups: pd.Series,
                      params: dict, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions: each group (e.g. day) is predicted by a model fitted on the others."""
    pred = np.full(len(df), np.nan)
    group_values = groups.to_numpy()
    for g in np.unique(group_values):
        test = group_values == g
        model = _model(params, seed).fit(df.loc[~test, features], df.loc[~test, target])
        pred[test] = model.predict(df.loc[test, features])
    return pred


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str,
                params: dict, seed: int = 0) -> np.ndarray:
    model = _model(params, seed).fit(train[features], train[target])
    return model.predict(test[features])
