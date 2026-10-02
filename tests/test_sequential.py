import numpy as np
import pandas as pd

from fmf.baselines.context_model import cross_fit_predict
from fmf.detectors.sequential import (conformal_e_statistic, count_alarms,
                                      gaussian_cusum_statistic)

PARAMS = {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 5}


def test_count_alarms_restarts_after_each_alarm():
    x = np.array([0, 0, 5, 0, 0, 5, 5, 0], dtype=float)
    alarms = count_alarms(lambda z: np.cumsum(z), x, threshold=5.0)
    assert alarms == [2, 5, 6]


def test_missing_scores_leave_statistics_unchanged():
    cal = np.random.default_rng(0).standard_normal(500)
    z = np.array([3.0, np.nan, np.nan, 3.0])
    stat = conformal_e_statistic(z, cal, [0.5], dkw_delta=None)
    assert stat[1] == stat[0] and stat[2] == stat[0]
    g = gaussian_cusum_statistic(z, 1.0)
    assert g[1] == g[0] and g[2] == g[0]


def test_cross_fit_predicts_each_group_out_of_fold():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.uniform(size=300), "g": np.repeat(np.arange(3), 100)})
    df["y"] = 2.0 * df.x + np.where(df.g == 2, 5.0, 0.0)  # group 2 has an offset
    pred = cross_fit_predict(df, ["x"], "y", df.g, PARAMS)
    # Each group is predicted by a model fitted on the others: group 2 never saw
    # its own offset, group 0 inherits half of it from the training groups.
    assert np.mean(df.y[df.g == 2] - pred[df.g == 2]) > 4.0
    assert np.mean(df.y[df.g == 0] - pred[df.g == 0]) < -1.0
