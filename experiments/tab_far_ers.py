"""Table: false alarms on real fault-free days (ERS twin AHU, RP-1312 baseline days).

For each actuator, the baseline model u_hat(z) is fitted on the first half of
each season's days (residuals out of fold by day), an AR model whitens the
residuals, and the test half is monitored by
  - a Gaussian CUSUM on the innovations, tuned to a command shift,
  - the conformal e-detector (kappa-grid mixture of e-CUSUMs) with and without
    the DKW correction, on the same innovations,
  - the Gaussian CUSUM and the conformal e-detector (with and without DKW) on
    hourly means of the residual, standardized by their calibration spread.
Every detector is two-sided (one stream per sign, alpha / 2 each) and restarts
after each alarm and at the start of each day. The table compares false
alarms per day of operation with the nominal rate 1 / ARL0.
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from _common import RESULTS, TABLES, load_config, write_numbers, write_results
from fmf.baselines.context_model import cross_fit_predict, fit_predict
from fmf.datasets.ghalamsiah2026 import ers_days, load_ers
from fmf.detectors.prewhitening import ar_prewhiten, fit_ar
from fmf.detectors.sequential import (conformal_e_statistic, count_alarms,
                                      gaussian_cusum_statistic)
from fmf.theory.delay import gaussian_cusum_threshold

NAME = "tab_far_ers"


def load_days(seasons):
    frames = []
    for _, r in ers_days().query("season in @seasons").iterrows():
        d = load_ers(r.folder)
        d["day"], d["season"] = r.date, r.season
        d["hour"] = d.index.hour + d.index.minute / 60.0
        frames.append(d[(d.occupied > 0.5) & (d.fan_on > 0.5)])
    return pd.concat(frames)


def chronological_split(df):
    cal, test = [], []
    for _, g in df.groupby("season"):
        days = sorted(g.day.unique())
        k = (len(days) + 1) // 2
        cal += days[:k]
        test += days[k:]
    return df[df.day.isin(cal)], df[df.day.isin(test)]


def per_day(df, values):
    return [np.asarray(values)[(df.day == d).to_numpy()] for d in sorted(df.day.unique())]


def padded(series):
    n = max(len(s) for s in series)
    out = np.full((n, len(series)), np.nan)
    for j, s in enumerate(series):
        out[: len(s), j] = s
    return out


def block_means(x, size):
    k = len(x) // size
    return x[: k * size].reshape(k, size).mean(axis=1)


def evaluate_target(cfg, target, seasons):
    df = load_days(seasons)
    cal, test = chronological_split(df)
    feats, params = cfg["features"], cfg["lightgbm"]
    cal_res = cal[target].to_numpy() - cross_fit_predict(cal, feats, target, cal.day, params,
                                                         cfg["seed"])
    test_res = test[target].to_numpy() - fit_predict(cal, test, feats, target, params,
                                                     cfg["seed"])
    cal_days, test_days = per_day(cal, cal_res), per_day(test, test_res)

    ar = fit_ar(padded(cal_days), cfg["ar_order"])
    w_cal = [ar_prewhiten(x, ar) / ar.sigma_w for x in cal_days]
    w_test = [ar_prewhiten(x, ar) / ar.sigma_w for x in test_days]
    w_cal_flat = np.concatenate(w_cal)
    w_cal_flat = w_cal_flat[np.isfinite(w_cal_flat)]

    block = cfg["block_minutes"]
    b_cal = np.concatenate([block_means(x, block) for x in cal_days])
    b_scale = b_cal.std()
    b_test = [block_means(x, block) / b_scale for x in test_days]
    b_cal = b_cal / b_scale

    samples_per_day = np.mean([len(x) for x in test_days])
    shift = cfg["design_shift"] * ar.gain / ar.sigma_w
    b_shift = cfg["design_shift"] / b_scale
    rows = []
    for arl0_days in cfg["arl0_days"]:
        arl0 = arl0_days * samples_per_day  # samples
        h_gauss = gaussian_cusum_threshold(2.0 * arl0, shift)
        h_e = np.log(2.0 * arl0)
        h_block = np.log(2.0 * arl0 / block)
        detectors = {
            "Gaussian CUSUM": (w_test, lambda z: gaussian_cusum_statistic(z, shift), h_gauss),
            "e-detector, DKW": (w_test, lambda z: conformal_e_statistic(
                z, w_cal_flat, cfg["kappas"], cfg["dkw_delta"]), h_e),
            "e-detector, no DKW": (w_test, lambda z: conformal_e_statistic(
                z, w_cal_flat, cfg["kappas"], None), h_e),
            "Gaussian CUSUM, hourly": (b_test, lambda z: gaussian_cusum_statistic(z, b_shift),
                                       gaussian_cusum_threshold(2.0 * arl0 / block, b_shift)),
            "e-detector, hourly": (b_test, lambda z: conformal_e_statistic(
                z, b_cal, cfg["block_kappas"], cfg["dkw_delta"]), h_block),
            "e-detector, hourly, no DKW": (b_test, lambda z: conformal_e_statistic(
                z, b_cal, cfg["block_kappas"], None), h_block),
        }
        for name, (streams, stat, h) in detectors.items():
            alarms = 0
            for x in streams:
                for sign in (1.0, -1.0):
                    alarms += len(count_alarms(stat, sign * x, h))
            rows.append({"target": target, "detector": name, "arl0_days": arl0_days,
                         "alarms": alarms, "test_days": len(streams),
                         "alarms_per_day": alarms / len(streams),
                         "nominal_per_day": 1.0 / arl0_days})
    day_means = np.array([x.mean() for x in cal_days])
    info = {"target": target, "seasons": seasons, "cal_days": len(cal_days),
            "test_days": len(test_days), "residual_std": float(cal_res.std()),
            "day_offset_std": float(day_means.std()), "ar_gain": ar.gain,
            "ar_sigma_w": ar.sigma_w, "innovation_excess_kurtosis": float(kurtosis(w_cal_flat)),
            "calibration_size": int(w_cal_flat.size), "samples_per_day": float(samples_per_day)}
    return rows, info


def main(tables_only=False):
    cfg = load_config(NAME)
    if tables_only:
        payload = json.loads((RESULTS / f"{NAME}.json").read_text())
    else:
        rows, infos = [], []
        for target, seasons in cfg["targets"].items():
            r, info = evaluate_target(cfg, target, seasons)
            rows += r
            infos.append(info)
        payload = {"config": cfg, "rows": rows, "targets": infos}
        write_results(NAME, payload)
    table = pd.DataFrame(payload["rows"])
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / f"{NAME}.csv", index=False)
    print(pd.DataFrame(payload["targets"]).round(4).to_string())
    print(table.pivot_table(index=["target", "detector"], columns="arl0_days",
                            values="alarms_per_day").round(3).to_string())

    def rate(target, detector, arl0):
        sel = table.query("target == @target and detector == @detector and arl0_days == @arl0")
        return float(sel.alarms_per_day.iloc[0])

    infos = {i["target"]: i for i in payload["targets"]}
    numbers = {}
    for target, key in (("ccv_cmd", "cooling"), ("hcv_cmd", "heating"), ("fan_cmd", "fan")):
        numbers[f"{key} test days"] = f"{infos[target]['test_days']}"
        numbers[f"{key} kurtosis"] = f"{infos[target]['innovation_excess_kurtosis']:.0f}"
        numbers[f"{key} gauss rate"] = f"{rate(target, 'Gaussian CUSUM', 30):.2f}"
        numbers[f"{key} dkw rate"] = f"{rate(target, 'e-detector, DKW', 30):.2f}"
        numbers[f"{key} hourly rate"] = f"{rate(target, 'e-detector, hourly', 30):.2f}"
        numbers[f"{key} hourly gauss rate"] = f"{rate(target, 'Gaussian CUSUM, hourly', 30):.2f}"
        numbers[f"{key} hourly nodkw rate"] = (
            f"{rate(target, 'e-detector, hourly, no DKW', 30):.2f}")
        numbers[f"{key} day offset"] = f"{infos[target]['day_offset_std']:.3f}"
    write_numbers(NAME, numbers)


if __name__ == "__main__":
    main(tables_only="--tables-only" in sys.argv)
