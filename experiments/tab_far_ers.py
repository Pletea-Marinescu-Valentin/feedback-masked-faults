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

from _common import GENERATED, RESULTS, TABLES, load_config, write_numbers, write_results
from fmf.baselines.context_model import cross_fit_predict, fit_predict
from fmf.datasets.ghalamsiah2026 import ers_days, load_ers
from fmf.detectors.prewhitening import ar_prewhiten, fit_ar
from fmf.detectors.sequential import (conformal_e_statistic, count_alarms,
                                      gaussian_cusum_statistic)
from fmf.theory.delay import e_detector_min_delay, gaussian_cusum_threshold

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

    samples_per_day = np.mean([len(x) for x in test_days])
    shift = cfg["design_shift"] * ar.gain / ar.sigma_w
    blocks = {}
    for size in cfg["block_minutes"]:
        b_cal = np.concatenate([block_means(x, size) for x in cal_days])
        scale = b_cal.std()
        blocks[size] = (b_cal / scale, [block_means(x, size) / scale for x in test_days],
                        cfg["design_shift"] / scale)

    rows = []
    for arl0_days in cfg["arl0_days"]:
        arl0 = arl0_days * samples_per_day  # samples
        detectors = {
            "Gaussian CUSUM, 1 min": (w_test, lambda z: gaussian_cusum_statistic(z, shift),
                                      gaussian_cusum_threshold(2.0 * arl0, shift), 1),
            "e-detector, 1 min": (w_test, lambda z: conformal_e_statistic(
                z, w_cal_flat, cfg["kappas"], cfg["dkw_delta"]), np.log(2.0 * arl0), 1),
            "e-detector, 1 min, no DKW": (w_test, lambda z: conformal_e_statistic(
                z, w_cal_flat, cfg["kappas"], None), np.log(2.0 * arl0), 1),
        }
        for size, (b_cal, b_test, b_shift) in blocks.items():
            arl0_b = 2.0 * arl0 / size
            detectors[f"Gaussian CUSUM, {size} min"] = (
                b_test, lambda z, b=b_shift: gaussian_cusum_statistic(z, b),
                gaussian_cusum_threshold(arl0_b, b_shift), size)
            detectors[f"e-detector, {size} min"] = (
                b_test, lambda z, c=b_cal: conformal_e_statistic(
                    z, c, cfg["block_kappas"], cfg["dkw_delta"]), np.log(arl0_b), size)
            detectors[f"e-detector, {size} min, no DKW"] = (
                b_test, lambda z, c=b_cal: conformal_e_statistic(
                    z, c, cfg["block_kappas"], None), np.log(arl0_b), size)
        for name, (streams, stat, h, size) in detectors.items():
            alarms = 0
            for x in streams:
                for sign in (1.0, -1.0):
                    alarms += len(count_alarms(stat, sign * x, h))
            kappas = cfg["kappas"] if size == 1 else cfg["block_kappas"]
            n_cal = w_cal_flat.size if size == 1 else blocks[size][0].size
            d_min = (e_detector_min_delay(n_cal, size / (2.0 * arl0), kappas,
                                          cfg["dkw_delta"] if "no DKW" not in name else None)
                     if name.startswith("e-detector") else float("nan"))
            rows.append({"target": target, "detector": name, "block_minutes": size,
                         "arl0_days": arl0_days, "alarms": alarms, "test_days": len(streams),
                         "alarms_per_day": alarms / len(streams),
                         "nominal_per_day": 1.0 / arl0_days,
                         "min_delay_hours": d_min * size / 60.0})
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
    print(table.query("arl0_days == 30").pivot_table(index="detector", columns="target",
                                                     values="min_delay_hours").round(1))

    def rate(target, detector, arl0):
        sel = table.query("target == @target and detector == @detector and arl0_days == @arl0")
        return float(sel.alarms_per_day.iloc[0])

    infos = {i["target"]: i for i in payload["targets"]}
    words = {15: "fifteen", 30: "thirty", 60: "sixty", 120: "onetwenty"}
    numbers = {}
    for target, key in (("ccv_cmd", "cooling"), ("hcv_cmd", "heating"), ("fan_cmd", "fan")):
        info = infos[target]
        numbers[f"{key} test days"] = f"{info['test_days']}"
        numbers[f"{key} kurtosis"] = f"{info['innovation_excess_kurtosis']:.0f}"
        numbers[f"{key} day offset"] = f"{info['day_offset_std']:.3f}"
        numbers[f"{key} gauss minute rate"] = f"{rate(target, 'Gaussian CUSUM, 1 min', 30):.2f}"
        numbers[f"{key} e minute rate"] = f"{rate(target, 'e-detector, 1 min', 30):.2f}"
        for size in cfg["block_minutes"]:
            w = words[size]
            numbers[f"{key} e {w} rate"] = f"{rate(target, f'e-detector, {size} min', 30):.2f}"
            numbers[f"{key} e {w} nodkw rate"] = (
                f"{rate(target, f'e-detector, {size} min, no DKW', 30):.2f}")
            numbers[f"{key} gauss {w} rate"] = (
                f"{rate(target, f'Gaussian CUSUM, {size} min', 30):.2f}")
            sel = table[(table.target == target) & (table.arl0_days == 30)
                        & (table.detector == f"e-detector, {size} min")]
            numbers[f"{key} e {w} min delay"] = f"{float(sel.min_delay_hours.iloc[0]):.1f}"
    write_numbers(NAME, numbers)
    write_latex_table(table)


ROWS = [("Gaussian CUSUM", "1 min", "Gaussian CUSUM, 1 min"),
        ("Gaussian CUSUM", "60 min", "Gaussian CUSUM, 60 min"),
        ("e-detector, no DKW", "1 min", "e-detector, 1 min, no DKW"),
        ("e-detector, no DKW", "60 min", "e-detector, 60 min, no DKW"),
        ("e-detector, DKW", "1 min", "e-detector, 1 min"),
        ("e-detector, DKW", "15 min", "e-detector, 15 min"),
        ("e-detector, DKW", "30 min", "e-detector, 30 min"),
        ("e-detector, DKW", "60 min", "e-detector, 60 min")]


def write_latex_table(table):
    """Table body for the paper: false alarms per day at nominal ARL0 = 30 days."""
    sel = table[table.arl0_days == 30]
    lines = ["% Generated by experiments/tab_far_ers.py -- do not edit.",
             r"{\setlength{\tabcolsep}{3.6pt}", r"\begin{tabular}{llcccc}", r"\hline",
             r"Detector & Scale & Cool. & Heat. & Fan & $D_{\min}$ [h] \\", r"\hline"]
    for label, block, name in ROWS:
        cells = []
        for target in ("ccv_cmd", "hcv_cmd", "fan_cmd"):
            row = sel[(sel.detector == name) & (sel.target == target)].iloc[0]
            cells.append(f"{row.alarms_per_day:.2f}")
        dmin = sel[(sel.detector == name) & (sel.target == "ccv_cmd")].min_delay_hours.iloc[0]
        dmin = "--" if np.isnan(dmin) else f"{dmin:.1f}"
        lines.append(f"{label} & {block} & {' & '.join(cells)} & {dmin} " + r"\\")
    lines += [r"\hline", r"\end{tabular}}"]
    (GENERATED / "tab_far_ers_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(tables_only="--tables-only" in sys.argv)
