"""Table: alarms on real faulted ERS days, effort residual vs controlled-variable residual.

Detectors are calibrated on all fault-free days of the twin AHU (unit B; out-of-
fold residuals by day) and run on the faulted unit (unit A) of each test day.
Two residuals are monitored with the hourly conformal e-detector (DKW, two-
sided, nominal ARL0 = 30 days):
  - the control-effort residual u - u_hat(z) of the actuator that compensates
    the fault (cooling valve, heating valve or supply fan);
  - the controlled-variable residual y - y_hat(z) of the supply-air
    temperature or duct static pressure, as a CV-only virtual sensor would.
Each fault lasts one day, so each faulted day is a separate run. Unit A has no
fault-free days in this dataset, so differences between the two units are not
separable from fault effects. As a control, unit B is monitored with the same
detectors calibrated on the other chronological half of its days (within
seasons, both ways); the stream runs through its days in order and restarts
only after an alarm, so that no control day is part of its own
calibration set and evidence can accumulate beyond the minimum delay.
"""

import json
import sys

import numpy as np
import pandas as pd

from _common import RESULTS, TABLES, load_config, write_numbers, write_results
from fmf.baselines.context_model import cross_fit_predict, fit_predict
from fmf.datasets.ghalamsiah2026 import ers_days, load_ers
from fmf.detectors.sequential import conformal_e_statistic, count_alarms

NAME = "tab_detection_ers"


def prepare(d, date, season):
    d = d[(d.occupied > 0.5) & (d.fan_on > 0.5)].copy()
    d["day"], d["season"] = date, season
    d["hour"] = d.index.hour + d.index.minute / 60.0
    return d


def hourly(x, size):
    k = len(x) // size
    return x[: k * size].reshape(k, size).mean(axis=1)


def alarms_by_sign(z, calibration, cfg, threshold):
    """Alarm indices of the two one-sided detectors (restart after each alarm)."""
    return {sign: count_alarms(lambda s: conformal_e_statistic(
        s, sign * calibration, cfg["block_kappas"], cfg["dkw_delta"]), sign * z, threshold)
        for sign in (1.0, -1.0)}


def first_alarm(z, calibration, cfg, threshold):
    """Hour index of the first alarm on either side, and its sign (+1, -1 or 0)."""
    best = (np.inf, 0)
    for sign, alarms in alarms_by_sign(z, calibration, cfg, threshold).items():
        if alarms and alarms[0] < best[0]:
            best = (alarms[0], int(sign))
    return best


def halves(days):
    """Chronological halves of the test days within each season (first rounded up)."""
    first, second = [], []
    for _, g in days.groupby("season"):
        dates = sorted(g.date)
        k = (len(dates) + 1) // 2
        first += dates[:k]
        second += dates[k:]
    return first, second


def control_alarms(base, days, spec, cfg, threshold):
    """Alarms on the fault-free unit, each half monitored with the other half as calibration."""
    target, feats, block = spec["target"], spec["features"], cfg["block_minutes"]
    first, second = halves(days)
    alarms = 0
    for cal_days, ctrl_days in ((first, second), (second, first)):
        cal, ctrl = base[base.day.isin(cal_days)], base[base.day.isin(ctrl_days)]
        oof = cal[target].to_numpy() - cross_fit_predict(cal, feats, target, cal.day,
                                                         cfg["lightgbm"], cfg["seed"])
        res = ctrl[target].to_numpy() - fit_predict(cal, ctrl, feats, target, cfg["lightgbm"],
                                                    cfg["seed"])
        b_cal = np.concatenate([hourly(oof[(cal.day == d).to_numpy()], block)
                                for d in sorted(cal_days)])
        scale = b_cal.std()
        z = np.concatenate([hourly(res[(ctrl.day == d).to_numpy()], block)
                            for d in sorted(ctrl_days)]) / scale
        found = alarms_by_sign(z, b_cal / scale, cfg, threshold)
        alarms += sum(len(a) for a in found.values())
    return alarms


def main(tables_only=False):
    cfg = load_config(NAME)
    if tables_only:
        payload = json.loads((RESULTS / f"{NAME}.json").read_text())
        rows, controls = payload["rows"], payload["controls"]
    else:
        rows, controls = [], []
        days = ers_days()
        for spec in cfg["monitors"]:
            seasons, target, feats = spec["seasons"], spec["target"], spec["features"]
            sel = days[days.season.isin(seasons)]
            base = pd.concat([prepare(load_ers(r.folder), r.date, r.season)
                              for _, r in sel.iterrows()])
            oof = base[target].to_numpy() - cross_fit_predict(base, feats, target, base.day,
                                                              cfg["lightgbm"], cfg["seed"])
            block = cfg["block_minutes"]
            per_day = {d: oof[(base.day == d).to_numpy()] for d in base.day.unique()}
            day_sd = float(np.std([x.mean() for x in per_day.values()]))
            b_cal = np.concatenate([hourly(x, block) for x in per_day.values()])
            scale = b_cal.std()
            b_cal = b_cal / scale
            samples_per_day = len(base) / base.day.nunique()
            threshold = np.log(2.0 * cfg["arl0_days"] * samples_per_day / block)
            for _, r in sel.iterrows():
                fault = prepare(load_ers(r.folder, "fault"), r.date, r.season)
                res = fault[target].to_numpy() - fit_predict(base, fault, feats, target,
                                                             cfg["lightgbm"], cfg["seed"])
                z = hourly(res, block) / scale
                hour, sign = first_alarm(z, b_cal, cfg, threshold)
                rows.append({
                    "monitor": spec["name"], "kind": spec["kind"], "season": r.season,
                    "date": str(r.date.date()), "fault": r.fault,
                    "mean_residual": float(res.mean()), "day_sd": day_sd,
                    "z_day": float(res.mean() / day_sd), "hours": int(len(z)),
                    "alarm_hour": None if np.isinf(hour) else int(hour) + 1,
                    "alarm_sign": sign,
                })
            controls.append({"monitor": spec["name"], "kind": spec["kind"],
                             "days": int(len(sel)),
                             "hours": int(sum(len(x) // block for x in per_day.values())),
                             "alarms": control_alarms(base, sel, spec, cfg, threshold)})
        write_results(NAME, {"config": cfg, "rows": rows, "controls": controls})
    table = pd.DataFrame(rows)
    control = pd.DataFrame(controls)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / f"{NAME}.csv", index=False)
    control.to_csv(TABLES / f"{NAME}_control.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.to_string())
    print(control.to_string())

    def summary(monitor, faults):
        sel = table[(table.monitor == monitor) & table.fault.isin(faults)]
        hit = sel.alarm_hour.notna()
        return int(hit.sum()), int(len(sel)), (float(sel.alarm_hour[hit].median()) if hit.any()
                                               else float("nan"))

    numbers = {}
    for key, faults, effort, cv in (
            ("leak", cfg["masked"]["leak"], "cooling valve", "supply air temperature cooling"),
            ("capacity", cfg["masked"]["capacity"], "heating valve",
             "supply air temperature heating"),
            ("airside", cfg["masked"]["airside"], "supply fan", "duct static pressure")):
        hits, n, med = summary(effort, faults)
        cv_hits, _, _ = summary(cv, faults)
        numbers[f"{key} days"] = f"{n}"
        numbers[f"{key} effort hits"] = f"{hits}"
        numbers[f"{key} cv hits"] = f"{cv_hits}"
        numbers[f"{key} median hour"] = "--" if np.isnan(med) else f"{med:.0f}"
    leak = table[table.fault.isin(cfg["masked"]["leak"])]
    effort_z = leak[leak.monitor == "cooling valve"].z_day
    cv_z = leak[leak.monitor == "supply air temperature cooling"].z_day.abs()
    numbers["leak effort z min"] = f"{effort_z.min():.1f}"
    numbers["leak effort z max"] = f"{effort_z.max():.1f}"
    numbers["leak cv z max"] = f"{cv_z.max():.1f}"
    numbers["control alarms"] = f"{int(control.alarms.sum())}"
    numbers["control days"] = f"{int(control.days.sum())}"
    write_numbers(NAME, numbers)


if __name__ == "__main__":
    main(tables_only="--tables-only" in sys.argv)
