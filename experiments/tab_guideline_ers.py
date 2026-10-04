"""Baseline: Guideline 36 AFDD alarms on real ERS days (fault-free twin and faulted unit).

For every test day, the G36 fault conditions run on the fault-free unit and on
the faulted unit with the guideline's default thresholds, AlarmDelay and
ModeDelay. The table lists, per day, which fault conditions raised an alarm.
"""

import json
import sys

import numpy as np
import pandas as pd

from _common import RESULTS, TABLES, load_config, write_numbers, write_results
from fmf.datasets.ghalamsiah2026 import ers_days, load_ers
from fmf.rules.g36_afdd import G36Params, alarms, fault_conditions

NAME = "tab_guideline_ers"


SENSOR_SETS = {
    # ERS has dedicated sensors between the coils (heating- and cooling-coil discharge).
    "dedicated coil sensors": ({"ccet": "hcdat", "cclt": "ccdat", "hcet": "mat", "hclt": "hcdat"},
                               False),
    # Typical AHU: only MAT and SAT, with the supply fan between them.
    "MAT and SAT only": ({"ccet": "mat", "cclt": "sat", "hcet": "mat", "hclt": "sat"}, True),
}


def prepare(d, mapping):
    d = d.copy()
    for target, source in mapping.items():
        d[target] = d[source]
    return d


def run_day(d, min_oa, params):
    fc = fault_conditions(d, min_oa=min_oa, p=params)
    operating = (d.occupied > 0.5) & (d.fan_on > 0.5)
    mode_start = operating & ~operating.shift(fill_value=False)
    out = alarms(fc, operating, mode_start, params)
    fired = {c: bool(out[c].any()) for c in out.columns}
    first = {c: (out[c].idxmax() - d.index[0]).total_seconds() / 3600.0
             for c in out.columns if fired[c]}
    return fired, first


def main(tables_only=False):
    cfg = load_config(NAME)
    if tables_only:
        rows = json.loads((RESULTS / f"{NAME}.json").read_text())["rows"]
    else:
        days = ers_days()
        base = {r.folder: load_ers(r.folder) for _, r in days.iterrows()}
        min_oa = {}
        for season, g in days.groupby("season"):
            pos = pd.concat([base[f].oad_cmd[(base[f].occupied > 0.5) & (base[f].oad_cmd < 0.99)]
                             for f in g.folder])
            min_oa[season] = float(pos.round(2).mode().iloc[0])
        rows = []
        faulted = {r.folder: load_ers(r.folder, "fault") for _, r in days.iterrows()}
        for sensors, (mapping, fan_between) in SENSOR_SETS.items():
            params = G36Params(eps_dsp=cfg["eps_dsp"], fan_heat_between_cc_sensors=fan_between,
                               fan_heat_between_hc_sensors=fan_between)
            for _, r in days.iterrows():
                for unit, d in (("fault-free", base[r.folder]), ("faulted", faulted[r.folder])):
                    fired, first = run_day(prepare(d, mapping), min_oa[r.season], params)
                    alarmed = sorted(c for c, v in fired.items() if v)
                    rows.append({"sensors": sensors, "season": r.season,
                                 "date": str(r.date.date()), "fault": r.fault, "unit": unit,
                                 "alarms": alarmed,
                                 "first_alarm_hour": min(first.values()) if first else None})
        write_results(NAME, {"config": cfg, "min_oa": min_oa, "rows": rows})
    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.assign(alarms=table.alarms.apply(" ".join)).to_csv(TABLES / f"{NAME}.csv", index=False)
    pd.set_option("display.width", 200)
    masked = sum(cfg["masked"].values(), [])
    print(table[table.fault.isin(masked) | (table.unit == "fault-free")]
          .assign(alarms=table.alarms.apply(" ".join)).to_string())

    numbers = {}
    for sensors, word in (("dedicated coil sensors", "dedicated"), ("MAT and SAT only", "plain")):
        sub = table[table.sensors == sensors]
        free = sub[sub.unit == "fault-free"]
        numbers[f"{word} fault free days"] = f"{len(free)}"
        numbers[f"{word} fault free days alarmed"] = f"{int(free.alarms.apply(bool).sum())}"
        faulted = sub[sub.unit == "faulted"]
        for key, faults in cfg["masked"].items():
            sel = faulted[faulted.fault.isin(faults)]
            numbers[f"{word} {key} days"] = f"{len(sel)}"
            numbers[f"{word} {key} days alarmed"] = f"{int(sel.alarms.apply(bool).sum())}"
            codes = sorted({c for a in sel.alarms for c in a}, key=lambda c: int(c[2:]))
            numbers[f"{word} {key} conditions"] = (
                ", ".join("FC" + chr(92) + "#" + c[2:] for c in codes) or "none")
    write_numbers(NAME, numbers)
    for k, v in numbers.items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main(tables_only="--tables-only" in sys.argv)
