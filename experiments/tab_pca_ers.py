"""Baseline: PCA T^2/SPE monitoring on real ERS fault-free and faulted days."""

import json
import sys

import numpy as np
import pandas as pd

from _common import RESULTS, TABLES, load_config, write_numbers, write_results
from fmf.baselines.pca_monitor import PCAMonitor
from fmf.datasets.ghalamsiah2026 import ers_days, load_ers

NAME = "tab_pca_ers"


def occupied(d):
    return d[(d.occupied > 0.5) & (d.fan_on > 0.5)]


def events(flags):
    """Number of runs of consecutive True values."""
    f = np.asarray(flags, dtype=bool)
    return int(np.sum(f[1:] & ~f[:-1]) + (1 if f.size and f[0] else 0))


def main(tables_only=False):
    cfg = load_config(NAME)
    if tables_only:
        payload = json.loads((RESULTS / f"{NAME}.json").read_text())
    else:
        days = ers_days()
        rows, groups = [], {}
        for group, seasons in cfg["groups"].items():
            sel = days[days.season.isin(seasons)]
            cal_days, test_days = [], []
            for _, g in sel.groupby("season"):
                g = g.sort_values("date")
                k = (len(g) + 1) // 2
                cal_days += list(g.folder[:k])
                test_days += list(g.folder[k:])
            feats = cfg["features"]
            cal = pd.concat([occupied(load_ers(f))[feats] for f in cal_days])
            model = PCAMonitor.fit(cal.to_numpy(), cfg["variance"])
            samples_per_day = len(cal) / len(cal_days)
            alpha = 1.0 / (cfg["arl0_days"] * samples_per_day)
            t2_lim, spe_lim = model.limits(alpha / 2.0)
            groups[group] = {"components": model.k, "calibration_days": len(cal_days),
                             "test_days": len(test_days), "t2_limit": t2_lim, "spe_limit": spe_lim}
            for folder in test_days:
                d = occupied(load_ers(folder))[feats].to_numpy()
                t2, spe = model.statistics(d)
                rows.append({"group": group, "unit": "fault-free test", "folder": str(folder),
                             "fault": None, "events": events((t2 > t2_lim) | (spe > spe_lim))})
            for _, r in sel.iterrows():
                d = occupied(load_ers(r.folder, "fault"))
                t2, spe = model.statistics(d[feats].to_numpy())
                flags = (t2 > t2_lim) | (spe > spe_lim)
                first = (d.index[int(np.argmax(flags))] - d.index[0]).total_seconds() / 3600.0 \
                    if flags.any() else None
                rows.append({"group": group, "unit": "faulted", "folder": str(r.folder),
                             "fault": r.fault, "events": events(flags),
                             "first_alarm_hours_after_start": first})
        payload = {"config": cfg, "groups": groups, "rows": rows}
        write_results(NAME, payload)
    table = pd.DataFrame(payload["rows"])
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / f"{NAME}.csv", index=False)
    print(pd.DataFrame(payload["groups"]).T)
    free = table[table.unit == "fault-free test"]
    print(free.groupby("group").events.agg(["sum", "count"]))
    numbers = {}
    for group in ("cooling", "heating"):
        f = free[free.group == group]
        numbers[f"{group} events per day"] = f"{f.events.sum() / len(f):.1f}"
        numbers[f"{group} test days"] = f"{len(f)}"
    faulted = table[table.unit == "faulted"]
    for key, faults in cfg["masked"].items():
        sel = faulted[faulted.fault.isin(faults)]
        numbers[f"{key} days alarmed"] = f"{int((sel.events > 0).sum())}"
        numbers[f"{key} days"] = f"{len(sel)}"
    allf = faulted.drop_duplicates("folder")
    numbers["faulted days alarmed"] = f"{int((allf.events > 0).sum())}"
    numbers["faulted days"] = f"{len(allf)}"
    write_numbers(NAME, numbers)
    for k, v in numbers.items():
        print(f"{k:26s} {v}")


if __name__ == "__main__":
    main(tables_only="--tables-only" in sys.argv)
