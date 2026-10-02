"""Fig. 2: where and when a masked fault carries information over a year (G36-Degrad).

Signatures: hourly means of u_fault - u_baseline during occupied hours, for each
actuator of the supply-air sequence (heating valve, outdoor-air damper,
cooling valve) and the supply fan, replaced by the median of the same hour of
day within each week. Both simulations share weather and schedules, so the
difference is the fault's effect on the command, apart from isolated hours in
which a tiny perturbation shifts a discrete mode switch; the weekly median
removes those.
Noise: hourly residual spread of the same actuator on the real ERS fault-free
days (out-of-fold baseline model), treated as independent across hours.
Information per hour I_k = mu_k^2 / (2 sigma^2); a CUSUM matched to the
signature alarms to first order when sum I_k reaches ln(ARL0). Measured alarm
times come from matched CUSUMs on signature + noise replicas, one per actuator
and one combining all actuators of the scenario.
"""

import json
import sys

import numpy as np
import pandas as pd

from _common import (AQUA, BLUE, COLUMN_WIDTH, INK_MUTED, ORANGE, RESULTS, figure, load_config,
                     save_figure, write_numbers, write_results)
from fmf.baselines.context_model import cross_fit_predict
from fmf.datasets.ghalamsiah2026 import ers_days, load_ers, load_g36_degrad
from fmf.detectors.cusum import cusum, gaussian_llr
from fmf.metrics.run_length import first_crossing

NAME = "fig_seasonal"
LABELS = {"hcv_cmd": "heating valve", "oad_cmd": "outdoor-air damper",
          "ccv_cmd": "cooling valve", "fan_cmd": "supply fan"}
COLORS = {"hcv_cmd": ORANGE, "oad_cmd": AQUA, "ccv_cmd": BLUE, "fan_cmd": INK_MUTED}
STYLES = {"hcv_cmd": "--", "oad_cmd": ":", "ccv_cmd": "-", "fan_cmd": "-."}


def ers_hourly_noise(cfg):
    """Hourly spread of the out-of-fold command residual on the ERS fault-free days."""
    noise = {}
    days = ers_days()
    for target, spec in cfg["monitors"].items():
        frames = []
        for _, r in days[days.season.isin(spec["seasons"])].iterrows():
            d = load_ers(r.folder)
            d = d[(d.occupied > 0.5) & (d.fan_on > 0.5)].copy()
            d["day"], d["hour"] = r.date, d.index.hour + d.index.minute / 60.0
            frames.append(d)
        df = pd.concat(frames)
        res = df[target].to_numpy() - cross_fit_predict(df, spec["features"], target, df.day,
                                                        cfg["lightgbm"])
        hourly = (pd.Series(res, index=df.index).groupby([df.day, df.index.hour]).mean())
        noise[target] = float(hourly.std())
    return noise


def hourly_signatures(scenario, actuators, block):
    base = load_g36_degrad("baseline", actuators + ["occupied", "fan_on"])
    fault = load_g36_degrad(scenario, actuators)
    on = (base.occupied > 0.5) & (base.fan_on > 0.5)
    diff = (fault[actuators] - base[actuators])[on]
    day = np.floor(diff.index.to_numpy() / 86400.0).astype(int)
    hour = np.floor((diff.index.to_numpy() % 86400.0) / 3600.0).astype(int)
    grouped = diff.groupby([day, hour])
    means = grouped.mean()[grouped.size() >= block // 2]
    d = means.index.get_level_values(0).to_numpy()
    h = means.index.get_level_values(1).to_numpy()
    # Robust signature: median of the same hour of day within each week. A
    # systematic fault effect recurs daily; isolated hours where the two runs
    # switch modes at slightly different times do not.
    robust = means.groupby([d // 7, h]).transform("median")
    return d + h / 24.0, robust


def main(plot_only=False):
    cfg = load_config(NAME)
    if plot_only:
        payload = json.loads((RESULTS / f"{NAME}.json").read_text())
    else:
        rng = np.random.default_rng(cfg["seed"])
        noise = ers_hourly_noise(cfg["ers_noise"])
        threshold = np.log(cfg["arl0_days"] * cfg["hours_per_day"])
        payload = {"config": cfg, "noise": noise, "threshold": threshold, "scenarios": {}}
        for scenario, spec in cfg["scenarios"].items():
            acts = spec["actuators"]
            days, mu = hourly_signatures(scenario, acts, cfg["block_samples"])
            out = {"days": days.tolist(), "mu": {a: mu[a].tolist() for a in acts}, "alarms": {},
                   "predicted": {}}
            delta = {a: mu[a].to_numpy() / noise[a] for a in acts}
            info = {a: delta[a] ** 2 / 2.0 for a in acts}
            info["combined"] = sum(info[a] for a in acts)
            for key, inf in info.items():
                hit = np.nonzero(np.cumsum(inf) >= threshold)[0]
                out["predicted"][key] = float(days[hit[0]]) if hit.size else None
            n, r = days.size, cfg["replicas"]
            z = {a: delta[a][:, None] + rng.standard_normal((n, r)) for a in acts}
            llr = {a: gaussian_llr(z[a], delta[a][:, None]) for a in acts}
            llr["combined"] = sum(llr[a] for a in acts)
            for key, inc in llr.items():
                idx = first_crossing(cusum(inc), threshold)
                alarm = np.where(idx >= 0, days[np.maximum(idx, 0)], np.nan)
                out["alarms"][key] = alarm.tolist()
            payload["scenarios"][scenario] = out
        write_results(NAME, payload)

    fig, axes = figure(height=3.3, nrows=3, sharex=True)
    for ax, (scenario, spec) in zip(axes[:2], cfg["scenarios"].items()):
        out = payload["scenarios"][scenario]
        days = np.asarray(out["days"])
        for a in spec["actuators"]:
            weekly = pd.Series(out["mu"][a]).groupby(np.floor(days / 7.0)).mean()
            ax.plot(weekly.index * 7.0 + 3.5, weekly.to_numpy(), color=COLORS[a], ls=STYLES[a],
                    lw=1.0, label=LABELS[a])
        ax.axhline(0.0, color=INK_MUTED, lw=0.5, zorder=0)
        ax.set_ylabel(r"$\Delta u$")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.55 * (hi - lo))
        ax.legend(loc="upper left", ncol=3 if len(spec["actuators"]) > 2 else 2,
                  columnspacing=0.8, handlelength=1.6, title=spec["label"],
                  title_fontsize=7, alignment="left")
    ax = axes[2]
    out = payload["scenarios"]["sat_bias_plus"]
    days = np.asarray(out["days"])
    noise = payload["noise"]
    for a in cfg["scenarios"]["sat_bias_plus"]["actuators"]:
        inf = (np.asarray(out["mu"][a]) / noise[a]) ** 2 / 2.0
        ax.plot(days, np.cumsum(inf), color=COLORS[a], ls=STYLES[a], lw=1.0, label=LABELS[a])
        med = np.nanmedian(out["alarms"][a])
        if np.isfinite(med):
            ax.plot(med, payload["threshold"], "v", color=COLORS[a], ms=4)
    total = sum((np.asarray(out["mu"][a]) / noise[a]) ** 2 / 2.0
                for a in cfg["scenarios"]["sat_bias_plus"]["actuators"])
    ax.plot(days, np.cumsum(total), color="#0b0b0b", lw=1.0, label="all three")
    ax.text(230, np.cumsum(total)[np.searchsorted(days, 230)] * 2.2, "all three", ha="center",
            va="bottom", fontsize=7)
    ax.text(362, payload["threshold"] * 0.55, r"$\ln\,\mathrm{ARL}_0$", ha="right", va="top",
            fontsize=7, color=INK_MUTED)
    ax.plot(np.nanmedian(out["alarms"]["combined"]), payload["threshold"], "v", color="#0b0b0b",
            ms=4)
    ax.axhline(payload["threshold"], color=INK_MUTED, lw=0.6, ls="--", zorder=0)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 4e4)
    ax.set_ylabel(r"$\sum I_k$ [nat]")
    ax.set_xlabel("Day of year")
    ax.set_xlim(0, 365)
    for a, tag in zip(axes, ("(a)", "(b)", "(c)")):
        a.text(0.995, 0.96, tag, transform=a.transAxes, ha="right", va="top")
    fig.align_ylabels(axes)
    save_figure(fig, NAME)

    numbers = {}
    for scenario, key in (("sat_bias_plus", "bias"), ("coil_fouling_14", "fouling")):
        out = payload["scenarios"][scenario]
        for actuator, word in (("hcv_cmd", "heating"), ("oad_cmd", "damper"), ("ccv_cmd", "cooling"),
                               ("fan_cmd", "fan"), ("combined", "combined")):
            if actuator in out["alarms"]:
                alarms = np.asarray(out["alarms"][actuator], dtype=float)
                pred = out["predicted"][actuator]
                numbers[f"{key} {word} alarm day"] = (
                    f"{np.nanmedian(alarms):.0f}" if np.isfinite(alarms).any() else "--")
                numbers[f"{key} {word} predicted day"] = "--" if pred is None else f"{pred:.0f}"
                numbers[f"{key} {word} missed"] = f"{np.mean(~np.isfinite(alarms)):.2f}"
    for actuator, word in (("hcv_cmd", "heating"), ("oad_cmd", "damper"), ("ccv_cmd", "cooling"),
                           ("fan_cmd", "fan")):
        numbers[f"noise {word}"] = f"{payload['noise'][actuator]:.3f}"
    write_numbers(NAME, numbers)
    print(json.dumps({k: v for k, v in numbers.items()}, indent=1))


if __name__ == "__main__":
    main(plot_only="--plot-only" in sys.argv)
