"""Fig. 3: price of the false-alarm guarantee on real residual noise.

Hourly means of the out-of-fold cooling-valve residual on real fault-free ERS
days form the noise; a step of known size is added from the onset. Test
streams resample whole test days (hourly blocks) with replacement. We compare
the delay of the hourly e-detector with DKW correction, whose false-alarm rate
held on these days, with a Gaussian CUSUM tuned to the shift, whose rate did
not (Table 1), over a range of nominal ARL0, together with the minimum delay
(3) of the e-detector.
"""

import json
import sys

import numpy as np

from _common import (BLUE, COLUMN_WIDTH, INK_MUTED, ORANGE, AQUA, RESULTS, figure, load_config,
                     save_figure, write_numbers, write_results)
from fmf.baselines.context_model import cross_fit_predict
from fmf.detectors.sequential import conformal_e_statistic, gaussian_cusum_statistic
from fmf.metrics.run_length import first_crossing
from fmf.theory.delay import e_detector_min_delay, gaussian_cusum_threshold
from tab_far_ers import block_means, chronological_split, load_days, per_day

NAME = "fig_delay_arl"
SHIFT_COLORS = (AQUA, BLUE, ORANGE)


def main(plot_only=False):
    cfg = load_config(NAME)
    if plot_only:
        payload = json.loads((RESULTS / f"{NAME}.json").read_text())
    else:
        rng = np.random.default_rng(cfg["seed"])
        df = load_days(cfg["seasons"])
        cal, test = chronological_split(df)
        res_cal = cal[cfg["target"]].to_numpy() - cross_fit_predict(
            cal, cfg["features"], cfg["target"], cal.day, cfg["lightgbm"])
        res_test = test[cfg["target"]].to_numpy() - cross_fit_predict(
            test, cfg["features"], cfg["target"], test.day, cfg["lightgbm"])
        size = cfg["block_minutes"]
        b_cal = np.concatenate([block_means(x, size) for x in per_day(cal, res_cal)])
        scale = b_cal.std()
        b_cal = b_cal / scale
        test_days = [block_means(x, size) / scale for x in per_day(test, res_test)]
        blocks_per_day = float(np.mean([len(d) for d in test_days]))

        h, r = cfg["horizon_blocks"], cfg["replicas"]
        streams = np.empty((h, r))
        for j in range(r):
            chunks, n = [], 0
            while n < h:
                day = test_days[rng.integers(len(test_days))]
                chunks.append(day)
                n += len(day)
            streams[:, j] = np.concatenate(chunks)[:h]

        payload = {"config": cfg, "scale": scale, "blocks_per_day": blocks_per_day,
                   "calibration_blocks": int(b_cal.size), "curves": []}
        for shift in cfg["shifts"]:
            z = streams + shift / scale
            delta = shift / scale
            e_stat = conformal_e_statistic(z, b_cal, cfg["block_kappas"], cfg["dkw_delta"])
            g_stat = gaussian_cusum_statistic(z, delta)
            for arl0_days in cfg["arl0_days"]:
                arl0 = arl0_days * blocks_per_day
                # one-sided streams of a two-sided detector run at alpha / 2
                for name, stat, thr in (
                        ("e-detector", e_stat, np.log(2.0 * arl0)),
                        ("Gaussian CUSUM", g_stat, gaussian_cusum_threshold(2.0 * arl0, delta))):
                    idx = first_crossing(stat, thr)
                    rl = np.where(idx >= 0, idx + 1, h).astype(float)
                    payload["curves"].append({
                        "detector": name, "shift": shift, "arl0_days": arl0_days,
                        "mean_delay_hours": float(rl.mean() * size / 60.0),
                        "censored": float(np.mean(idx < 0))})
        payload["min_delay_hours"] = {
            str(a): e_detector_min_delay(int(b_cal.size), 1.0 / (2.0 * a * blocks_per_day),
                                         cfg["block_kappas"], cfg["dkw_delta"]) * size / 60.0
            for a in cfg["arl0_days"]}
        write_results(NAME, payload)

    curves = payload["curves"]
    fig, ax = figure(height=1.95)
    arls = np.asarray(cfg["arl0_days"], dtype=float)
    for shift, color in zip(cfg["shifts"], SHIFT_COLORS):
        for name, style, marker in (("e-detector", "-", "o"), ("Gaussian CUSUM", "--", "s")):
            ys = [c["mean_delay_hours"] for c in curves
                  if c["detector"] == name and c["shift"] == shift]
            ax.plot(arls, ys, ls=style, marker=marker, color=color, ms=3.0, lw=1.0, mfc="white"
                    if name == "Gaussian CUSUM" else color)
        ys = [c["mean_delay_hours"] for c in curves
              if c["detector"] == "e-detector" and c["shift"] == shift]
        ax.text(arls[-1] * 1.25, ys[-1], rf"$\Delta u={shift:.2f}$", color=color, va="center",
                fontsize=7)
    dmin = [payload["min_delay_hours"][str(a)] for a in cfg["arl0_days"]]
    ax.plot(arls, dmin, color=INK_MUTED, lw=0.8, ls=":")
    ax.text(arls[-1] * 1.25, dmin[-1], r"$D_{\min}$", color=INK_MUTED, fontsize=7, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(arls[0] * 0.8, arls[-1] * 3.5)
    ax.set_xlabel(r"Nominal ARL$_0$ [operating days]")
    ax.set_ylabel("Mean delay [h]")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=INK_MUTED, marker="o", ms=3, label="e-detector, DKW"),
                       Line2D([], [], color=INK_MUTED, ls="--", marker="s", mfc="white", ms=3,
                              label="Gaussian CUSUM")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.0)
    save_figure(fig, NAME)

    def delay(name, shift, arl):
        return next(c["mean_delay_hours"] for c in curves if c["detector"] == name
                    and c["shift"] == shift and c["arl0_days"] == arl)
    numbers = {"scale": f"{payload['scale']:.3f}",
               "calibration blocks": f"{payload['calibration_blocks']}"}
    for shift, word in zip(cfg["shifts"], ("small", "medium", "large")):
        numbers[f"{word} shift"] = f"{shift:.2f}"
        for arl, aw in ((30, "thirty"), (3000, "fleet")):
            numbers[f"e {word} {aw}"] = f"{delay('e-detector', shift, arl):.0f}"
            numbers[f"gauss {word} {aw}"] = f"{delay('Gaussian CUSUM', shift, arl):.0f}"
    numbers["max censored"] = f"{max(c['censored'] for c in curves):.2f}"
    write_numbers(NAME, numbers)
    for k, v in numbers.items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main(plot_only="--plot-only" in sys.argv)
