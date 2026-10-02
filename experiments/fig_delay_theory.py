"""Fig. 4: the delay is set by m_eq and the long-run CV-referred noise.

For each loop (valve law, PI gain), fault (sensor bias or coil fouling) and
disturbance level, a fault-free calibration run gives an AR model of the
command residual; replicas with a step fault are prewhitened with it and fed
to a CUSUM tuned to the quasi-static command shift. The measured information
rate (threshold increase per sample of delay) and the delay at the design
ARL0 (the first entry of arl0_days) are compared with first-principles predictions that use only

    I* = m_eq^2 / (2 S),   S = sigma_n^2 + sigma_d^2 (1 + rho_d) / (1 - rho_d),

with m_eq = b for a bias and delta / (1 - delta) * dT for fouling (dT: air
temperature drop across the coil). The naive prediction uses the marginal
variance of the residual, as if it were white; the asymptotic one uses
ln(ARL0) / I*.
"""

import json
import sys

import numpy as np
from matplotlib.lines import Line2D

from _common import (BLUE, COLUMN_WIDTH, INK_MUTED, ORANGE, RESULTS, figure, load_config,
                     save_figure,
                     write_numbers, write_results)
from fmf.detectors.cusum import cusum, gaussian_llr
from fmf.detectors.prewhitening import ar_prewhiten, fit_ar
from fmf.metrics.run_length import first_crossing
from fmf.sim import LoopParams, NoiseModel, constant_context, simulate, step_profile
from fmf.theory.delay import (asymptotic_delay, brownian_cusum_delay, gaussian_cusum_arl,
                              gaussian_cusum_threshold)
from fmf.theory.masking import bias_command_shift, fouling_command_shift, plant_dc_gain
from fmf.valves import EqualPercentageValve, LinearValve

NAME = "fig_delay_theory"
VALVES = {"linear": LinearValve(), "equal_percentage": EqualPercentageValve(50.0)}
MDOT, SETPOINT = 5.0, 13.0


def run_condition(cfg, loop, fault, sd, rng):
    spd = 86400.0 / cfg["dt"]
    rho = cfg["disturbance_rho"]
    params = LoopParams(kp=LoopParams().kp * loop["kp_scale"], valve=VALVES[loop["valve"]])
    noise = NoiseModel(sensor_std=cfg["sensor_std"], disturbance_std=sd, disturbance_rho=rho)

    s_cv = cfg["sensor_std"] ** 2 + sd**2 * (1.0 + rho) / (1.0 - rho)
    arl0s = [d * spd for d in cfg["arl0_days"]]
    onset = cfg["burn_in_samples"]
    delays, censored, slopes, lrv_cv, naive = [], [], [], [], []
    for _ in range(cfg["calibrations"]):
        cal = simulate(params, constant_context(int(cfg["calibration_days"] * spd),
                                                t_in=cfg["t_in"]), noise=noise, rng=rng)
        e_cal = cal.residual[onset:]
        model = fit_ar(e_cal, cfg["ar_order"])

        load = float(cal.load[0])
        if fault["type"] == "bias":
            mu = float(bias_command_shift(fault["size"], load, params.theta0, MDOT,
                                          params.valve))
            m_eq = fault["size"]
            faults = {"bias": fault["size"]}
        else:
            mu = float(fouling_command_shift(fault["size"], load, params.theta0, params.valve))
            m_eq = fault["size"] / (1.0 - fault["size"]) * (cfg["t_in"] - SETPOINT)
            faults = {"fouling": fault["size"]}
        info_pred = m_eq**2 / (2.0 * s_cv)
        naive.append(mu**2 / (2.0 * e_cal.var()))
        gain = abs(float(plant_dc_gain(params.theta0, MDOT, cal.u_nominal[0], params.valve)))
        lrv_cv.append(model.long_run_variance * gain**2)

        delta_w = mu * model.gain / model.sigma_w
        hs = [gaussian_cusum_threshold(a, delta_w) for a in arl0s]
        horizon = int(min(cfg["max_horizon_days"] * spd,
                          max(5.0 * brownian_cusum_delay(info_pred, arl0s[-1]), spd)))
        n = onset + horizon
        profile = {k: step_profile(n, onset, v) for k, v in faults.items()}
        test = simulate(params, constant_context(n, t_in=cfg["t_in"]), noise=noise,
                        n_replicas=cfg["replicas"], rng=rng, **profile)
        w = ar_prewhiten(test.residual, model)[onset:] / model.sigma_w
        stat = cusum(gaussian_llr(w, delta_w))
        d_cal = []
        for h in hs:
            idx = first_crossing(stat, h)
            d_cal.append(float(np.where(idx >= 0, idx + 1, horizon).mean()))
            censored.append(float(np.mean(idx < 0)))
        delays.append(d_cal)
        slopes.append((hs[-1] - hs[0]) / (d_cal[-1] - d_cal[0]))
    delays = np.mean(delays, axis=0)
    info_naive = float(np.mean(naive))
    delta_pred = np.sqrt(2.0 * info_pred)

    h_pred = gaussian_cusum_threshold(arl0s[0], delta_pred)
    return {
        **loop, **fault, "disturbance_std": sd,
        "m_eq": m_eq, "s_cv": s_cv, "mu": mu,
        "lrv_cv_estimated": float(np.mean(lrv_cv)),
        "info_pred": info_pred, "info_naive": info_naive,
        "info_meas": float(np.mean(slopes)),
        "delay_meas": float(delays[0]), "delay_meas_all": delays.tolist(),
        "censored": censored,
        "delay_pred": gaussian_cusum_arl(h_pred, delta_pred, actual_shift=delta_pred),
        "delay_brownian": brownian_cusum_delay(info_pred, arl0s[0]),
        "delay_asymptotic": float(asymptotic_delay(arl0s[0], info_pred)),
        "delay_naive": gaussian_cusum_arl(
            gaussian_cusum_threshold(arl0s[0], np.sqrt(2 * info_naive)),
            np.sqrt(2 * info_naive), actual_shift=np.sqrt(2 * info_naive)),
    }


def main(plot_only=False):
    cfg = load_config(NAME)
    if plot_only:
        rows = json.loads((RESULTS / f"{NAME}.json").read_text())["rows"]
    else:
        rng = np.random.default_rng(cfg["seed"])
        rows = [run_condition(cfg, loop, fault, sd, rng) for sd in cfg["disturbance_std"]
                for loop in cfg["loops"] for fault in cfg["faults"]]
        write_results(NAME, {"config": cfg, "rows": rows})

    per_hour = 3600.0 / cfg["dt"]
    fig, (ax1, ax2) = figure(width=COLUMN_WIDTH, height=1.72, ncols=2)
    for r in rows:
        color = BLUE if r["type"] == "bias" else ORANGE
        marker = "o" if r["valve"] == "linear" else "s"
        ax1.plot(r["info_pred"] * per_hour, r["info_meas"] * per_hour, marker, color=color,
                 mew=0.7, ms=3.2, alpha=0.85)
        ax1.plot(r["info_naive"] * per_hour, r["info_meas"] * per_hour, "x", color=INK_MUTED,
                 ms=2.6, mew=0.6)
        ax2.plot(r["delay_pred"] / per_hour, r["delay_meas"] / per_hour, marker, color=color,
                 mew=0.7, ms=3.2, alpha=0.85)
        ax2.plot(r["delay_asymptotic"] / per_hour, r["delay_meas"] / per_hour, "+",
                 color=INK_MUTED, ms=3.0, mew=0.6)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo, hi = ax.get_xlim()[0], ax.get_xlim()[1]
        lo, hi = min(lo, ax.get_ylim()[0]), max(hi, ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=0.6, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    ax1.set_xticks([0.1, 1, 10, 100])
    ax1.set_yticks([0.1, 1, 10, 100])
    ax1.set_xlabel("Predicted information [nat/h]")
    ax1.set_ylabel("Measured [nat/h]")
    ax2.set_xlabel("Predicted delay [h]")
    ax2.set_ylabel("Measured delay [h]")
    handles = [Line2D([], [], ls="", marker="o", color=BLUE, ms=3.2, label="bias"),
               Line2D([], [], ls="", marker="o", color=ORANGE, ms=3.2, label="fouling"),
               Line2D([], [], ls="", marker="s", color=INK_MUTED, ms=3.2, label="eq.-% valve"),
               Line2D([], [], ls="", marker="x", color=INK_MUTED, ms=2.6, label="white-noise"),
               Line2D([], [], ls="", marker="+", color=INK_MUTED, ms=3.0, label=r"$\ln$ ARL$_0/I$")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.09),
               handletextpad=0.1, columnspacing=0.6, fontsize=6.5)
    for ax, tag in ((ax1, "(a)"), (ax2, "(b)")):
        ax.text(0.04, 0.96, tag, transform=ax.transAxes, ha="left", va="top")
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, NAME)

    info_ratio = np.array([r["info_meas"] / r["info_pred"] for r in rows])
    naive_ratio = np.array([r["info_meas"] / r["info_naive"] for r in rows])
    delay_ratio = np.array([r["delay_meas"] / r["delay_pred"] for r in rows])
    asym_ratio = np.array([r["delay_asymptotic"] / r["delay_meas"] for r in rows])
    lrv_ratio = np.array([r["lrv_cv_estimated"] / r["s_cv"] for r in rows])
    write_numbers(NAME, {
        "conditions": f"{len(rows)}",
        "replicas": f"{cfg['replicas']}",
        "info ratio median": f"{np.median(info_ratio):.2f}",
        "info ratio min": f"{info_ratio.min():.2f}",
        "info ratio max": f"{info_ratio.max():.2f}",
        "naive ratio min": f"{naive_ratio.min():.2f}",
        "naive ratio max": f"{naive_ratio.max():.2f}",
        "delay ratio median": f"{np.median(delay_ratio):.2f}",
        "delay ratio min": f"{delay_ratio.min():.2f}",
        "delay ratio max": f"{delay_ratio.max():.2f}",
        "asymptotic ratio min": f"{asym_ratio.min():.1f}",
        "asymptotic ratio max": f"{asym_ratio.max():.1f}",
        "lrv ratio min": f"{lrv_ratio.min():.2f}",
        "lrv ratio max": f"{lrv_ratio.max():.2f}",
        "max censored": f"{max(max(r['censored']) for r in rows):.2f}",
    })


if __name__ == "__main__":
    main(plot_only="--plot-only" in sys.argv)
