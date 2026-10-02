"""Fig. 1: integral action hides a CV sensor bias and coil fouling from the measured CV.

Three runs of the synthetic cooling-coil loop share context and noise: fault
free, a supply-air sensor bias, and a capacity loss. The measured CV stays at
the setpoint in all three; the fault appears only in the command residual
u - u_hat(z), where it matches the quasi-static prediction of
fmf.theory.masking.
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from _common import AQUA, BLUE, INK_MUTED, ORANGE, figure, load_config, save_figure, write_numbers
from fmf.sim import LoopParams, NoiseModel, seasonal_context, simulate, step_profile
from fmf.theory.masking import bias_command_shift, fouling_command_shift

NAME = "fig_masking"


def smooth(x, k):
    """Centered moving average; NaN where the window is incomplete."""
    return pd.Series(x).rolling(k, center=True, min_periods=k).mean().to_numpy()


def main():
    cfg = load_config(NAME)
    params = LoopParams()
    ctx = seasonal_context(cfg["days"], dt=cfg["dt"], start_day=cfg["start_day"])
    onset = int(cfg["onset_day"] * 86400 / cfg["dt"])
    noise = NoiseModel(**cfg["noise"])
    runs = {
        "fault-free": simulate(params, ctx, noise=noise, rng=cfg["seed"]),
        "bias": simulate(params, ctx, bias=step_profile(ctx.n, onset, cfg["bias"]), noise=noise,
                         rng=cfg["seed"]),
        "fouling": simulate(params, ctx, fouling=step_profile(ctx.n, onset, cfg["fouling"]),
                            noise=noise, rng=cfg["seed"]),
    }
    load = ctx.load()
    pred_bias = np.where(np.arange(ctx.n) >= onset,
                         bias_command_shift(cfg["bias"], load, params.theta0, ctx.mdot,
                                            params.valve), 0.0)
    pred_foul = np.where(np.arange(ctx.n) >= onset,
                         fouling_command_shift(cfg["fouling"], load, params.theta0,
                                               params.valve), 0.0)

    hours = ctx.t / 3600.0
    k = int(cfg["smoothing_min"] * 60 / cfg["dt"])
    fig, (ax1, ax2) = figure(height=2.45, nrows=2, sharex=True)
    for name, color, style in (("fault-free", BLUE, "-"), ("bias", ORANGE, "--"),
                               ("fouling", AQUA, ":")):
        ax1.plot(hours, smooth(runs[name].y_meas, k), color=color, ls=style, lw=0.9,
                 label=f"measured, {name}")
    ax1.plot(hours, smooth(runs["bias"].y, k), color=ORANGE, lw=0.9, ls="-.",
             label="true, bias")
    ax1.axhline(13.0, color=INK_MUTED, lw=0.6, zorder=0)
    ax1.set_ylabel("Supply air [°C]")
    ax1.set_ylim(10.55, 13.45)
    ax1.legend(ncol=2, loc="lower center", columnspacing=1.0)

    for name, color, style, pred in (("fault-free", BLUE, "-", None),
                                     ("bias", ORANGE, "--", pred_bias),
                                     ("fouling", AQUA, ":", pred_foul)):
        ax2.plot(hours, smooth(runs[name].residual, k), color=color, ls=style, lw=0.9,
                 label=name)
        if pred is not None:
            ax2.plot(hours, pred, color=color, lw=1.6, alpha=0.35)
    ax2.set_ylabel(r"$u-\hat u(z)$")
    ax2.set_xlabel("Time [h]")
    ax2.set_xlim(0, hours[-1])
    ax2.set_xticks(np.arange(0, hours[-1] + 1, 12))
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK_MUTED, lw=1.6, alpha=0.5))
    labels.append("quasi-static")
    ax2.set_ylim(-0.04, 0.3)
    ax2.legend(handles, labels, ncol=2, loc="upper left", columnspacing=1.0)
    for ax, tag in ((ax1, "(a)"), (ax2, "(b)")):
        ax.text(0.995, 0.96, tag, transform=ax.transAxes, ha="right", va="top")
    save_figure(fig, NAME)

    post = slice(onset + 120, None)
    b, f = runs["bias"], runs["fouling"]
    write_numbers(NAME, {
        "bias": f"{cfg['bias']:.1f}",
        "fouling": f"{cfg['fouling']:.2f}",
        "bias shift": f"{np.mean(pred_bias[post]):.3f}",
        "bias shift measured": f"{np.mean(b.residual[post] - runs['fault-free'].residual[post]):.3f}",
        "fouling shift min": f"{pred_foul[post].min():.3f}",
        "fouling shift max": f"{pred_foul[post].max():.3f}",
        "cv offset bias": f"{abs(np.mean(b.y_meas[post] - runs['fault-free'].y_meas[post])):.3f}",
        "cv offset fouling": f"{abs(np.mean(f.y_meas[post] - runs['fault-free'].y_meas[post])):.3f}",
    })


if __name__ == "__main__":
    main()
