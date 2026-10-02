"""Design numbers: fleet alarm budget and minimum alarm delay vs calibration length.

By linearity of expectation (no independence between loops needed), N loops
monitored with statistics every dt and a budget of B false alarms over T need
ARL0 >= N T / (dt B) per loop. The minimum delay of the two-sided hourly
e-detector follows from fmf.theory.delay.e_detector_min_delay with
n = calibration days x operating hours.
"""

from _common import load_config, write_numbers
from fmf.theory.delay import e_detector_min_delay

NAME = "tab_design"
WORDS = {30: "thirty", 90: "ninety", 365: "year", 3000: "fleet"}


def main():
    cfg = load_config(NAME)
    arl0_fleet = (cfg["fleet_loops"] * cfg["budget_horizon_days"]
                  / (cfg["statistic_period_days"] * cfg["budget_alarms"]))
    numbers = {
        "fleet loops": f"{cfg['fleet_loops']}",
        "fleet budget": f"{cfg['budget_alarms']}",
        "fleet arl": f"{arl0_fleet:.0f}",
        "fleet arl years": f"{arl0_fleet / 365.0:.1f}",
    }
    blocks_per_day = cfg["hours_per_day"] / cfg["block_hours"]
    for days in cfg["calibration_days"]:
        n = int(days * blocks_per_day)
        for arl0 in cfg["arl0_days"]:
            alpha = 1.0 / (2.0 * arl0 * blocks_per_day)  # two one-sided streams
            d = e_detector_min_delay(n, alpha, cfg["kappas"], cfg["dkw_delta"])
            numbers[f"min delay {WORDS[days]} cal {WORDS[arl0]} arl"] = (
                f"{d * cfg['block_hours']:.0f}")
    write_numbers(NAME, numbers)
    for k, v in numbers.items():
        print(f"{k:40s} {v}")


if __name__ == "__main__":
    main()
