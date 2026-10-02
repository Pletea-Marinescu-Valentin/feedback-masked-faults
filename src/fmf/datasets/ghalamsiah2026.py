"""Loaders for Ghalamsiah et al. (2026), doi:10.6084/m9.figshare.29297999.v3.

Column names are taken from the CSV headers of the dataset; each loader maps
them to the canonical names used across fmf (temperatures in degC, actuator
commands as fractions in [0, 1]).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = (Path(__file__).resolve().parents[3] / "data" / "raw" / "ghalamsiah2026"
        / "Public_ScientificData_AHUFaults" / "Public_ScientificData_AHUFaults")

# Simulated G36 datasets (G36-1wk, G36-5wk, G36-Degrad, G36-Cyber, G36-HIL): 5-min samples.
G36_COLUMNS = {
    "time_s": "time (s)",
    "sat": "AHU Supply Air Temperature (°K)",
    "sat_sp": "AHU Supply Air Temperature Setpoint (°K)",
    "oat": "AHU Outdoor Air Dry Bulb Temperature (°K)",
    "oat_wb": "AHU Outdoor Air Wet Bulb Temperature (°K)",
    "mat": "AHU Mixed Air Temperature (°K)",
    "rat": "AHU Return Air Temperature (°K)",
    "ccdat": "Cooling Coil Discharge Air Temperature (°K)",
    "fan_on": "AHU Supply Air Fan Status (0: off, 1: on)",
    "fan_cmd": ("AHU Supply Air Fan Speed Control Signal (fraction, 0: fan speed should be 0% "
                "to 1:fan speed should be 100%)"),
    "oad_cmd": ("Control Signal for AHU Outdoor Air Damper from BAS (fraction, 0: damper should "
                "be fully closed to 1: damper should be fully open)"),
    "sa_mdot": "Supply Air Mass Flow Rate (kg/s)",
    "oa_mdot": "Outdoor Air Mass Flow Rate (kg/s)",
    "ccv_cmd": ("Control Signal for AHU Cooling Coil Valve from BAS (fraction, 0: valve should "
                "be fully closed to 1: valve should be fully open)"),
    "hcv_cmd": ("Control Signal for AHU Heating Coil Valve (fraction, 0: valve should be fully "
                "closed to 1: valve should be fully open)"),
    "dsp": "Measured AHU Supply Air Duct Static Pressure (Pa)",
    "dsp_sp": "AHU Supply Air Duct Static Pressure Setpoint (Pa)",
    "cc_q": "Cooling Coil Heat Transfer Rate (W)",
    "hc_q": "Heating Coil Heat Transfer Rate (W)",
    "occupied": ("Indicator if the System Operates in Occupied Mode (0: unoccupied mode, "
                 "1: occupied mode)"),
    "chw_supply": "Temperature of the Water Leaving the Chilled Water Loop (°K)",
    "chw_return": "Temperature of the Water Entering the Chilled Water Loop (°K)",
    "chw_sp": "Setpoint of Temperature of the Water Leaving the Chilled Water Loop (°K)",
    "chw_cc_mdot": "Chilled Water Flow Rate of the Chilled Water Loop into the Cooling Coil (kg/s)",
}
KELVIN = {"sat", "sat_sp", "oat", "oat_wb", "mat", "rat", "ccdat", "chw_supply", "chw_return",
          "chw_sp"}

G36_DEGRAD = ROOT / "06_G36-Degrad"
G36_DEGRAD_SCENARIOS = {
    "baseline": "BaselineSystem.csv",
    "coil_fouling_7": "CooCoiAirFou_hea7_pre30.csv",
    "coil_fouling_14": "CooCoiAirFou_hea14_pre200.csv",
    "sat_bias_plus": "TSup_p1.csv",
    "sat_bias_minus": "TSup_m1.csv",
}


def load_g36(path: str | Path, points: list[str] | None = None) -> pd.DataFrame:
    """Load a simulated G36 CSV with canonical columns, indexed by time from the start [s]."""
    points = list(G36_COLUMNS) if points is None else ["time_s", *points]
    usecols = [G36_COLUMNS[p] for p in dict.fromkeys(points)]
    raw = pd.read_csv(path, usecols=usecols, encoding="utf-8")
    df = raw.rename(columns={v: k for k, v in G36_COLUMNS.items()})[list(dict.fromkeys(points))]
    for col in KELVIN.intersection(df.columns):
        df[col] = df[col] - 273.15
    return df.set_index("time_s")


def load_g36_degrad(scenario: str, points: list[str] | None = None) -> pd.DataFrame:
    return load_g36(G36_DEGRAD / G36_DEGRAD_SCENARIOS[scenario], points)


def day_of_year(index: pd.Index) -> np.ndarray:
    return np.asarray(index, dtype=float) / 86400.0


# RBC-ASHRAE1312, real part: ERS (Iowa), 1-min samples, one folder per test day
# holding baseline.csv (the fault-free twin AHU on that day), the fault file and
# weather.csv. Fault files carry OA-TEMP = 0, so the outdoor temperature is taken
# from weather.csv.
ERS_REAL = ROOT / "01_RBC-ASHRAE1312" / "Real"
ERS_COLUMNS = {
    "occupied": "SYS-CTL",
    "fan_on": "SF-SST",
    "sat": "SA-TEMP",
    "sat_sp": "SAT_SPT",
    "mat": "MA-TEMP",
    "rat": "RA-TEMP",
    "oat": "OA-TEMP",
    "ccdat": "CHWC-DAT",
    "hcdat": "HWC-DAT",
    "chw_ewt": "CHWC-EWT",
    "hw_ewt": "HWC-EWT",
    "sa_cfm": "SA-CFM",
    "oa_cfm": "OA-CFM",
    "dsp": "SA-SP",
    "dsp_sp": "SA_SPSPT",
    "ra_humd": "RA-HUMD",
    "sa_humd": "SA-HUMD",
    "ccv_cmd": "CHWC-VLV",  # % open
    "hcv_cmd": "HWC-VLV",  # % closed
    "fan_cmd": "SF-SPD",  # % speed
    "oad_cmd": "OA-DMPR",  # % open
}
ERS_FAHRENHEIT = {"sat", "sat_sp", "mat", "rat", "oat", "ccdat", "hcdat", "chw_ewt", "hw_ewt"}
ERS_SEASONS = ("Summer", "Transition", "Winter")


def ers_days() -> pd.DataFrame:
    """One row per ERS test day: season, date, folder and fault file name."""
    rows = []
    for season in ERS_SEASONS:
        for folder in sorted((ERS_REAL / season).iterdir()):
            names = sorted(f.name for f in folder.iterdir())
            fault = [n for n in names if n not in ("baseline.csv", "weather.csv")]
            rows.append({"season": season, "date": pd.Timestamp(folder.name), "folder": folder,
                         "fault": fault[0].removesuffix(".csv") if fault else None})
    return pd.DataFrame(rows)


def load_ers(folder: str | Path, which: str = "baseline") -> pd.DataFrame:
    """Load one ERS day: which='baseline' (fault-free twin AHU) or 'fault'."""
    folder = Path(folder)
    if which == "baseline":
        path = folder / "baseline.csv"
    else:
        path = next(f for f in folder.iterdir() if f.name not in ("baseline.csv", "weather.csv"))
    raw = pd.read_csv(path)
    date = pd.Timestamp(folder.name)
    hhmm = raw["TIME"].astype(int)
    index = date + pd.to_timedelta(hhmm // 100 * 60 + hhmm % 100, unit="min")
    df = pd.DataFrame({k: raw[v].to_numpy(dtype=float) for k, v in ERS_COLUMNS.items()},
                      index=pd.DatetimeIndex(index, name="time"))
    weather = pd.read_csv(folder / "weather.csv")
    df["oat"] = weather["OA-TEMP"].to_numpy(dtype=float)[: len(df)]
    for col in ERS_FAHRENHEIT:
        df[col] = (df[col] - 32.0) / 1.8
    df["ccv_cmd"] = df["ccv_cmd"] / 100.0
    df["hcv_cmd"] = 1.0 - df["hcv_cmd"] / 100.0
    df["fan_cmd"] = df["fan_cmd"] / 100.0
    df["oad_cmd"] = df["oad_cmd"] / 100.0
    return df
