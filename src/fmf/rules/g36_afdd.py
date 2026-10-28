"""ASHRAE Guideline 36 AFDD for multiple-zone VAV air-handling units.

Implements the fault conditions FC#1-FC#15 of ASHRAE Guideline 36-2021,
Section 5.16.14, as amended by Addendum p (2024): operating states from the
commanded heating valve, cooling valve and outdoor-air damper positions (the
table for a single common minimum-OA/economizer damper), five-minute rolling
averages of the measured temperatures and duct static pressure, the default
internal variables, the operating states in which each condition applies,
ModeDelay suppression and AlarmDelay persistence. Comparisons written as
equalities in the guideline (valve "= 0", damper "= 100%", "= MinOA-P") use
the tolerance `position_tol`, an implementation choice.

All temperatures in degC, positions and fan speed as fractions in [0, 1],
pressures in consistent units (eps_dsp in the same unit as DSP).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class G36Params:
    dt_sf: float = 1.0  # temperature rise across the supply fan [K]
    dt_min: float = 6.0  # minimum |OAT - RAT| to evaluate economizer errors (FC#6) [K]
    eps_sat: float = 1.0
    eps_rat: float = 1.0
    eps_mat: float = 3.0
    eps_oat: float = 1.0  # local OAT sensor at the unit
    eps_f: float = 0.30  # airflow (outdoor-air fraction) error threshold
    eps_vfdspd: float = 0.05
    eps_dsp: float = 25.0  # duct static pressure error threshold [Pa]
    eps_ccet: float | None = None  # None: eps_mat
    eps_cclt: float | None = None  # None: eps_sat
    eps_hcet: float | None = None  # None: eps_mat
    eps_hclt: float | None = None  # None: eps_sat
    os_max_changes: int = 7  # per 60-minute moving window
    mode_delay_min: int = 30
    alarm_delay_min: int = 30
    avg_window_min: int = 5
    position_tol: float = 0.01
    fan_heat_between_cc_sensors: bool = False  # fan heat lowers the FC#14 drop threshold
    fan_heat_between_hc_sensors: bool = False  # fan heat raises the FC#15 rise threshold


FC_STATES = {
    1: (1, 2, 3, 4, 5), 2: (1, 2, 3, 4, 5), 3: (1, 2, 3, 4, 5), 4: (1, 2, 3, 4, 5),
    5: (1,), 6: (1, 4), 7: (1,), 8: (2,), 9: (2,), 10: (3,), 11: (3,),
    12: (2, 3, 4), 13: (3, 4), 14: (1, 2), 15: (2, 3, 4),
}


def operating_state(hc: pd.Series, cc: pd.Series, oad: pd.Series, min_oa: float,
                    tol: float = 0.01) -> pd.Series:
    """OS#1-#5 from commanded positions (single common minimum-OA/economizer damper)."""
    hc_on, cc_on = hc > tol, cc > tol
    oad_min = (oad - min_oa).abs() <= tol
    oad_full = oad >= 1.0 - tol
    oad_mod = (oad > min_oa + tol) & (oad < 1.0 - tol)
    state = pd.Series(5, index=hc.index, dtype=int)
    state[hc_on & ~cc_on & oad_min] = 1
    state[~hc_on & ~cc_on & oad_mod] = 2
    state[~hc_on & cc_on & oad_full] = 3
    state[~hc_on & cc_on & oad_min] = 4
    return state


def fault_conditions(df: pd.DataFrame, min_oa: float, p: G36Params = G36Params(),
                     oa_min_fraction: pd.Series | None = None) -> pd.DataFrame:
    """Instantaneous truth of FC#1-#15 (False outside the applicable operating states).

    df: 1-min samples indexed by time with columns sat, mat, rat, oat, dsp,
    sat_sp, dsp_sp, hcv_cmd, ccv_cmd, fan_cmd, oad_cmd and, for FC#14/#15,
    ccet, cclt, hcet, hclt (coil entering/leaving temperatures).
    oa_min_fraction: active minimum outdoor-air fraction %OA_min; if None,
    FC#6 is not evaluated.
    """
    w = f"{p.avg_window_min}min"
    avg = {c: df[c].rolling(w, min_periods=1).mean()
           for c in ("sat", "mat", "rat", "oat", "dsp", "ccet", "cclt", "hcet", "hclt")
           if c in df}
    state = operating_state(df.hcv_cmd, df.ccv_cmd, df.oad_cmd, min_oa, p.position_tol)
    changes = (state != state.shift()).astype(int)
    changes.iloc[0] = 0
    n_changes = changes.rolling("60min", min_periods=1).sum()
    e_ccet = p.eps_mat if p.eps_ccet is None else p.eps_ccet
    e_cclt = p.eps_sat if p.eps_cclt is None else p.eps_cclt
    e_hcet = p.eps_mat if p.eps_hcet is None else p.eps_hcet
    e_hclt = p.eps_sat if p.eps_hclt is None else p.eps_hclt
    sat, mat, rat, oat = avg["sat"], avg["mat"], avg["rat"], avg["oat"]

    fc = {
        1: (avg["dsp"] < df.dsp_sp - p.eps_dsp) & (df.fan_cmd >= 0.99 - p.eps_vfdspd),
        2: mat + p.eps_mat < np.minimum(rat - p.eps_rat, oat - p.eps_oat),
        3: mat - p.eps_mat > np.maximum(rat + p.eps_rat, oat + p.eps_oat),
        4: n_changes > p.os_max_changes,
        5: sat + p.eps_sat <= mat - p.eps_mat + p.dt_sf,
        7: (sat < df.sat_sp - p.eps_sat) & (df.hcv_cmd >= 0.99),
        8: (sat - p.dt_sf - mat).abs() > np.hypot(p.eps_sat, p.eps_mat),
        9: oat - p.eps_oat > df.sat_sp - p.dt_sf + p.eps_sat,
        10: (mat - oat).abs() > np.hypot(p.eps_mat, p.eps_oat),
        11: oat + p.eps_oat < df.sat_sp - p.dt_sf - p.eps_sat,
        12: sat - p.eps_sat - p.dt_sf >= mat + p.eps_mat,
        13: (sat > df.sat_sp + p.eps_sat) & (df.ccv_cmd >= 0.99),
    }
    if oa_min_fraction is not None:
        oa_frac = (mat - rat) / (oat - rat)
        fc[6] = ((rat - oat).abs() >= p.dt_min) & ((oa_frac - oa_min_fraction).abs() > p.eps_f)
    if "ccet" in avg and "cclt" in avg:
        fc[14] = (avg["ccet"] - avg["cclt"]
                  >= np.hypot(e_ccet, e_cclt) - (p.dt_sf if p.fan_heat_between_cc_sensors else 0.0))
    if "hcet" in avg and "hclt" in avg:
        fc[15] = (avg["hclt"] - avg["hcet"]
                  >= np.hypot(e_hcet, e_hclt) + (p.dt_sf if p.fan_heat_between_hc_sensors else 0.0))
    out = pd.DataFrame(index=df.index)
    for k in sorted(fc):
        out[f"FC{k}"] = fc[k].fillna(False).astype(bool) & state.isin(FC_STATES[k])
    out["state"] = state
    return out


def alarms(conditions: pd.DataFrame, operating: pd.Series, mode_start: pd.Series,
           p: G36Params = G36Params()) -> pd.DataFrame:
    """Reported alarms: a condition must hold for AlarmDelay minutes while the AHU
    operates, outside the ModeDelay window after a mode change.

    operating: True while the AHU runs. mode_start: True at samples where the
    zone-group mode changes (e.g. start of occupancy).
    """
    since_mode = pd.Series(np.nan, index=operating.index)
    last = None
    for i, (t, flag) in enumerate(mode_start.items()):
        if flag:
            last = t
        if last is not None:
            since_mode.iloc[i] = (t - last).total_seconds() / 60.0
    evaluable = operating & (since_mode >= p.mode_delay_min)
    out = pd.DataFrame(index=conditions.index)
    for col in [c for c in conditions.columns if c.startswith("FC")]:
        active = conditions[col] & evaluable
        run = active.astype(int).groupby((~active).cumsum()).cumsum()
        out[col] = run >= p.alarm_delay_min
    return out
