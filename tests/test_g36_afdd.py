import numpy as np
import pandas as pd
import pytest

from fmf.rules.g36_afdd import G36Params, alarms, fault_conditions, operating_state


def frame(n=120, **cols):
    idx = pd.date_range("2026-01-01 06:00", periods=n, freq="1min")
    base = dict(sat=13.0, mat=24.0, rat=23.0, oat=30.0, dsp=250.0, sat_sp=13.0, dsp_sp=250.0,
                hcv_cmd=0.0, ccv_cmd=0.5, fan_cmd=0.7, oad_cmd=0.4,
                ccet=24.0, cclt=12.0, hcet=24.0, hclt=24.0)
    base.update(cols)
    return pd.DataFrame({k: np.broadcast_to(v, n).astype(float) for k, v in base.items()},
                        index=idx)


def test_operating_states():
    s = pd.Series
    hc = s([0.3, 0.0, 0.0, 0.0, 0.2])
    cc = s([0.0, 0.0, 0.6, 0.6, 0.6])
    oad = s([0.4, 0.7, 1.0, 0.4, 0.4])
    np.testing.assert_array_equal(operating_state(hc, cc, oad, 0.4), [1, 2, 3, 4, 5])


def test_fault_free_cooling_raises_nothing():
    fc = fault_conditions(frame(), min_oa=0.4)
    assert (fc["state"] == 4).all()
    assert not fc.filter(like="FC").any().any()


def test_full_cooling_with_high_sat_triggers_fc13():
    fc = fault_conditions(frame(sat=15.0, ccv_cmd=1.0), min_oa=0.4)
    assert fc["FC13"].all()


def test_heating_coil_leak_triggers_fc15_only_above_threshold():
    p = G36Params()
    threshold = np.hypot(p.eps_mat, p.eps_sat)
    small = fault_conditions(frame(hclt=24.0 + 0.8 * threshold), min_oa=0.4)
    large = fault_conditions(frame(hclt=24.0 + 1.2 * threshold), min_oa=0.4)
    assert not small["FC15"].any()
    assert large["FC15"].all()


def test_conditions_outside_their_states_are_not_evaluated():
    # SAT far below SATSP with the heating valve full open: FC#7 only in OS#1.
    heating = fault_conditions(frame(hcv_cmd=1.0, ccv_cmd=0.0, sat=10.0, sat_sp=20.0, mat=8.0,
                                     oat=-5.0, hcet=8.0, hclt=9.0), min_oa=0.4)
    assert (heating["state"] == 1).all() and heating["FC7"].all()
    cooling = fault_conditions(frame(hcv_cmd=0.0, ccv_cmd=0.5, sat=10.0, sat_sp=20.0),
                               min_oa=0.4)
    assert not cooling["FC7"].any()


def test_alarm_needs_persistence_and_mode_delay():
    df = frame(n=120, sat=15.0, ccv_cmd=1.0)
    fc = fault_conditions(df, min_oa=0.4)
    operating = pd.Series(True, index=df.index)
    mode_start = pd.Series(False, index=df.index)
    mode_start.iloc[0] = True
    out = alarms(fc, operating, mode_start)
    first = int(np.argmax(out["FC13"].to_numpy()))
    # ModeDelay (30 min) then AlarmDelay (30 min of persistence).
    assert first == pytest.approx(30 + 29, abs=1)
