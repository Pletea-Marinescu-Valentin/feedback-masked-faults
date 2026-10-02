import numpy as np
import pytest

from fmf.theory.masking import (
    CP_AIR,
    bias_command_shift,
    critical_capacity,
    fouling_command_shift,
    measured_cv_offset,
    plant_dc_gain,
    required_load,
    steady_state_command,
)
from fmf.valves import EqualPercentageValve, LinearValve

LIN = LinearValve()
MDOT = 5.0
THETA0 = 100.0


def test_load_sign_convention():
    # Cooling (sign -1) needs capacity when the inlet is warmer than the setpoint.
    assert required_load(24.0, 13.0, MDOT, sign=-1) == pytest.approx(MDOT * CP_AIR * 11.0)
    # Heating (sign +1) needs capacity when the inlet is colder.
    assert required_load(5.0, 20.0, MDOT, sign=+1) == pytest.approx(MDOT * CP_AIR * 15.0)


@pytest.mark.parametrize("sign", [-1, +1])
def test_bias_shift_linear_valve(sign):
    load, b = 50.0, 0.8
    du = bias_command_shift(b, load, THETA0, MDOT, LIN, sign=sign)
    assert du == pytest.approx(-sign * MDOT * CP_AIR * b / THETA0)


def test_fouling_shift_linear_valve():
    load, delta = 50.0, 0.25
    u_n = load / THETA0
    du = fouling_command_shift(delta, load, THETA0, LIN)
    assert du == pytest.approx(u_n * delta / (1.0 - delta))


@pytest.mark.parametrize("valve", [LIN, EqualPercentageValve(50.0)])
def test_fouling_shift_vanishes_without_load(valve):
    loads = np.array([-20.0, 0.0])
    np.testing.assert_allclose(fouling_command_shift(0.3, loads, THETA0, valve), 0.0)


def test_fouling_shift_grows_with_load():
    loads = np.linspace(5.0, 60.0, 12)
    du = fouling_command_shift(0.2, loads, THETA0, EqualPercentageValve(50.0))
    assert np.all(np.diff(du) > 0.0)


def test_measured_cv_is_masked_until_saturation():
    # The bias adds mdot * cp * b = 2.5 kW, so 77 kW stays just below theta = 80 kW.
    loads = np.array([10.0, 50.0, 77.0])
    theta = THETA0 * (1.0 - 0.2)
    np.testing.assert_allclose(measured_cv_offset(loads, theta, MDOT, LIN, bias=0.5), 0.0,
                               atol=1e-12)
    # Overload of a cooling coil: the supply air ends up warmer than the setpoint.
    overload = 90.0
    offset = measured_cv_offset(overload, theta, MDOT, LIN)
    assert offset == pytest.approx((overload - theta) / (MDOT * CP_AIR))
    # Idle cooling coil: no capacity to remove, the measured CV sits below the setpoint.
    assert measured_cv_offset(-15.0, theta, MDOT, LIN) == pytest.approx(-15.0 / (MDOT * CP_AIR))


def test_steady_state_command_clips_to_actuator_range():
    u = steady_state_command(np.array([-5.0, 50.0, 150.0]), THETA0, LIN)
    np.testing.assert_allclose(u, [0.0, 0.5, 1.0])


def test_critical_capacity_and_dc_gain():
    valve = EqualPercentageValve(50.0)
    assert critical_capacity(80.0, valve, u_max=0.9) == pytest.approx(80.0 / valve.phi(0.9))
    k = plant_dc_gain(THETA0, MDOT, 0.4, LIN)
    assert k == pytest.approx(-THETA0 / (MDOT * CP_AIR))
