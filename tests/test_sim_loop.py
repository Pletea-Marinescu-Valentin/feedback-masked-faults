"""The synthetic loop reproduces the quasi-static masking relations of fmf.theory.masking."""

import numpy as np
import pytest

from fmf.sim import (
    LoopParams,
    NoiseModel,
    constant_context,
    seasonal_context,
    simulate,
    step_profile,
)
from fmf.theory.masking import (
    CP_AIR,
    bias_command_shift,
    command_long_run_variance,
    fouling_command_shift,
    measured_cv_offset,
    plant_dc_gain,
)
from fmf.theory.sigma_eff import batch_means_lrv
from fmf.valves import EqualPercentageValve, LinearValve

N, ONSET = 600, 200
VALVES = [LinearValve(), EqualPercentageValve(50.0)]
# (sign, inlet temperature, setpoint): cooling and heating coils, both unsaturated.
LOOPS = [(-1, 24.0, 13.0), (+1, 5.0, 20.0)]


def run(sign, t_in, setpoint, valve, **faults):
    params = LoopParams(sign=sign, valve=valve)
    ctx = constant_context(N, t_in=t_in, setpoint=setpoint)
    return params, ctx, simulate(params, ctx, **faults)


@pytest.mark.parametrize("valve", VALVES)
@pytest.mark.parametrize("sign,t_in,setpoint", LOOPS)
def test_fault_free_loop_holds_setpoint(sign, t_in, setpoint, valve):
    _, _, res = run(sign, t_in, setpoint, valve)
    np.testing.assert_allclose(res.y_meas, setpoint, atol=1e-9)
    np.testing.assert_allclose(res.residual, 0.0, atol=1e-9)


@pytest.mark.parametrize("valve", VALVES)
@pytest.mark.parametrize("sign,t_in,setpoint", LOOPS)
def test_sensor_bias_is_masked_in_cv_and_shifts_command(sign, t_in, setpoint, valve):
    b = 1.0
    params, ctx, res = run(sign, t_in, setpoint, valve, bias=step_profile(N, ONSET, b))
    end = slice(-50, None)
    # Measured CV back at the setpoint, true CV off by -b.
    np.testing.assert_allclose(res.y_meas[end], setpoint, atol=1e-6)
    np.testing.assert_allclose(res.y[end], setpoint - b, atol=1e-6)
    expected = bias_command_shift(b, ctx.load(sign=sign)[0], params.theta0, 5.0, valve,
                                  sign=sign)
    np.testing.assert_allclose(res.residual[end], expected, atol=1e-6)
    if isinstance(valve, LinearValve):
        assert expected == pytest.approx(-sign * 5.0 * CP_AIR * b / params.theta0)


@pytest.mark.parametrize("valve", VALVES)
@pytest.mark.parametrize("sign,t_in,setpoint", LOOPS)
def test_fouling_is_masked_in_cv_and_shifts_command(sign, t_in, setpoint, valve):
    delta = 0.2
    params, ctx, res = run(sign, t_in, setpoint, valve,
                           fouling=step_profile(N, ONSET, delta))
    end = slice(-50, None)
    np.testing.assert_allclose(res.y_meas[end], setpoint, atol=1e-6)
    load = ctx.load(sign=sign)[0]
    expected = fouling_command_shift(delta, load, params.theta0, valve)
    np.testing.assert_allclose(res.residual[end], expected, atol=1e-6)
    if isinstance(valve, LinearValve):
        u_n = load / params.theta0
        assert expected == pytest.approx(u_n * delta / (1.0 - delta))


def test_fouling_of_idle_coil_leaves_no_trace():
    # Cooling coil with the inlet below the setpoint: the actuator rests at 0.
    _, _, res = run(-1, 10.0, 13.0, LinearValve(), fouling=step_profile(N, ONSET, 0.3))
    np.testing.assert_allclose(res.u, 0.0, atol=1e-12)
    np.testing.assert_allclose(res.residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(res.y_meas, 10.0, atol=1e-9)


@pytest.mark.parametrize("t_in,bias,fouling", [(30.0, 0.0, 0.3), (10.0, 1.0, 0.3)])
def test_saturation_opens_the_loop_and_reveals_the_fault(t_in, bias, fouling):
    params, ctx, res = run(-1, t_in, 13.0, LinearValve(), bias=step_profile(N, ONSET, bias),
                           fouling=step_profile(N, ONSET, fouling))
    end = slice(-50, None)
    u_sat = params.u_max if t_in > 13.0 else params.u_min
    np.testing.assert_allclose(res.u[end], u_sat, atol=1e-12)
    expected = measured_cv_offset(ctx.load()[0], params.theta0 * (1 - fouling), 5.0,
                                  params.valve, bias=bias)
    assert abs(expected) > 1.0
    np.testing.assert_allclose(res.y_meas[end] - 13.0, expected, atol=1e-6)


def test_replicas_share_context_and_faults():
    params = LoopParams()
    ctx = constant_context(100)
    res = simulate(params, ctx, bias=0.5, n_replicas=4)
    assert res.u.shape == (100, 4)
    np.testing.assert_array_equal(res.u, res.u[:, :1].repeat(4, axis=1))
    noisy = simulate(params, ctx, noise=NoiseModel(sensor_std=0.1), n_replicas=4, rng=0)
    assert np.all(np.std(noisy.u, axis=1)[1:] > 0.0)


def test_simulation_is_reproducible_with_seed():
    params, ctx = LoopParams(), constant_context(200)
    noise = NoiseModel(sensor_std=0.1, quantization=0.05, disturbance_std=0.3,
                       disturbance_rho=0.95)
    a = simulate(params, ctx, noise=noise, rng=42)
    b = simulate(params, ctx, noise=noise, rng=42)
    np.testing.assert_array_equal(a.u, b.u)


@pytest.mark.parametrize("kp,ti", [(0.0168, 120.0), (0.0335, 120.0), (0.067, 120.0),
                                   (0.0335, 60.0), (0.0335, 300.0)])
def test_command_noise_long_run_variance_does_not_depend_on_tuning(kp, ti):
    # With integral action, CV sensor noise reaches u through the DC gain -1/P(0):
    # the long-run variance of the residual is sigma_n^2 / P(0)^2 for any stable
    # tuning, although its marginal variance and autocorrelation change.
    params = LoopParams(kp=kp, ti=ti)
    ctx = constant_context(20500, t_in=24.0)
    res = simulate(params, ctx, noise=NoiseModel(sensor_std=0.2), n_replicas=200,
                   rng=7)
    lrv = batch_means_lrv(res.residual[500:], 400).mean()
    gain = plant_dc_gain(params.theta0, 5.0, res.u_nominal[0], params.valve)
    assert lrv == pytest.approx(command_long_run_variance(0.2**2, gain), rel=0.06)


def test_seasonal_context_has_idle_windows():
    ctx = seasonal_context(365, dt=600.0)
    assert ctx.t_in.min() == pytest.approx(7.0, abs=0.1)
    assert ctx.t_in.max() == pytest.approx(29.0, abs=0.1)
    load = ctx.load()
    assert 0.1 < np.mean(load <= 0.0) < 0.9
    du = fouling_command_shift(0.2, load, 100.0, LinearValve())
    assert np.all(du[load <= 0.0] == 0.0)
