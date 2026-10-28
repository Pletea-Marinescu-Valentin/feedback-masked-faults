import numpy as np
import pytest

from fmf.valves import EqualPercentageValve, LinearValve

VALVES = [LinearValve(), EqualPercentageValve(50.0), EqualPercentageValve(5.0)]


@pytest.mark.parametrize("valve", VALVES)
def test_characteristic_is_normalized_and_increasing(valve):
    u = np.linspace(0.0, 1.0, 101)
    q = valve.phi(u)
    assert q[0] == pytest.approx(0.0, abs=1e-15)
    assert q[-1] == pytest.approx(1.0, abs=1e-15)
    assert np.all(np.diff(q) > 0.0)


@pytest.mark.parametrize("valve", VALVES)
def test_inverse_and_derivative(valve):
    u = np.linspace(0.01, 0.99, 50)
    np.testing.assert_allclose(valve.phi_inv(valve.phi(u)), u, atol=1e-12)
    h = 1e-6
    fd = (valve.phi(u + h) - valve.phi(u - h)) / (2 * h)
    np.testing.assert_allclose(valve.dphi(u), fd, rtol=1e-6)


def test_rangeability_must_exceed_one():
    with pytest.raises(ValueError):
        EqualPercentageValve(1.0)
