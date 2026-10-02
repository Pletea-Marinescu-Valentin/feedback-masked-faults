"""CUSUM delay against ln(ARL0) / I for a known Gaussian shift."""

import numpy as np
import pytest

from fmf.detectors.cusum import cusum, gaussian_llr
from fmf.metrics.run_length import first_crossing, run_lengths
from fmf.theory.delay import (
    cumulative_information_delay,
    gaussian_cusum_arl,
    gaussian_cusum_threshold,
    gaussian_information,
)


def mean_run_length(shift, actual_shift, h, horizon, n_runs, rng):
    x = actual_shift + rng.standard_normal((horizon, n_runs))
    rl = run_lengths(first_crossing(cusum(gaussian_llr(x, shift)), h), horizon)
    assert np.all(rl < horizon), "censored runs bias the mean"
    return rl.mean()


def test_cusum_matches_recursion():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 3)) - 0.2
    w = np.zeros(3)
    expected = []
    for row in x:
        w = np.maximum(0.0, w + row)
        expected.append(w)
    np.testing.assert_allclose(cusum(x), np.array(expected), atol=1e-12)


def test_arl0_matches_siegmund():
    rng = np.random.default_rng(1)
    h = 3.0
    arl0 = mean_run_length(1.0, 0.0, h, 2000, 2000, rng)
    assert arl0 == pytest.approx(gaussian_cusum_arl(h, 1.0), rel=0.08)


@pytest.mark.parametrize("h", [4.0, 12.0])
def test_delay_matches_siegmund(h):
    rng = np.random.default_rng(2)
    delay = mean_run_length(1.0, 1.0, h, 200, 4000, rng)
    assert delay == pytest.approx(gaussian_cusum_arl(h, 1.0, actual_shift=1.0), rel=0.06)


@pytest.mark.parametrize("shift", [0.5, 1.0])
def test_delay_grows_as_log_arl0_over_information(shift):
    # D ~ ln(ARL0) / I: the asymptotic slope survives at finite ARL0, the offset does not.
    rng = np.random.default_rng(3)
    h1, h2 = 4.0, 12.0
    horizon = int(100.0 / gaussian_information(shift, 1.0))
    d1 = mean_run_length(shift, shift, h1, horizon, 4000, rng)
    d2 = mean_run_length(shift, shift, h2, horizon, 4000, rng)
    log_arl0_gap = np.log(gaussian_cusum_arl(h2, shift) / gaussian_cusum_arl(h1, shift))
    slope = (d2 - d1) / log_arl0_gap
    assert slope == pytest.approx(1.0 / gaussian_information(shift, 1.0), rel=0.06)


@pytest.mark.parametrize("shift", [0.25, 0.5, 1.0])
def test_siegmund_expansion_offset(shift):
    # ARL1 = (ln ARL0 - ln(2 / delta^2) - 1) / I + o(1) for large thresholds.
    info = gaussian_information(shift, 1.0)
    h = 40.0
    arl0 = gaussian_cusum_arl(h, shift)
    arl1 = gaussian_cusum_arl(h, shift, actual_shift=shift)
    predicted = (np.log(arl0) - np.log(2.0 / shift**2) - 1.0) / info
    assert arl1 == pytest.approx(predicted, rel=1e-6)


def test_threshold_inverts_arl0():
    for shift in (0.3, 1.0):
        h = gaussian_cusum_threshold(1e4, shift)
        assert gaussian_cusum_arl(h, shift) == pytest.approx(1e4, rel=1e-8)


def test_cumulative_information_pauses_in_zero_information_windows():
    info = np.full(40, 0.5)
    arl0 = np.exp(5.0)
    assert cumulative_information_delay(info, arl0) == 9
    info[3:13] = 0.0
    assert cumulative_information_delay(info, arl0) == 19
    assert cumulative_information_delay(np.zeros(10), arl0) == -1
