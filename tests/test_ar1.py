"""AR(1) residuals: information per sample scales by (1 - rho) / (1 + rho)."""

import numpy as np
import pytest

from fmf.detectors.cusum import cusum, gaussian_llr
from fmf.detectors.prewhitening import ar1_prewhiten
from fmf.metrics.run_length import first_crossing, run_lengths
from fmf.sim.noise import ar1
from fmf.theory.sigma_eff import (
    ar1_information_factor,
    ar1_long_run_variance,
    batch_means_lrv,
    lag1_autocorrelation,
)


def test_ar1_generator_is_stationary():
    rng = np.random.default_rng(0)
    x = ar1(4000, 0.8, 2.0, size=200, rng=rng)
    assert x.var() == pytest.approx(4.0, rel=0.03)
    assert lag1_autocorrelation(x).mean() == pytest.approx(0.8, abs=0.01)
    assert x[0].var() == pytest.approx(4.0, rel=0.3)


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_prewhitened_shift_and_variance(rho):
    rng = np.random.default_rng(1)
    mu = 1.0
    w = ar1_prewhiten(mu + ar1(20001, rho, 1.0, size=50, rng=rng), rho)
    assert w.mean() == pytest.approx(mu * (1 - rho), abs=0.01)
    assert w.var() == pytest.approx(1 - rho**2, rel=0.02)
    assert abs(lag1_autocorrelation(w).mean()) < 0.01


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_batch_means_recovers_ar1_long_run_variance(rho):
    rng = np.random.default_rng(2)
    lrv = batch_means_lrv(ar1(20000, rho, 1.0, size=200, rng=rng), 500).mean()
    assert lrv == pytest.approx(ar1_long_run_variance(1.0, rho), rel=0.05)


def delay_slope(x, shift, sigma, horizon, h1=4.0, h2=12.0):
    llr = gaussian_llr(x, shift, sigma)
    delays = []
    for h in (h1, h2):
        rl = run_lengths(first_crossing(cusum(llr), h), horizon)
        assert np.all(rl < horizon)
        delays.append(rl.mean())
    return (delays[1] - delays[0]) / (h2 - h1)


@pytest.mark.parametrize("rho,n_runs,horizon", [(0.5, 4000, 400), (0.9, 1000, 3000)])
def test_delay_scales_with_ar1_factor(rho, n_runs, horizon):
    # Same marginal variance and shift; the delay per unit of threshold, 1 / I,
    # grows by (1 + rho) / (1 - rho) once the residual is AR(1).
    rng = np.random.default_rng(3)
    mu = 1.0
    w = ar1_prewhiten(mu + ar1(horizon + 1, rho, 1.0, size=n_runs, rng=rng), rho)
    slope_ar = delay_slope(w, mu * (1 - rho), np.sqrt(1 - rho**2), horizon)
    slope_iid = delay_slope(mu + rng.standard_normal((200, 4000)), mu, 1.0, 200)
    assert slope_ar / slope_iid == pytest.approx(1.0 / ar1_information_factor(rho), rel=0.08)


def test_fit_ar_recovers_ar2_and_whitens():
    from scipy.signal import lfilter

    from fmf.detectors.prewhitening import ar_prewhiten, fit_ar

    rng = np.random.default_rng(4)
    phi = np.array([1.2, -0.35])
    x = lfilter([0.5], [1.0, -phi[0], -phi[1]], rng.standard_normal((20000, 20)), axis=0)[500:]
    model = fit_ar(x, 2)
    np.testing.assert_allclose(model.phi, phi, atol=0.01)
    assert model.sigma_w == pytest.approx(0.5, rel=0.01)
    assert model.long_run_variance == pytest.approx(0.25 / (1 - phi.sum()) ** 2, rel=0.03)
    w = ar_prewhiten(x, model)
    assert np.all(np.isnan(w[:2]))
    assert abs(lag1_autocorrelation(w[2:]).mean()) < 0.01


def test_ar_prewhiten_skips_gaps():
    from fmf.detectors.prewhitening import ARModel, ar_prewhiten

    model = ARModel(phi=np.array([0.5]), sigma_w=1.0, mean=0.0)
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    np.testing.assert_allclose(ar_prewhiten(x, model), [np.nan, 1.5, np.nan, np.nan, 3.0])
