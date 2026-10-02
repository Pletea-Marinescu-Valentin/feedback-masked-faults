"""Conformal p-values, DKW correction and e-detectors: ARL0 >= 1 / alpha on i.i.d. data."""

import numpy as np
import pytest
from scipy.integrate import quad

from fmf.detectors.conformal import conformal_pvalues, dkw_adjust, dkw_epsilon
from fmf.detectors.e_detector import (
    log_e_cusum,
    log_e_shiryaev_roberts,
    mixture_calibrator,
    power_calibrator,
)
from fmf.metrics.run_length import first_crossing, run_lengths

ALPHA = 0.02


# E[e(U)] is integrated over x = -ln p on [0, X]; beyond X, exp(-x) underflows, so
# the tail is added analytically.
X_MAX = 700.0


@pytest.mark.parametrize("kappa", [0.2, 0.5, 0.8])
def test_power_calibrator_integrates_to_one(kappa):
    value, _ = quad(lambda x: power_calibrator(np.exp(-x), kappa) * np.exp(-x), 0, X_MAX,
                    limit=200)
    assert value + np.exp(-kappa * X_MAX) == pytest.approx(1.0, rel=1e-6)


def test_mixture_calibrator():
    # The integrand is (1 - exp(-x) - x exp(-x)) / x^2, whose tail beyond X is 1 / X.
    value, _ = quad(lambda x: mixture_calibrator(np.exp(-x)) * np.exp(-x), 0, X_MAX,
                    limit=200)
    assert value + 1.0 / X_MAX == pytest.approx(1.0, rel=1e-6)
    for p in (1e-6, 0.01, 0.3, 0.99):
        direct, _ = quad(lambda k: power_calibrator(p, k), 0, 1)
        assert mixture_calibrator(p) == pytest.approx(direct, rel=1e-6)
    assert mixture_calibrator(1.0) == pytest.approx(0.5)


def test_conformal_pvalues_small_example():
    p = conformal_pvalues([1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0])
    np.testing.assert_allclose(p, [1.0, 0.8, 0.2])


def test_conformal_pvalues_are_marginally_uniform():
    rng = np.random.default_rng(0)
    n, draws = 19, 50000
    scores = rng.standard_normal((draws, n + 1))
    p = np.array([conformal_pvalues(row[:n], row[n]) for row in scores])
    levels = np.arange(1, n + 2) / (n + 1)
    freq = np.array([np.mean(p <= a + 1e-12) for a in levels])
    np.testing.assert_allclose(freq, levels, atol=4 * np.sqrt(0.25 / draws))


def test_dkw_gives_calibration_conditional_validity():
    # Given the calibration set, P(p <= j / (n + 1)) = V_(j), the j-th smallest of
    # n uniforms. Conditional validity at every level means V_(j) <= j / (n + 1) + eps.
    rng = np.random.default_rng(1)
    n, draws, delta = 200, 4000, 0.1
    v = np.sort(rng.uniform(size=(draws, n)), axis=1)
    excess = np.max(v - np.arange(1, n + 1) / (n + 1), axis=1)
    assert np.mean(excess > 0.0) > 0.9  # unadjusted: almost never conditionally valid
    se = np.sqrt(delta * (1 - delta) / draws)
    assert np.mean(excess > dkw_epsilon(n, delta)) <= delta + 3 * se


def test_dkw_epsilon_requires_small_delta():
    with pytest.raises(ValueError):
        dkw_epsilon(100, 0.6)


def e_detector_run_lengths(log_e, detector, horizon):
    stat = log_e_cusum(log_e) if detector == "cusum" else log_e_shiryaev_roberts(log_e)
    return run_lengths(first_crossing(stat, np.log(1.0 / ALPHA)), horizon)


@pytest.mark.parametrize("detector", ["cusum", "sr"])
@pytest.mark.parametrize("calibrator", ["power", "mixture"])
def test_arl0_at_least_inverse_alpha_with_calibrated_conformal_pvalues(detector, calibrator):
    rng = np.random.default_rng(2)
    n_cal, delta, horizon, n_runs = 1000, 0.1, 2000, 1000
    calibration = rng.standard_normal(n_cal)
    p = dkw_adjust(conformal_pvalues(calibration, rng.standard_normal((horizon, n_runs))),
                   n_cal, delta)
    e = power_calibrator(p, 0.5) if calibrator == "power" else mixture_calibrator(p)
    # Censoring at the horizon only lowers the estimate.
    assert e_detector_run_lengths(np.log(e), detector, horizon).mean() >= 1.0 / ALPHA


@pytest.mark.parametrize("detector", ["cusum", "sr"])
def test_arl0_at_least_inverse_alpha_with_exact_pvalues(detector):
    # Uniform p-values make E[e] = 1 exactly: e-SR is then close to its bound.
    rng = np.random.default_rng(3)
    horizon, n_runs = 2000, 1000
    log_e = np.log(power_calibrator(rng.uniform(size=(horizon, n_runs)), 0.5))
    assert e_detector_run_lengths(log_e, detector, horizon).mean() >= 1.0 / ALPHA


def test_e_cusum_detects_a_shift():
    rng = np.random.default_rng(4)
    n_cal, horizon = 1000, 400
    calibration = rng.standard_normal(n_cal)
    p = dkw_adjust(conformal_pvalues(calibration, 1.0 + rng.standard_normal((horizon, 2000))),
                   n_cal, 0.1)
    rl = e_detector_run_lengths(np.log(power_calibrator(p, 0.5)), "cusum", horizon)
    assert np.all(rl < horizon)
    assert rl.mean() < 40.0


def test_log_space_detectors_match_recursions():
    rng = np.random.default_rng(5)
    e = np.exp(rng.normal(-0.1, 0.8, size=(30, 2)))
    m = np.zeros(2)
    r = np.zeros(2)
    cusum_ref, sr_ref = [], []
    for row in e:
        m = row * np.maximum(m, 1.0)
        r = row * (1.0 + r)
        cusum_ref.append(np.maximum(np.log(m), 0.0))
        sr_ref.append(np.log(r))
    np.testing.assert_allclose(log_e_cusum(np.log(e)), np.array(cusum_ref), atol=1e-12)
    np.testing.assert_allclose(log_e_shiryaev_roberts(np.log(e)), np.array(sr_ref), atol=1e-12)


def test_unfloored_e_cusum_matches_recursion():
    from fmf.detectors.e_detector import log_e_cusum_unfloored

    rng = np.random.default_rng(6)
    e = np.exp(rng.normal(-0.1, 0.8, size=40))
    m, ref = 0.0, []
    for x in e:
        m = x * max(m, 1.0)
        ref.append(np.log(m))
    np.testing.assert_allclose(log_e_cusum_unfloored(np.log(e)), ref, atol=1e-12)


def test_kappa_mixture_keeps_arl0_and_detects_small_shifts():
    from scipy.stats import norm

    from fmf.detectors.e_detector import log_e_cusum_unfloored, log_mixture

    rng = np.random.default_rng(7)
    kappas = np.array([0.5, 0.8, 0.9, 0.95, 0.98])

    def run(p, threshold, horizon):
        stats = np.stack([log_e_cusum_unfloored(np.log(power_calibrator(p, k))) for k in kappas],
                         axis=-1)
        return run_lengths(first_crossing(log_mixture(stats), threshold), horizon)

    h0 = run(rng.uniform(size=(1500, 600)), np.log(1.0 / ALPHA), 1500)
    assert h0.mean() >= 1.0 / ALPHA
    # A 0.3-sigma shift at ARL0 >= 1e4: kappa = 0.5 has a negative drift and relies on
    # fluctuations; the mixture follows the kappa with positive drift.
    p_shift = norm.sf(0.3 + rng.standard_normal((3000, 400)))
    threshold = np.log(1e4)
    rl_half = run_lengths(first_crossing(log_e_cusum(np.log(power_calibrator(p_shift, 0.5))),
                                         threshold), 3000)
    rl_mix = run(p_shift, threshold, 3000)
    assert np.all(rl_mix < 3000)
    assert rl_mix.mean() < 0.6 * rl_half.mean()
