import numpy as np
import pytest

from fmf.baselines.pca_monitor import PCAMonitor


def correlated_gaussian(n, rng):
    latent = rng.standard_normal((n, 3))
    mix = np.array([[1.0, 0.5, 0.0, 0.2, 0.0, 0.3],
                    [0.0, 1.0, 0.8, 0.0, 0.4, 0.0],
                    [0.3, 0.0, 0.0, 1.0, 0.6, 0.5]])
    return latent @ mix + 0.3 * rng.standard_normal((n, 6))


@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_limits_have_nominal_exceedance_on_gaussian_data(alpha):
    rng = np.random.default_rng(0)
    model = PCAMonitor.fit(correlated_gaussian(20000, rng), variance=0.9)
    t2, spe = model.statistics(correlated_gaussian(200000, rng))
    t2_lim, spe_lim = model.limits(alpha)
    assert np.mean(t2 > t2_lim) == pytest.approx(alpha, rel=0.15)
    assert np.mean(spe > spe_lim) == pytest.approx(alpha, rel=0.15)


def test_shift_breaking_the_correlation_raises_spe():
    rng = np.random.default_rng(1)
    model = PCAMonitor.fit(correlated_gaussian(20000, rng))
    x = correlated_gaussian(2000, rng)
    x[:, 2] += 1.5  # breaks the latent structure
    _, spe = model.statistics(x)
    assert np.mean(spe > model.limits(0.01)[1]) > 0.5
