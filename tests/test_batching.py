import numpy as np
import pytest

from fmf.detectors.batching import batch_means, standardize


def test_batch_means_with_gate():
    x = np.arange(12.0)
    gate = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1], dtype=bool)
    means, counts = batch_means(x, 4, gate=gate, min_fill=0.5)
    np.testing.assert_allclose(means, [1.0, np.nan, 9.5])
    np.testing.assert_array_equal(counts, [3, 0, 4])


def test_batch_means_broadcasts_gate_over_replicas():
    x = np.arange(16.0).reshape(8, 2)
    means, counts = batch_means(x, 4, gate=np.ones(8, dtype=bool))
    np.testing.assert_allclose(means, [[3.0, 4.0], [11.0, 12.0]])
    assert counts.shape == (2, 2)


def test_standardized_batch_means_of_ar1_have_unit_variance():
    from fmf.sim.noise import ar1
    from fmf.theory.sigma_eff import ar1_long_run_variance

    x = ar1(40000, 0.5, 1.0, size=50, rng=np.random.default_rng(0))
    means, counts = batch_means(x, 200)
    z = standardize(means, counts, ar1_long_run_variance(1.0, 0.5))
    assert z.var() == pytest.approx(1.0, rel=0.05)
