"""
Unit tests for src/signal_separation (SignalSeparator).
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.signal_separation import SignalSeparator


def _make_mixed_signals(n_samples=2000, n_components=2, seed=0):
    """Generate synthetic mixed independent sources for testing."""
    rng = np.random.default_rng(seed)
    sources = rng.standard_normal((n_samples, n_components))
    A = np.array([[0.8, 0.4], [0.3, 0.9]])
    return sources @ A.T, sources, A


class TestSignalSeparatorInit:
    def test_default_construction(self):
        sep = SignalSeparator()
        assert sep.n_components == 2

    def test_custom_components(self):
        sep = SignalSeparator(n_components=3)
        assert sep.n_components == 3


class TestFitTransform:
    def test_output_shape(self):
        mixed, _, _ = _make_mixed_signals()
        sep = SignalSeparator(n_components=2)
        out = sep.fit_transform(mixed)
        assert out.shape == (mixed.shape[0], 2)

    def test_components_stored(self):
        mixed, _, _ = _make_mixed_signals()
        sep = SignalSeparator(n_components=2)
        sep.fit_transform(mixed)
        assert sep.components_ is not None
        assert sep.mixing_matrix_ is not None

    def test_reproducibility(self):
        mixed, _, _ = _make_mixed_signals()
        sep1 = SignalSeparator(n_components=2, random_state=7)
        sep2 = SignalSeparator(n_components=2, random_state=7)
        out1 = sep1.fit_transform(mixed)
        out2 = sep2.fit_transform(mixed)
        np.testing.assert_array_almost_equal(np.abs(out1), np.abs(out2), decimal=5)


class TestGetters:
    def setup_method(self):
        mixed, _, _ = _make_mixed_signals()
        self.sep = SignalSeparator(n_components=2)
        self.sep.fit_transform(mixed)

    def test_get_sources(self):
        srcs = self.sep.get_sources()
        assert srcs.shape[1] == 2

    def test_get_mixing_matrix(self):
        M = self.sep.get_mixing_matrix()
        assert M.shape == (2, 2)

    def test_get_unmixing_matrix(self):
        W = self.sep.get_unmixing_matrix()
        assert W.shape == (2, 2)

    def test_not_fitted_raises(self):
        sep = SignalSeparator()
        try:
            sep.get_sources()
            assert False, 'Expected ValueError when calling get_sources before fit_transform'
        except ValueError:
            pass


class TestQualityEstimate:
    def test_high_correlation_for_perfect_reconstruction(self):
        mixed, _, _ = _make_mixed_signals(n_samples=3000)
        sep = SignalSeparator(n_components=2)
        comps = sep.fit_transform(mixed)
        reconstructed = comps @ sep.get_mixing_matrix().T
        quality = sep.estimate_quality(mixed, reconstructed)
        assert quality['correlation'] > 0.95
        assert quality['nmse'] < 0.1
