#Unit tests for src/preprocessing (ECGPreprocessor).
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import ECGPreprocessor


FS = 500  # Hz


def _sine_signal(freq_hz=10.0, duration_s=2.0, fs=FS):
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t)


class TestECGPreprocessorInit:
    def test_default_construction(self):
        pp = ECGPreprocessor()
        assert pp.fs == 500
        assert pp.lowcut == 0.5
        assert pp.highcut == 40

    def test_custom_params(self):
        pp = ECGPreprocessor(sampling_rate=250, lowcut=1.0, highcut=30)
        assert pp.fs == 250


class TestFilterSignal:
    def test_1d_output_shape(self):
        pp  = ECGPreprocessor(sampling_rate=FS)
        sig = _sine_signal()
        out = pp.filter_signal(sig)
        assert out.shape == sig.shape

    def test_2d_multichannel_output_shape(self):
        pp  = ECGPreprocessor(sampling_rate=FS)
        sig = np.column_stack([_sine_signal(5), _sine_signal(15)])
        out = pp.filter_signal(sig)
        assert out.shape == sig.shape

    def test_highfreq_attenuated(self):
        #A 45 Hz sine should be heavily attenuated by the 40 Hz bandpass."""
        pp  = ECGPreprocessor(sampling_rate=FS, highcut=40)
        sig = _sine_signal(freq_hz=45.0, duration_s=4.0)
        out = pp.filter_signal(sig)
        assert np.std(out) < 0.1 * np.std(sig)

    def test_passband_preserved(self):
        #A 5 Hz sine (well within band) should survive filtering.
        pp  = ECGPreprocessor(sampling_rate=FS, lowcut=0.5, highcut=40)
        sig = _sine_signal(freq_hz=5.0, duration_s=4.0)
        out = pp.filter_signal(sig)
        assert np.std(out) > 0.5 * np.std(sig)


class TestNormalizeSignal:
    def test_zscore_mean_zero(self):
        pp  = ECGPreprocessor()
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = pp.normalize_signal(sig, method='zscore')
        assert abs(np.mean(out)) < 1e-10

    def test_zscore_std_one(self):
        pp  = ECGPreprocessor()
        sig = np.random.randn(500) * 5 + 10
        out = pp.normalize_signal(sig, method='zscore')
        assert abs(np.std(out) - 1.0) < 0.01

    def test_minmax_range(self):
        pp  = ECGPreprocessor()
        sig = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        out = pp.normalize_signal(sig, method='minmax')
        assert abs(out.min()) < 1e-10
        assert abs(out.max() - 1.0) < 1e-10

    def test_invalid_method_raises(self):
        pp = ECGPreprocessor()
        try:
            pp.normalize_signal(np.ones(10), method='invalid')
            assert False, 'Expected ValueError for invalid normalization method'
        except ValueError:
            pass


class TestRemoveBaselineWander:
    def test_output_shape(self):
        pp  = ECGPreprocessor(sampling_rate=FS)
        sig = _sine_signal()
        out = pp.remove_baseline_wander(sig)
        assert out.shape == sig.shape
