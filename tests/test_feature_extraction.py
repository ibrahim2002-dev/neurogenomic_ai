"""
Unit tests for src/feature_extraction (HRVExtractor).
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feature_extraction import HRVExtractor


FS = 500  # Hz


def _synthetic_ecg(heart_rate_bpm=75, duration_s=10, fs=FS, noise_std=0.02, seed=42):
    """Generate a synthetic ECG with Gaussian R-peak pulses."""
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    rr  = 60.0 / heart_rate_bpm
    ecg = np.zeros_like(t)
    for bt in np.arange(0, duration_s, rr):
        ecg += np.exp(-((t - bt) ** 2) / (2 * 0.02 ** 2))
    ecg += noise_std * rng.standard_normal(len(t))
    return ecg


class TestRPeakDetection:
    def test_correct_beat_count_75bpm(self):
        ecg = _synthetic_ecg(heart_rate_bpm=75, duration_s=10)
        ext = HRVExtractor(sampling_rate=FS)
        peaks = ext.detect_r_peaks(ecg)
        # expect ~75 beats in 10 s (allow ±5)
        assert 70 <= len(peaks) <= 80

    def test_correct_beat_count_145bpm(self):
        ecg = _synthetic_ecg(heart_rate_bpm=145, duration_s=10)
        ext = HRVExtractor(sampling_rate=FS)
        peaks = ext.detect_r_peaks(ecg)
        # expect ~145 beats in 10 s (allow ±10)
        assert 135 <= len(peaks) <= 155

    def test_peaks_within_signal_bounds(self):
        ecg = _synthetic_ecg()
        ext = HRVExtractor(sampling_rate=FS)
        peaks = ext.detect_r_peaks(ecg)
        assert peaks.min() >= 0
        assert peaks.max() < len(ecg)


class TestExtractFeatures:
    def setup_method(self):
        self.ecg = _synthetic_ecg(heart_rate_bpm=75, duration_s=15)
        self.ext = HRVExtractor(sampling_rate=FS)
        self.feats = self.ext.extract_features(self.ecg)

    def test_returns_dict(self):
        assert isinstance(self.feats, dict)

    def test_expected_keys_present(self):
        for key in ('num_beats', 'heart_rate_mean', 'heart_rate_std',
                    'rr_interval_mean', 'rr_interval_std', 'rmssd', 'pnn50',
                    'r_peaks', 'rr_intervals'):
            assert key in self.feats, f'Missing key: {key}'

    def test_heart_rate_plausible(self):
        hr = self.feats['heart_rate_mean']
        assert 60 <= hr <= 90

    def test_rmssd_non_negative(self):
        assert self.feats['rmssd'] >= 0

    def test_pnn50_in_percent_range(self):
        assert 0.0 <= self.feats['pnn50'] <= 100.0

    def test_rr_intervals_positive(self):
        assert np.all(self.feats['rr_intervals'] > 0)


class TestEmptyFeatures:
    def test_flat_signal_returns_nan(self):
        ext   = HRVExtractor(sampling_rate=FS)
        feats = ext.extract_features(np.zeros(1000))
        assert feats['num_beats'] == 0
        assert np.isnan(feats['heart_rate_mean'])
        assert np.isnan(feats['rmssd'])
