"""
17 classic features per series (statistics, shape, trend, spectral energy, autocorrelation, zero-crossings, peaks)
"""
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch

from .base import FeatureExtractor, register_feature_extractor


@register_feature_extractor("handcrafted")
class HandcraftedFeatures(FeatureExtractor):
    """Calculation of traditional features: basic statistics + shape + frequency"""
    def __init__(self, **kwargs):
        pass

    def fit(self, X: np.ndarray) -> "HandcraftedFeatures":
        return self  # pas d'état à apprendre

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for row in X:
            row = row.astype(np.float64)
            diff = np.diff(row)

            nperseg = min(256, len(row))
            freqs, psd = welch(row, nperseg=nperseg)
            fft_energy = np.sum(psd)
            dominant_freq = float(freqs[np.argmax(psd[1:]) + 1]) if len(psd) > 1 else 0.0

            feats.append(
                [
                    row.mean(),
                    row.std(),
                    row.min(),
                    row.max(),
                    np.sqrt(np.mean(row ** 2)),                 # rms

                    skew(row),
                    kurtosis(row),

                    np.ptp(row),
                    diff.mean() if len(diff) else 0.0,          # average trend
                    diff.std() if len(diff) else 0.0,           # volatilité locale
                    np.sum(np.abs(diff)),                       # total length of the curve

                    fft_energy,
                    dominant_freq,

                    _autocorr(row, lag=1),                      # short-term autocorrelation
                    _autocorr(row, lag=max(1, len(row) // 10)), # long-term autocorrelation
                    _zcr(row),                                  # zero-crossing-rate
                    _n_peaks(row),                              # number of local maxima
                ]
            )
        
        feats_array = np.asarray(feats, dtype=np.float32)
        feats_array = np.nan_to_num(feats_array, nan=0.0, posinf=0.0, neginf=0.0)
        return feats_array


def _autocorr(row: np.ndarray, lag: int) -> float:
    if lag >= len(row):
        return 0.0
    a, b = row[:-lag], row[lag:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _zcr(row: np.ndarray) -> float:
    centered = row - row.mean()
    return float(np.mean(np.diff(np.sign(centered)) != 0))


def _n_peaks(row: np.ndarray) -> int:
    return int(np.sum((row[1:-1] > row[:-2]) & (row[1:-1] > row[2:])))
