"""
Log-spectrogram features, flattened, with optional PCA reduction
"""

import numpy as np

from scipy.signal import spectrogram
from sklearn.decomposition import PCA

from .base import FeatureExtractor, register_feature_extractor


@register_feature_extractor("spectrogram")
class SpectrogramFeatures(FeatureExtractor):
  def __init__(self, use_pca: bool, n_components: int, **kwargs):
    self.use_pca = use_pca
    self.n_components = n_components
    self.pca = None

  def fit(self, X: np.ndarray) -> "SpectrogramFeatures":
    if self.use_pca:
      X_feats = self._spectrogram(X)
      n_comp = min(self.n_components, X_feats.shape[0], X_feats.shape[1])
      self.pca = PCA(n_components=n_comp, random_state=42)
      self.pca.fit(X_feats)
    return self
    
  def transform(self, X: np.ndarray) -> np.ndarray:
    X_feats = self._spectrogram(X)
    if self.use_pca and self.pca is not None:
      X_feats = self.pca.transform(X_feats)
    return X_feats

  def _spectrogram(self, X: np.ndarray) -> np.ndarray:
    feats = []
    for row in X:
      row = row.astype(np.float64)
      seq_len = row.shape[0]
      nperseg = max(16, min(256, seq_len // 4))
      noverlap = nperseg // 2

      f, t, row_spec = spectrogram(row, nperseg=nperseg, noverlap=noverlap)

      row_spec_db = 10 * np.log10(row_spec + 1e-10)

      row_feats = row_spec_db.flatten()
      feats.append(row_feats)
      
    feats_array = np.asarray(feats, dtype=np.float32)
    feats_array = np.nan_to_num(feats_array, nan=0.0, posinf=0.0, neginf=0.0)
    return feats_array