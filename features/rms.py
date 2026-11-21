"""
Single-feature baseline using the root mean square of each series
"""

import numpy as np
from .base import FeatureExtractor, register_feature_extractor

@register_feature_extractor("rms")
class RMSFeatures(FeatureExtractor):
  def __init__(self, **kwargs):
    pass
  
  def fit(self, X: np.ndarray) -> "RMSFeatures":
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    feats = []
    for row in X:
      row = row.astype(np.float64)
      rms = np.sqrt(np.mean(row ** 2))
      feats.append(rms)
    
    feats_array = np.asarray(feats, dtype=np.float32).reshape(-1, 1)
    return feats_array