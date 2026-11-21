"""
Pipeline wrapping any registered feature extractor
"""

import numpy as np

from features.base import FEATURE_REGISTRY

from .base import ClusteringPipeline, register_pipeline


@register_pipeline("feature_based")
class FeatureBasedPipeline(ClusteringPipeline):
    def __init__(self, extractor_name: str, **extractor_kwargs):
        self.extractor = FEATURE_REGISTRY[extractor_name](**extractor_kwargs)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test=None, y_test=None, log_dir=None, **kwargs) -> "FeatureBasedPipeline":
        self.extractor.fit(X_train)
        return self

    def embed(self, X: np.ndarray) -> np.ndarray:
        return self.extractor.transform(X)
