"""
FeatureExtractor interface (fit / transform) and its registry
"""

from abc import ABC, abstractmethod

import numpy as np

FEATURE_REGISTRY = {}


def register_feature_extractor(name: str):
    """Decorator: saves a feature extractor class under a name"""

    def deco(cls):
        FEATURE_REGISTRY[name] = cls
        return cls

    return deco


class FeatureExtractor(ABC):
    """Common interface: X (N, seq_len) -> features (N, n_components)"""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "FeatureExtractor":
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
