"""
ClusteringAlgorithm interface (fit_predict / predict) and its registry
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

CLUSTERING_REGISTRY = {}


def register_clustering(name: str):
    """Decorator: saves a clustering algorithm under a name"""

    def deco(cls):
        CLUSTERING_REGISTRY[name] = cls
        return cls

    return deco


class ClusteringAlgorithm(ABC):
    """Common interface for all clustering algorithms in the benchmark
    n_clusters: ignored by algorithms that do not need it (e.g., HDBSCAN),
    but always accepted to maintain a uniform constructor
    """

    def __init__(self, n_clusters: Optional[int], seed: int, **kwargs: Dict[str, Any]):
        self.n_clusters = n_clusters
        self.seed = seed
        self.kwargs = kwargs
        self.model = None

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fits to X and returns the cluster labels for X"""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts clusters for new data (e.g., the test set)
        Override if the algorithm does not have a built-in prediction method (see HDBSCAN)
        """
        if self.model is None or not hasattr(self.model, "predict"):
            raise NotImplementedError(
                f"{type(self).__name__} does not have a native `predict()` method—override this method"
            )
        return self.model.predict(X)
