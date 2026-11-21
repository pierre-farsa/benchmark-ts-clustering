"""
ClusteringPipeline interface (fit / embed) and its registry. This is the main extension point
"""

from abc import ABC, abstractmethod

import numpy as np

PIPELINE_REGISTRY = {}


def register_pipeline(name: str):
    """Decorator: saves an embedding pipeline under a name.
    This is the main extension point of the benchmark: any new
    approach (manual features—variant X, autoencoder—variant Y, hybrid
    method, pre-trained embeddings...) becomes available in runner.py /
    the notebook simply by creating a class here and decorating it.
    """

    def deco(cls):
        PIPELINE_REGISTRY[name] = cls
        return cls

    return deco


class ClusteringPipeline(ABC):
    """A common process for all approaches: raw signal → embedding.

    Clustering (algorithm + scaler) is then applied in exactly
    the same way to the embedding, regardless of which pipeline produced it.
    It is this decoupling that allows us to compare feature-based and autoencoder
    methods using the same clustering/evaluation code.
    """

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None, **kwargs) -> "ClusteringPipeline":
        """y_train/y_test are never used for the training itself
        (neither the manually selected features nor the autoencoder are supervised) —
        they are only accepted in the signature for pipelines that
        technically require them (e.g., TensorDataset from DataModule)
        """
        ...

    @abstractmethod
    def embed(self, X: np.ndarray) -> np.ndarray:
        ...
