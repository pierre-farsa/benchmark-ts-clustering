"""
KMeans, Gaussian Mixture and HDBSCAN implementations (HDBSCAN labels noise as -1)
"""

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture

from .base import ClusteringAlgorithm, register_clustering


@register_clustering("kmeans")
class KMeansClustering(ClusteringAlgorithm):
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10, **self.kwargs)
        return self.model.fit_predict(X)


@register_clustering("gmm")
class GMMClustering(ClusteringAlgorithm):
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)

        params = {"reg_covar": 1e-3, "n_init": 10}
        params.update(self.kwargs)

        self.model = GaussianMixture(n_components=self.n_clusters, random_state=self.seed, **params)
        return self.model.fit_predict(X)


@register_clustering("hdbscan")
class HDBSCANClustering(ClusteringAlgorithm):
    """Noise is labeled -1 (see evaluation/metrics.py)"""
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        params = {"min_cluster_size": 5, "min_samples": None}
        params.update(self.kwargs)
        
        self.model = HDBSCAN(**params, copy=True)
        return self.model.fit_predict(X)