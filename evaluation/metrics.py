"""
Computes external metrics (accuracy via Hungarian matching, ARI, NMI)
and internal metrics (silhouette, Calinski-Harabasz, Davies-Bouldin), excluding HDBSCAN noise
"""

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy after optimal cluster-to-true-class matching (Hungarian
    algorithm), the only correct way to calculate “accuracy” in
    clustering, since cluster indices have no inherent meaning.
    HDBSCAN outliers (label -1) are excluded from the calculation
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_pred != -1
    y_true, y_pred = y_true[mask], y_pred[mask] # exclude HDBSCAN noise
    if len(y_pred) == 0:
        return float("nan")

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost = np.zeros((len(labels_pred), len(labels_true)), dtype=np.int64)
    
    for i, lp in enumerate(labels_pred):
        for j, lt in enumerate(labels_true):
            cost[i, j] = -np.sum((y_pred == lp) & (y_true == lt))

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {labels_pred[r]: labels_true[c] for r, c in zip(row_ind, col_ind)}
    y_pred_mapped = np.array([mapping.get(p, -999) for p in y_pred])
    return float(np.mean(y_pred_mapped == y_true))


def compute_all_metrics(X_embed: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": clustering_accuracy(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "n_clusters_found": int(len(set(y_pred) - {-1})),
        "noise_ratio": float(np.mean(y_pred == -1)),
    }

    mask = y_pred != -1
    n_valid_clusters = len(set(y_pred[mask]))
    if n_valid_clusters > 1 and mask.sum() > n_valid_clusters:
        metrics["silhouette"] = float(silhouette_score(X_embed[mask], y_pred[mask]))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_embed[mask], y_pred[mask]))
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_embed[mask], y_pred[mask]))
    else:
        metrics["silhouette"] = float("nan")
        metrics["calinski_harabasz"] = float("nan")
        metrics["davies_bouldin"] = float("nan")

    return metrics
