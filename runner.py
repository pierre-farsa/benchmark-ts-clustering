"""
Orchestrates the benchmark. For each config: load data → fit pipeline → embed → scale → cluster → compute metrics.
Importing it registers all implementations
"""

import traceback
from typing import List
from tqdm import tqdm

import pandas as pd

import features.handcrafted
import features.rms
import features.spectrogram
import models.mlp_ae
import models.conv1d_ae
import models.conv2d_ae
import clustering.algorithms
import pipelines.feature_based
import pipelines.autoencoder_based
import pipelines.series2vec_based

from clustering.base import CLUSTERING_REGISTRY
from config import ExperimentConfig
from preprocessing.datasets import load_dataset, load_dataset_merged
from preprocessing.scalers import get_scaler
from evaluation.metrics import compute_all_metrics
from pipelines.base import PIPELINE_REGISTRY


def run_experiment(config: ExperimentConfig, data_dir: str, verbose: bool, log_dir=None, weights_dir=None) -> dict:
    """Runs a single benchmark test and returns its metrics"""
    if config.merge_train_test:
        X_train, y_train, X_test, y_test = load_dataset_merged(
            config.dataset_name,
            data_dir,
            test_size=config.merge_test_size,
            seed=config.seed
        )
    else:
        X_train, y_train, X_test, y_test = load_dataset(
          name=config.dataset_name,
          data_dir=data_dir,
          test_size=config.merge_test_size,
          seed=config.seed
          )

    # Embedding (feature-based or autoencoder—either term will be used interchangeably hereafter)
    pipeline_cls = PIPELINE_REGISTRY[config.pipeline_name]
    pipeline = pipeline_cls(**config.pipeline_params)
    pipeline.fit(X_train, y_train, X_test, y_test, log_dir=log_dir, weights_dir=weights_dir, run_name=config.run_name())  # log_dir, weights_dir, run_name are ignored by feature_based

    Z_train = pipeline.embed(X_train)
    Z_test = pipeline.embed(X_test)

    # Scaling the l'embedding
    scaler = get_scaler(config.scaler_name)
    if scaler is not None:
        Z_train = scaler.fit_transform(Z_train)
        Z_test = scaler.transform(Z_test)

    # Clustering: fitting on the training set, prediction on the test set
    n_clusters = config.n_clusters or len(set(y_train))
    clustering_cls = CLUSTERING_REGISTRY[config.clustering_name]
    clustering = clustering_cls(n_clusters=n_clusters, seed=config.seed, **config.clustering_params)
    clustering.fit_predict(Z_train)

    try:
        y_pred_test = clustering.predict(Z_test)
    except NotImplementedError:
        # Algorithms without native prediction (e.g., certain variants) -> direct `fit_predict` on the test
        y_pred_test = clustering.fit_predict(Z_test)

    # Test evaluation, using the actual label (never used before)
    metrics = compute_all_metrics(Z_test, y_test, y_pred_test)
    metrics.update(
        {
            "dataset": config.dataset_name,
            "pipeline": config.display_name,
            "scaler": config.scaler_name,
            "clustering": config.clustering_name,
            "run_name": config.run_name(),
        }
    )

    tqdm.write(f"[RES] {config.run_name()} -> acc={metrics["accuracy"]:.3f}  ari={metrics["ari"]:.3f}  nmi={metrics["nmi"]:.3f}  silhouette={metrics["silhouette"]:.3f}  db={metrics["davies_bouldin"]:.3f}  ch={metrics["calinski_harabasz"]:.3f}")
    return metrics


def run_benchmark(configs: List[ExperimentConfig], data_dir: str, verbose: bool, log_dir=None, weights_dir=None) -> pd.DataFrame:
    """Runs all benchmark configurations"""
    results = []
    pbar = tqdm(configs, desc="Benchmark execution")

    for config in pbar:
        pbar.set_postfix(
          ds=config.dataset_name,
          pipe=config.display_name,
          scale=config.scaler_name,
          clust=config.clustering_name
        )

        try:
          res = run_experiment(config=config, data_dir=data_dir, verbose=verbose, log_dir=log_dir, weights_dir=weights_dir)
          results.append(res)
        except Exception as e:
          tqdm.write(f"[ERROR] Configuration error '{config.run_name()}' : {e}")
          if verbose:
            tqdm.write(traceback.format_exc())

    return pd.DataFrame(results)
