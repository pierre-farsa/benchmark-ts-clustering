# benchmark-ts-clustering

Benchmark of time series clustering approaches. It compares several **embeddings** (handcrafted features, autoencoders, Series2Vec) combined with several **scalers** and **clustering algorithms** (KMeans, Gaussian Mixture, HDBSCAN), on univariate datasets from the UCR archive.

## How it works

```
raw signal → pipeline (embedding) → scaler → clustering → metrics
```

- Every experiment is one combination `dataset × pipeline × scaler × clustering`, generated from `config.yaml`.
- The embedding is fitted on the train set, then the clustering is fitted on the train embedding and applied to the test embedding (HDBSCAN has no native `predict`, so it is fitted directly on the test set).
- **Labels are never used for training**, only for evaluation (accuracy with Hungarian matching, ARI, NMI) next to internal metrics (silhouette, Calinski-Harabasz, Davies-Bouldin).

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<your-user>/benchmark-ts-clustering.git
cd benchmark-ts-clustering
uv sync
```

> `torch` is installed with its CUDA libraries by default on Linux (several GB). The benchmark runs on CPU, except Series2Vec (see [Known limitations](#known-limitations)).

## Data

Download the datasets from [timeseriesclassification.com](https://www.timeseriesclassification.com/) and place the `.arff` files directly in `data/`, using this naming convention:

```
data/{NAME}_TRAIN.arff
data/{NAME}_TEST.arff
```

The example config uses **SonyAIBORobotSurface1** (621 series, length 70, 2 classes). Only univariate datasets are supported.

## Usage

1. Edit `config.yaml`: datasets (with their number of classes), pipelines and their hyperparameters, scalers, clustering algorithms. Comment or uncomment entries to disable or enable them.
2. Run the benchmark:

   ```bash
   uv run python main.py
   ```

3. Results are written to `results/run__<timestamp>/`:
   - `benchmark_results_<timestamp>.csv`: one row of metrics per experiment (a copy is also saved as `results/benchmark_results_latest_run.csv`);
   - `logs/`: TensorBoard logs of the neural networks;
   - `weights/`: best checkpoints of the neural networks.

4. Analyse the results with `notebook_results.ipynb` (select the project `.venv` as kernel): best methods per dataset, 3D UMAP visualisation of the clusters, TensorBoard.

   Set `CSV_PATH` to the timestamped CSV of the run, not to `latest_run`, so the notebook can find the `weights/` folder.

## Project structure

```
main.py                    Entry point: loads the config, runs the benchmark, saves the results
config.py                  ExperimentConfig + config.yaml parsing (one config per combination)
runner.py                  Orchestrates data → pipeline → scaler → clustering → metrics
config.yaml                Experiment definition
notebook_results.ipynb     Results analysis

preprocessing/
  datasets.py              .arff loading, label encoding, train/test merge and split
  scalers.py               Scalers registry (standard, minmax, none)
features/
  base.py                  FeatureExtractor interface and registry
  handcrafted.py           17 statistical / shape / spectral features per series
  rms.py                   Single-feature baseline (root mean square)
  spectrogram.py           Log-spectrogram features with optional PCA
models/
  base.py                  Neural network registry
  datamodule.py            Lightning DataModule (train/val split, loaders)
  mlp_ae.py                Fully connected autoencoder
  conv1d_ae.py             1D convolutional autoencoder
  conv2d_ae.py             2D convolutional autoencoder on a log-spectrogram
  series2vec_model/        Series2Vec network, training loop, Soft-DTW, optimizers
pipelines/
  base.py                  ClusteringPipeline interface and registry (main extension point)
  feature_based.py         Handcrafted features pipeline
  autoencoder_based.py     Autoencoder training (early stopping, checkpoints) + latent embedding
  series2vec_based.py      Series2Vec training + representation extraction
clustering/
  base.py                  ClusteringAlgorithm interface and registry
  algorithms.py            KMeans, GMM, HDBSCAN (noise labelled -1)
evaluation/
  metrics.py               External and internal clustering metrics
```

## Adding a method

Every component is registered by name with a decorator, so no other part of the code has to change:

| To add…                  | Subclass / write                         | Decorator                           |
| ------------------------ | ---------------------------------------- | ----------------------------------- |
| a feature extractor      | `FeatureExtractor` in `features/`        | `@register_feature_extractor("x")`  |
| an autoencoder           | a `LightningModule` in `models/`         | `@register_model("x")`              |
| a clustering algorithm   | `ClusteringAlgorithm` in `clustering/`   | `@register_clustering("x")`         |
| a new kind of pipeline   | `ClusteringPipeline` in `pipelines/`     | `@register_pipeline("x")`           |

Then import the module in `runner.py` (this triggers the registration) and add the name, with its hyperparameters, to `config.yaml`.

## Known limitations

- **Series2Vec requires an NVIDIA GPU**: CUDA is hard-coded in its training and embedding code. It is commented out in `config.yaml` by default.
- **`rms` is degenerate on UCR datasets**: they are already z-normalised per series, so the feature is almost constant. It is kept as an "uninformative" baseline.
- **HDBSCAN accuracy excludes noise points**: always read it together with `noise_ratio` and `n_clusters_found` in the CSV.
- Each experiment retrains its pipeline, so a neural network is trained once per `scaler × clustering` combination.

## Credits and licenses

Part of the code is adapted from external projects. Please refer to their repositories for the original licenses and keep their copyright headers:

- **Series2Vec** (`models/series2vec_model/`): Foumani et al., *Series2Vec: Similarity-based Self-supervised Representation Learning for Time Series Classification* (https://github.com/Navidfoumani/Series2Vec).
- **pytorch-softdtw-cuda** (`soft_dtw_cuda.py`): Soft-DTW CUDA implementation by M. Maghoumi (https://github.com/Maghoumi/pytorch-softdtw-cuda).
- **RAdam** (`optimizers.py`): L. Liu et al. (https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam/radam.py)

Datasets come from the [UCR/UEA time series classification archive](https://www.timeseriesclassification.com/).
