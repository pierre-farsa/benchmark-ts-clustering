"""
Entry point. Loads config.yaml, creates a timestamped run folder in results/ (logs, weights),
runs the whole benchmark and saves the metrics to CSV
"""

import logging
import warnings
import pandas as pd

from pathlib import Path
from datetime import datetime
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from numba.core.errors import NumbaPerformanceWarning

from config import build_config
from runner import run_benchmark

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*")


def main():
  print("Chargement de la configuration depuis config.yaml...")
  configs, data_dir, results_dir, verbose = build_config("config.yaml")

  print(f"Launch of the benchmark: {len(configs)} possible configurations ready for testing")
  print(f"Source data file: {data_dir}\n")

  timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

  run_output_dir = Path(results_dir) / f"run__{timestamp}"
  run_output_dir.mkdir(parents=True, exist_ok=True)

  log_dir = run_output_dir / "logs"
  log_dir.mkdir(parents=True, exist_ok=True)
  weights_dir = run_output_dir / "weights"
  weights_dir.mkdir(parents=True, exist_ok=True)

  results_df = run_benchmark(configs=configs, data_dir=data_dir, verbose=verbose, log_dir=log_dir, weights_dir=weights_dir)

  output_file = run_output_dir / f"benchmark_results_{timestamp}.csv"
  results_df.to_csv(output_file, index=False)

  latest_file = Path(results_dir) / "benchmark_results_latest_run.csv"
  results_df.to_csv(latest_file, index=False)

  print(f"\nResults saved in: {run_output_dir}")
  print(f"To launch TensorBoard for this run, type the following in the terminal: \ntensorboard --logdir {log_dir}")


if __name__ == "__main__":
  main()