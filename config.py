"""
Defines ExperimentConfig and build_config(), which parses config.yaml
and expands it into one config per dataset × pipeline × scaler × clustering combination.
"""

import yaml
import itertools

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

from features.base import FEATURE_REGISTRY
from models.base import MODEL_REGISTRY


@dataclass
class ExperimentConfig:
  dataset_name: str

  pipeline_name: str
  display_name: str
  scaler_name: str
  clustering_name: str

  merge_train_test: bool
  merge_test_size: float

  seed: int

  n_clusters: Optional[int] = None

  pipeline_params: Dict[str, Any] = field(default_factory=dict)
  clustering_params: Dict[str, Any] = field(default_factory=dict)

  def run_name(self) -> str:
    return f"{self.dataset_name}__{self.display_name}__{self.scaler_name}__{self.clustering_name}"


def load_yaml_config(yaml_path: str = "config.yaml") -> Dict[str, Any]:
  """Load the raw content of the YAML file"""
  path = Path(yaml_path)
  if not path.exists():
    raise FileNotFoundError(f"Config file '{yaml_path}' not found")

  with open(path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


def build_config(yaml_path: str = "config.yaml") -> Tuple[list[ExperimentConfig], str, str, bool]:
  """Returns:
    - a list of all ExperimentConfig objects
    - the path to the raw data folder (data_dir)
    - the path to the results folder (results_dir)
    - the boolean value verbose (detailed error output in the console)
  """
  cfg = load_yaml_config(yaml_path)

  paths = cfg.get("paths", {})
  base_dir = Path(paths.get("base_dir", "."))
  data_dir = str(base_dir / paths.get("data_dir", "data"))
  results_dir = str(base_dir / paths.get("results_dir", "results"))

  settings = cfg.get("settings", {})
  merge_train_test = settings.get("merge_train_test", "True")
  merge_test_size = settings.get("merge_test_size", 0.2)
  seed = settings.get("seed", 42)
  verbose = settings.get("verbose", True)

  datasets = cfg.get("datasets", {})
  pipelines = cfg.get("pipelines", {})
  scalers = cfg.get("scalers", [])
  clusterings = cfg.get("clusterings", {})

  configs = []

  for (dataset_name, k), (alias, pipeline_params), scaler_name, (clustering_name, clustering_params) in itertools.product(
    datasets.items(),
    pipelines.items(),
    scalers,
    clusterings.items()
  ):
    p_params_copy = dict(pipeline_params)

    if alias in FEATURE_REGISTRY:   # case 1: it is an extractor (e.g., feature-based)
      backend_pipeline = "feature_based"
      p_params_copy["extractor_name"] = alias
      display_name = alias
    elif alias in MODEL_REGISTRY:   # case 2: it is an autoencoder (e.g. ae_based)
      backend_pipeline = "ae_based"
      p_params_copy ["model_name"] = alias
      display_name = alias
    else:
      backend_pipeline = p_params_copy.pop("pipeline_name", alias)
      display_name = alias
    
    configs.append(
      ExperimentConfig(
        dataset_name=dataset_name,
        pipeline_name=backend_pipeline,
        display_name=display_name,
        scaler_name=scaler_name,
        clustering_name=clustering_name,
        n_clusters=k,
        merge_train_test=merge_train_test,
        merge_test_size=merge_test_size,
        pipeline_params=p_params_copy,
        clustering_params=clustering_params,
        seed=seed
      )
    )
  
  return configs, data_dir, results_dir, verbose