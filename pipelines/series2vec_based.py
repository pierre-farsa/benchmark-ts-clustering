"""
Trains Series2Vec, saves/loads its weights and returns its representations as embedding
"""

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset

from models.series2vec_model.Series2Vec import Series2Vec
from models.series2vec_model.s2v_training import S2V_SS_Trainer, S2V_make_representation, SS_train_runner
from models.series2vec_model.optimizers import get_optimizer
from .base import ClusteringPipeline, register_pipeline


class S2VDataset(Dataset):
  def __init__(self, X: np.ndarray, y: np.ndarray):
    self.X = torch.tensor(X, dtype=torch.float32)
    if self.X.ndim == 2:
      self.X = self.X.unsqueeze(1)
    self.y = torch.tensor(np.asarray(y), dtype=torch.long)

  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, idx):
    return self.X[idx], self.y[idx], torch.tensor(idx)

  
@register_pipeline("series2vec")
class Series2VecPipeline(ClusteringPipeline):
  def __init__(
    self,
    emb_size: int,
    num_heads: int,
    dim_ff: int,
    rep_size: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer_name: str,
    seed: int,
    accelerator: str,
    **kwargs
  ):
    self.emb_size = emb_size
    self.num_heads = num_heads
    self.dim_ff = dim_ff
    self.rep_size = rep_size
    self.layers = layers
    self.dropout = dropout
    self.epochs = epochs
    self.batch_size = batch_size
    self.lr = lr
    self.optimizer_name = optimizer_name
    self.seed = seed

    if accelerator == "auto":
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
      self.device = torch.device(accelerator)
      
    self.model = None

  def fit(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    log_dir: str = None,
    weights_dir: str = None,
    run_name: str = None,
    **kwargs
  ) -> "Series2VecPipeline":

    torch.manual_seed(self.seed)
    np.random.seed(self.seed)

    train_ds = S2VDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False, drop_last=True)

    sample_shape = train_ds[0][0].shape
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
    config = {
      "data_shape": (self.batch_size, sample_shape[0], sample_shape[1]),
      "emb_size": self.emb_size,
      "num_heads": self.num_heads,
      "dim_ff": self.dim_ff,
      "rep_size": self.rep_size,
      "layers": self.layers,
      "dropout": self.dropout,
      "epochs": self.epochs,
      "device": self.device,
      "output_dir": "results",
      "log_dir": log_dir,
      "weights_dir": weights_dir,
      "run_name": run_name,
      "problem": "benchmark"
    }

    num_classes = len(np.unique(y_train))
    self.model = Series2Vec(config, num_classes=num_classes).to(self.device)

    opt_cls = get_optimizer(self.optimizer_name)
    optimizer = opt_cls(self.model.parameters(), lr=self.lr)
    config["optimizer"] = optimizer
    config["loss_module"] = None
      
    trainer = S2V_SS_Trainer(
      model=self.model,
      train_loader=train_loader,
      test_loader=test_loader,
      config=config,
      optimizer=optimizer
    )

    SS_train_runner(config=config, model=self.model, trainer=trainer)

    self.model.eval()
    return self

  def load(self, weights_file, X_sample, num_classes):
    if not weights_file.exists():
      raise FileNotFoundError(f"File not found {weights_file}")

    seq_len = X_sample.shape[-1]

    config = {
      "data_shape": (1, 1, seq_len),
      "emb_size": self.emb_size,
      "num_heads": self.num_heads,
      "dim_ff": self.dim_ff,
      "rep_size": self.rep_size,
      "layers": self.layers,
      "dropout": self.dropout
    } 

    self.model = Series2Vec(config, num_classes).to(self.device)

    checkpoint = torch.load(weights_file, map_location=self.device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    self.model.load_state_dict(state_dict)
    self.model.eval()

    return self
  
  def embed(self, X: np.ndarray) -> np.ndarray:
    eval_ds = S2VDataset(X, np.zeros(len(X)))
    eval_loader = DataLoader(eval_ds, batch_size=self.batch_size, shuffle=False)

    with torch.no_grad():
      rep, _ = S2V_make_representation(self.model.to(self.device), eval_loader)

    return rep.cpu().numpy()