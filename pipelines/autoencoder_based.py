"""
Trains a registered autoencoder with Lightning (early stopping, checkpointing, TensorBoard)
and returns the latent space as embedding
"""

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.base import MODEL_REGISTRY
from models.datamodule import TimeSeriesDataModule

from .base import ClusteringPipeline, register_pipeline


@register_pipeline("ae_based")
class AutoencoderPipeline(ClusteringPipeline):
    def __init__(
        self,
        model_name: str,
        latent_dim: int,
        max_epochs: int,
        batch_size: int,
        lr: float,
        val_size: float,
        patience: int,
        seed: int,
        accelerator: str,
        num_workers: int,
        **kwargs  # hidden_dims, in_channels, kernel_size etc.
    ):
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_size = val_size
        self.patience = patience
        self.seed = seed
        self.accelerator = accelerator
        self.num_workers = num_workers

        self.model_params = kwargs

        self.datamodule = None
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test=None, y_test=None, log_dir=None, weights_dir=None, run_name=None, **kwargs) -> "AutoencoderPipeline":
        pl.seed_everything(self.seed, verbose=False)

        self.datamodule = TimeSeriesDataModule(
          X_train, y_train, X_test, y_test,
          batch_size=self.batch_size,
          val_size=self.val_size,
          seed=self.seed,
          num_workers=self.num_workers
        )

        if self.model_name not in MODEL_REGISTRY:
          raise ValueError(f"Unknown model: '{self.model_name}'. Registered: {list(MODEL_REGISTRY)}")
        
        model_cls = MODEL_REGISTRY[self.model_name]

        self.model = model_cls(
          seq_len=X_train.shape[1],
          latent_dim=self.latent_dim,
          lr=self.lr,
          **self.model_params
        )

        save_logs_path = str(log_dir) if log_dir else "lightning_logs"
        logger = TensorBoardLogger(save_dir=save_logs_path, name=f"logs__{run_name}")
        save_weights_path = str(weights_dir) if weights_dir else "weights"

        checkpoint = ModelCheckpoint(
          dirpath=save_weights_path,
          filename=f"weights__{run_name}",
          monitor="val_loss",
          save_top_k=1,
          mode="min"
        )

        early_stopping = EarlyStopping(
          monitor="val_loss",
          patience=self.patience,
          mode="min"
        )

        trainer = pl.Trainer(
          max_epochs=self.max_epochs,
          accelerator=self.accelerator,
          logger=logger,
          enable_checkpointing=True,
          enable_progress_bar=False,
          enable_model_summary=False,
          callbacks=[checkpoint, early_stopping]
        )

        trainer.fit(self.model, datamodule=self.datamodule)
        self.model.eval()
        return self


    def embed(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).to(self.model.device)
            z, _x_hat = self.model(x_t)
        return z.cpu().numpy()
