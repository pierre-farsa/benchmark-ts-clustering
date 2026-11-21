"""
Lightning DataModule handling the train/val split, the dataloaders
and the full-train/test loaders used for embedding extraction
"""

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from preprocessing.datasets import train_valid_split


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int,
        val_size: float,
        seed: int,
        num_workers: int,
    ):
        super().__init__()
        self.X_train_full, self.y_train_full = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.batch_size = batch_size
        self.val_size = val_size
        self.seed = seed
        self.num_workers = num_workers

    def setup(self, stage=None):
        X_tr, X_val, y_tr, y_val = train_valid_split(
            self.X_train_full, self.y_train_full, val_size=self.val_size, seed=self.seed
        )
        self.train_ds = self._to_dataset(X_tr, y_tr)
        self.val_ds = self._to_dataset(X_val, y_val)
        self.full_train_ds = self._to_dataset(self.X_train_full, self.y_train_full)
        self.test_ds = self._to_dataset(self.X_test, self.y_test)

    @staticmethod
    def _to_dataset(X, y) -> TensorDataset:
        return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(np.asarray(y), dtype=torch.long))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)  # (seq_len) -> (B, seq_len)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)  # (seq_len) -> (B, seq_len)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers) # (seq_len) -> (B, seq_len)

    def embedding_dataloader(self, split: str):
        ds = self.full_train_ds if split == "train" else self.test_ds
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
