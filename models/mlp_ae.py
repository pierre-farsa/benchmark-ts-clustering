"""
Fully connected autoencoder on the raw series
"""

import pytorch_lightning as pl
import torch
from torch import nn

from .base import register_model

class MLPEncoder(nn.Module):
  def __init__(self, seq_len: int, hidden_dims: list, latent_dim: int):
    super().__init__()
    dims = [seq_len, *hidden_dims]
    layers = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
      layers += [nn.Linear(d_in, d_out), nn.ReLU()]
    layers += [nn.Linear(dims[-1], latent_dim)]
    self.net = nn.Sequential(*layers)
  
  def forward(self, x):
    return self.net(x)


class MLPDecoder(nn.Module):
  def __init__(self, latent_dim: int, hidden_dims: list, seq_len: int):
    super().__init__()
    dims = [latent_dim, *hidden_dims]
    layers = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
      layers += [nn.Linear(d_in, d_out), nn.ReLU()]
    layers += [nn.Linear(dims[-1], seq_len)]
    self.net = nn.Sequential(*layers)

  def forward(self, z):
    return self.net(z)


@register_model("mlp_ae")
class MLPAutoencoder(pl.LightningModule):
  def __init__(self, seq_len: int, hidden_dims: list, latent_dim: int, lr: float, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    self.encoder = MLPEncoder(seq_len=seq_len, hidden_dims=hidden_dims, latent_dim=latent_dim)
    self.decoder = MLPDecoder(latent_dim=latent_dim, hidden_dims=tuple(reversed(hidden_dims)), seq_len=seq_len)

  def forward(self, x):
    z = self.encoder(x)
    x_hat = self.decoder(z)
    return z, x_hat
  
  def _step(self, batch, stage: str):
    x, _y = batch
    _z, x_hat = self(x)
    loss = nn.functional.mse_loss(x_hat, x)
    self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
    return loss

  def training_step(self, batch, batch_idx):
    return self._step(batch, "train")

  def validation_step(self, batch, batch_idx):
    return self._step(batch, "val")

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)