"""
1D convolutional autoencoder on the raw series
"""

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from .base import register_model


class Conv1dEncoder(nn.Module):
  def __init__(
    self,
    in_channels: int, hidden_dims: list, latent_dim: int,
    kernel_size: int, stride: int, padding: int,
    seq_len: int
  ):
    super().__init__()
    dims = [in_channels, *hidden_dims]

    self.encoder = nn.Sequential(
      nn.Conv1d(in_channels=dims[0], out_channels=dims[1], kernel_size=kernel_size, stride=stride, padding=padding),  # e.g. (B, 1, seq_len) -> (B, 32, seq_len // 2)
      nn.ReLU(),
      nn.Conv1d(in_channels=dims[1], out_channels=dims[2], kernel_size=kernel_size, stride=stride, padding=padding),  # e.g. (B, 32, seq_len // 2) 6> (B, 64, seq_len // 4)
      nn.ReLU(),
      nn.Conv1d(in_channels=dims[2], out_channels=dims[3], kernel_size=kernel_size, stride=stride, padding=padding),  # e.g. (B, 64, seq_len // 4) -> (B, 128, seq_len // 8)
      nn.ReLU() 
    )

    # Passage à vide pour que PyTorch détermine la taille exacte de la sortie
    with torch.no_grad():
      dummy_input = torch.zeros(1, in_channels, seq_len)
      dummy_output = self.encoder(dummy_input)
      self.output_shape = dummy_output.shape[1:]  # récupère (canaux, longueur)
      flatten_size = dummy_output.shape[1] * dummy_output.shape[2]

    self.norm = nn.LayerNorm(flatten_size)

    self.fc = nn.Linear(in_features=flatten_size, out_features=latent_dim)
  
  def forward(self, x):
    x = self.encoder(x) # e.g. (B, 1, seq_len) -> (B, 128, _)
    x = x.flatten(1)    # e.g. (B, 128, _) -> (B, flatten_size)
    x = self.norm(x)
    z = self.fc(x)      # e.g. (B, flatten_size) -> (B, latent_dim)
    return z
  

class Conv1dDecoder(nn.Module):
  def __init__(
    self,
    latent_dim: int, hidden_dims: list, in_channels: int,
    kernel_size: int, stride: int, padding: int, output_padding: int,
    encoder_shape: tuple
  ):
    super().__init__()
    self.encoder_shape = encoder_shape
    self.stride = stride
    self.dims = [in_channels, *hidden_dims]

    flatten_size = self.encoder_shape[0] * self.encoder_shape[1]
    self.fc = nn.Linear(latent_dim, flatten_size)

    self.decoder = nn.Sequential(
      nn.ConvTranspose1d(in_channels=self.dims[3], out_channels=self.dims[2], kernel_size=kernel_size, stride=self.stride, padding=padding, output_padding=output_padding), # e.g. (B, 128, seq_len // 8) -> (B, 64, seq_len // 4)
      nn.ReLU(),
      nn.ConvTranspose1d(in_channels=self.dims[2], out_channels=self.dims[1], kernel_size=kernel_size, stride=self.stride, padding=padding, output_padding=output_padding), # e.g. (B, 64, seq_len // 4) -> (B, 32, seq_len // 2)
      nn.ReLU(),
      nn.ConvTranspose1d(in_channels=self.dims[1], out_channels=self.dims[0], kernel_size=kernel_size, stride=self.stride, padding=padding, output_padding=output_padding), # e.g. (B, 32, seq_len // 2) -> (B, 1, seq_len)
    )

  def forward(self, z):
    z = self.fc(z)  # e.g. (B, latent_dim) -> (B, flatten_size)
    z = z.view(z.size(0), *self.encoder_shape)  # e.g. (B, flatten_size) -> (B, 128, _)
    x_hat = self.decoder(z) # e.g. (B, 128, _) -> (B, 1, seq_len)
    return x_hat

  
@register_model("conv1d_ae")
class Conv1dAutoencoder(pl.LightningModule):
  def __init__(
    self,
    in_channels: int, hidden_dims: list, latent_dim: int,
    kernel_size: int, stride: int, padding: int, output_padding: int,
    seq_len: int, lr: float,
    **kwargs
  ):
    super().__init__()

    self.save_hyperparameters()
    self.seq_len = seq_len
    self.lr = lr

    self.encoder = Conv1dEncoder(in_channels, hidden_dims, latent_dim, kernel_size, stride, padding, self.seq_len)
    self.decoder = Conv1dDecoder(latent_dim, hidden_dims, in_channels, kernel_size, stride, padding, output_padding, self.encoder.output_shape)

    self.loss = nn.MSELoss()

  def forward(self, x):
    if x.ndim == 2:
      x = x.unsqueeze(1)

    z = self.encoder(x)
    x_hat = self.decoder(z)

    if x_hat.size(-1) > self.seq_len:
      x_hat = x_hat[:, :, :self.seq_len]
    
    elif x_hat.size(-1) < self.seq_len:
      pad = self.seq_len - x_hat.size(-1)
      x_hat = F.pad(x_hat, (0, pad))

    x_hat = x_hat.squeeze(1)
    return z, x_hat

  def training_step(self, batch, batch_idx):
    x, _y = batch
    _z, x_hat = self(x)
    loss = self.loss(x_hat, x)
    self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
    return loss
  
  def validation_step(self, batch, batch_idx):
    x, _y = batch
    _z, x_hat = self(x)
    loss = self.loss(x_hat, x)
    self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)