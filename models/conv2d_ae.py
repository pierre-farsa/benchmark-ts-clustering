"""
2D convolutional autoencoder working on a resized log-spectrogram computed in torch
"""

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from .base import register_model


class Conv2dEncoder(nn.Module):
  def __init__(
    self,
    in_channels: int, hidden_dims: list, latent_dim: int,
    kernel_size: int, stride: int, padding: int,
    input_dim: list
  ):
    super().__init__()
    H, W = input_dim[0], input_dim[1]
    dims = [in_channels, *hidden_dims]

    assert H % 8 == 0 and W % 8 == 0, ("H et W doivent être divisibles par 8 (3 stride-2 pooling)")

    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels=dims[0], out_channels=dims[1], kernel_size=kernel_size, padding=padding),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=stride, stride=stride),  # e.g. (B, 1, H, W) -> (B, 16, H//2, W//2)
      nn.Conv2d(in_channels=dims[1], out_channels=dims[2], kernel_size=kernel_size, padding=padding),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=stride, stride=stride),  # e.g. (B, 16, H//2, W//2) -> (B, 32, H//4, W//4)
      nn.Conv2d(in_channels=dims[2], out_channels=dims[3], kernel_size=kernel_size, padding=padding),
      nn.ReLU() ,
      nn.MaxPool2d(kernel_size=stride, stride=stride),  # e.g. (B, 32, H//4, W//4) -> (B, 64, H//8, W//8)
    )

    self.fc = nn.Linear(in_features=dims[3] * H//8 * W//8, out_features=latent_dim)
  
  def forward(self, x):
    x = self.encoder(x) # e.g. (B, 1, H, W -> (B, 64, H//8, W//8)
    x = x.flatten(1)    # e.g. (B, 64, H//8, W//8) -> (B, 64 * H//8 * W//8)
    z = self.fc(x)      # e.g. (B, 64 * H//8 * W//8) -> (B, latent_dim)
    return z
  

class Conv2dDecoder(nn.Module):
  def __init__(
    self,
    latent_dim: int, hidden_dims: list, in_channels: int,
    kernel_size: int, stride: int,
    input_dim: list
  ):
    super().__init__()
    H, W = input_dim[0], input_dim[1]
    self.dims = [in_channels, *hidden_dims]
    self.pooled_H, self.pooled_W = H//8, W//8

    self.fc = nn.Linear(in_features=latent_dim, out_features=self.dims[3] * H//8 * W//8)

    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(in_channels=self.dims[3], out_channels=self.dims[2], kernel_size=kernel_size, stride=stride), # e.g. (B, 64, H//8, W//8) -> (B, 32, H//4, W//4)
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=self.dims[2], out_channels=self.dims[1], kernel_size=kernel_size, stride=stride), # e.g. (B, 32, H//4, W//4) -> (B, 16, H//2, W//2)
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=self.dims[1], out_channels=self.dims[0], kernel_size=kernel_size, stride=stride), # e.g. (B, 16, H//2, W//2) -> (B, 1, H, W)
    )

  def forward(self, z):
    z = self.fc(z)  # e.g. (B, latent_dim) -> (B, 64 * H//8 * W//8)
    z = z.view(z.size(0), self.dims[3], self.pooled_H, self.pooled_W)  # e.g. (B, 64 * H//8 * W//8) -> (B, 64, H//8, W//8)
    x_hat = self.decoder(z) # e.g. (B, 64, H//8, W//8) -> (B, 1, H, W)
    return x_hat

  
@register_model("conv2d_ae")
class Conv2dAutoencoder(pl.LightningModule):
  def __init__(
    self,
    in_channels: int, hidden_dims: list, latent_dim: int,
    kernel_size: int, stride: int, padding: int,
    input_dim: list, lr: float,
    **kwargs
  ):
    super().__init__()

    self.save_hyperparameters()
    self.H, self.W = input_dim
    self.lr = lr

    self.encoder = Conv2dEncoder(in_channels, hidden_dims, latent_dim, kernel_size, stride, padding, input_dim)
    self.decoder = Conv2dDecoder(latent_dim, hidden_dims, in_channels, kernel_size, stride, input_dim)

    self.loss = nn.MSELoss()
  
  def _torch_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = x.shape

    n_fft = int(max(16, min(256, seq_len // 4)))
    hop_length = max(1, n_fft // 2)

    window = torch.hann_window(n_fft, device=x.device)

    stft = torch.stft(
      x,
      n_fft=n_fft,
      hop_length=hop_length,
      win_length=n_fft,
      window=window,
      return_complex=True
    )

    spec = torch.abs(stft)
    spec_db = 10 * torch.log10(spec + 1e-10)
    spec_db = spec_db.unsqueeze(1)  # (B, n_freqs, n_times) -> (B, 1, n_freqs, n_times)

    spec_resized = F.interpolate(spec_db, size=(self.H, self.W), mode="bilinear", align_corners=False)  # (B, 1, n_freqs, n_times) -> (B, 1, H, W)

    return spec_resized

  def forward(self, x):
    x_spec = self._torch_spectrogram(x) # (B, seq_len) -> (B, 1, H, W)

    z = self.encoder(x_spec)
    x_hat = self.decoder(z)

    if x_hat.shape != x_spec.shape:
      x_hat = F.interpolate(x_hat, size=(self.H, self.W), mode="bilinear", align_corners=False)
    
    return z, x_hat

  def training_step(self, batch, batch_idx):
    x, _y = batch
    _z, x_hat = self(x)
    x_spec = self._torch_spectrogram(x)
    loss = self.loss(x_hat, x_spec)
    self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
    return loss
  
  def validation_step(self, batch, batch_idx):
    x, _y = batch
    _z, x_hat = self(x)
    x_spec = self._torch_spectrogram(x)
    loss = self.loss(x_hat, x_spec)
    self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.lr)