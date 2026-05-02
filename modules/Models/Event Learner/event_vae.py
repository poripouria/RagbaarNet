import torch
from torch import nn
from torch.nn import functional as F


class SequenceVAE(nn.Module):
	"""Generic VAE for fixed-size vectors (e.g., flattened driving/segmentation sequences).

	This is a minimal baseline to keep the music + event VAEs structurally similar.
	"""

	def __init__(
		self,
		input_dim: int,
		latent_dim: int = 64,
		hidden_dims: tuple[int, ...] = (512, 256),
	) -> None:
		super().__init__()
		if input_dim <= 0:
			raise ValueError("input_dim must be > 0")
		if latent_dim <= 0:
			raise ValueError("latent_dim must be > 0")

		self.input_dim = int(input_dim)
		self.latent_dim = int(latent_dim)

		encoder_layers: list[nn.Module] = []
		prev_dim = self.input_dim
		for h in hidden_dims:
			encoder_layers.append(nn.Linear(prev_dim, int(h)))
			encoder_layers.append(nn.ReLU())
			prev_dim = int(h)
		self.encoder = nn.Sequential(*encoder_layers)
		self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
		self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)

		decoder_layers: list[nn.Module] = []
		prev_dim = self.latent_dim
		for h in reversed(hidden_dims):
			decoder_layers.append(nn.Linear(prev_dim, int(h)))
			decoder_layers.append(nn.ReLU())
			prev_dim = int(h)
		decoder_layers.append(nn.Linear(prev_dim, self.input_dim))
		self.decoder = nn.Sequential(*decoder_layers)

	def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		h = self.encoder(x)
		return self.fc_mu(h), self.fc_logvar(h)

	@staticmethod
	def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z: torch.Tensor) -> torch.Tensor:
		return self.decoder(z)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon = self.decode(z)
		return recon, mu, logvar


class VideoClipVAE(nn.Module):
	"""A compact VAE for fixed-length RGB video clips.

	The architecture mirrors the music VAE pattern: a convolutional encoder,
	a latent bottleneck, and a mirrored decoder that reconstructs a fixed clip.
	"""

	def __init__(
		self,
		clip_frames: int,
		frame_size: int = 64,
		input_channels: int = 32,
		latent_dim: int = 64,
		hidden_dims: tuple[int, ...] = (32, 64, 128),
		dropout: float = 0.1,
	) -> None:
		super().__init__()
		if clip_frames <= 0:
			raise ValueError("clip_frames must be > 0")
		if frame_size <= 0:
			raise ValueError("frame_size must be > 0")
		if input_channels <= 0:
			raise ValueError("input_channels must be > 0")
		if latent_dim <= 0:
			raise ValueError("latent_dim must be > 0")
		if len(hidden_dims) < 2:
			raise ValueError("hidden_dims must contain at least 2 layers")

		self.clip_frames = int(clip_frames)
		self.frame_size = int(frame_size)
		self.channels = int(input_channels)
		self.latent_dim = int(latent_dim)
		self.hidden_dims = tuple(int(h) for h in hidden_dims)
		self.dropout = float(dropout)

		encoder_layers: list[nn.Module] = []
		in_channels = self.channels
		for idx, out_channels in enumerate(self.hidden_dims):
			stride = (1, 2, 2) if idx < len(self.hidden_dims) - 1 else (1, 1, 1)
			encoder_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
			encoder_layers.append(nn.BatchNorm3d(out_channels))
			encoder_layers.append(nn.ReLU())
			if self.dropout > 0.0:
				encoder_layers.append(nn.Dropout3d(self.dropout))
			in_channels = out_channels
		self.encoder = nn.Sequential(*encoder_layers)

		with torch.no_grad():
			dummy = torch.zeros(1, self.channels, self.clip_frames, self.frame_size, self.frame_size)
			encoded = self.encoder(dummy)
			self._encoded_shape = tuple(int(v) for v in encoded.shape[1:])
			encoded_dim = int(encoded.flatten(start_dim=1).shape[1])

		self.fc_mu = nn.Linear(encoded_dim, self.latent_dim)
		self.fc_logvar = nn.Linear(encoded_dim, self.latent_dim)
		self.decoder_input = nn.Linear(self.latent_dim, encoded_dim)

		decoder_layers: list[nn.Module] = []
		decoder_channels = list(reversed(self.hidden_dims))
		for idx in range(len(decoder_channels) - 1):
			in_ch = decoder_channels[idx]
			out_ch = decoder_channels[idx + 1]
			decoder_layers.append(
				nn.ConvTranspose3d(
					in_ch,
					out_ch,
					kernel_size=(3, 4, 4),
					stride=(1, 2, 2),
					padding=(1, 1, 1),
				)
			)
			decoder_layers.append(nn.BatchNorm3d(out_ch))
			decoder_layers.append(nn.ReLU())
			if self.dropout > 0.0:
				decoder_layers.append(nn.Dropout3d(self.dropout))
		decoder_layers.append(nn.Conv3d(decoder_channels[-1], self.channels, kernel_size=3, padding=1))
		self.decoder = nn.Sequential(*decoder_layers)

	def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		h = self.encoder(x)
		h = h.flatten(start_dim=1)
		return self.fc_mu(h), self.fc_logvar(h)

	@staticmethod
	def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def decode(self, z: torch.Tensor) -> torch.Tensor:
		h = self.decoder_input(z)
		h = h.view(-1, *self._encoded_shape)
		return self.decoder(h)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon = self.decode(z)
		return recon, mu, logvar


def vae_loss(
	recon: torch.Tensor,
	x: torch.Tensor,
	mu: torch.Tensor,
	logvar: torch.Tensor,
	beta: float = 1.0,
	pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
	"""Compute reconstruction + KL loss for one-hot segmentation clips."""
	if pos_weight is not None:
		recon_loss = F.binary_cross_entropy_with_logits(recon, x, reduction="mean", pos_weight=pos_weight)
	else:
		recon_loss = F.binary_cross_entropy_with_logits(recon, x, reduction="mean")
	kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
	total = recon_loss + float(beta) * kl
	return total, {"recon": recon_loss.detach(), "kl": kl.detach()}

