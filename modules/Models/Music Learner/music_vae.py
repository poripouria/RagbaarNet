import torch
from torch import nn


class MusicVAE(nn.Module):
	"""A convolutional VAE for fixed-length piano-roll segments.

	Input/Output: flattened piano-roll vector of shape (segment_steps * 128,).
	Decoder returns logits (use BCEWithLogitsLoss for reconstruction).
	"""

	def __init__(
		self,
		sequence_steps: int,
		note_dim: int = 128,
		latent_dim: int = 64,
		hidden_dims: tuple[int, ...] = (64, 128, 256),
		dropout: float = 0.1,
	) -> None:
		super().__init__()
		if sequence_steps <= 0:
			raise ValueError("sequence_steps must be > 0")
		if note_dim <= 0:
			raise ValueError("note_dim must be > 0")
		if latent_dim <= 0:
			raise ValueError("latent_dim must be > 0")
		if len(hidden_dims) < 2:
			raise ValueError("hidden_dims must contain at least 2 layers")

		self.sequence_steps = int(sequence_steps)
		self.note_dim = int(note_dim)
		self.input_dim = self.sequence_steps * self.note_dim
		self.latent_dim = int(latent_dim)
		self.hidden_dims = tuple(int(h) for h in hidden_dims)
		self.dropout = float(dropout)

		encoder_layers: list[nn.Module] = []
		in_channels = self.note_dim
		for out_channels in self.hidden_dims:
			encoder_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
			encoder_layers.append(nn.BatchNorm1d(out_channels))
			encoder_layers.append(nn.ReLU())
			if self.dropout > 0.0:
				encoder_layers.append(nn.Dropout(self.dropout))
			in_channels = out_channels
		self.encoder = nn.Sequential(*encoder_layers)
		self._encoded_steps = self._downsample_steps(self.sequence_steps, len(self.hidden_dims))
		encoded_dim = self.hidden_dims[-1] * self._encoded_steps
		self.fc_mu = nn.Linear(encoded_dim, self.latent_dim)
		self.fc_logvar = nn.Linear(encoded_dim, self.latent_dim)

		self.decoder_input = nn.Linear(self.latent_dim, encoded_dim)
		decoder_layers: list[nn.Module] = []
		decoder_channels = list(reversed(self.hidden_dims))
		for idx in range(len(decoder_channels) - 1):
			in_ch = decoder_channels[idx]
			out_ch = decoder_channels[idx + 1]
			decoder_layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
			decoder_layers.append(nn.BatchNorm1d(out_ch))
			decoder_layers.append(nn.ReLU())
			if self.dropout > 0.0:
				decoder_layers.append(nn.Dropout(self.dropout))
		decoder_layers.append(nn.ConvTranspose1d(decoder_channels[-1], self.note_dim, kernel_size=4, stride=2, padding=1))
		self.decoder = nn.Sequential(*decoder_layers)

	@staticmethod
	def _downsample_steps(sequence_steps: int, num_layers: int) -> int:
		steps = int(sequence_steps)
		for _ in range(num_layers):
			steps = (steps + 1) // 2
		return max(1, steps)

	def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		x = x.view(-1, self.sequence_steps, self.note_dim).transpose(1, 2)
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
		h = h.view(-1, self.hidden_dims[-1], self._encoded_steps)
		logits = self.decoder(h)
		return logits.transpose(1, 2).reshape(-1, self.input_dim)

	def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		logits = self.decode(z)
		return logits, mu, logvar


def vae_loss(
	recon_logits: torch.Tensor,
	x: torch.Tensor,
	mu: torch.Tensor,
	logvar: torch.Tensor,
	beta: float = 1.0,
	pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
	"""Compute VAE loss = reconstruction (BCE) + beta * KL."""
	# Reconstruction term (weighted binary piano-roll)
	if pos_weight is not None:
		recon = nn.functional.binary_cross_entropy_with_logits(recon_logits, x, reduction="mean", pos_weight=pos_weight)
	else:
		recon = nn.functional.binary_cross_entropy_with_logits(recon_logits, x, reduction="mean")
	# KL divergence term
	kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
	total = recon + float(beta) * kl
	return total, {"recon": recon.detach(), "kl": kl.detach()}

