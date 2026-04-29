import torch
from torch import nn


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

