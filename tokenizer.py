"""
VQ-VAE (Vector-Quantized Variational Autoencoder) Visual Tokenizer.
Compresses continuous environment visual frames into discrete token IDs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TokenizerConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with Straight-Through Estimator.
    Maps continuous vectors to discrete codebook indices.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Uniform initialization of codebook
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs: torch.Tensor):
        # inputs shape: (B, C, H, W) -> permute to (B, H, W, C)
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs_permuted.shape
        flat_input = inputs_permuted.view(-1, self.embedding_dim)

        # Calculate L2 distances to all codebook embeddings: (x - e)^2 = x^2 + e^2 - 2xe
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )

        # Get closest codebook indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices.squeeze(1)).view(input_shape)

        # Loss: commitment loss + codebook loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs_permuted)
        q_latent_loss = F.mse_loss(quantized, inputs_permuted.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator: gradient flows directly to encoder
        quantized = inputs_permuted + (quantized - inputs_permuted).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Reshape indices to (B, H*W)
        tokens = encoding_indices.view(input_shape[0], -1)

        return quantized, loss, tokens

    def indices_to_latents(self, indices: torch.Tensor, h: int = 8, w: int = 8) -> torch.Tensor:
        """
        Convert token indices back to spatial feature map (B, C, H, W).
        """
        b = indices.shape[0]
        quantized = self.embedding(indices.view(-1))
        quantized = quantized.view(b, h, w, self.embedding_dim)
        return quantized.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    """
    Visual Tokenizer that encodes 64x64 frames into an 8x8 grid of discrete tokens (64 tokens).
    """

    def __init__(self, config: TokenizerConfig | None = None):
        super().__init__()
        config = config or TokenizerConfig()
        self.config = config

        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(config.in_channels, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=False),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, config.latent_dim, kernel_size=1),
        )

        # Vector Quantizer Codebook
        self.quantizer = VectorQuantizer(
            num_embeddings=config.vocab_size,
            embedding_dim=config.latent_dim,
            commitment_cost=config.commitment_cost,
        )

        # Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64
        self.decoder = nn.Sequential(
            nn.Conv2d(config.latent_dim, 256, kernel_size=1),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, config.in_channels, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Sigmoid(),  # Images normalized to [0, 1]
        )

    def encode(self, x: torch.Tensor):
        """Encodes frame (B, C, 64, 64) -> discrete token indices (B, 64)."""
        z = self.encoder(x)
        _, _, tokens = self.quantizer(z)
        return tokens

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decodes token indices (B, 64) -> reconstructed frames (B, C, 64, 64)."""
        quantized = self.quantizer.indices_to_latents(tokens, h=8, w=8)
        reconstruction = self.decoder(quantized)
        return reconstruction

    def forward(self, x: torch.Tensor):
        """Full autoencoder forward pass with reconstruction + quantization loss."""
        z = self.encoder(x)
        quantized, vq_loss, tokens = self.quantizer(z)
        reconstruction = self.decoder(quantized)
        recon_loss = F.mse_loss(reconstruction, x)
        total_loss = recon_loss + vq_loss
        return {
            "reconstruction": reconstruction,
            "tokens": tokens,
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "vq_loss": vq_loss,
        }
