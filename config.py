"""
Configuration presets for World Models and Tokenizer.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TokenizerConfig:
    """VQ-VAE Visual Tokenizer Configuration"""
    in_channels: int = 3                # RGB frames
    hidden_dims: Tuple[int, ...] = (64, 128, 256) # Downsampling conv channels
    latent_dim: int = 128               # Dimension of quantized vectors
    vocab_size: int = 2048              # Size of the discrete visual codebook
    commitment_cost: float = 0.25       # Weight for commitment loss
    img_size: int = 64                  # 64x64 input images
    tokens_per_frame: int = 64          # 8x8 latent grid = 64 tokens per frame


@dataclass
class WorldModelConfig:
    """
    Mundus 0.1 (230M) Causal Transformer Configuration.
    Default: 232M Parameters.
    """
    model_name: str = "Mundus-0.1-230M"
    version: str = "0.1"
    d_model: int = 1024
    n_layers: int = 18
    n_heads: int = 16
    hidden_dim: int = 2752              # SwiGLU hidden dimension
    vocab_size: int = 2048              # Visual token vocabulary size
    num_actions: int = 16               # Discrete actions supported
    max_seq_len: int = 2048             # Maximum context tokens
    tokens_per_frame: int = 64          # Visual tokens per frame
    dropout: float = 0.05
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    """Training Configuration"""
    batch_size: int = 2                 # Optimal for CPU training and RAM limits
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000               # Default run steps (extendable to 25k)
    grad_clip: float = 1.0
    num_cpu_threads: int = 8            # Leverage all 8 CPU cores
    device: str = "cpu"                 # "cpu" or "cuda"
    checkpoint_dir: str = "/home/tahary/.gemini/antigravity/scratch/world_model_230m/checkpoints"
    save_interval: int = 500
    log_interval: int = 20


# Preset helpers
def get_config_85m() -> WorldModelConfig:
    return WorldModelConfig(
        model_name="WorldModel-85M",
        d_model=768,
        n_layers=12,
        n_heads=12,
        hidden_dim=2048,
        vocab_size=2048,
        num_actions=16,
    )

def get_config_130m() -> WorldModelConfig:
    return WorldModelConfig(
        model_name="WorldModel-130M",
        d_model=768,
        n_layers=18,
        n_heads=12,
        hidden_dim=2048,
        vocab_size=2048,
        num_actions=16,
    )

def get_config_230m() -> WorldModelConfig:
    return WorldModelConfig(
        model_name="Mundus-0.1-230M",
        version="0.1",
        d_model=1024,
        n_layers=18,
        n_heads=16,
        hidden_dim=2752,
        vocab_size=2048,
        num_actions=16,
    )
