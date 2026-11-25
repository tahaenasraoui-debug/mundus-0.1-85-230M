"""
Mundus 0.1 - Causal Transformer World Model with Action Conditioning and KV Cache.
Implements the latent dynamics engine for environment simulation and imagination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import WorldModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for relative sequence awareness."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

        # inv_freq: (dim // 2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin caches
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        # (max_seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        d_2 = x.shape[-1] // 2
        x1 = x[..., :d_2]
        x2 = x[..., d_2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k shape: (B, num_heads, seq_len, head_dim)
        seq_len = q.shape[2]
        cos = self.cos_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class SwiGLU(nn.Module):
    """SwiGLU FeedForward network (LLaMA / PaLM style)."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(w_gate(x)) * w_up(x)
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE and KV-Cache."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        assert self.head_dim * self.n_heads == self.d_model, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.max_seq_len, theta=config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, S, D = x.shape

        # Linear projections
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        offset = past_kv[0].shape[2] if past_kv is not None else 0
        q, k = self.rope(q, k, offset=offset)

        # Update KV cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        # Scaled dot-product attention
        # For single token autoregressive step with cache, attend to all past tokens
        is_causal = past_kv is None and S > 1
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        return out, new_kv


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block with RMSNorm, Self-Attention, and SwiGLU."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.hidden_dim, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Residual connection with pre-norm
        attn_out, new_kv = self.attn(self.attn_norm(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv


class WorldModel(nn.Module):
    """
    Mundus 0.1 Transformer World Model.
    Models environment dynamics: P(Next Tokens, Reward, Done | History, Actions)
    """

    def __init__(self, config: WorldModelConfig | None = None):
        super().__init__()
        config = config or WorldModelConfig()
        self.config = config

        # Embeddings:
        # Visual tokens: [0, vocab_size - 1]
        # Action tokens: [vocab_size, vocab_size + num_actions - 1]
        self.total_vocab = config.vocab_size + config.num_actions
        self.token_embedding = nn.Embedding(self.total_vocab, config.d_model)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.gradient_checkpointing = False

        # Prediction Heads
        # 1. Next visual token logits (predicts over visual vocabulary 0..vocab_size-1)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 2. Reward prediction head (scalar regression)
        self.reward_head = nn.Sequential(
            nn.Linear(config.d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # 3. Done / Episode termination prediction head (binary classification)
        self.done_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Initialize weights with standard normal (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def action_to_token(self, actions: torch.Tensor) -> torch.Tensor:
        """Offset action IDs to the action subspace in the unified vocabulary."""
        return actions + self.config.vocab_size

    def gradient_checkpointing_enable(self):
        """Enables activation checkpointing to drastically reduce VRAM during training."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        token_seq: torch.Tensor,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor | list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Forward pass through the dynamics model.
        Args:
            token_seq: (B, seq_len) sequence of interleaved visual and action token IDs.
            past_kvs: Optional cached key-values from previous steps.
            use_cache: Whether to return updated KV caches.
        Returns:
            dict containing:
                - logits: (B, seq_len, vocab_size) next visual token logits
                - rewards: (B, seq_len, 1) predicted rewards
                - dones: (B, seq_len, 1) predicted termination logits
                - new_kvs: updated KV caches if use_cache is True
        """
        x = self.token_embedding(token_seq)

        new_kvs = [] if use_cache else None
        use_ckpt = self.gradient_checkpointing and self.training and past_kvs is None and not use_cache

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_kvs[i] if past_kvs is not None else None
            if use_ckpt:
                def make_custom_forward(mod):
                    def custom_forward(hidden):
                        out, _ = mod(hidden, past_kv=None, use_cache=False)
                        return out
                    return custom_forward
                x = torch.utils.checkpoint.checkpoint(make_custom_forward(layer), x, use_reentrant=False)
            else:
                x, new_kv = layer(x, past_kv=layer_past_kv, use_cache=use_cache)
                if use_cache:
                    new_kvs.append(new_kv)

        x = self.final_norm(x)

        logits = self.lm_head(x)
        rewards = self.reward_head(x)
        dones = self.done_head(x)

        return {
            "logits": logits,
            "rewards": rewards,
            "dones": dones,
            "new_kvs": new_kvs,
        }

    def compute_loss(
        self,
        token_seq: torch.Tensor,
        target_tokens: torch.Tensor,
        target_rewards: torch.Tensor,
        target_dones: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Computes multi-task loss for dynamics, reward, and episode termination.
        """
        out = self.forward(token_seq)
        logits = out["logits"]
        pred_rewards = out["rewards"]
        pred_dones = out["dones"]

        # 1. Next token cross-entropy (computed only on visual token positions)
        flat_logits = logits.view(-1, self.config.vocab_size)
        flat_targets = target_tokens.view(-1)
        flat_mask = visual_mask.view(-1).bool()

        if flat_mask.sum() > 0:
            token_loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
        else:
            token_loss = torch.tensor(0.0, device=token_seq.device)

        # 2. Reward MSE loss
        reward_loss = F.mse_loss(pred_rewards.squeeze(-1), target_rewards)

        # 3. Done BCE loss with logits
        done_loss = F.binary_cross_entropy_with_logits(pred_dones.squeeze(-1), target_dones)

        total_loss = token_loss + 0.5 * reward_loss + 0.5 * done_loss

        return {
            "total_loss": total_loss,
            "token_loss": token_loss,
            "reward_loss": reward_loss,
            "done_loss": done_loss,
        }

    @torch.no_grad()
    def imagine_step(
        self,
        input_token: torch.Tensor,
        past_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Single auto-regressive step with KV caching.
        Returns sampled next token, predicted reward, predicted done, and new KV cache.
        """
        out = self.forward(input_token, past_kvs=past_kvs, use_cache=True)
        logits = out["logits"][:, -1, :] / max(temperature, 1e-5)
        reward = out["rewards"][:, -1, :].squeeze(-1)
        done_prob = torch.sigmoid(out["dones"][:, -1, :]).squeeze(-1)

        # Top-k sampling
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token, reward, done_prob, out["new_kvs"]

    def count_parameters(self) -> dict[str, int]:
        """Returns parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        layers = sum(p.numel() for layer in self.layers for p in layer.parameters())
        embed = self.token_embedding.weight.numel()
        heads = (
            sum(p.numel() for p in self.lm_head.parameters())
            + sum(p.numel() for p in self.reward_head.parameters())
            + sum(p.numel() for p in self.done_head.parameters())
        )
        return {
            "total": total,
            "transformer_layers": layers,
            "embeddings": embed,
            "prediction_heads": heads,
        }
