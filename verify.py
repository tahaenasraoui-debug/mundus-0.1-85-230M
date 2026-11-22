"""
Smoke test and parameter verification script for the 230M World Model.
Validates parameter counts, forward/backward passes, VQ-VAE encode/decode, and KV-cache.
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

from config import TokenizerConfig, WorldModelConfig, get_config_230m
from tokenizer import VQVAE
from world_model import WorldModel


def verify_all():
    print("==================================================")
    print("      MUNDUS 0.1 (230M) VERIFICATION SUITE       ")
    print("==================================================")
    device = "cpu"
    torch.set_num_threads(8)

    # 1. Parameter Count Verification
    print("\n[1/5] Checking Architecture & Parameter Counts...")
    wm_config = get_config_230m()
    tok_config = TokenizerConfig()

    tokenizer = VQVAE(tok_config).to(device)
    world_model = WorldModel(wm_config).to(device)

    wm_params = world_model.count_parameters()
    tok_params = sum(p.numel() for p in tokenizer.parameters())

    print(f"  • World Model Total Parameters: {wm_params['total']:,} ({wm_params['total']/1e6:.2f}M)")
    print(f"    - 18x SwiGLU Transformer Blocks: {wm_params['transformer_layers']:,}")
    print(f"    - Embeddings (Vocab + Actions):  {wm_params['embeddings']:,}")
    print(f"    - Multi-Task Prediction Heads:   {wm_params['prediction_heads']:,}")
    print(f"  • VQ-VAE Tokenizer Parameters:     {tok_params:,} ({tok_params/1e6:.2f}M)")
    print(f"  • Combined System Parameters:      {(wm_params['total'] + tok_params):,} ({(wm_params['total'] + tok_params)/1e6:.2f}M)")

    assert 225_000_000 <= wm_params["total"] <= 240_000_000, f"Expected ~230M params, got {wm_params['total']}"
    print("  --> Parameter count test PASSED (Target ~230M confirmed)!")

    # 2. VQ-VAE Encode/Decode Test
    print("\n[2/5] Testing VQ-VAE Visual Tokenizer...")
    dummy_frame = torch.randn(2, 3, 64, 64)
    tokens = tokenizer.encode(dummy_frame)
    assert tokens.shape == (2, 64), f"Expected tokens shape (2, 64), got {tokens.shape}"
    assert (tokens >= 0).all() and (tokens < tok_config.vocab_size).all()
    
    reconstruction = tokenizer.decode_tokens(tokens)
    assert reconstruction.shape == (2, 3, 64, 64), f"Expected recon shape (2, 3, 64, 64), got {reconstruction.shape}"
    print(f"  --> Tokenizer successfully compressed (2, 3, 64, 64) into (2, 64) tokens and reconstructed back. PASSED!")

    # 3. World Model Forward & Loss Test
    print("\n[3/5] Testing 230M Transformer Forward Pass & Multi-Task Loss...")
    B, S = 2, 128
    dummy_seq = torch.randint(0, wm_config.vocab_size + wm_config.num_actions, (B, S))
    target_tok = torch.randint(0, wm_config.vocab_size, (B, S))
    target_rew = torch.randn(B, S)
    target_don = torch.randint(0, 2, (B, S)).float()
    visual_mask = torch.ones(B, S)

    t0 = time.time()
    losses = world_model.compute_loss(
        token_seq=dummy_seq,
        target_tokens=target_tok,
        target_rewards=target_rew,
        target_dones=target_don,
        visual_mask=visual_mask,
    )
    dt_fwd = time.time() - t0

    assert "total_loss" in losses
    print(f"  • Forward pass time: {dt_fwd:.3f}s")
    print(f"  • Total Loss: {losses['total_loss'].item():.4f}")
    print(f"  • Token Loss: {losses['token_loss'].item():.4f}")
    print(f"  • Reward Loss: {losses['reward_loss'].item():.4f}")
    print(f"  • Done Loss: {losses['done_loss'].item():.4f}")
    print("  --> Multi-task loss computation PASSED!")

    # 4. Backward Pass & Gradient Updates Test
    print("\n[4/5] Testing Backward Pass & Optimizer Step...")
    optimizer = torch.optim.AdamW(world_model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    t0 = time.time()
    losses["total_loss"].backward()
    optimizer.step()
    dt_bwd = time.time() - t0
    print(f"  • Backward pass + Adam step time: {dt_bwd:.3f}s")
    print("  --> Backward pass and gradient update PASSED!")

    # 5. KV-Cache & Autoregressive Step Test
    print("\n[5/5] Testing KV Cache Autoregressive Step...")
    prompt = torch.randint(0, wm_config.vocab_size, (1, 10))
    out = world_model(prompt, use_cache=True)
    past_kvs = out["new_kvs"]

    next_tok = torch.randint(0, wm_config.vocab_size, (1, 1))
    step_tok, rew, done, new_kvs = world_model.imagine_step(next_tok, past_kvs=past_kvs)

    assert step_tok.shape == (1, 1)
    assert rew.numel() == 1
    assert done.numel() == 1
    assert len(new_kvs) == wm_config.n_layers
    assert new_kvs[0][0].shape[2] == 11 # 10 prompt tokens + 1 step token
    print(f"  • KV-cache length after 1 step: {new_kvs[0][0].shape[2]} tokens")
    print("  --> KV-cache autoregressive step PASSED!")

    print("\n==================================================")
    print("  ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY! ")
    print("==================================================")


if __name__ == "__main__":
    verify_all()
