"""Tests for VQVAE tokenizer and WorldModelDataset sequence/mask shapes."""

import torch

from config import TokenizerConfig, get_config_85m
from dataset import WorldModelDataset, collect_trajectories
from doom_env import collect_doom_trajectories
from tokenizer import VQVAE


def test_vqvae_encode_decode_and_forward_shapes():
    tok_cfg = TokenizerConfig()
    tokenizer = VQVAE(tok_cfg)
    tokenizer.eval()

    frames = torch.rand(2, 3, 64, 64)
    tokens = tokenizer.encode(frames)
    assert tokens.shape == (2, tok_cfg.tokens_per_frame)
    assert tokens.dtype == torch.long
    assert (tokens >= 0).all()
    assert (tokens < tok_cfg.vocab_size).all()

    recon = tokenizer.decode_tokens(tokens)
    assert recon.shape == (2, 3, 64, 64)
    assert (recon >= 0.0).all()
    assert (recon <= 1.0).all()

    out = tokenizer(frames)
    assert set(out) == {"reconstruction", "tokens", "total_loss", "recon_loss", "vq_loss"}
    assert out["reconstruction"].shape == (2, 3, 64, 64)
    assert out["tokens"].shape == (2, 64)
    assert out["total_loss"].ndim == 0


def test_world_model_dataset_interleaved_shapes_and_masks():
    wm_cfg = get_config_85m()
    tokenizer = VQVAE(TokenizerConfig())
    seq_frames = 3

    arcade_eps = collect_trajectories(num_episodes=2, episode_length=8)
    doom_eps = collect_doom_trajectories(num_episodes=2, episode_length=8)

    for episodes, max_action in ((arcade_eps, 2), (doom_eps, 5)):
        ds = WorldModelDataset(episodes, tokenizer, wm_cfg, seq_frames=seq_frames, device="cpu")
        assert len(ds) > 0

        item = ds[0]
        k = wm_cfg.tokens_per_frame
        expected_len = k + seq_frames * (1 + k) - 1

        assert item["token_seq"].shape == (expected_len,)
        assert item["target_tokens"].shape == (expected_len,)
        assert item["target_rewards"].shape == (expected_len,)
        assert item["target_dones"].shape == (expected_len,)
        assert item["visual_mask"].shape == (expected_len,)

        assert item["token_seq"].dtype == torch.long
        assert item["target_tokens"].dtype == torch.long
        assert item["target_rewards"].dtype == torch.float32
        assert item["target_dones"].dtype == torch.float32
        assert item["visual_mask"].dtype == torch.float32

        # Initial prompt frame positions are not supervised
        assert (item["visual_mask"][:k] == 0.0).all()
        # Action token sits immediately after the prompt frame
        act_tok = item["token_seq"][k].item()
        assert wm_cfg.vocab_size <= act_tok <= wm_cfg.vocab_size + max_action
        # Targets belong to visual vocabulary
        assert (item["target_tokens"] >= 0).all()
        assert (item["target_tokens"] < wm_cfg.vocab_size).all()
