"""Tests for model presets, portable path defaults, WorldModel forward/KV-cache, and training helpers."""

from pathlib import Path

import pytest
import torch

import config
import train
from config import (
    BASE_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_TOKENIZER_PATH,
    TokenizerConfig,
    TrainConfig,
    WorldModelConfig,
    get_config,
    get_config_85m,
    get_config_130m,
    get_config_230m,
)
from dataset import WorldModelDataset, collect_trajectories
from tokenizer import VQVAE
from world_model import WorldModel


def test_portable_default_paths():
    repo_root = Path(config.__file__).resolve().parent
    assert BASE_DIR == repo_root
    assert DEFAULT_CHECKPOINT_DIR == repo_root / "checkpoints"
    assert DEFAULT_TOKENIZER_PATH == repo_root / "checkpoints" / "tokenizer.pt"
    assert Path(TrainConfig().checkpoint_dir) == DEFAULT_CHECKPOINT_DIR


@pytest.mark.parametrize(
    ("key", "expected_fn", "expected_name", "d_model", "n_layers", "n_heads", "hidden_dim", "min_p", "max_p"),
    [
        ("85m", get_config_85m, "Mundus-0.1-85M", 768, 12, 12, 2048, 85_000_000, 92_000_000),
        ("130m", get_config_130m, "Mundus-0.1-130M", 768, 18, 12, 2048, 128_000_000, 135_000_000),
        ("230m", get_config_230m, "Mundus-0.1-230M", 1024, 18, 16, 2752, 225_000_000, 240_000_000),
    ],
)
def test_model_config_presets_and_param_counts(
    key: str,
    expected_fn,
    expected_name: str,
    d_model: int,
    n_layers: int,
    n_heads: int,
    hidden_dim: int,
    min_p: int,
    max_p: int,
):
    cfg = get_config(f"  {key.upper()} ")
    direct = expected_fn()
    assert cfg == direct
    assert cfg.model_name == expected_name
    assert cfg.d_model == d_model
    assert cfg.n_layers == n_layers
    assert cfg.n_heads == n_heads
    assert cfg.hidden_dim == hidden_dim

    model = WorldModel(cfg)
    counts = model.count_parameters()
    assert min_p <= counts["total"] <= max_p
    assert counts["total"] == (
        counts["transformer_layers"] + counts["embeddings"] + counts["prediction_heads"] + d_model
    )
    del model


def test_unknown_model_preset_raises():
    with pytest.raises(ValueError, match="Unknown model preset"):
        get_config("400m")


def test_world_model_loss_and_kv_cache_tiny():
    cfg = WorldModelConfig(
        model_name="Mundus-Test-Tiny",
        d_model=64,
        n_layers=2,
        n_heads=4,
        hidden_dim=128,
        vocab_size=64,
        num_actions=16,
        max_seq_len=128,
        tokens_per_frame=16,
    )
    model = WorldModel(cfg)
    seq = torch.randint(0, cfg.vocab_size + cfg.num_actions, (2, 20))
    targets = torch.randint(0, cfg.vocab_size, (2, 20))
    rewards = torch.zeros(2, 20)
    dones = torch.zeros(2, 20)
    mask = torch.ones(2, 20)

    losses = model.compute_loss(seq, targets, rewards, dones, mask)
    assert set(losses) == {"total_loss", "token_loss", "reward_loss", "done_loss"}
    assert torch.isfinite(losses["total_loss"])
    losses["total_loss"].backward()

    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model(prompt, use_cache=True)
    next_tok, rew, done_prob, new_kvs = model.imagine_step(
        torch.tensor([[cfg.vocab_size + 1]]), past_kvs=out["new_kvs"]
    )
    assert next_tok.shape == (1, 1)
    assert 0 <= next_tok.item() < cfg.vocab_size
    assert rew.shape == (1,)
    assert 0.0 <= done_prob.item() <= 1.0
    assert len(new_kvs) == cfg.n_layers
    assert new_kvs[0][0].shape[2] == 9


def test_train_helpers_save_tokenizer_and_dynamic_model_slugs(tmp_path: Path, capsys):
    assert train._model_slug("Mundus-0.1-85M") == "mundus_0.1_85m"
    assert train._model_slug("Mundus-0.1-130M") == "mundus_0.1_130m"
    assert train._model_slug("Mundus-0.1-230M") == "mundus_0.1_230m"

    episodes = collect_trajectories(num_episodes=2, episode_length=6)
    tok_cfg = TokenizerConfig(vocab_size=64, latent_dim=16)
    tokenizer = VQVAE(tok_cfg)

    tok_path = train.train_tokenizer(
        tokenizer,
        episodes,
        epochs=1,
        batch_size=4,
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    assert tok_path == tmp_path / "tokenizer.pt"
    assert tok_path.exists()

    wm_cfg = WorldModelConfig(
        model_name="Mundus-0.1-130M",
        d_model=64,
        n_layers=2,
        n_heads=4,
        hidden_dim=128,
        vocab_size=64,
        num_actions=16,
        max_seq_len=512,
        tokens_per_frame=64,
    )
    dataset = WorldModelDataset(episodes, tokenizer, wm_cfg, seq_frames=2, device="cpu")
    model = WorldModel(wm_cfg)
    train_cfg = TrainConfig(
        batch_size=2,
        max_steps=2,
        save_interval=1,
        log_interval=1,
        checkpoint_dir=str(tmp_path),
    )

    train.train_world_model(model, dataset, train_cfg, device="cpu")
    captured = capsys.readouterr().out

    assert "STAGE 2: Training Mundus-0.1-130M" in captured
    assert "[Mundus-0.1-130M] Step 00001/00002" in captured
    assert (tmp_path / "mundus_0.1_130m_latest.pt").exists()
    assert (tmp_path / "mundus_0.1_130m_step_2.pt").exists()
