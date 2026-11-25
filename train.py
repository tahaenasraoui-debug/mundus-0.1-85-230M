"""
Training pipeline for Mundus 0.1 (85M / 130M / 230M) World Model and VQ-VAE Tokenizer.
Includes Stage 1 (Visual Codebook Training) and Stage 2 (Latent Dynamics Training).
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Prevent CUDA memory fragmentation on large model allocations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import DEFAULT_CHECKPOINT_DIR, TokenizerConfig, TrainConfig, get_config
from dataset import WorldModelDataset
from games import GAME_REGISTRY, collect_game_trajectories, list_games
from tokenizer import VQVAE
from world_model import WorldModel

# Ensure immediate line-buffered output when logging to files
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def _model_slug(model_name: str) -> str:
    """Normalizes a model name like 'Mundus-0.1-85M' to 'mundus_0.1_85m'."""
    return model_name.strip().lower().replace("-", "_").replace(" ", "_")


def train_tokenizer(
    tokenizer: VQVAE,
    episodes: list,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint_dir: str | Path | None = None,
) -> Path:
    """
    Stage 1: Train VQ-VAE codebook on visual frames and save checkpoints/tokenizer.pt.
    """
    print("\n==========================================")
    print("STAGE 1: Training VQ-VAE Visual Tokenizer")
    print("==========================================")

    # Gather all frames from episodes into a flat tensor
    all_frames = [f for ep in episodes for f in ep["frames"]]
    frames_tensor = torch.from_numpy(np.stack(all_frames)).permute(0, 3, 1, 2).float()  # (N, 3, 64, 64)

    dataset = torch.utils.data.TensorDataset(frames_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(tokenizer.parameters(), lr=lr)
    tokenizer.train()

    for epoch in range(epochs):
        total_loss = 0.0
        t0 = time.time()
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            out = tokenizer(x)
            loss = out["total_loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        dt = time.time() - t0
        print(f"Epoch {epoch + 1}/{epochs} | VQ-VAE Loss: {avg_loss:.5f} | Time: {dt:.1f}s")

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / "tokenizer.pt"
    torch.save(tokenizer.state_dict(), save_path)
    print(f"--> Saved VQ-VAE visual tokenizer to {save_path}")
    print("Stage 1 complete! Visual tokenizer is ready.\n")
    return save_path


def train_world_model(
    world_model: WorldModel,
    dataset: WorldModelDataset,
    train_cfg: TrainConfig,
    device: str = "cpu",
    amp: bool = True,
    grad_accum_steps: int = 1,
    use_adam8bit: bool = False,
    grad_checkpoint: bool = True,
):
    """
    Stage 2: Train the Transformer World Model on latent trajectory rollouts.
    """
    model_name = world_model.config.model_name
    model_slug = _model_slug(model_name)

    print("==========================================")
    print(f"STAGE 2: Training {model_name}")
    print("==========================================")
    params_breakdown = world_model.count_parameters()
    total_p = params_breakdown["total"]
    scale_str = f"{total_p / 1e9:.2f}B" if total_p >= 1e9 else f"{total_p / 1e6:.2f}M"
    print(f"Total Parameters: {total_p:,} ({scale_str})")
    print(f"  - Transformer Layers ({world_model.config.n_layers}x): {params_breakdown['transformer_layers']:,}")
    print(f"  - Vocabulary & Action Embeddings: {params_breakdown['embeddings']:,}")
    print(f"  - Multi-Task Prediction Heads: {params_breakdown['prediction_heads']:,}")
    print("==========================================\n")

    # Enable activation gradient checkpointing to save VRAM
    if grad_checkpoint and hasattr(world_model, "gradient_checkpointing_enable"):
        world_model.gradient_checkpointing_enable()
        print("[Memory] Gradient Checkpointing ENABLED (saving activation VRAM)")

    use_cuda = device.startswith("cuda")
    amp_enabled = use_cuda and amp
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

    # Cast model parameters to 16-bit / bfloat16 to cut weights & gradient memory by 50%
    if use_cuda and amp_enabled:
        world_model.to(dtype=amp_dtype)
        print(f"[Memory] World Model cast to {amp_dtype} (50% parameter & gradient memory savings)")

    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    if use_adam8bit:
        try:
            import bitsandbytes as bnb
            if hasattr(bnb.optim, "PagedAdamW8bit"):
                optimizer = bnb.optim.PagedAdamW8bit(
                    world_model.parameters(),
                    lr=train_cfg.lr,
                    weight_decay=train_cfg.weight_decay,
                    betas=(0.9, 0.95),
                )
                print("[Optimizer] Using Paged 8-bit AdamW (PagedAdamW8bit) — pages states to host RAM (prevents OOM on 15GB T4 GPUs)!")
            else:
                optimizer = bnb.optim.AdamW8bit(
                    world_model.parameters(),
                    lr=train_cfg.lr,
                    weight_decay=train_cfg.weight_decay,
                    betas=(0.9, 0.95),
                )
                print("[Optimizer] Using 8-bit AdamW (bitsandbytes) for ultra-low VRAM!")
        except ImportError:
            print("[Optimizer] bitsandbytes not installed, falling back to standard AdamW.")
            optimizer = optim.AdamW(
                world_model.parameters(),
                lr=train_cfg.lr,
                weight_decay=train_cfg.weight_decay,
                betas=(0.9, 0.95),
            )
    else:
        optimizer = optim.AdamW(
            world_model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.95),
        )

    # GradScaler is essential for float16 to prevent underflow, but bypassed for bfloat16
    scaler_enabled = amp_enabled and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    if amp_enabled:
        status_scaler = "active" if scaler_enabled else "bypassed for native bfloat16"
        print(f"[AMP] Mixed Precision enabled (dtype: {amp_dtype}, GradScaler: {status_scaler})")
    if grad_accum_steps > 1:
        print(f"[Grad Accum] Effective batch size: {train_cfg.batch_size * grad_accum_steps} (accum={grad_accum_steps})")

    # Cosine annealing with warmup
    def get_lr(step: int) -> float:
        if step < train_cfg.warmup_steps:
            return train_cfg.lr * (step + 1) / train_cfg.warmup_steps
        progress = (step - train_cfg.warmup_steps) / max(1, train_cfg.max_steps - train_cfg.warmup_steps)
        return train_cfg.min_lr + 0.5 * (train_cfg.lr - train_cfg.min_lr) * (1.0 + math.cos(math.pi * progress))

    world_model.train()
    step = 0
    running_loss = 0.0
    running_tok_loss = 0.0
    running_rew_loss = 0.0
    running_don_loss = 0.0

    t_start = time.time()
    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step_prefix = f"{model_slug}_step_"
    optimizer.zero_grad()

    while step < train_cfg.max_steps:
        for batch in loader:
            step += 1
            cur_lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            token_seq = batch["token_seq"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            target_rewards = batch["target_rewards"].to(device)
            target_dones = batch["target_dones"].to(device)
            visual_mask = batch["visual_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                losses = world_model.compute_loss(
                    token_seq=token_seq,
                    target_tokens=target_tokens,
                    target_rewards=target_rewards,
                    target_dones=target_dones,
                    visual_mask=visual_mask,
                )
                total_loss = losses["total_loss"]
                loss_to_backprop = total_loss / grad_accum_steps

            if scaler_enabled:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if step % grad_accum_steps == 0:
                if scaler_enabled:
                    if train_cfg.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(world_model.parameters(), train_cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if train_cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(world_model.parameters(), train_cfg.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += total_loss.item()
            running_tok_loss += losses["token_loss"].item()
            running_rew_loss += losses["reward_loss"].item()
            running_don_loss += losses["done_loss"].item()

            if step % train_cfg.log_interval == 0:
                avg_l = running_loss / train_cfg.log_interval
                avg_tok = running_tok_loss / train_cfg.log_interval
                avg_rew = running_rew_loss / train_cfg.log_interval
                avg_don = running_don_loss / train_cfg.log_interval
                elapsed = time.time() - t_start
                steps_per_sec = step / max(1.0, elapsed)

                print(
                    f"[{model_name}] Step {step:05d}/{train_cfg.max_steps:05d} | "
                    f"Loss: {avg_l:.4f} (Tok: {avg_tok:.4f}, Rew: {avg_rew:.4f}, Done: {avg_don:.4f}) | "
                    f"LR: {cur_lr:.2e} | "
                    f"Speed: {steps_per_sec:.2f} steps/s"
                )

                running_loss = 0.0
                running_tok_loss = 0.0
                running_rew_loss = 0.0
                running_don_loss = 0.0

            # Smart rolling checkpoint saving (protects disk space on long 45k runs)
            if step % train_cfg.save_interval == 0 or step == train_cfg.max_steps:
                latest_ckpt_path = ckpt_dir / f"{model_slug}_latest.pt"
                step_ckpt_path = ckpt_dir / f"{step_prefix}{step}.pt"

                checkpoint_data = {
                    "model_name": model_name,
                    "step": step,
                    "world_model": world_model.state_dict(),
                    "config": world_model.config,
                }

                # Always update the latest checkpoint
                torch.save(checkpoint_data, latest_ckpt_path)

                # For milestones every 2500 steps, save named checkpoint and keep last 2 milestones
                if step % 2500 == 0 or step == train_cfg.max_steps:
                    torch.save(checkpoint_data, step_ckpt_path)
                    print(f"--> Saved milestone checkpoint to {step_ckpt_path}")

                    # Prune older step checkpoints to keep disk usage bounded
                    existing_ckpts = sorted(
                        [f for f in os.listdir(ckpt_dir) if f.startswith(step_prefix) and f.endswith(".pt")],
                        key=lambda x: int(x.removeprefix(step_prefix).removesuffix(".pt")),
                    )
                    if len(existing_ckpts) > 2:
                        for old in existing_ckpts[:-2]:
                            try:
                                (ckpt_dir / old).unlink()
                            except OSError:
                                pass
                else:
                    print(f"--> Updated latest checkpoint: {latest_ckpt_path}")

            if step >= train_cfg.max_steps:
                break


def main():
    parser = argparse.ArgumentParser(description="Train Mundus World Model")
    parser.add_argument(
        "--model",
        type=str,
        default="85m",
        choices=["85m", "130m", "230m", "3b"],
        help="Model preset scale ('85m', '130m', '230m', '3b')",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="all",
        choices=list_games(),
        help=f"Game environment ({list_games()})",
    )
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to record")
    parser.add_argument("--episode_length", type=int, default=40, help="Steps per trajectory episode")
    parser.add_argument("--steps", type=int, default=25000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--amp", action="store_true", default=True, help="Use Mixed Precision (FP16/BF16)")
    parser.add_argument("--no_amp", action="store_false", dest="amp", help="Disable Mixed Precision")
    parser.add_argument("--adam8bit", action="store_true", help="Use 8-bit Adam via bitsandbytes (vital for 3B on 16GB GPUs)")
    parser.add_argument(
        "--grad_checkpoint",
        action="store_true",
        default=True,
        help="Enable activation gradient checkpointing (saves massive VRAM for 3B)",
    )
    parser.add_argument(
        "--no_grad_checkpoint",
        action="store_false",
        dest="grad_checkpoint",
        help="Disable activation gradient checkpointing",
    )
    parser.add_argument("--tokenizer_epochs", type=int, default=4, help="Stage 1 VQ-VAE tokenizer epochs")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory to save tokenizer and world model checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    device = args.device
    wm_cfg = get_config(args.model)
    if device == "cpu":
        torch.set_num_threads(8)
        print(f"Running {wm_cfg.model_name} on 8 CPU cores...")
    else:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else device
        print(f"Running {wm_cfg.model_name} on GPU: {gpu_name}...")

    print("Initializing environment and models...")
    tok_cfg = TokenizerConfig()
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.steps,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    tokenizer = VQVAE(tok_cfg).to(device)
    world_model = WorldModel(wm_cfg).to(device)

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"[Resume] Loading checkpoint weights from {resume_path}...")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            if "world_model" in ckpt:
                world_model.load_state_dict(ckpt["world_model"])
            if "tokenizer" in ckpt:
                tokenizer.load_state_dict(ckpt["tokenizer"])
            print(f"[Resume] Successfully resumed from step {ckpt.get('step', 0)}!")
        else:
            print(f"[Resume] Warning: Checkpoint path '{resume_path}' not found. Starting from scratch.")

    if args.env == "all":
        print(f"Collecting {args.episodes} training trajectories across ALL 4 game universes...")
        episodes = collect_game_trajectories("all", num_episodes=args.episodes, episode_length=args.episode_length)
    else:
        game_meta = GAME_REGISTRY[args.env]
        print(f"Collecting {args.episodes} training trajectories from {game_meta['title']}...")
        episodes = collect_game_trajectories(args.env, num_episodes=args.episodes, episode_length=args.episode_length)

    # Stage 1: Train and save VQ-VAE visual tokenizer
    train_tokenizer(
        tokenizer,
        episodes,
        epochs=args.tokenizer_epochs,
        batch_size=16,
        device=device,
        checkpoint_dir=train_cfg.checkpoint_dir,
    )

    # Stage 2: Tokenize trajectories and train dynamics
    print("Pre-tokenizing trajectories for fast dynamics training...")
    dataset = WorldModelDataset(episodes, tokenizer, wm_cfg, seq_frames=4, device=device)
    print(f"Generated {len(dataset)} transition chunks for training.")

    # Free memory from tokenizer training before starting dynamics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_world_model(
        world_model,
        dataset,
        train_cfg,
        device=device,
        amp=args.amp,
        grad_accum_steps=args.grad_accum,
        use_adam8bit=args.adam8bit,
        grad_checkpoint=args.grad_checkpoint,
    )


if __name__ == "__main__":
    main()
