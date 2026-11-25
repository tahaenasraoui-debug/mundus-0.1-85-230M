"""
Pre-trains and permanently freezes the VQ-VAE Visual Tokenizer.
Saves to checkpoints/tokenizer.pt so the visual codebook is fixed forever.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_CHECKPOINT_DIR, TokenizerConfig  # noqa: E402
from dataset import collect_trajectories  # noqa: E402
from doom_env import collect_doom_trajectories  # noqa: E402
from tokenizer import VQVAE  # noqa: E402


def train_and_freeze_tokenizer(
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
    env_name: str = "arcade",
    num_episodes: int = 80,
    episode_length: int = 40,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Path:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CHECKPOINT_DIR

    print(f"Training permanent VQ-VAE Visual Tokenizer on device: {resolved_device}...")
    tok_cfg = TokenizerConfig()
    tokenizer = VQVAE(tok_cfg).to(resolved_device)

    # 1. Collect rich visual frames
    print(f"Collecting {num_episodes} {env_name} episodes for visual training...")
    if env_name == "doom":
        episodes = collect_doom_trajectories(num_episodes=num_episodes, episode_length=episode_length)
    else:
        episodes = collect_trajectories(num_episodes=num_episodes, episode_length=episode_length)

    all_frames = [f for ep in episodes for f in ep["frames"]]
    frames_tensor = torch.from_numpy(np.stack(all_frames)).permute(0, 3, 1, 2).float()  # (N, 3, 64, 64)
    print(f"Collected {len(frames_tensor)} total visual frames.")

    loader = DataLoader(TensorDataset(frames_tensor), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(tokenizer.parameters(), lr=lr)

    tokenizer.train()
    for ep in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        for batch in loader:
            x = batch[0].to(resolved_device)
            optimizer.zero_grad()
            out = tokenizer(x)
            loss = out["total_loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        dt = time.time() - t0
        print(f"Epoch {ep + 1}/{epochs} | Loss: {avg_loss:.5f} | Time: {dt:.1f}s")

    # 2. Save permanent tokenizer checkpoint
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / "tokenizer.pt"
    torch.save(tokenizer.state_dict(), save_path)
    print(f"\n--> Successfully saved permanent tokenizer to: {save_path}")

    # 3. Save a visual reconstruction check image
    tokenizer.eval()
    with torch.no_grad():
        sample_idx = min(10, len(frames_tensor) - 1)
        test_frame = frames_tensor[sample_idx : sample_idx + 1].to(resolved_device)
        recon = tokenizer(test_frame)["reconstruction"].cpu().squeeze(0).permute(1, 2, 0).numpy()
        orig = test_frame.cpu().squeeze(0).permute(1, 2, 0).numpy()

        # Side by side: Original vs Reconstructed
        side_by_side = np.concatenate([orig, recon], axis=1)
        side_img = (np.clip(side_by_side, 0.0, 1.0) * 255.0).astype(np.uint8)
        img_path = ckpt_dir / "reconstruction_check.png"
        Image.fromarray(side_img).resize((512, 256), Image.NEAREST).save(img_path)
        print(f"--> Saved visual verification image to: {img_path}")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Pre-train and freeze the Mundus VQ-VAE visual tokenizer")
    parser.add_argument("--env", type=str, default="arcade", choices=["arcade", "doom"], help="Environment to sample")
    parser.add_argument("--episodes", type=int, default=80, help="Number of episodes to collect")
    parser.add_argument("--episode_length", type=int, default=40, help="Steps per episode")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory to write tokenizer.pt and reconstruction_check.png",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override ('cpu' or 'cuda')")
    args = parser.parse_args()

    train_and_freeze_tokenizer(
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        env_name=args.env,
        num_episodes=args.episodes,
        episode_length=args.episode_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
