"""
Pre-trains and permanently freezes the VQ-VAE Visual Tokenizer.
Saves to checkpoints/tokenizer.pt so the visual codebook is fixed forever.
"""

import os
import sys
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image

sys.path.append("/home/tahary/.gemini/antigravity/scratch/world_model_230m")
from config import TokenizerConfig
from tokenizer import VQVAE
from dataset import PhysicsGameEnv, collect_trajectories


def train_and_freeze_tokenizer(device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Training permanent VQ-VAE Visual Tokenizer on device: {device}...")
    tok_cfg = TokenizerConfig()
    tokenizer = VQVAE(tok_cfg).to(device)

    # 1. Collect rich visual frames
    print("Collecting 80 game episodes for visual training...")
    episodes = collect_trajectories(num_episodes=80, episode_length=40)
    all_frames = []
    for ep in episodes:
        for f in ep["frames"]:
            all_frames.append(f)
    frames_tensor = torch.from_numpy(np.stack(all_frames)).permute(0, 3, 1, 2).float() # (N, 3, 64, 64)
    print(f"Collected {len(frames_tensor)} total visual frames.")

    loader = DataLoader(TensorDataset(frames_tensor), batch_size=32, shuffle=True)
    optimizer = optim.Adam(tokenizer.parameters(), lr=1e-3)

    tokenizer.train()
    epochs = 8
    for ep in range(epochs):
        t0 = time.time()
        total_loss = 0.0
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
        print(f"Epoch {ep+1}/{epochs} | Loss: {avg_loss:.5f} | Time: {dt:.1f}s")

    # 2. Save permanent tokenizer checkpoint
    ckpt_dir = "/home/tahary/.gemini/antigravity/scratch/world_model_230m/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, "tokenizer.pt")
    torch.save(tokenizer.state_dict(), save_path)
    print(f"\n--> Successfully saved permanent tokenizer to: {save_path}")

    # 3. Save a visual reconstruction check image
    tokenizer.eval()
    with torch.no_grad():
        test_frame = frames_tensor[10:11].to(device)
        recon = tokenizer(test_frame)["reconstruction"].cpu().squeeze(0).permute(1, 2, 0).numpy()
        orig = test_frame.cpu().squeeze(0).permute(1, 2, 0).numpy()

        # Side by side: Original vs Reconstructed
        side_by_side = np.concatenate([orig, recon], axis=1)
        side_img = (np.clip(side_by_side, 0.0, 1.0) * 255.0).astype(np.uint8)
        img_path = "/home/tahary/.gemini/antigravity/scratch/world_model_230m/checkpoints/reconstruction_check.png"
        Image.fromarray(side_img).resize((512, 256), Image.NEAREST).save(img_path)
        print(f"--> Saved visual verification image to: {img_path}")


if __name__ == "__main__":
    train_and_freeze_tokenizer()
