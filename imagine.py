"""
Latent Imagination and Simulation Engine.
Hallucinates environment rollouts given an initial frame and sequence of actions.
Outputs animated GIFs and reward trajectory predictions without interacting with the real world.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import DEFAULT_TOKENIZER_PATH, TokenizerConfig, get_config, get_config_230m
from games import get_game_env, list_games
from tokenizer import VQVAE
from world_model import WorldModel


def save_gif(frames: list[np.ndarray], output_path: str | Path, fps: int = 10):
    """Saves a list of (H, W, 3) float [0, 1] frames as an animated GIF."""
    pil_images = []
    for f in frames:
        # Scale to 256x256 for clear visualization
        img = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img).resize((256, 256), Image.NEAREST)
        pil_images.append(pil_img)

    if pil_images:
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil_images[0].save(
            out_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        print(f"Saved imagination rollout GIF: {out_path}")


@torch.no_grad()
def imagine_trajectory(
    world_model: WorldModel,
    tokenizer: VQVAE,
    initial_frame: np.ndarray,  # (64, 64, 3)
    actions: list[int],  # Sequence of action integers
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu",
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """
    Rolls out future frames purely in the world model's imagination.
    """
    world_model.eval()
    tokenizer.eval()

    # 1. Encode prompt frame into tokens
    frame_t = torch.from_numpy(initial_frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
    prompt_tokens = tokenizer.encode(frame_t)  # (1, K) where K=64
    K = world_model.config.tokens_per_frame

    # 2. Prime the KV cache with the prompt frame tokens
    out = world_model.forward(prompt_tokens, use_cache=True)
    past_kvs = out["new_kvs"]

    imagined_frames = [initial_frame]
    predicted_rewards = []
    predicted_dones = []

    print(f"Beginning imagination rollout for {len(actions)} steps...")

    # 3. Step forward through time
    for step_idx, act in enumerate(actions):
        # Action token
        act_token = torch.tensor([[act + world_model.config.vocab_size]], device=device)

        # Feed action to predict first visual token
        next_tok, rew, done, past_kvs = world_model.imagine_step(
            act_token, past_kvs=past_kvs, temperature=temperature, top_k=top_k
        )
        predicted_rewards.append(rew.item())
        predicted_dones.append(done.item())

        frame_tokens = [next_tok]

        # Autoregressively generate remaining K - 1 visual tokens
        curr_tok = next_tok
        for _ in range(K - 1):
            curr_tok, _, _, past_kvs = world_model.imagine_step(
                curr_tok, past_kvs=past_kvs, temperature=temperature, top_k=top_k
            )
            frame_tokens.append(curr_tok)

        # Decode tokens to reconstructed image
        all_frame_tokens = torch.cat(frame_tokens, dim=1)  # (1, K)
        recon_frame = tokenizer.decode_tokens(all_frame_tokens).squeeze(0).permute(1, 2, 0).cpu().numpy()
        imagined_frames.append(recon_frame)

        print(
            f"  Step {step_idx + 1:02d} | Action: {act} | "
            f"Pred Reward: {rew.item():+.2f} | Done Prob: {done.item():.2f}"
        )

    return imagined_frames, predicted_rewards, predicted_dones


def main():
    parser = argparse.ArgumentParser(description="Simulate futures with Mundus 0.1 World Model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to Mundus 0.1 checkpoint")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(DEFAULT_TOKENIZER_PATH),
        help="Path to standalone VQ-VAE tokenizer checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        choices=["auto", "85m", "130m", "230m", "3b"],
        help="Model preset when no checkpoint config is present",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="doom",
        choices=list_games(),
        help=f"Game environment ({list_games()})",
    )
    parser.add_argument("--steps", type=int, default=15, help="Number of steps to imagine")
    parser.add_argument("--output_gif", type=str, default="mundus_imagination.gif", help="Output GIF path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_config = TokenizerConfig()
    tokenizer = VQVAE(tok_config).to(device)

    # Check for standalone permanent tokenizer
    tok_path = Path(args.tokenizer)
    tok_loaded = False
    if tok_path.exists():
        tokenizer.load_state_dict(torch.load(tok_path, map_location=device, weights_only=False))
        tok_loaded = True

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if ckpt_path is not None and ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "config" in ckpt:
            wm_config = ckpt["config"]
        else:
            wm_config = get_config(args.model) if args.model != "auto" else get_config_230m()
        world_model = WorldModel(wm_config).to(device)
        world_model.load_state_dict(ckpt["world_model"])
        if "tokenizer" in ckpt and not tok_loaded:
            tokenizer.load_state_dict(ckpt["tokenizer"])
    else:
        wm_config = get_config(args.model) if args.model != "auto" else get_config_230m()
        world_model = WorldModel(wm_config).to(device)

    # Grab a starting seed frame from the environment
    env = get_game_env(args.env)
    seed_frame = env.reset()

    # Generate an action script (e.g. Move right for 5 steps, stay for 3, left for remainder)
    action_sequence = [2] * 5 + [0] * 3 + [1] * max(0, args.steps - 8)
    action_sequence = action_sequence[: args.steps]

    frames, _, _ = imagine_trajectory(
        world_model,
        tokenizer,
        initial_frame=seed_frame,
        actions=action_sequence,
        device=device,
    )

    save_gif(frames, args.output_gif)


if __name__ == "__main__":
    main()
