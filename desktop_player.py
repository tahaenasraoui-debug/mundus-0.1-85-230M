"""
Mundus 0.1 (230M) - Native Desktop Pygame Player.
Requires python-pygame (sudo pacman -S python-pygame).
Controls: Left Arrow / A (Left), Right Arrow / D (Right), Down / S (Stay).
"""

import sys
import os
import time
import argparse
import numpy as np
import torch

# Suppress CUDA mismatch warnings
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from config import get_config_85m, get_config_130m, get_config_230m, TokenizerConfig
from tokenizer import VQVAE
from world_model import WorldModel
from dataset import PhysicsGameEnv


def main():
    try:
        import pygame
    except ImportError:
        print("Error: pygame is not installed.")
        print("To run the native desktop player on Arch Linux, run:")
        print("  sudo pacman -S python-pygame")
        print("\nOr use the zero-dependency browser player instead:")
        print("  python3 web_player.py --port 8080")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Mundus 0.1 Desktop Player")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    args = parser.parse_args()

    device = "cpu"
    torch.set_num_threads(8)

    print("Initializing Mundus Engine...")
    tok_config = TokenizerConfig()
    tokenizer = VQVAE(tok_config).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        wm_config = ckpt.get("config", get_config_85m())
        print(f"Detected architecture: {wm_config.model_name}")
        world_model = WorldModel(wm_config).to(device)
        world_model.load_state_dict(ckpt["world_model"])
        if "tokenizer" in ckpt:
            tokenizer.load_state_dict(ckpt["tokenizer"])
    else:
        wm_config = get_config_85m()
        world_model = WorldModel(wm_config).to(device)

    world_model.eval()
    tokenizer.eval()

    env = PhysicsGameEnv()
    obs = env.reset()

    # Prime KV-cache
    frame_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        prompt_tokens = tokenizer.encode(frame_t)
        out = world_model(prompt_tokens, use_cache=True)
        past_kvs = out["new_kvs"]

    pygame.init()
    screen_size = 512
    display = pygame.display.set_mode((screen_size, screen_size + 60))
    pygame.display.set_caption("🌌 Mundus 0.1 (230M) - Neural Game Engine")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    running = True
    action = 0
    total_score = 0.0
    latency_ms = 0.0

    print("\n[Controls] A / Left: Move Left | D / Right: Move Right | S / Down: Stay | R: Reset | ESC: Quit")

    while running:
        action = 0 # Default stay
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs = env.reset()
                    frame_t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    with torch.no_grad():
                        prompt_tokens = tokenizer.encode(frame_t)
                        out = world_model(prompt_tokens, use_cache=True)
                        past_kvs = out["new_kvs"]
                    total_score = 0.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = 1
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = 2

        t0 = time.time()
        # Single step through Mundus 0.1 imagination
        with torch.no_grad():
            act_tok = torch.tensor([[action + world_model.config.vocab_size]], device=device)
            next_tok, rew, done, past_kvs = world_model.imagine_step(act_tok, past_kvs=past_kvs)
            tokens = [next_tok]
            curr_tok = next_tok
            for _ in range(world_model.config.tokens_per_frame - 1):
                curr_tok, _, _, past_kvs = world_model.imagine_step(curr_tok, past_kvs=past_kvs)
                tokens.append(curr_tok)

            all_toks = torch.cat(tokens, dim=1)
            frame_np = tokenizer.decode_tokens(all_toks).squeeze(0).permute(1, 2, 0).cpu().numpy()

        latency_ms = (time.time() - t0) * 1000.0
        total_score += rew.item()

        # Render to Pygame surface
        rgb_img = (np.clip(frame_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        # Transpose for pygame surface: (H, W, 3) -> (W, H, 3)
        surf = pygame.surfarray.make_surface(np.transpose(rgb_img, (1, 0, 2)))
        scaled_surf = pygame.transform.scale(surf, (screen_size, screen_size))

        display.fill((15, 23, 42))
        display.blit(scaled_surf, (0, 0))

        # Draw HUD bar
        hud_text = f"Action: {action} | Pred Score: {total_score:+.1f} | Latency: {latency_ms:.0f}ms | Mundus 0.1"
        txt_surf = font.render(hud_text, True, (56, 189, 248))
        display.blit(txt_surf, (10, screen_size + 20))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
