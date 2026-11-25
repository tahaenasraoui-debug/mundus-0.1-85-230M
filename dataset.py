"""
Self-contained Interactive Environment & Trajectory Dataset Generator.
Simulates arcade physics (paddle, bouncing ball, bricks, rewards) to train the world model.
Requires zero external ROMs or gym dependencies.
"""

import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from config import WorldModelConfig


class PhysicsGameEnv:
    """
    Self-contained 2D Physics Arcade Environment.
    Features:
      - Screen: 64x64 RGB
      - Controllable Paddle: Actions: 0 = Stay, 1 = Move Left, 2 = Move Right
      - Ball with continuous position and velocity (elastic collisions with walls & paddle)
      - Bricks at top: ball destruction yields +2 reward
      - Paddle catch yields +1 reward
      - Missed ball yields -1 reward and episode termination (done=True)
    """

    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        self.paddle_w = 14
        self.paddle_h = 3
        self.ball_radius = 2.0
        self.reset()

    def reset(self) -> np.ndarray:
        self.paddle_x = self.width // 2
        self.ball_x = float(self.width // 2)
        self.ball_y = float(self.height // 3)
        angle = random.uniform(math.pi / 4, 3 * math.pi / 4)
        speed = 2.0
        self.ball_vx = speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)
        self.score = 0.0
        self.steps = 0
        self.max_steps = 100

        # Initialize bricks: 2 rows of 6 bricks
        self.bricks = []
        for row in range(2):
            for col in range(6):
                bx = 6 + col * 9
                by = 10 + row * 6
                self.bricks.append([bx, by, 7, 4, True])  # x, y, w, h, alive

        return self.render()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """
        Actions: 0 = Stay, 1 = Left, 2 = Right
        """
        self.steps += 1
        reward = 0.0
        done = False

        # 1. Update paddle
        if action == 1:
            self.paddle_x = max(self.paddle_w // 2, self.paddle_x - 3)
        elif action == 2:
            self.paddle_x = min(self.width - self.paddle_w // 2, self.paddle_x + 3)

        # 2. Update ball physics
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Wall collisions
        if self.ball_x <= self.ball_radius:
            self.ball_x = self.ball_radius
            self.ball_vx = abs(self.ball_vx)
        elif self.ball_x >= self.width - self.ball_radius:
            self.ball_x = self.width - self.ball_radius
            self.ball_vx = -abs(self.ball_vx)

        if self.ball_y <= self.ball_radius:
            self.ball_y = self.ball_radius
            self.ball_vy = abs(self.ball_vy)

        # Paddle collision
        paddle_top = self.height - 8
        if (paddle_top - self.paddle_h <= self.ball_y <= paddle_top) and (
            abs(self.ball_x - self.paddle_x) <= self.paddle_w // 2 + self.ball_radius
        ):
            self.ball_y = paddle_top - self.ball_radius
            self.ball_vy = -abs(self.ball_vy)
            # Add English based on hit position
            offset = (self.ball_x - self.paddle_x) / (self.paddle_w / 2)
            self.ball_vx += offset * 0.8
            reward += 1.0

        # Brick collisions
        for brick in self.bricks:
            if brick[4]:  # alive
                bx, by, bw, bh = brick[:4]
                if bx <= self.ball_x <= bx + bw and by <= self.ball_y <= by + bh:
                    brick[4] = False
                    self.ball_vy = -self.ball_vy
                    reward += 2.0
                    break

        # Miss / Fall off screen
        if self.ball_y >= self.height - 2:
            reward -= 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        self.score += reward
        obs = self.render()
        return obs, reward, done

    def render(self) -> np.ndarray:
        """Renders 64x64x3 RGB image (float32 normalized to [0, 1])."""
        img = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Background: Dark navy blue
        img[:, :] = [0.05, 0.07, 0.12]

        # Draw Bricks (Gold / Orange)
        for brick in self.bricks:
            if brick[4]:
                bx, by, bw, bh = brick[:4]
                img[by : by + bh, bx : bx + bw] = [0.95, 0.65, 0.15]

        # Draw Paddle (Cyan)
        px1 = max(0, self.paddle_x - self.paddle_w // 2)
        px2 = min(self.width, self.paddle_x + self.paddle_w // 2)
        py = self.height - 8
        img[py : py + self.paddle_h, px1:px2] = [0.2, 0.85, 0.95]

        # Draw Ball (Bright White)
        bx = round(self.ball_x)
        by = round(self.ball_y)
        r = int(self.ball_radius)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    ny, nx = by + dy, bx + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        img[ny, nx] = [1.0, 1.0, 1.0]

        return img


def collect_trajectories(num_episodes: int = 50, episode_length: int = 30) -> list[dict[str, np.ndarray]]:
    """
    Simulates environment and records trajectories of (frames, actions, rewards, dones).
    """
    env = PhysicsGameEnv()
    dataset = []

    for _ in range(num_episodes):
        frames = []
        actions = []
        rewards = []
        dones = []

        obs = env.reset()
        frames.append(obs)

        # Heuristic policy with random exploration
        for _ in range(episode_length):
            # 80% tracker, 20% random exploration
            if random.random() < 0.8:
                if env.ball_x < env.paddle_x - 1:
                    action = 1  # Left
                elif env.ball_x > env.paddle_x + 1:
                    action = 2  # Right
                else:
                    action = 0  # Stay
            else:
                action = random.choice([0, 1, 2])

            next_obs, reward, done = env.step(action)
            frames.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))

            if done:
                break

        dataset.append(
            {
                "frames": np.stack(frames, axis=0),  # (T+1, 64, 64, 3)
                "actions": np.array(actions, dtype=np.int64),  # (T,)
                "rewards": np.array(rewards, dtype=np.float32),  # (T,)
                "dones": np.array(dones, dtype=np.float32),  # (T,)
            }
        )

    return dataset


class WorldModelDataset(Dataset):
    """
    Prepares interleaved token sequences for the World Model:
    Sequence format per transition:
        [Visual_Tokens_t, Action_Token_t, Visual_Tokens_{t+1}, ...]
    """

    def __init__(
        self,
        episodes: list[dict[str, np.ndarray]],
        tokenizer,
        config: WorldModelConfig,
        seq_frames: int = 4,  # number of transitions per slice
        device: str = "cpu",
    ):
        self.config = config
        self.seq_frames = seq_frames
        self.device = device
        self.samples = []

        # Pre-tokenize all frames to integers to make training fast
        tokenizer.eval()
        with torch.no_grad():
            for ep in episodes:
                frames_np = ep["frames"]  # (T+1, 64, 64, 3)
                frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().to(device)
                tokens = tokenizer.encode(frames_t).cpu()  # (T+1, tokens_per_frame)

                actions = ep["actions"]
                rewards = ep["rewards"]
                dones = ep["dones"]
                T = len(actions)

                # Slice into sub-sequences of length seq_frames
                for i in range(max(1, T - seq_frames + 1)):
                    sub_tokens = tokens[i : i + seq_frames + 1]  # (seq_frames + 1, K)
                    sub_actions = actions[i : i + seq_frames]  # (seq_frames,)
                    sub_rewards = rewards[i : i + seq_frames]  # (seq_frames,)
                    sub_dones = dones[i : i + seq_frames]  # (seq_frames,)

                    if len(sub_actions) == seq_frames:
                        self.samples.append(
                            {
                                "tokens": sub_tokens,
                                "actions": torch.from_numpy(sub_actions),
                                "rewards": torch.from_numpy(sub_rewards),
                                "dones": torch.from_numpy(sub_dones),
                            }
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tokens = sample["tokens"]  # (F+1, K) where K = 64
        actions = sample["actions"]  # (F,)
        rewards = sample["rewards"]  # (F,)
        dones = sample["dones"]  # (F,)

        K = self.config.tokens_per_frame
        F = len(actions)

        # Build interleaved input sequence:
        # Step 0: Tokens_0 (K) -> Action_0 (1) -> Tokens_1 (K) -> Action_1 (1) ...
        # Total length = (K + 1) * F + K
        input_tokens = []
        target_tokens = []
        target_rewards = []
        target_dones = []
        visual_mask = []

        # Initial frame tokens
        for k in range(K):
            input_tokens.append(tokens[0, k].item())
            target_tokens.append(tokens[0, k].item())
            target_rewards.append(0.0)
            target_dones.append(0.0)
            visual_mask.append(0.0)  # Do not supervise prompt frame

        # For each transition:
        for t in range(F):
            # 1. Action token
            act_tok = actions[t].item() + self.config.vocab_size
            input_tokens.append(act_tok)
            # Targets at action token: predicts first token of next frame
            target_tokens.append(tokens[t + 1, 0].item())
            target_rewards.append(rewards[t].item())
            target_dones.append(dones[t].item())
            visual_mask.append(1.0)

            # 2. Next frame visual tokens (k = 0 .. K-1)
            for k in range(K):
                input_tokens.append(tokens[t + 1, k].item())
                # Target is next visual token (or 0 at end)
                if k + 1 < K:
                    target_tokens.append(tokens[t + 1, k + 1].item())
                else:
                    target_tokens.append(0)
                target_rewards.append(rewards[t].item())
                target_dones.append(dones[t].item())
                visual_mask.append(1.0 if k + 1 < K else 0.0)

        # Truncate input and target to same length
        seq_len = len(input_tokens) - 1
        inp = torch.tensor(input_tokens[:seq_len], dtype=torch.long)
        tgt = torch.tensor(target_tokens[:seq_len], dtype=torch.long)
        rew = torch.tensor(target_rewards[:seq_len], dtype=torch.float32)
        don = torch.tensor(target_dones[:seq_len], dtype=torch.float32)
        msk = torch.tensor(visual_mask[:seq_len], dtype=torch.float32)

        return {
            "token_seq": inp,
            "target_tokens": tgt,
            "target_rewards": rew,
            "target_dones": don,
            "visual_mask": msk,
        }
