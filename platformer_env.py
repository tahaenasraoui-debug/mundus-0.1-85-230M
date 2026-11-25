"""
PlatformerEnv: A 2D Side-Scrolling Platformer Game Environment (Super Mario style).
Simulates continuous gravity, parabolic jump arcs, running momentum, tile collisions,
interactive '?' mystery blocks, floating coins, and stomping enemies.
Zero external ROM or C++ dependencies: 100% native Python & NumPy.
"""

import math
import os
import random
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image


class PlatformerEnv:
    """
    2D Side-Scrolling Physics Platformer.
    Actions:
      0 = Idle
      1 = Move Left
      2 = Move Right
      3 = Jump
      4 = Crouch
    """
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        self.ground_y = 48
        self.reset()

    def reset(self) -> np.ndarray:
        self.px = 16.0
        self.py = float(self.ground_y - 8)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.facing_right = True
        self.crouching = False
        self.score = 0
        self.coins_collected = 0
        self.steps = 0
        self.max_steps = 180

        # World layout: platforms, blocks, coins, enemies
        # Blocks: [world_x, world_y, is_question_block, hit_state]
        self.blocks = [
            [36, 30, False, False],
            [44, 30, True, False],
            [52, 30, False, False],
            [80, 24, True, False],
            [96, 32, False, False],
            [104, 32, True, False],
            [130, 26, False, False],
            [138, 26, False, False],
        ]
        # Coins: [world_x, world_y, collected]
        self.coins = [
            [44, 18, False],
            [65, 38, False],
            [80, 14, False],
            [104, 22, False],
            [120, 38, False],
        ]
        # Enemies: [world_x, world_y, vx, alive, squish_timer]
        self.enemies = [
            [60.0, float(self.ground_y - 6), -0.5, True, 0],
            [110.0, float(self.ground_y - 6), -0.5, True, 0],
            [150.0, float(self.ground_y - 6), -0.6, True, 0],
        ]

        self.camera_x = 0.0
        return self.render()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.steps += 1
        reward = 0.0
        done = False

        accel = 0.55
        friction = 0.82
        gravity = 0.45
        jump_power = -3.8

        self.crouching = (action == 4)

        # 1. Action processing
        if action == 1:  # Run Left
            self.vx -= accel
            self.facing_right = False
        elif action == 2:  # Run Right
            self.vx += accel
            self.facing_right = True
            reward += 0.05  # Exploration reward for advancing right
        elif action == 3 and self.on_ground:  # Jump!
            self.vy = jump_power
            self.on_ground = False

        # Physics integration
        self.vx *= friction
        self.vx = max(-2.4, min(2.4, self.vx))
        self.vy += gravity
        self.vy = min(4.5, self.vy)

        self.px += self.vx
        self.py += self.vy

        # Prevent moving left beyond world boundary
        if self.px < 8:
            self.px = 8
            self.vx = 0

        # Ground collision
        if self.py >= self.ground_y - 8:
            self.py = float(self.ground_y - 8)
            self.vy = 0.0
            self.on_ground = True

        # Block collisions
        player_w = 6
        player_h = 8 if not self.crouching else 5
        self.on_ground = (self.py >= self.ground_y - 8)

        for b in self.blocks:
            bx, by, is_q, hit = b
            # Check overlap
            if abs(self.px - (bx + 4)) < (player_w / 2 + 4):
                # Landing on top of block
                if self.vy >= 0 and abs((self.py + player_h) - by) < 4:
                    self.py = float(by - player_h)
                    self.vy = 0.0
                    self.on_ground = True
                # Hitting block from underneath
                elif self.vy < 0 and abs(self.py - (by + 8)) < 4:
                    self.py = float(by + 8)
                    self.vy = 0.5
                    if is_q and not hit:
                        b[3] = True  # Block has been hit
                        reward += 1.5  # Mystery coin reward!
                        self.coins_collected += 1

        # Coin collection
        for c in self.coins:
            if not c[2]:
                if math.hypot(self.px - c[0], self.py - c[1]) < 6.0:
                    c[2] = True
                    reward += 1.0
                    self.coins_collected += 1

        # Enemy updates & collisions
        for e in self.enemies:
            if e[3]:  # Alive
                e[0] += e[2]
                # Reverse at boundaries
                if e[0] < 30 or e[0] > 180:
                    e[2] *= -1

                dx = self.px - e[0]
                dy = self.py - e[1]

                # Stomp enemy from above
                if abs(dx) < 7.0 and -8.0 < dy < 2.0 and self.vy > 0:
                    e[3] = False
                    e[4] = 4  # Squish timer
                    self.vy = jump_power * 0.7  # Bounce off enemy!
                    reward += 2.0
                # Hit enemy from side
                elif abs(dx) < 6.0 and abs(dy) < 5.0:
                    reward -= 2.0
                    done = True

        # Goal reached
        if self.px >= 170:
            reward += 5.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        # Smooth camera tracking
        target_cam = max(0.0, self.px - self.width / 3.0)
        self.camera_x += (target_cam - self.camera_x) * 0.2

        obs = self.render()
        return obs, reward, done

    def render(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)
        cam = self.camera_x

        # 1. Vibrant Sky Gradient
        for y in range(self.ground_y):
            f = y / float(self.ground_y)
            frame[y, :] = [0.35 * (1.0 - f) + 0.50, 0.60 * (1.0 - f) + 0.75, 0.95]

        # 2. Clouds (Parallax effect)
        for cy, world_cx, r in [(10, 30, 8), (14, 90, 10), (12, 160, 9)]:
            screen_cx = int(world_cx - cam * 0.4)
            if -r < screen_cx < self.width + r:
                for y in range(max(0, cy - r), min(self.ground_y, cy + r)):
                    for x in range(max(0, screen_cx - r), min(self.width, screen_cx + r)):
                        if (x - screen_cx)**2 + (y - cy)**2 * 2.0 < r**2:
                            frame[y, x] = [0.98, 0.98, 1.0]

        # 3. Ground & Underground Soil
        for y in range(self.ground_y, self.height):
            if y == self.ground_y:
                frame[y, :] = [0.20, 0.85, 0.25]  # Bright Grass
            elif y == self.ground_y + 1:
                frame[y, :] = [0.15, 0.65, 0.20]  # Grass edge
            else:
                frame[y, :] = [0.65, 0.40, 0.22]  # Terracotta soil
                for x in range(self.width):
                    if (int(x + cam) * 7 + y * 13) % 11 == 0:
                        frame[y, x] = [0.45, 0.25, 0.12]

        # 4. Floating Bricks & Mystery Blocks
        for bx, by, is_q, hit in self.blocks:
            sx = int(bx - cam)
            if -8 <= sx < self.width:
                for y in range(by, by + 8):
                    for x in range(sx, sx + 8):
                        if 0 <= x < self.width and 0 <= y < self.height:
                            if is_q:
                                if hit:
                                    # Used brown block
                                    frame[y, x] = [0.55, 0.35, 0.20]
                                    if (x in (sx, sx+7)) or (y in (by, by+7)):
                                        frame[y, x] = [0.35, 0.20, 0.10]
                                else:
                                    # Glowing Golden '?' Block
                                    frame[y, x] = [0.95, 0.80, 0.10]
                                    if (x in (sx, sx+7)) or (y in (by, by+7)):
                                        frame[y, x] = [0.70, 0.55, 0.05]
                                    if (x == sx+4 and by+2 <= y <= by+4) or (x == sx+4 and y == by+6):
                                        frame[y, x] = [1.0, 1.0, 1.0]
                            else:
                                # Terracotta Brick
                                frame[y, x] = [0.80, 0.32, 0.18]
                                if (x in (sx, sx+7)) or (y in (by, by+7)) or y == by+4:
                                    frame[y, x] = [0.45, 0.15, 0.08]

        # 5. Floating Gold Coins
        for cx, cy, collected in self.coins:
            if not collected:
                scx = int(cx - cam)
                if -4 <= scx < self.width + 4:
                    for y in range(cy - 3, cy + 4):
                        for x in range(scx - 2, scx + 3):
                            if 0 <= x < self.width and 0 <= y < self.height:
                                if (x - scx)**2 + (y - cy)**2 <= 6:
                                    frame[y, x] = [1.0, 0.90, 0.15]
                                    if x == scx and y == cy:
                                        frame[y, x] = [1.0, 1.0, 0.6]

        # 6. Goomba Enemies
        for gx, gy, _, alive, sq in self.enemies:
            sgx = int(gx - cam)
            if -8 <= sgx < self.width + 8:
                if alive:
                    for y in range(int(gy), int(gy) + 6):
                        for x in range(sgx - 3, sgx + 4):
                            if 0 <= x < self.width and 0 <= y < self.height:
                                frame[y, x] = [0.55, 0.28, 0.12]
                                if y == int(gy) + 2 and (x == sgx - 1 or x == sgx + 1):
                                    frame[y, x] = [1.0, 1.0, 1.0]
                                if y == int(gy) + 3 and (x == sgx - 1 or x == sgx + 1):
                                    frame[y, x] = [0.0, 0.0, 0.0]
                elif sq > 0:
                    # Squished flat goomba
                    for x in range(sgx - 4, sgx + 5):
                        if 0 <= x < self.width:
                            frame[int(gy) + 4 : int(gy) + 6, x] = [0.45, 0.22, 0.10]

        # 7. Hero Adventurer (Red Cap, Overalls)
        spx = int(self.px - cam)
        spy = int(self.py)

        if 0 <= spx < self.width and 0 <= spy < self.height:
            # Red Cap
            frame[spy : spy + 2, max(0, spx - 3) : min(self.width, spx + 4)] = [0.92, 0.15, 0.15]
            # Face Skin
            frame[spy + 2 : spy + 4, max(0, spx - 2) : min(self.width, spx + 3)] = [0.95, 0.75, 0.60]
            # Eye & Mustache (directed by facing)
            eye_offset = 1 if self.facing_right else -1
            if 0 <= spx + eye_offset < self.width:
                frame[spy + 2, spx + eye_offset] = [0.0, 0.0, 0.0]
                frame[spy + 3, spx + eye_offset] = [0.2, 0.1, 0.05]
            # Blue Overalls
            frame[spy + 4 : spy + 7, max(0, spx - 2) : min(self.width, spx + 3)] = [0.15, 0.35, 0.85]
            # Brown Shoes
            frame[spy + 7, max(0, spx - 3) : min(self.width, spx - 1)] = [0.4, 0.2, 0.1]
            frame[spy + 7, max(0, spx + 1) : min(self.width, spx + 3)] = [0.4, 0.2, 0.1]

        # 8. HUD (Coin counter & score at top left)
        # Mini coin icon
        frame[3 : 6, 3 : 5] = [1.0, 0.9, 0.2]
        # Coin count pips
        for c in range(min(10, self.coins_collected)):
            frame[4, 7 + c * 3] = [1.0, 0.9, 0.2]

        return frame.astype(np.float32)


def collect_platformer_trajectories(num_episodes: int = 50, episode_length: int = 35) -> List[Dict[str, np.ndarray]]:
    """Simulates autonomous platformer agent traversing the level."""
    env = PlatformerEnv()
    dataset = []

    for _ in range(num_episodes):
        frames, actions, rewards, dones = [], [], [], []
        obs = env.reset()
        frames.append(obs)

        for _ in range(episode_length):
            # Intelligent platformer policy:
            # Advance right (action 2)
            # If an obstacle or enemy is ahead or randomly, JUMP (action 3)
            # Occasionally idle or duck
            p = random.random()
            if p < 0.65:
                action = 2  # Run Right
            elif p < 0.88:
                action = 3  # Jump!
            elif p < 0.95:
                action = 1  # Tap Left
            else:
                action = 0  # Idle

            next_obs, reward, done = env.step(action)
            frames.append(next_obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))

            if done:
                break

        dataset.append({
            "frames": np.stack(frames, axis=0),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
        })

    return dataset


if __name__ == "__main__":
    env = PlatformerEnv()
    frame = env.reset()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "assets"), exist_ok=True)
    img = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img).resize((384, 384), Image.NEAREST).save(os.path.join(base_dir, "assets", "platformer_preview.png"))
    print("Saved platformer preview to assets/platformer_preview.png")
