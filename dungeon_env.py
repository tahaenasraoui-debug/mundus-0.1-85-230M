"""
DungeonEnv: A Top-Down Action-Adventure RPG Dungeon Crawler (The Legend of Zelda style).
Simulates dynamic radial torchlight illumination, room exploration, sword combat arcs,
skeleton and slime monsters, treasure chests, and health mechanics.
Zero external ROM or C++ dependencies: 100% native Python & NumPy.
"""

import math
import os
import random
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image


class DungeonEnv:
    """
    Top-Down Action RPG Dungeon Crawler.
    Actions:
      0 = Idle
      1 = Move Up
      2 = Move Down
      3 = Move Left
      4 = Move Right
      5 = Sword Attack
    """
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        self.reset()

    def reset(self) -> np.ndarray:
        self.px = 32.0
        self.py = 36.0
        self.player_hp = 100
        self.facing = 0  # 0=Up, 1=Down, 2=Left, 3=Right
        self.attack_timer = 0
        self.torch_phase = 0.0
        self.score = 0
        self.steps = 0
        self.max_steps = 180

        # Monsters: [x, y, hp, alive, flash_timer]
        self.monsters = [
            {"x": 20.0, "y": 18.0, "hp": 20, "alive": True, "flash": 0, "type": "skeleton"},
            {"x": 44.0, "y": 44.0, "hp": 15, "alive": True, "flash": 0, "type": "slime"},
            {"x": 18.0, "y": 46.0, "hp": 20, "alive": True, "flash": 0, "type": "skeleton"},
        ]
        # Chests: [x, y, opened]
        self.chests = [
            {"x": 48.0, "y": 16.0, "opened": False},
        ]

        return self.render()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.steps += 1
        reward = 0.0
        done = False
        speed = 2.2

        self.torch_phase += 0.2
        if self.attack_timer > 0:
            self.attack_timer -= 1

        for m in self.monsters:
            if m["flash"] > 0:
                m["flash"] -= 1

        # 1. Action processing
        dx, dy = 0.0, 0.0
        if action == 1:  # Move Up
            dy = -speed
            self.facing = 0
        elif action == 2:  # Move Down
            dy = speed
            self.facing = 1
        elif action == 3:  # Move Left
            dx = -speed
            self.facing = 2
        elif action == 4:  # Move Right
            dx = speed
            self.facing = 3
        elif action == 5:  # Sword Attack!
            self.attack_timer = 3
            # Check sword hitbox in front of player
            sword_reach = 10.0
            ax = self.px + (sword_reach if self.facing == 3 else (-sword_reach if self.facing == 2 else 0))
            ay = self.py + (sword_reach if self.facing == 1 else (-sword_reach if self.facing == 0 else 0))

            for m in self.monsters:
                if m["alive"]:
                    if math.hypot(m["x"] - ax, m["y"] - ay) < 9.0:
                        m["hp"] -= 15
                        m["flash"] = 3
                        reward += 0.8
                        if m["hp"] <= 0:
                            m["alive"] = False
                            reward += 2.5  # Monster slain!

            for ch in self.chests:
                if not ch["opened"]:
                    if math.hypot(ch["x"] - ax, ch["y"] - ay) < 10.0:
                        ch["opened"] = True
                        reward += 3.0  # Opened treasure chest!

        # Wall collisions (Dungeon room bounds [8, 56])
        nx = max(8.0, min(float(self.width - 8), self.px + dx))
        ny = max(8.0, min(float(self.height - 8), self.py + dy))
        self.px = nx
        self.py = ny

        # 2. Monster AI (approach hero)
        alive_count = 0
        for m in self.monsters:
            if m["alive"]:
                alive_count += 1
                dist = math.hypot(self.px - m["x"], self.py - m["y"])
                if dist < 32.0:
                    mx = (self.px - m["x"]) / dist * 0.9
                    my = (self.py - m["y"]) / dist * 0.9
                    m["x"] = max(8.0, min(float(self.width - 8), m["x"] + mx))
                    m["y"] = max(8.0, min(float(self.height - 8), m["y"] + my))

                if dist < 6.0 and self.attack_timer == 0:
                    self.player_hp -= 5
                    reward -= 0.5

        if alive_count == 0 and all(ch["opened"] for ch in self.chests):
            reward += 5.0
            done = True
        elif self.player_hp <= 0 or self.steps >= self.max_steps:
            done = True

        obs = self.render()
        return obs, reward, done

    def render(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # 1. Cobblestone Floor & Stone Walls
        for y in range(self.height):
            for x in range(self.width):
                is_wall = (x < 6 or x >= self.width - 6 or y < 6 or y >= self.height - 6)
                if is_wall:
                    frame[y, x] = [0.25, 0.28, 0.32]
                    if (x in (5, self.width-6)) or (y in (5, self.height-6)) or (x % 8 == 0):
                        frame[y, x] = [0.12, 0.14, 0.16]
                else:
                    c = 0.16 + 0.04 * math.sin(x * 1.5 + y * 1.2)
                    frame[y, x] = [c, c * 0.95, c * 0.85]
                    if (x % 6 == 0) or (y % 6 == 0):
                        frame[y, x] = [0.10, 0.09, 0.08]

        # 2. Dynamic Radial Torchlight
        flicker = 1.0 + 0.05 * math.sin(self.torch_phase)
        for y in range(self.height):
            for x in range(self.width):
                d = math.hypot(x - self.px, y - self.py)
                light = max(0.15, min(1.3, (1.25 - (d / 28.0)**1.5) * flicker))
                frame[y, x] = np.clip(frame[y, x] * light * np.array([1.15, 1.05, 0.90]), 0.0, 1.0)

        # 3. Wall Torches
        for tx, ty in [(6, 16), (self.width - 7, 16), (6, 48), (self.width - 7, 48)]:
            frame[ty : ty + 3, tx - 1 : tx + 2] = [0.4, 0.25, 0.1]
            frame[ty - 2 : ty, tx] = [1.0, 0.7 + 0.2 * math.sin(self.torch_phase * 2.0), 0.1]

        # 4. Treasure Chests
        for ch in self.chests:
            cx, cy = int(ch["x"]), int(ch["y"])
            if ch["opened"]:
                # Open chest with glittering gems inside
                frame[cy : cy + 5, cx - 4 : cx + 4] = [0.65, 0.45, 0.10]
                frame[cy + 1 : cy + 3, cx - 2 : cx + 2] = [0.2, 0.85, 1.0] # Diamond glow
            else:
                # Closed golden chest with ruby lock
                frame[cy : cy + 6, cx - 4 : cx + 4] = [0.85, 0.65, 0.15]
                frame[cy + 2, cx - 4 : cx + 4] = [0.2, 0.15, 0.05]
                frame[cy + 2 : cy + 4, cx - 1 : cx + 1] = [0.95, 0.1, 0.1]

        # 5. Monsters
        for m in self.monsters:
            if m["alive"]:
                mx, my = int(m["x"]), int(m["y"])
                if m["flash"] > 0:
                    frame[max(0, my-3) : min(self.height, my+5), max(0, mx-3) : min(self.width, mx+4)] = [1.0, 1.0, 1.0]
                    continue

                if m["type"] == "skeleton":
                    # Skeleton warrior (White skull & ribs)
                    frame[my - 3 : my, mx - 2 : mx + 3] = [0.92, 0.92, 0.92]
                    frame[my - 2, mx - 1] = [0.1, 0.1, 0.1] # Eyes
                    frame[my - 2, mx + 1] = [0.1, 0.1, 0.1]
                    frame[my : my + 4, mx - 2 : mx + 3] = [0.75, 0.75, 0.75] # Ribcage
                    frame[my + 4 : my + 6, mx - 2] = [0.8, 0.8, 0.8]
                    frame[my + 4 : my + 6, mx + 2] = [0.8, 0.8, 0.8]
                else:
                    # Toxic green slime
                    frame[my - 2 : my + 3, mx - 3 : mx + 4] = [0.20, 0.85, 0.25]
                    frame[my - 1, mx - 1] = [0.1, 0.1, 0.1]
                    frame[my - 1, mx + 1] = [0.1, 0.1, 0.1]

        # 6. Hero Adventurer
        px, py = int(self.px), int(self.py)
        # Green Tunic
        frame[py - 4 : py - 2, px - 2 : px + 3] = [0.15, 0.75, 0.25]
        # Face skin
        frame[py - 2 : py, px - 2 : px + 2] = [0.95, 0.78, 0.60]
        # Eye
        frame[py - 1, px] = [0.1, 0.1, 0.1]
        # Tunic body
        frame[py : py + 4, px - 3 : px + 3] = [0.15, 0.75, 0.25]
        # Blue Shield
        frame[py - 1 : py + 3, px - 5 : px - 3] = [0.20, 0.45, 0.85]

        # Silver Sword Attack Arc
        if self.attack_timer > 0:
            # Sweeping silver slash
            if self.facing == 3:  # Right
                frame[py - 4 : py + 5, px + 3 : px + 8] = [0.90, 0.92, 1.0]
            elif self.facing == 2:  # Left
                frame[py - 4 : py + 5, px - 8 : px - 3] = [0.90, 0.92, 1.0]
            elif self.facing == 0:  # Up
                frame[py - 8 : py - 3, px - 4 : px + 5] = [0.90, 0.92, 1.0]
            else:  # Down
                frame[py + 3 : py + 8, px - 4 : px + 5] = [0.90, 0.92, 1.0]
        else:
            # Resting sword on right
            frame[py - 3 : py + 1, px + 3 : px + 6] = [0.85, 0.88, 0.92]

        # 7. Health Heart Pips (Top Left)
        hearts = max(0, int(self.player_hp // 25))
        for h in range(hearts):
            frame[2 : 5, 2 + h * 5 : 5 + h * 5] = [0.95, 0.15, 0.15]

        return frame.astype(np.float32)


def collect_dungeon_trajectories(num_episodes: int = 50, episode_length: int = 35) -> List[Dict[str, np.ndarray]]:
    """Simulates autonomous dungeon crawling agents exploring and fighting."""
    env = DungeonEnv()
    dataset = []

    for _ in range(num_episodes):
        frames, actions, rewards, dones = [], [], [], []
        obs = env.reset()
        frames.append(obs)

        for _ in range(episode_length):
            # Check if monster or chest is nearby -> attack (action 5)
            near_target = False
            for m in env.monsters:
                if m["alive"] and math.hypot(env.px - m["x"], env.py - m["y"]) < 12.0:
                    near_target = True
                    break
            for ch in env.chests:
                if not ch["opened"] and math.hypot(env.px - ch["x"], env.py - ch["y"]) < 12.0:
                    near_target = True
                    break

            if near_target and random.random() < 0.75:
                action = 5  # Sword attack!
            else:
                p = random.random()
                if p < 0.25: action = 1    # Up
                elif p < 0.50: action = 2  # Down
                elif p < 0.75: action = 3  # Left
                elif p < 0.95: action = 4  # Right
                else: action = 0           # Idle

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
    env = DungeonEnv()
    frame = env.reset()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "assets"), exist_ok=True)
    img = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img).resize((384, 384), Image.NEAREST).save(os.path.join(base_dir, "assets", "dungeon_preview.png"))
    print("Saved dungeon preview to assets/dungeon_preview.png")
