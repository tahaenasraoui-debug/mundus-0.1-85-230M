"""
DoomMiniEnv: A self-contained 3D First-Person Raycasting Game Environment.
Inspired by Wolfenstein 3D & Classic Doom (like Google's GameNGen).
Zero external ROM or C++ dependencies: 100% native Python & NumPy.
"""

import math
import random
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image


class DoomMiniEnv:
    """
    3D First-Person Raycaster Environment (DDA Engine).
    Features:
      - 64x64 First-person 3D perspective projection with DDA raycasting
      - Directional lighting (X/Y wall shading) & depth atmospheric fog
      - Textured brick mortar patterns for walls
      - 3D Billboard enemy demon sprites with glowing eyes & hit reactions
      - First-person double-barrel shotgun with bob, muzzle flash, and crosshair
      - Hitscan shooting mechanics with blood impact rewards

    Actions:
      0 = Idle
      1 = Turn Left
      2 = Turn Right
      3 = Move Forward
      4 = Move Backward
      5 = Fire Weapon
    """
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        self.fov = math.pi / 2.7  # ~66 degrees field of view

        # 12x12 Dungeon Map (1 = Stone Wall, 2 = Rust Tech Wall, 3 = Crimson Wall)
        self.map_grid = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 2, 2, 0, 0, 0, 0, 3, 3, 0, 1],
            [1, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 1],
            [1, 0, 3, 3, 0, 0, 0, 0, 2, 2, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ], dtype=np.int32)

        self.reset()

    def reset(self) -> np.ndarray:
        # Spawn in open hallway looking straight down corridor (+X direction)
        self.player_x = 2.5
        self.player_y = 5.5
        self.player_angle = 0.0
        self.player_hp = 100
        self.muzzle_flash = 0
        self.bob_phase = 0.0
        self.steps = 0
        self.max_steps = 200

        # Enemies: [x, y, hp, alive, flash]
        self.enemies = [
            {"x": 6.5, "y": 5.5, "hp": 30, "alive": True, "flash": 0},
            {"x": 8.5, "y": 2.5, "hp": 30, "alive": True, "flash": 0},
            {"x": 8.5, "y": 8.5, "hp": 30, "alive": True, "flash": 0},
        ]
        self.visited_tiles = set()
        self.visited_tiles.add((int(self.player_x), int(self.player_y)))

        return self.render()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Takes discrete action and returns (obs, reward, done).
        Actions: 0=Stay, 1=Turn Left, 2=Turn Right, 3=Forward, 4=Backward, 5=Fire
        """
        self.steps += 1
        reward = 0.0
        done = False
        move_speed = 0.28
        rot_speed = 0.18

        if self.muzzle_flash > 0:
            self.muzzle_flash -= 1

        for e in self.enemies:
            if e["flash"] > 0:
                e["flash"] -= 1

        # 1. Action processing
        if action == 1:  # Turn Left
            self.player_angle -= rot_speed
        elif action == 2:  # Turn Right
            self.player_angle += rot_speed
        elif action == 3:  # Move Forward
            dx = math.cos(self.player_angle) * move_speed
            dy = math.sin(self.player_angle) * move_speed
            # Collision check with margin
            nx = self.player_x + dx
            ny = self.player_y + dy
            if 0 < nx < 11 and self.map_grid[int(self.player_y), int(nx + math.copysign(0.2, dx))] == 0:
                self.player_x = nx
            if 0 < ny < 11 and self.map_grid[int(ny + math.copysign(0.2, dy)), int(self.player_x)] == 0:
                self.player_y = ny
            self.bob_phase += 0.4
        elif action == 4:  # Move Backward
            dx = math.cos(self.player_angle) * move_speed * 0.7
            dy = math.sin(self.player_angle) * move_speed * 0.7
            nx = self.player_x - dx
            ny = self.player_y - dy
            if 0 < nx < 11 and self.map_grid[int(self.player_y), int(nx - math.copysign(0.2, dx))] == 0:
                self.player_x = nx
            if 0 < ny < 11 and self.map_grid[int(ny - math.copysign(0.2, dy)), int(self.player_x)] == 0:
                self.player_y = ny
            self.bob_phase += 0.3
        elif action == 5:  # Fire Weapon
            self.muzzle_flash = 2
            hit_enemy = self._check_hitscan()
            if hit_enemy:
                hit_enemy["hp"] -= 15
                hit_enemy["flash"] = 3
                if hit_enemy["hp"] <= 0 and hit_enemy["alive"]:
                    hit_enemy["alive"] = False
                    reward += 2.0  # Reward for kill
                else:
                    reward += 0.5  # Reward for hit

        # 2. Exploration bonus
        tile = (int(self.player_x), int(self.player_y))
        if tile not in self.visited_tiles:
            self.visited_tiles.add(tile)
            reward += 0.2

        # 3. Enemy AI (Patrol / approach player)
        alive_count = 0
        for e in self.enemies:
            if e["alive"]:
                alive_count += 1
                dist = math.hypot(self.player_x - e["x"], self.player_y - e["y"])
                if dist < 5.0:  # In sight
                    edx = (self.player_x - e["x"]) / dist * 0.08
                    edy = (self.player_y - e["y"]) / dist * 0.08
                    if self.map_grid[int(e["y"]), int(e["x"] + edx)] == 0:
                        e["x"] += edx
                    if self.map_grid[int(e["y"] + edy), int(e["x"])] == 0:
                        e["y"] += edy

                    # Attack if adjacent
                    if dist < 0.9:
                        self.player_hp -= 10
                        reward -= 1.0

        if alive_count == 0:
            reward += 5.0
            done = True
        elif self.player_hp <= 0 or self.steps >= self.max_steps:
            done = True

        obs = self.render()
        return obs, reward, done

    def _check_hitscan(self) -> dict:
        """Checks if crosshairs hit an enemy in line of sight."""
        closest_enemy = None
        min_dist = 8.0
        for e in self.enemies:
            if not e["alive"]:
                continue
            dx = e["x"] - self.player_x
            dy = e["y"] - self.player_y
            dist = math.hypot(dx, dy)
            angle_to_e = math.atan2(dy, dx) - self.player_angle
            while angle_to_e > math.pi: angle_to_e -= 2 * math.pi
            while angle_to_e < -math.pi: angle_to_e += 2 * math.pi

            if abs(angle_to_e) < 0.25 and dist < min_dist:
                min_dist = dist
                closest_enemy = e
        return closest_enemy

    def render(self) -> np.ndarray:
        """
        Renders a 64x64x3 RGB image with true perspective DDA raycasting,
        wall depth shading, textured brick mortar, 3D billboard enemies,
        crosshair, and animated double-barrel shotgun.
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)
        half_h = self.height // 2

        # 1. Ceiling (Blood Doom Sky Gradient)
        for y in range(half_h):
            factor = y / half_h
            frame[y, :] = [0.22 * (1.0 - factor) + 0.04, 0.05, 0.08]

        # 2. Floor (Textured Industrial Slate Gradient)
        for y in range(half_h, self.height):
            factor = (y - half_h) / half_h
            frame[y, :] = [0.08 * factor + 0.03, 0.09 * factor + 0.03, 0.12 * factor + 0.04]

        # Direction and camera plane vectors
        dir_x = math.cos(self.player_angle)
        dir_y = math.sin(self.player_angle)
        plane_scale = math.tan(self.fov / 2.0)
        plane_x = -dir_y * plane_scale
        plane_y = dir_x * plane_scale

        # 3. DDA Raycast Walls
        z_buffer = [100.0] * self.width
        for col in range(self.width):
            camera_x = 2.0 * col / self.width - 1.0
            ray_dir_x = dir_x + plane_x * camera_x
            ray_dir_y = dir_y + plane_y * camera_x

            map_x = int(self.player_x)
            map_y = int(self.player_y)

            delta_dist_x = 1e30 if ray_dir_x == 0 else abs(1.0 / ray_dir_x)
            delta_dist_y = 1e30 if ray_dir_y == 0 else abs(1.0 / ray_dir_y)

            if ray_dir_x < 0:
                step_x = -1
                side_dist_x = (self.player_x - map_x) * delta_dist_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1.0 - self.player_x) * delta_dist_x

            if ray_dir_y < 0:
                step_y = -1
                side_dist_y = (self.player_y - map_y) * delta_dist_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1.0 - self.player_y) * delta_dist_y

            hit = 0
            side = 0
            for _ in range(24):
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side = 0
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side = 1

                if 0 <= map_x < 12 and 0 <= map_y < 12:
                    if self.map_grid[map_y, map_x] > 0:
                        hit = self.map_grid[map_y, map_x]
                        break
                else:
                    hit = 1
                    break

            if side == 0:
                perp_wall_dist = max(0.1, side_dist_x - delta_dist_x)
                wall_x = self.player_y + perp_wall_dist * ray_dir_y
            else:
                perp_wall_dist = max(0.1, side_dist_y - delta_dist_y)
                wall_x = self.player_x + perp_wall_dist * ray_dir_x

            wall_x -= math.floor(wall_x)
            z_buffer[col] = perp_wall_dist

            wall_h = int(min(self.height, self.height / perp_wall_dist))
            y1 = max(0, half_h - wall_h // 2)
            y2 = min(self.height, half_h + wall_h // 2)

            # Atmospheric depth shading & directional wall shading
            depth_shade = max(0.12, 1.0 - (perp_wall_dist / 9.5))
            if side == 1:
                depth_shade *= 0.75  # Darken Y-walls for 3D realism

            # Mortar column lines
            tex_u = int(wall_x * 8)
            is_mortar_x = (tex_u == 0 or tex_u == 7)

            if hit == 1:
                base_color = np.array([0.45, 0.50, 0.60])  # Slate Stone
            elif hit == 2:
                base_color = np.array([0.70, 0.40, 0.15])  # Rust Tech
            else:
                base_color = np.array([0.75, 0.15, 0.15])  # Crimson Pillar

            wall_color = base_color * depth_shade
            for y in range(y1, y2):
                y_tex = (y - (half_h - wall_h // 2)) / max(1, wall_h)
                row_idx = int(y_tex * 8)
                c = wall_color.copy()
                if is_mortar_x or (row_idx % 2 == 0 and abs(y_tex * 8 - row_idx) < 0.15):
                    c *= 0.6  # Dark mortar line
                frame[y, col] = c

        # 4. Render 3D Billboard Enemy Sprites (Demons)
        det = plane_x * dir_y - dir_x * plane_y
        inv_det = 1.0 / (det if abs(det) > 1e-6 else 1e-6)
        for e in self.enemies:
            if not e["alive"]:
                continue
            dx = e["x"] - self.player_x
            dy = e["y"] - self.player_y

            # Transform sprite with camera matrix
            transform_x = inv_det * (dir_y * dx - dir_x * dy)
            transform_y = inv_det * (-plane_y * dx + plane_x * dy)

            if transform_y > 0.3:
                screen_x = int((self.width / 2) * (1 + transform_x / transform_y))
                sprite_h = int(abs(self.height / transform_y) * 0.8)
                sprite_w = int(sprite_h * 0.65)

                sy1 = max(0, half_h - sprite_h // 2)
                sy2 = min(self.height, half_h + sprite_h // 2)
                sx1 = max(0, screen_x - sprite_w // 2)
                sx2 = min(self.width, screen_x + sprite_w // 2)

                for sx in range(sx1, sx2):
                    if 0 <= sx < self.width and transform_y < z_buffer[sx]:
                        # Demon body silhouette
                        body_c = [1.0, 1.0, 1.0] if e["flash"] > 0 else [0.85, 0.15, 0.15]
                        frame[sy1:sy2, sx] = body_c

                        # Glowing demon eyes
                        eye_y = sy1 + int(sprite_h * 0.25)
                        if sy1 <= eye_y < sy2:
                            frame[eye_y, sx] = [1.0, 0.95, 0.1]

        # 5. Render First-Person Double-Barrel Shotgun
        bob_y = int(math.sin(self.bob_phase) * 2.0)
        bob_x = int(math.cos(self.bob_phase * 0.5) * 2.0)
        wx = self.width // 2 + bob_x
        wy = self.height - 12 + bob_y
        bw = 6

        # Gun metal base
        frame[wy : self.height, max(0, wx - bw) : min(self.width, wx + bw)] = [0.25, 0.27, 0.30]
        # Inner twin barrels
        frame[wy : wy + 3, max(0, wx - 3) : min(self.width, wx - 1)] = [0.05, 0.05, 0.05]
        frame[wy : wy + 3, max(0, wx + 1) : min(self.width, wx + 3)] = [0.05, 0.05, 0.05]

        # 6. Muzzle Flash Animation
        if self.muzzle_flash > 0:
            flash_w = 8
            flash_y = max(0, wy - 6)
            for fx in range(wx - flash_w, wx + flash_w):
                if 0 <= fx < self.width:
                    frame[flash_y : wy, fx] = [1.0, 0.85, 0.2]
            frame = np.clip(frame * 1.35, 0.0, 1.0)

        # 7. Crosshairs
        cx = self.width // 2
        cy = half_h
        frame[cy, max(0, cx - 2) : min(self.width, cx + 3)] = [0.0, 1.0, 0.8]
        frame[max(0, cy - 2) : min(self.height, cy + 3), cx] = [0.0, 1.0, 0.8]

        return frame.astype(np.float32)


def collect_doom_trajectories(num_episodes: int = 50, episode_length: int = 35) -> List[Dict[str, np.ndarray]]:
    """
    Simulates autonomous Doom explorer agents and records trajectories.
    """
    env = DoomMiniEnv()
    dataset = []

    for _ in range(num_episodes):
        frames = []
        actions = []
        rewards = []
        dones = []

        obs = env.reset()
        frames.append(obs)

        for _ in range(episode_length):
            enemy_in_sight = env._check_hitscan()
            if enemy_in_sight and random.random() < 0.7:
                action = 5  # Fire!
            else:
                p = random.random()
                if p < 0.55:
                    action = 3  # Forward
                elif p < 0.75:
                    action = 1  # Turn Left
                elif p < 0.95:
                    action = 2  # Turn Right
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
    print("Testing DoomMiniEnv rendering...")
    env = DoomMiniEnv()
    frame = env.reset()
    print("Initial frame shape:", frame.shape)

    img = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "assets"), exist_ok=True)
    Image.fromarray(img).resize((384, 384), Image.NEAREST).save(os.path.join(base_dir, "doom_preview.png"))
    Image.fromarray(img).resize((384, 384), Image.NEAREST).save(os.path.join(base_dir, "assets", "doom_preview.png"))
    print("Saved preview to doom_preview.png and assets/doom_preview.png")
