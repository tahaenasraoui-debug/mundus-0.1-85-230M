"""
DoomMiniEnv: High-Octane 3D First-Person Raycasting Game Engine.
Inspired by Wolfenstein 3D & Classic Doom (like Google's GameNGen).
Zero external ROM or C++ dependencies: 100% native Python & NumPy.
"""

import math
import os
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
      - Procedural wall shaders: Gothic stone brick, Tech steel with hazard stripes, Demonic rune pillars
      - Perspective textured floor & blood dusk sky with jagged mountains
      - 3D Billboard demons with horns, glowing yellow eyes, bloody teeth, and death collapse
      - First-person double-barrel shotgun with wood stock, twin steel barrels, bead sight, and recoil muzzle flash
      - Dynamic crosshair with hitscan feedback
      - Classic Doom status bar HUD with Doomguy avatar, health, and ammo counters

    Actions:
      0 = Idle
      1 = Turn Left
      2 = Turn Right
      3 = Move Forward
      4 = Move Backward
      5 = Fire Shotgun
    """
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        self.hud_h = 10
        self.play_h = height - self.hud_h
        self.half_play_h = self.play_h // 2
        self.fov = math.pi / 2.7  # ~66 degrees field of view

        # 12x12 Dungeon Map (1 = Stone Brick, 2 = Tech Steel + Hazard, 3 = Demonic Rune)
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
        self.player_ammo = 24
        self.muzzle_flash = 0
        self.recoil_offset = 0
        self.hit_flash = 0
        self.bob_phase = 0.0
        self.steps = 0
        self.max_steps = 200

        # Enemies: [x, y, hp, alive, flash, death_anim]
        self.enemies = [
            {"x": 6.5, "y": 5.5, "hp": 30, "alive": True, "flash": 0, "collapse": 0},
            {"x": 8.5, "y": 2.5, "hp": 30, "alive": True, "flash": 0, "collapse": 0},
            {"x": 8.5, "y": 8.5, "hp": 30, "alive": True, "flash": 0, "collapse": 0},
        ]
        self.particles: List[Dict] = []
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
        if self.recoil_offset > 0:
            self.recoil_offset -= 1
        if self.hit_flash > 0:
            self.hit_flash -= 1

        for e in self.enemies:
            if e["flash"] > 0:
                e["flash"] -= 1
            if not e["alive"] and e["collapse"] < 4:
                e["collapse"] += 1

        # 1. Action processing
        if action == 1:  # Turn Left
            self.player_angle -= rot_speed
        elif action == 2:  # Turn Right
            self.player_angle += rot_speed
        elif action == 3:  # Move Forward
            dx = math.cos(self.player_angle) * move_speed
            dy = math.sin(self.player_angle) * move_speed
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
        elif action == 5:  # Fire Shotgun
            if self.player_ammo > 0:
                self.player_ammo -= 1
                self.muzzle_flash = 2
                self.recoil_offset = 3
                hit_enemy = self._check_hitscan()
                if hit_enemy:
                    hit_enemy["hp"] -= 15
                    hit_enemy["flash"] = 3
                    self.hit_flash = 2
                    if hit_enemy["hp"] <= 0 and hit_enemy["alive"]:
                        hit_enemy["alive"] = False
                        reward += 2.0  # Reward for demon kill!
                    else:
                        reward += 0.5  # Reward for hitscan connection

        # 2. Exploration bonus
        tile = (int(self.player_x), int(self.player_y))
        if tile not in self.visited_tiles:
            self.visited_tiles.add(tile)
            reward += 0.2

        # 3. Enemy AI (Patrol / approach player / attack)
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
        """Checks if weapon crosshairs hit an enemy in line of sight."""
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
        Renders a 64x64x3 RGB image with:
          - DDA Raycast 3D perspective
          - Procedural wall shaders (Gothic brick, Tech steel hazard, Demonic rune)
          - Textured perspective floor & blood dusk mountain sky
          - 3D Billboard demons with horns, yellow eyes, teeth, and death animation
          - Double-barrel shotgun with wood stock, hands, and muzzle blast
          - Classic Doom status bar (Health bar, Ammo, Doomguy face)
        """
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # 1. Sky & Mountain Silhouettes
        for y in range(self.half_play_h):
            f = y / self.half_play_h
            frame[y, :] = [0.30 * (1.0 - f) + 0.08, 0.04, 0.06]
            mountain_h = 4 + int(math.sin(y * 0.8) * 2)
            if y >= self.half_play_h - mountain_h:
                for x in range(self.width):
                    if (x * 7 + y * 3) % 11 < 8:
                        frame[y, x] = [0.10, 0.02, 0.04]

        # 2. Floor with Perspective Tile Shading
        for y in range(self.half_play_h, self.play_h):
            dist = self.play_h / max(1e-3, 2 * y - self.play_h)
            darkness = max(0.08, min(1.0, 1.0 - dist / 7.0))
            for x in range(self.width):
                tile_pattern = ((x // 4) + (y // 3)) % 2
                base = [0.14, 0.15, 0.18] if tile_pattern == 0 else [0.08, 0.09, 0.11]
                frame[y, x] = [c * darkness for c in base]

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

            step_x = -1 if ray_dir_x < 0 else 1
            side_dist_x = (self.player_x - map_x) * delta_dist_x if ray_dir_x < 0 else (map_x + 1.0 - self.player_x) * delta_dist_x

            step_y = -1 if ray_dir_y < 0 else 1
            side_dist_y = (self.player_y - map_y) * delta_dist_y if ray_dir_y < 0 else (map_y + 1.0 - self.player_y) * delta_dist_y

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

            wall_h = int(min(self.play_h, self.play_h / perp_wall_dist))
            y1 = max(0, self.half_play_h - wall_h // 2)
            y2 = min(self.play_h, self.half_play_h + wall_h // 2)

            depth_shade = max(0.10, 1.0 - (perp_wall_dist / 9.0))
            if side == 1:
                depth_shade *= 0.72  # 3D shadow for Y-walls

            tex_u = int(wall_x * 8) % 8

            for y in range(y1, y2):
                y_norm = (y - y1) / max(1, wall_h)
                tex_v = int(y_norm * 8) % 8

                if hit == 1:  # Dungeon Stone Brick
                    row_stagger = (tex_u + (4 if (tex_v // 4) % 2 == 1 else 0)) % 8
                    is_crevice = (tex_v % 4 == 0) or (row_stagger == 0)
                    if is_crevice:
                        c = np.array([0.15, 0.16, 0.18])
                    else:
                        stone_shade = 0.85 + 0.15 * math.sin(tex_u * 2 + tex_v)
                        c = np.array([0.48, 0.46, 0.52]) * stone_shade

                elif hit == 2:  # Tech Steel with Hazard Stripes
                    if tex_v <= 1 or tex_v >= 6:
                        is_yellow = ((tex_u + tex_v) % 4) < 2
                        c = np.array([0.85, 0.75, 0.10]) if is_yellow else np.array([0.15, 0.15, 0.15])
                    else:
                        is_rivet = (tex_u in (1, 6)) and (tex_v in (2, 5))
                        c = np.array([0.90, 0.90, 0.95]) if is_rivet else np.array([0.35, 0.40, 0.48])

                else:  # Demonic Rune Pillar
                    rune_eye = (3 <= tex_u <= 4) and (3 <= tex_v <= 4)
                    if rune_eye:
                        c = np.array([1.0, 0.2, 0.1])
                    else:
                        c = np.array([0.65, 0.18, 0.15]) * (0.8 + 0.2 * math.sin(tex_u * 3))

                frame[y, col] = np.clip(c * depth_shade, 0.0, 1.0)

        # 4. Render 3D Billboard Demons
        det = plane_x * dir_y - dir_x * plane_y
        inv_det = 1.0 / (det if abs(det) > 1e-6 else 1e-6)

        for e in self.enemies:
            dx = e["x"] - self.player_x
            dy = e["y"] - self.player_y

            transform_x = inv_det * (dir_y * dx - dir_x * dy)
            transform_y = inv_det * (-plane_y * dx + plane_x * dy)

            if transform_y > 0.4:
                screen_x = int((self.width / 2) * (1 + transform_x / transform_y))
                sprite_h = int(abs(self.play_h / transform_y) * 0.95)
                sprite_w = int(sprite_h * 0.75)

                if e["alive"]:
                    sy1 = max(0, self.half_play_h - sprite_h // 2)
                    sy2 = min(self.play_h, self.half_play_h + sprite_h // 2)
                    sx1 = max(0, screen_x - sprite_w // 2)
                    sx2 = min(self.width, screen_x + sprite_w // 2)

                    for sx in range(sx1, sx2):
                        if 0 <= sx < self.width and transform_y < z_buffer[sx]:
                            u = (sx - sx1) / max(1, sprite_w)
                            for sy in range(sy1, sy2):
                                v = (sy - sy1) / max(1, sprite_h)

                                if e["flash"] > 0:
                                    frame[sy, sx] = [1.0, 1.0, 0.8] # Hit flash
                                    continue

                                is_horn = (0.15 <= u <= 0.30 and 0.05 <= v <= 0.25) or (0.70 <= u <= 0.85 and 0.05 <= v <= 0.25)
                                is_head = 0.28 <= u <= 0.72 and 0.18 <= v <= 0.45
                                is_eye = (0.35 <= u <= 0.44 or 0.56 <= u <= 0.65) and (0.25 <= v <= 0.30)
                                is_maw = 0.40 <= u <= 0.60 and 0.35 <= v <= 0.42
                                is_torso = 0.22 <= u <= 0.78 and 0.45 <= v <= 0.80
                                is_legs = (0.26 <= u <= 0.45 or 0.55 <= u <= 0.74) and v >= 0.80

                                if is_eye:
                                    frame[sy, sx] = [1.0, 0.95, 0.10]
                                elif is_maw:
                                    frame[sy, sx] = [0.90, 0.10, 0.10]
                                elif is_horn:
                                    frame[sy, sx] = [0.25, 0.18, 0.12]
                                elif is_head or is_torso:
                                    frame[sy, sx] = [0.75, 0.22, 0.15]
                                elif is_legs:
                                    frame[sy, sx] = [0.55, 0.18, 0.12]

                else:
                    # Dead demon puddle on floor
                    cy = min(self.play_h - 1, self.half_play_h + sprite_h // 4)
                    pw = max(2, sprite_w // 2)
                    sx1 = max(0, screen_x - pw)
                    sx2 = min(self.width, screen_x + pw)
                    frame[cy : min(self.play_h, cy + 2), sx1 : sx2] = [0.65, 0.12, 0.12]

        # 5. First-Person Double-Barrel Shotgun & Hands
        bob_y = int(math.sin(self.bob_phase) * 1.5)
        bob_x = int(math.cos(self.bob_phase * 0.5) * 1.5)
        wx = self.width // 2 + bob_x
        wy = self.play_h - 13 + bob_y - self.recoil_offset

        # Player hands gripping weapon
        frame[wy + 6 : self.play_h, max(0, wx - 11) : min(self.width, wx - 5)] = [0.72, 0.48, 0.35]
        frame[wy + 6 : self.play_h, max(0, wx + 5) : min(self.width, wx + 11)] = [0.72, 0.48, 0.35]
        # Wooden stock / forearm
        frame[wy + 4 : self.play_h, max(0, wx - 6) : min(self.width, wx + 6)] = [0.42, 0.22, 0.12]
        # Twin steel barrels
        frame[wy : wy + 10, max(0, wx - 4) : min(self.width, wx + 4)] = [0.35, 0.38, 0.42]
        # Center barrel divider
        frame[wy : wy + 10, max(0, wx - 1) : min(self.width, wx + 1)] = [0.15, 0.16, 0.18]
        # Barrel bores
        frame[wy : wy + 3, max(0, wx - 3) : min(self.width, wx - 1)] = [0.02, 0.02, 0.02]
        frame[wy : wy + 3, max(0, wx + 1) : min(self.width, wx + 3)] = [0.02, 0.02, 0.02]
        # Brass bead sight pin
        if 0 <= wy < self.height and 0 <= wx < self.width:
            frame[max(0, wy - 1) : wy + 1, wx] = [0.95, 0.85, 0.20]

        # 6. Muzzle Flash Explosion
        if self.muzzle_flash > 0:
            flash_w = 10
            fy1 = max(0, wy - 7)
            for fx in range(wx - flash_w, wx + flash_w):
                if 0 <= fx < self.width:
                    frame[fy1 : wy, fx] = [1.0, 0.85, 0.25]
            frame = np.clip(frame * 1.35, 0.0, 1.0)

        # 7. Crosshair with Hit Marker
        cx, cy = self.width // 2, self.half_play_h
        ch_color = [1.0, 0.2, 0.2] if self.hit_flash > 0 else [0.0, 1.0, 0.8]
        frame[cy, max(0, cx - 2) : min(self.width, cx + 3)] = ch_color
        frame[max(0, cy - 2) : min(self.height, cy + 3), cx] = ch_color

        # 8. Classic Doom Status Bar (HUD)
        frame[self.play_h : self.height, :] = [0.10, 0.12, 0.15]
        frame[self.play_h, :] = [0.25, 0.30, 0.38]
        # Doomguy face avatar box
        box_x = self.width // 2 - 6
        frame[self.play_h + 1 : self.height - 1, box_x : box_x + 12] = [0.20, 0.22, 0.28]
        # Face: hair, skin, eyes
        frame[self.play_h + 2 : self.play_h + 8, box_x + 2 : box_x + 10] = [0.85, 0.65, 0.50]
        frame[self.play_h + 2 : self.play_h + 4, box_x + 2 : box_x + 10] = [0.35, 0.22, 0.12]
        frame[self.play_h + 4, box_x + 4] = [0.1, 0.1, 0.1]
        frame[self.play_h + 4, box_x + 7] = [0.1, 0.1, 0.1]
        # Health bar on left (proportional to HP)
        hp_w = int(max(0, min(15, (self.player_hp / 100.0) * 15)))
        frame[self.play_h + 3 : self.play_h + 7, 3 : 3 + hp_w] = [0.15, 0.75, 0.20]
        # Ammo bar on right (proportional to Ammo)
        ammo_w = int(max(0, min(15, (self.player_ammo / 24.0) * 15)))
        frame[self.play_h + 3 : self.play_h + 7, self.width - 3 - ammo_w : self.width - 3] = [0.85, 0.65, 0.10]

        return frame.astype(np.float32)


def collect_doom_trajectories(num_episodes: int = 50, episode_length: int = 35) -> List[Dict[str, np.ndarray]]:
    """Simulates autonomous Doom explorer agents and records trajectories."""
    env = DoomMiniEnv()
    dataset = []

    for _ in range(num_episodes):
        frames, actions, rewards, dones = [], [], [], []
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
    env = DoomMiniEnv()
    frame = env.reset()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "assets"), exist_ok=True)
    img = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img).resize((384, 384), Image.NEAREST).save(os.path.join(base_dir, "assets", "doom_preview.png"))
    print("Saved enhanced Doom preview to assets/doom_preview.png")
