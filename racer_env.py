"""
RacerEnv: A High-Speed Top-Down 2D Racing Game Environment (Micro Machines / Grand Prix style).
Simulates continuous vehicle dynamics (heading, acceleration, tire grip, drift friction),
track kerbs, checkered start/finish lines, AI rival cars, and lap checkpoints.
Zero external ROM or C++ dependencies: 100% native Python & NumPy.
"""

import math
import os
import random
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image


class RacerEnv:
    """
    Top-Down 2D Circuit Racing Engine.
    Actions:
      0 = Coast (Friction decel)
      1 = Steer Left
      2 = Steer Right
      3 = Accelerate (Gas)
      4 = Brake / Reverse
      5 = Turbo Boost
    """
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        self.center_x = 32.0
        self.center_y = 32.0
        self.r_inner = 14.0
        self.r_outer = 26.0
        self.reset()

    def reset(self) -> np.ndarray:
        # Spawn at bottom of track facing right (counter-clockwise)
        self.px = 32.0
        self.py = 52.0
        self.angle = 0.0  # 0 radians = facing +X (right)
        self.speed = 0.0
        self.max_speed = 2.4
        self.boost_timer = 0
        self.laps_completed = 0
        self.last_checkpoint = 0
        self.steps = 0
        self.max_steps = 180

        # AI rival car [x, y, angle, speed]
        self.rival_angle_pos = math.pi * 0.5  # Position on circle
        self.rival_speed = 1.6
        self.rival_x = self.center_x + math.cos(self.rival_angle_pos) * 20.0
        self.rival_y = self.center_y + math.sin(self.rival_angle_pos) * 20.0

        return self.render()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.steps += 1
        reward = 0.0
        done = False

        accel = 0.18
        brake = 0.25
        friction = 0.94
        turn_rate = 0.16

        if self.boost_timer > 0:
            self.boost_timer -= 1

        # 1. Action processing
        if action == 1:  # Steer Left
            self.angle -= turn_rate * max(0.3, abs(self.speed) / self.max_speed)
        elif action == 2:  # Steer Right
            self.angle += turn_rate * max(0.3, abs(self.speed) / self.max_speed)
        elif action == 3:  # Accelerate
            self.speed += accel
        elif action == 4:  # Brake / Reverse
            self.speed -= brake
        elif action == 5 and self.boost_timer == 0:  # Turbo Boost!
            self.speed += accel * 2.5
            self.boost_timer = 6
            reward += 0.1

        # Friction
        self.speed *= friction

        # Check track surface (Asphalt vs Grass)
        dx = self.px - self.center_x
        dy = self.py - self.center_y
        dist_from_center = math.hypot(dx, dy)
        on_track = (self.r_inner <= dist_from_center <= self.r_outer)

        if not on_track:
            self.speed *= 0.85  # Grass slow down!
            reward -= 0.05
        else:
            reward += 0.05 * (self.speed / self.max_speed)  # Reward forward speed on track

        # Cap speed
        top_spd = self.max_speed * (1.4 if self.boost_timer > 0 else 1.0)
        self.speed = max(-1.0, min(top_spd, self.speed))

        # Position update
        self.px += math.cos(self.angle) * self.speed
        self.py += math.sin(self.angle) * self.speed

        # Keep inside screen bounds
        self.px = max(4.0, min(float(self.width - 4), self.px))
        self.py = max(4.0, min(float(self.height - 4), self.py))

        # Checkpoint tracking around the oval: 4 quadrants
        curr_angle = math.atan2(self.py - self.center_y, self.px - self.center_x)
        # Map [-pi, pi] to [0, 1, 2, 3] quadrants
        quadrant = int(((curr_angle + math.pi) / (2 * math.pi)) * 4) % 4
        if quadrant == (self.last_checkpoint + 1) % 4:
            self.last_checkpoint = quadrant
            reward += 0.5
            if self.last_checkpoint == 0:
                self.laps_completed += 1
                reward += 3.0  # Lap completed!

        # 2. Rival AI car progression along track center
        self.rival_angle_pos += (self.rival_speed / 20.0)
        self.rival_x = self.center_x + math.cos(self.rival_angle_pos) * 20.0
        self.rival_y = self.center_y + math.sin(self.rival_angle_pos) * 20.0

        if self.laps_completed >= 3 or self.steps >= self.max_steps:
            done = True

        obs = self.render()
        return obs, reward, done

    def render(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # 1. Racetrack & Grass
        for y in range(self.height):
            for x in range(self.width):
                dx = x - self.center_x
                dy = y - self.center_y
                dist = math.hypot(dx, dy)

                if self.r_inner <= dist <= self.r_outer:
                    # Dark Asphalt Road
                    frame[y, x] = [0.22, 0.23, 0.25]

                    # Apex Curbs (Red and White alternating curbs)
                    if dist <= self.r_inner + 1.8 or dist >= self.r_outer - 1.8:
                        ang = math.atan2(dy, dx)
                        stripe = int(ang * 6.0) % 2
                        frame[y, x] = [0.90, 0.15, 0.15] if stripe == 0 else [0.95, 0.95, 0.95]

                    # Center Dashed Line
                    elif abs(dist - 20.0) < 0.6:
                        ang = math.atan2(dy, dx)
                        if int(ang * 8.0) % 2 == 0:
                            frame[y, x] = [0.95, 0.95, 0.95]
                else:
                    # Infield & Outfield Grass with texture
                    g = 0.55 + 0.05 * math.sin(x * 2.0 + y * 2.0)
                    frame[y, x] = [0.15, g, 0.18]

        # 2. Checkered Start/Finish Line at top (y: 6..18, x: ~32)
        for y in range(int(self.center_y - self.r_outer), int(self.center_y - self.r_inner)):
            for x in range(30, 34):
                chk = ((x // 2) + (y // 2)) % 2
                frame[y, x] = [1.0, 1.0, 1.0] if chk == 0 else [0.05, 0.05, 0.05]

        # 3. AI Rival Car (Yellow Racer)
        rx = int(self.rival_x)
        ry = int(self.rival_y)
        if 0 <= rx < self.width and 0 <= ry < self.height:
            frame[max(0, ry - 3) : min(self.height, ry + 4), max(0, rx - 3) : min(self.width, rx + 4)] = [0.95, 0.85, 0.10]
            frame[max(0, ry - 1) : min(self.height, ry + 2), max(0, rx - 1) : min(self.width, rx + 2)] = [0.2, 0.2, 0.2]

        # 4. Player Car (Red F1 Racer)
        cx, cy = int(self.px), int(self.py)
        ca = self.angle

        # Draw rotated car body (10x6 box)
        cos_a = math.cos(ca)
        sin_a = math.sin(ca)

        for ox in range(-5, 6):
            for oy in range(-3, 4):
                # Rotate local offset
                rx_rot = int(cx + ox * cos_a - oy * sin_a)
                ry_rot = int(cy + ox * sin_a + oy * cos_a)

                if 0 <= rx_rot < self.width and 0 <= ry_rot < self.height:
                    # Wheels (4 black tires)
                    if (ox in (-4, 3)) and (oy in (-3, 3)):
                        frame[ry_rot, rx_rot] = [0.1, 0.1, 0.1]
                    # Cockpit / Windshield
                    elif abs(ox) <= 1 and abs(oy) <= 1:
                        frame[ry_rot, rx_rot] = [0.2, 0.8, 0.95]
                    # Rear Spoiler
                    elif ox == -5:
                        frame[ry_rot, rx_rot] = [0.75, 0.1, 0.1]
                    # Red Chassis Body
                    else:
                        frame[ry_rot, rx_rot] = [0.95, 0.15, 0.15]

        # 5. Headlight beams if moving forward
        if self.speed > 0.5:
            for l_offset in (-2, 2):
                hx = int(cx + 6 * cos_a - l_offset * sin_a)
                hy = int(cy + 6 * sin_a + l_offset * cos_a)
                if 0 <= hx < self.width and 0 <= hy < self.height:
                    frame[hy, hx] = [1.0, 1.0, 0.6]

        # 6. Lap Counter (Pips in top left)
        for l in range(min(5, self.laps_completed)):
            frame[3, 3 + l * 4 : 5 + l * 4] = [0.95, 0.85, 0.10]

        return frame.astype(np.float32)


def collect_racer_trajectories(num_episodes: int = 50, episode_length: int = 35) -> List[Dict[str, np.ndarray]]:
    """Simulates autonomous race driver agents lapping the circuit."""
    env = RacerEnv()
    dataset = []

    for _ in range(num_episodes):
        frames, actions, rewards, dones = [], [], [], []
        obs = env.reset()
        frames.append(obs)

        for _ in range(episode_length):
            # Autonomous racing controller:
            # Desired heading is tangential to oval (counter-clockwise)
            dx = env.px - env.center_x
            dy = env.py - env.center_y
            target_angle = math.atan2(dy, dx) + math.pi / 2.0
            angle_diff = (target_angle - env.angle + math.pi) % (2 * math.pi) - math.pi

            # Steer towards tangent
            if angle_diff > 0.1:
                action = 2  # Steer Right
            elif angle_diff < -0.1:
                action = 1  # Steer Left
            else:
                p = random.random()
                if p < 0.75:
                    action = 3  # Gas!
                elif p < 0.90:
                    action = 5  # Turbo boost!
                else:
                    action = 0  # Coast

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
    env = RacerEnv()
    frame = env.reset()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, "assets"), exist_ok=True)
    img = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(img).resize((384, 384), Image.NEAREST).save(os.path.join(base_dir, "assets", "racer_preview.png"))
    print("Saved racer preview to assets/racer_preview.png")
