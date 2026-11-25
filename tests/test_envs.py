"""Tests for PhysicsGameEnv (2D Arcade) and DoomMiniEnv (3D Raycaster)."""

import math

import numpy as np

from dataset import PhysicsGameEnv, collect_trajectories
from doom_env import DoomMiniEnv, collect_doom_trajectories


def test_physics_game_env_basics_and_boundaries():
    env = PhysicsGameEnv()
    obs = env.reset()
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.float32
    assert 0.0 <= float(obs.min()) <= float(obs.max()) <= 1.0
    assert len(env.bricks) == 12
    assert all(b[4] for b in env.bricks)

    start_x = env.paddle_x
    env.step(0)
    assert env.paddle_x == start_x

    for _ in range(30):
        env.step(1)
    assert env.paddle_x == env.paddle_w // 2

    for _ in range(40):
        env.step(2)
    assert env.paddle_x == env.width - env.paddle_w // 2


def test_physics_game_env_collisions_and_termination():
    env = PhysicsGameEnv()
    env.reset()

    # Force a paddle catch
    paddle_top = env.height - 8
    env.paddle_x = 32
    env.ball_x = 32.0
    env.ball_y = float(paddle_top - 2)
    env.ball_vx = 0.0
    env.ball_vy = 1.5
    _, reward, done = env.step(0)
    assert reward >= 1.0
    assert not done

    # Force a brick destruction
    bx, by, bw, bh, _ = env.bricks[0]
    env.ball_x = float(bx + bw / 2)
    env.ball_y = float(by + bh / 2)
    env.ball_vx = 0.0
    env.ball_vy = 0.0
    _, reward, done = env.step(0)
    assert reward >= 2.0
    assert not env.bricks[0][4]
    assert not done

    # Force ball miss -> termination
    env.ball_y = float(env.height - 2)
    env.ball_vy = 1.0
    _, reward, done = env.step(0)
    assert reward == -1.0
    assert done


def test_collect_trajectories_shapes():
    episodes = collect_trajectories(num_episodes=2, episode_length=8)
    assert len(episodes) == 2
    for ep in episodes:
        t_steps = len(ep["actions"])
        assert 1 <= t_steps <= 8
        assert ep["frames"].shape == (t_steps + 1, 64, 64, 3)
        assert ep["actions"].shape == (t_steps,)
        assert ep["rewards"].shape == (t_steps,)
        assert ep["dones"].shape == (t_steps,)


def test_doom_env_movement_and_unobstructed_hitscan():
    env = DoomMiniEnv()
    obs = env.reset()
    assert obs.shape == (64, 64, 3)
    assert obs.dtype == np.float32
    assert 0.0 <= float(obs.min()) <= float(obs.max()) <= 1.0

    # Player spawns facing +X directly down corridor toward first enemy at (6.5, 5.5)
    target = env._check_hitscan()
    assert target is env.enemies[0]

    # Two shotgun blasts (15 dmg each) kill a 30 HP enemy
    _, rew1, done1 = env.step(5)
    assert rew1 == 0.5
    assert env.enemies[0]["hp"] == 15
    assert env.enemies[0]["alive"]
    assert not done1

    _, rew2, done2 = env.step(5)
    assert rew2 == 2.0
    assert not env.enemies[0]["alive"]
    assert not done2

    # Turning updates angle
    old_angle = env.player_angle
    env.step(1)
    assert env.player_angle < old_angle
    env.step(2)
    assert math.isclose(env.player_angle, old_angle)


def test_doom_env_wall_occlusion_blocks_shot():
    env = DoomMiniEnv()
    env.reset()

    # Place a solid stone wall between player (2.5, 5.5) and enemy (6.5, 5.5)
    env.map_grid[5, 4] = 1
    assert env._check_hitscan() is None

    initial_hp = env.enemies[0]["hp"]
    _, reward, done = env.step(5)
    assert env.enemies[0]["hp"] == initial_hp
    assert env.enemies[0]["alive"]
    assert reward == 0.0
    assert not done


def test_doom_env_raycast_miss_handled_safely():
    env = DoomMiniEnv()
    env.reset()

    # Zero-length ray safely reports miss
    hit_wall, side, dist, wall_x = env._cast_ray(0.0, 0.0)
    assert hit_wall == 0
    assert side == 0
    assert math.isinf(dist)
    assert wall_x == 0.0

    # Open map with zero walls: rays miss (hit_wall == 0), hitscan and render remain safe
    env.map_grid = np.zeros((12, 12), dtype=np.int32)
    hit_wall, _, dist, _ = env._cast_ray(1.0, 0.0)
    assert hit_wall == 0
    assert math.isinf(dist)

    assert env._check_hitscan() is env.enemies[0]
    frame = env.render()
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.float32
    assert np.isfinite(frame).all()


def test_collect_doom_trajectories_shapes():
    episodes = collect_doom_trajectories(num_episodes=2, episode_length=8)
    assert len(episodes) == 2
    for ep in episodes:
        t_steps = len(ep["actions"])
        assert 1 <= t_steps <= 8
        assert ep["frames"].shape == (t_steps + 1, 64, 64, 3)
        assert ep["actions"].shape == (t_steps,)
        assert ep["rewards"].shape == (t_steps,)
        assert ep["dones"].shape == (t_steps,)
