"""
Mundus Games Hub: Unified registry for all neural simulator environments.
Supports:
  1. 'doom'       - 3D First-Person Raycaster (Classic Doom / Wolfenstein 3D)
  2. 'platformer' - 2D Side-Scrolling Physics Platformer (Super Mario style)
  3. 'dungeon'    - Top-Down Action-Adventure RPG (The Legend of Zelda style)
  4. 'racer'      - Top-Down High-Speed Circuit Racer (Micro Machines style)
"""

from typing import Dict, Any, List
import numpy as np

from doom_env import DoomMiniEnv, collect_doom_trajectories
from platformer_env import PlatformerEnv, collect_platformer_trajectories
from dungeon_env import DungeonEnv, collect_dungeon_trajectories
from racer_env import RacerEnv, collect_racer_trajectories


GAME_REGISTRY: Dict[str, Dict[str, Any]] = {
    "doom": {
        "id": "doom",
        "title": "🔴 Doom 3D (First-Person Raycaster)",
        "short_title": "Doom 3D",
        "env_class": DoomMiniEnv,
        "collector": collect_doom_trajectories,
        "controls_html": "<b>Doom 3D Controls:</b> <b>W / S</b> = Move Fwd/Back &bull; <b>A / D</b> = Turn Left/Right &bull; <b>Space</b> = Fire Shotgun &bull; <b>R</b> = Reset<br>DDA 3D perspective &bull; Animated demons & shotgun &bull; Status HUD",
        "actions": [
            {"id": 0, "label": "Idle", "key": "S"},
            {"id": 1, "label": "← Turn L", "key": "A"},
            {"id": 3, "label": "↑ Fwd", "key": "W"},
            {"id": 4, "label": "↓ Back", "key": "S"},
            {"id": 2, "label": "Turn R →", "key": "D"},
            {"id": 5, "label": "🔥 FIRE", "key": "Space", "color": "#ef4444"},
        ],
    },
    "platformer": {
        "id": "platformer",
        "title": "🟢 Mundus Platformer (2D Mario Physics)",
        "short_title": "Platformer",
        "env_class": PlatformerEnv,
        "collector": collect_platformer_trajectories,
        "controls_html": "<b>Platformer Controls:</b> <b>A / D</b> or <b>Left/Right</b> = Run &bull; <b>Space / W / Up</b> = Jump &bull; <b>S</b> = Crouch &bull; <b>R</b> = Reset<br>Parabolic jump physics &bull; Mystery '?' blocks &bull; Stomping Goombas",
        "actions": [
            {"id": 1, "label": "← Run L", "key": "A"},
            {"id": 0, "label": "Idle", "key": "None"},
            {"id": 2, "label": "Run R →", "key": "D"},
            {"id": 3, "label": "⬆ JUMP", "key": "Space", "color": "#10b981"},
            {"id": 4, "label": "↓ Crouch", "key": "S"},
        ],
    },
    "dungeon": {
        "id": "dungeon",
        "title": "🗡️ Mundus Dungeon (Zelda Action RPG)",
        "short_title": "Dungeon RPG",
        "env_class": DungeonEnv,
        "collector": collect_dungeon_trajectories,
        "controls_html": "<b>Dungeon RPG Controls:</b> <b>W / S / A / D</b> = Move Cardinal &bull; <b>Space</b> = Sword Attack &bull; <b>R</b> = Reset<br>Radial torchlight illumination &bull; Skeletons & Slimes &bull; Treasure chests",
        "actions": [
            {"id": 1, "label": "↑ Up", "key": "W"},
            {"id": 2, "label": "↓ Down", "key": "S"},
            {"id": 3, "label": "← Left", "key": "A"},
            {"id": 4, "label": "Right →", "key": "D"},
            {"id": 5, "label": "⚔️ ATTACK", "key": "Space", "color": "#f59e0b"},
        ],
    },
    "racer": {
        "id": "racer",
        "title": "🏎️ Mundus Grand Prix (High-Speed Racer)",
        "short_title": "Grand Prix Racer",
        "env_class": RacerEnv,
        "collector": collect_racer_trajectories,
        "controls_html": "<b>Grand Prix Controls:</b> <b>W / Up</b> = Accelerate &bull; <b>S / Down</b> = Brake &bull; <b>A / D</b> = Steer &bull; <b>Space</b> = Turbo Boost &bull; <b>R</b> = Reset<br>Continuous vehicle momentum &bull; Apex kerbs &bull; Lap timing",
        "actions": [
            {"id": 1, "label": "← Steer L", "key": "A"},
            {"id": 3, "label": "🏎️ GAS", "key": "W", "color": "#0284c7"},
            {"id": 4, "label": "🛑 BRAKE", "key": "S", "color": "#e11d48"},
            {"id": 2, "label": "Steer R →", "key": "D"},
            {"id": 5, "label": "⚡ BOOST", "key": "Space", "color": "#8b5cf6"},
        ],
    },
}


def get_game_env(game_id: str):
    """Instantiates the environment for the given game ID."""
    gid = game_id.lower().strip()
    if gid not in GAME_REGISTRY:
        raise ValueError(f"Unknown game '{game_id}'. Available: {list(GAME_REGISTRY.keys())}")
    return GAME_REGISTRY[gid]["env_class"]()


def collect_all_games_trajectories(num_episodes_per_game: int = 150, episode_length: int = 40) -> List[Dict[str, np.ndarray]]:
    """Records demonstrations across all 4 games and mixes them into a single multi-game dataset."""
    import random
    combined = []
    print(f"\n[Multi-Game Dataset] Collecting demonstrations across all {len(GAME_REGISTRY)} game universes...")
    for gid, meta in GAME_REGISTRY.items():
        print(f"  --> Recording {num_episodes_per_game} episodes from {meta['title']}...")
        eps = meta["collector"](num_episodes=num_episodes_per_game, episode_length=episode_length)
        combined.extend(eps)
    random.shuffle(combined)
    print(f"[Multi-Game Dataset] Total combined dataset: {len(combined)} episodes across 4 games!\n")
    return combined


def collect_game_trajectories(game_id: str, num_episodes: int = 50, episode_length: int = 35) -> List[Dict[str, np.ndarray]]:
    """Records autonomous demonstration trajectories for the given game (or all games)."""
    gid = game_id.lower().strip()
    if gid == "all":
        eps_per_game = max(10, num_episodes // len(GAME_REGISTRY))
        return collect_all_games_trajectories(num_episodes_per_game=eps_per_game, episode_length=episode_length)
    if gid not in GAME_REGISTRY:
        raise ValueError(f"Unknown game '{game_id}'. Available: {list_games()}")
    collector_fn = GAME_REGISTRY[gid]["collector"]
    return collector_fn(num_episodes=num_episodes, episode_length=episode_length)


def list_games() -> List[str]:
    """Returns list of valid game identifiers."""
    return list(GAME_REGISTRY.keys()) + ["all"]
