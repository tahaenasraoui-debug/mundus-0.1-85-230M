# 🌌 Mundus 0.1: Autonomous Neural Game Engine

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Mundus--0.1-yellow)](https://huggingface.co/Tahary/mundus-0.1-world-model)

> *"Mundus"* — Latin for *World / Universe*.  
> A high-capacity, two-stage **Causal Transformer World Model** engineered for latent dynamics simulation, interactive environment rollouts, and model-based RL.

---

<p align="center">
  <img src="assets/dashboard.png" alt="Mundus 0.1 Live Neural Game Engine Dashboard" width="700" />
</p>

---

## 🎮 Supported Game Environments

Mundus includes 4 self-contained retro game simulators written in native NumPy (zero external ROM dependencies):

1. **🔴 Doom 3D (`doom_env.py`) — First-Person Raycaster**:
   - DDA 3D perspective projection with directional face lighting & atmospheric fog.
   - Procedural wall shaders: Gothic brick, Tech steel with hazard stripes, Demonic rune pillars.
   - 3D billboard demons with horns, glowing eyes, teeth, and death collapse animations.
   - First-person double-barrel shotgun with wood stock, hands, brass bead, and recoil muzzle blast.
   - Classic Doom status bar HUD with Doomguy face, health, and ammo counters.
   - **Controls**: `W` (Fwd), `S` (Back), `A` (Turn Left), `D` (Turn Right), `Space` (Fire Shotgun), `R` (Reset).

2. **🟢 Mundus Platformer (`platformer_env.py`) — 2D Super Mario Physics**:
   - Continuous Newtonian jumping arcs, momentum, gravity ($g = 0.45$), and camera scrolling.
   - Terracotta bricks, glowing '?' mystery blocks, floating coins, and stomping Goombas.
   - **Controls**: `A` / `D` or `Left` / `Right` (Run), `Space` / `W` / `Up` (Jump), `S` (Crouch), `R` (Reset).

3. **🗡️ Mundus Dungeon (`dungeon_env.py`) — Top-Down Zelda Action RPG**:
   - Dynamic radial torchlight illumination cone with dark dungeon corners.
   - Skeletons and slimes, golden treasure chests with ruby gems.
   - Adventurer hero with tunic, shield, and silver sword attack sweeps.
   - **Controls**: `W` / `S` / `A` / `D` (Move Cardinal), `Space` (Sword Attack), `R` (Reset).

4. **🏎️ Mundus Grand Prix (`racer_env.py`) — Top-Down High-Speed Circuit Racer**:
   - Continuous vehicle kinematics: acceleration, steering, tire drift friction, and grass runoff.
   - Asphalt oval track with red & white apex kerbs, checkered finish line, and AI rival cars.
   - **Controls**: `W` / `Up` (Gas), `S` / `Down` (Brake), `A` / `D` (Steer), `Space` (Turbo Boost), `R` (Reset).

---

## ⚡ Model Specifications

| Specification | Mundus-85M (Fast) | Mundus-230M (High-Capacity) | Mundus-3B (Cloud / Colab) |
| :--- | :--- | :--- | :--- |
| **Parameters** | **88.2 Million** | **232.1 Million** | **3.18 Billion** |
| **Layers** | 12 SwiGLU Blocks | 18 SwiGLU Blocks | 28 SwiGLU Blocks |
| **Hidden Dim ($d_{\text{model}}$)** | 768 | 1,024 | 3,072 |
| **Attention Heads** | 12 ($d_{\text{head}} = 64$) | 16 ($d_{\text{head}} = 64$) | 24 ($d_{\text{head}} = 128$) |
| **FeedForward Dim ($d_{\text{ff}}$)** | 2,048 (SwiGLU) | 2,752 (SwiGLU) | 8,192 (SwiGLU) |
| **Positional Encoding** | RoPE (Rotary) | RoPE (Rotary) | RoPE (Rotary) |
| **VQ-VAE Codebook** | 2,048 Visual Codes | 2,048 Visual Codes | 2,048 Visual Codes |
| **Training Footprint** | 1.68 GB VRAM (GTX 1050) | 3.52 GB VRAM / RAM | ~13.5 GB (Colab T4 w/ 8-bit Adam) |

---

## 🚀 Quickstart

### 1. Installation

```bash
cd mundus-world-model
source mundus_env/bin/activate
pip install -r requirements.txt
```

### 2. Play with the Live Web Player (Browser)

Launch the browser engine:

```bash
python3 web_player.py --port 8080 --env doom
```

Open **`http://localhost:8080`** in your browser.
* Switch between **Doom 3D**, **Platformer**, **Dungeon RPG**, and **Grand Prix Racer** using the dropdown at the top.
* Click the purple **Mode: Neural / Ground Truth** button anytime to toggle between the **100% Neural Hallucination** and the **Real Physics Simulator**.

### 3. Training on Google Colab (3B Model)

To train the 3.18B model on Google Colab GPUs (T4 16GB, L4 24GB, or A100 40GB), open [`train_colab.ipynb`](train_colab.ipynb) or run:

```bash
python3 train.py \
    --model 3b \
    --env doom \
    --episodes 100 \
    --steps 50000 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 2e-4 \
    --amp \
    --adam8bit \
    --device cuda
```

---

## 📦 Project Structure

```
mundus-0.1-85-230M/
├── .github/workflows/
│   └── ci.yml             # Ruff lint and CPU pytest CI workflow
├── assets/
│   ├── dashboard.png      # Web player interface screenshot
│   └── doom_preview.png   # 3D Doom environment preview
├── tests/                 # Unit tests for presets, envs, tokenizer, dataset, and web player
├── doom_env.py            # Self-contained 3D First-Person Raycaster (Doom-style)
├── dataset.py             # 2D Arcade physics game environment & trajectory dataset
├── config.py              # Architecture presets (85M, 130M, 230M)
├── tokenizer.py           # VQ-VAE visual compression model
├── world_model.py         # Causal Transformer with RoPE, SwiGLU & KV-cache
├── train.py               # Dynamics training loop supporting both Doom and Arcade
├── train_tokenizer.py     # Standalone visual codebook trainer
├── imagine.py             # Autoregressive rollout GIF generator
├── web_player.py          # Browser game engine streaming to HTML5 canvas
├── desktop_player.py      # Native desktop Pygame interface (2D Arcade / CPU)
├── upload_to_hf.py        # Hugging Face Hub checkpoint uploader
├── verify.py              # Automated 5-stage 230M verification script
├── pyproject.toml         # Project metadata, Ruff, and pytest configuration
├── requirements.txt       # Python package dependencies
├── LICENSE                # MIT License
└── README.md
```

---

## 📜 License

MIT License. Free for research, personal, and commercial experimentation.
