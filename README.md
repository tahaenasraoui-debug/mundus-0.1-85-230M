# 🌌 Mundus 0.1: Autonomous Neural Game Engine

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Mundus--0.1-yellow)](https://huggingface.co)

> *"Mundus"* — Latin for *World / Universe*.  
> A high-capacity, two-stage **Causal Transformer World Model** engineered for latent dynamics simulation, interactive environment rollouts, and model-based RL.

---

<p align="center">
  <img src="assets/dashboard.png" alt="Mundus 0.1 Live Neural Game Engine Dashboard" width="750" />
</p>

<p align="center">
  <em>Live interactive browser player running on Mundus 0.1: Arrow-key actions feed into the Transformer, which predicts game dynamics, collisions, and rewards in real-time.</em>
</p>

---

## ⚡ Model Specifications

| Specification | Mundus-85M (Active) | Mundus-230M (High-Capacity) |
| :--- | :--- | :--- |
| **Parameters** | **88.2 Million** | **232.1 Million** |
| **Layers** | 12 SwiGLU Transformer Blocks | 18 SwiGLU Transformer Blocks |
| **Hidden Dim ($d_{\text{model}}$)** | 768 | 1,024 |
| **Attention Heads** | 12 ($d_{\text{head}} = 64$) | 16 ($d_{\text{head}} = 64$) |
| **FeedForward Dim ($d_{\text{ff}}$)** | 2,048 (SwiGLU) | 2,752 (SwiGLU) |
| **Positional Encoding** | Rotary Position Embeddings (RoPE) | Rotary Position Embeddings (RoPE) |
| **VQ-VAE Codebook** | 2,048 Discrete Visual Codes | 2,048 Discrete Visual Codes |
| **Hardware Footprint** | 1.68 GB VRAM (GTX 1050 Ready) | 3.52 GB RAM (Multi-Core CPU / High VRAM) |

---

## 🏛️ Architecture & Pipeline

```
                                          +-------------------------------+
                                          |   Next Visual Token Logits    |
                                          +-------------------------------+
                                                         ^
                                                         |
[Frame t] ---> [ VQ-VAE Tokenizer ] ---> z_t             |
                                          |              |
                                          v              |
[History] ---> [     Mundus 0.1        ] ----------------+---> [ Predicted Reward  ]
                        ^                                |
                        |                                +---> [ Done Probability   ]
                    [Action a_t]
```

1. **VQ-VAE Visual Tokenizer (`tokenizer.py`)**:
   Compresses continuous $64 \times 64 \times 3$ image frames into an $8 \times 8$ grid of discrete codebook tokens ($K=64$ tokens per frame). Uses straight-through vector quantization with codebook size 2,048.
2. **Causal Dynamics Engine (`world_model.py`)**:
   Models environment transition dynamics:
   $$P(z_{t+1}, r_{t+1}, d_{t+1} \mid z_{\le t}, a_{\le t})$$
   Interleaves action tokens with visual token sequences so future frames are causally conditioned on agent decisions.
3. **Procedural Game Environment (`dataset.py`)**:
   Self-contained 2D arcade physics with paddle control, elastic ball bounces, brick destruction, and scoring rewards. Zero external ROM dependencies.
4. **Live Interactive Engine (`web_player.py`)**:
   Streams neural rollouts directly to an HTML5 Canvas interface with real-time telemetry (Predicted Reward, Score, Done Probability, and Neural Latency).

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/tahaenasraoui-debug/mundus-world-model.git
cd mundus-world-model

# Recommended: setup with uv
uv venv mundus_env --python 3.12
source mundus_env/bin/activate
uv pip install torch torchvision pillow numpy
```

### 2. Play with the Trained Model (Live Web Player)

Launch the browser engine:

```bash
python3 web_player.py --checkpoint checkpoints/mundus_0.1_step_45000.pt --port 8080
```

Open **`http://localhost:8080`** in your browser.
* **Controls**: **Arrow Keys $\leftarrow$ $\rightarrow$** or **`A` / `D`** (Move), **`S` / Down** (Stay).
* **Mode Toggle**: Click the purple **Mode button** to toggle between **100% Neural Hallucination** and **Real Physics Simulation**.

### 3. Generate an Imagination Rollout GIF

```bash
python3 imagine.py --checkpoint checkpoints/mundus_0.1_step_45000.pt --steps 25 --output_gif mundus_dream.gif
```

### 4. Training from Scratch

To train on your own trajectories:

```bash
# Pre-train and freeze the visual codebook
python3 train_tokenizer.py

# Train the dynamics transformer (e.g. 45k steps on GPU)
python3 train.py --model 85m --device cuda --episodes 300 --steps 45000 --batch_size 2
```

---

## 📦 Project Structure

```
mundus_world_model/
├── assets/
│   └── dashboard.png      # Web player interface screenshot
├── config.py              # Architecture presets (85M, 130M, 230M)
├── tokenizer.py           # VQ-VAE visual compression model
├── world_model.py         # Causal Transformer with RoPE, SwiGLU & KV-cache
├── dataset.py             # Self-contained arcade physics environment
├── train.py               # Dynamics training loop with smart rolling checkpoints
├── train_tokenizer.py     # Standalone visual codebook trainer
├── imagine.py             # Autoregressive rollout GIF generator
├── web_player.py          # Browser game engine streaming to HTML5 canvas
├── desktop_player.py      # Native desktop Pygame interface
├── verify.py              # Automated 5-stage test suite
└── README.md
```

---

## 📜 License

MIT License. Free for research, personal, and commercial experimentation.
