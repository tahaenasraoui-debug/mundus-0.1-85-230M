"""
Mundus 0.1 - Live Neural Game Engine Simulator.
Zero-dependency browser game engine streaming to an HTML5 canvas via standard HTTP/JSON.
Supports:
  - 100% Neural Mode (world model autoregressive imagination from past KVs)
  - Ground Truth Mode (direct physics simulation)
  - Multi-game suite: Doom 3D, Platformer, Dungeon RPG, Grand Prix Racer
"""

import argparse
import base64
import io
import json
import math
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_TOKENIZER_PATH,
    TokenizerConfig,
    WorldModelConfig,
    get_config,
    get_config_85m,
    get_config_230m,
    get_config_3b,
)
from games import GAME_REGISTRY, get_game_env, list_games
from tokenizer import VQVAE
from world_model import WorldModel


VALID_MODES = {"neural", "ground_truth"}


def get_safe_device(requested: str = "auto") -> str:
    """Safely determines if CUDA is actually usable without kernel mismatch crashes."""
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda") + 1.0
            return "cuda"
        except Exception:
            print("[NeuralEngine] CUDA detected, but current PyTorch build lacks compatible kernels. Falling back to CPU.")
            return "cpu"
    return "cpu"


class NeuralEngineState:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model_type: str = "auto",
        device: str = "auto",
        env_name: str = "doom",
        tokenizer_path: str | Path | None = None,
    ):
        self.device = get_safe_device(device)
        print(f"[NeuralEngine] Running on device: {self.device.upper()}")
        if self.device == "cpu":
            torch.set_num_threads(8)

        print("[NeuralEngine] Initializing Mundus Engine and VQ-VAE Tokenizer...")
        self.tok_config = TokenizerConfig()
        self.tokenizer = VQVAE(self.tok_config).to(self.device)

        tok_file = Path(tokenizer_path) if tokenizer_path is not None else DEFAULT_TOKENIZER_PATH
        tok_loaded = False
        if tok_file.exists():
            print(f"[NeuralEngine] Loading permanent visual codebook from {tok_file}...")
            self.tokenizer.load_state_dict(torch.load(tok_file, map_location=self.device, weights_only=False))
            tok_loaded = True

        ckpt_file = Path(checkpoint_path) if checkpoint_path is not None else None
        if ckpt_file is not None and ckpt_file.exists():
            print(f"[NeuralEngine] Loading weights from {ckpt_file}...")
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            if "config" in ckpt:
                self.wm_config = ckpt["config"]
                print(
                    f"[NeuralEngine] Detected model config: {self.wm_config.model_name} "
                    f"(Layers: {self.wm_config.n_layers}, Dim: {self.wm_config.d_model})"
                )
            else:
                self.wm_config = get_config(model_type) if model_type != "auto" else get_config_85m()

            self.world_model = WorldModel(self.wm_config)
            if "world_model" in ckpt:
                self.world_model.load_state_dict(ckpt["world_model"])
            if "tokenizer" in ckpt and not tok_loaded:
                self.tokenizer.load_state_dict(ckpt["tokenizer"])
            del ckpt

            if self.device == "cuda":
                infer_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                self.world_model = self.world_model.to(device=self.device, dtype=infer_dtype)
                print(f"[NeuralEngine] Loaded checkpoint in {infer_dtype} on CUDA!")
            else:
                self.world_model = self.world_model.to(self.device)
                print("[NeuralEngine] Loaded checkpoint successfully!")
        else:
            self.wm_config = get_config(model_type) if model_type != "auto" else get_config_85m()
            self.world_model = WorldModel(self.wm_config).to(self.device)
            print(f"[NeuralEngine] Running in live zero-shot mode ({self.wm_config.model_name}).")

        self.world_model.eval()
        self.tokenizer.eval()

        self.env_name = env_name if env_name in GAME_REGISTRY else "doom"
        self.env = get_game_env(self.env_name)
        self.reset()

    def valid_actions(self) -> range:
        num_acts = len(GAME_REGISTRY[self.env_name]["actions"])
        return range(0, num_acts)

    def validate_action(self, action: int) -> int:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError(f"Action must be an integer, got {type(action).__name__}")
        valid = self.valid_actions()
        if action not in valid:
            raise ValueError(
                f"Invalid action {action} for environment '{self.env_name}'. "
                f"Expected an integer in [{valid.start}, {valid.stop - 1}]."
            )
        return action

    def validate_mode(self, mode: str) -> str:
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of: {', '.join(VALID_MODES)}.")
        return mode

    def set_env(self, env_name: str):
        if env_name not in GAME_REGISTRY:
            raise ValueError(f"Unknown game '{env_name}'. Choose from: {list(GAME_REGISTRY.keys())}")
        self.env_name = env_name
        self.env = get_game_env(env_name)
        self.reset()

    def reset(self):
        self.current_frame = self.env.reset()
        self.past_kvs = None
        self.step_count = 0
        self.predicted_score = 0.0
        self.gt_score = 0.0
        self.done = False

        frame_t = torch.from_numpy(self.current_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            prompt_tokens = self.tokenizer.encode(frame_t)
            out = self.world_model(prompt_tokens, use_cache=True)
            self.past_kvs = out["new_kvs"]

    def step(self, action: int, mode="neural"):
        action = self.validate_action(action)
        mode = self.validate_mode(mode)

        if self.done:
            self.reset()

        self.step_count += 1
        t0 = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None

        if t0 is not None:
            t0.record()
            wall_t0 = None
        else:
            import time
            wall_t0 = time.perf_counter()

        gt_frame, gt_reward, gt_done = self.env.step(action)
        self.gt_score += float(gt_reward)
        self.done = bool(gt_done)

        with torch.no_grad():
            act_token = torch.tensor([[action + self.world_model.config.vocab_size]], device=self.device)
            next_tok, rew, done_prob, self.past_kvs = self.world_model.imagine_step(
                act_token, past_kvs=self.past_kvs, temperature=0.8, top_k=50
            )

            frame_tokens = [next_tok]
            curr_tok = next_tok
            K = self.world_model.config.tokens_per_frame
            for _ in range(K - 1):
                curr_tok, _, _, self.past_kvs = self.world_model.imagine_step(
                    curr_tok, past_kvs=self.past_kvs, temperature=0.8, top_k=50
                )
                frame_tokens.append(curr_tok)

            all_frame_tokens = torch.cat(frame_tokens, dim=1)
            neural_frame = self.tokenizer.decode_tokens(all_frame_tokens).squeeze(0).permute(1, 2, 0).cpu().numpy()

        if t1 is not None:
            t1.record()
            torch.cuda.synchronize()
            latency_ms = t0.elapsed_time(t1)
        else:
            import time
            latency_ms = (time.perf_counter() - wall_t0) * 1000.0

        self.predicted_score += rew.item()
        display_frame = neural_frame if mode == "neural" else gt_frame

        img_np = (np.clip(display_frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_np).resize((384, 384), Image.NEAREST)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "image": img_b64,
            "step": self.step_count,
            "predicted_reward": round(rew.item(), 3),
            "total_score": round(self.predicted_score, 2),
            "done_prob": round(done_prob.item(), 3),
            "latency_ms": round(latency_ms, 1),
            "gt_reward": round(float(gt_reward), 3),
            "gt_score": round(self.gt_score, 2),
            "gt_done": bool(gt_done),
            "env_name": self.env_name,
        }


ENGINE: NeuralEngineState | None = None


class NeuralEngineHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            total_params = ENGINE.world_model.count_parameters()["total"]
            scale_str = f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.1f}M"
            
            html = HTML_UI.replace("__MODEL_NAME__", ENGINE.wm_config.model_name)
            html = html.replace("__PARAMS_M__", scale_str)
            html = html.replace("__ACTIVE_ENV__", ENGINE.env_name)

            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/step":
            params = parse_qs(parsed.query)
            try:
                action = int(params.get("action", [0])[0])
            except (ValueError, TypeError):
                self._send_json(400, {"error": "Invalid action parameter; must be an integer."})
                return

            mode = params.get("mode", ["neural"])[0]
            try:
                result = ENGINE.step(action, mode=mode)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            self._send_json(200, result)

        elif parsed.path == "/reset":
            ENGINE.reset()
            self._send_json(200, {"status": "reset_complete", "env": ENGINE.env_name})

        elif parsed.path == "/set_env":
            params = parse_qs(parsed.query)
            env_choice = params.get("env", ["doom"])[0]
            try:
                ENGINE.set_env(env_choice)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, {"status": "env_changed", "env": ENGINE.env_name})

        else:
            self._send_json(404, {"error": f"Unknown endpoint '{parsed.path}'."})

    def log_message(self, format, *args):
        return


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🌌 __MODEL_NAME__ - Neural Game Engine</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #090b10;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 24px;
    }
    h1 {
      font-size: 24px;
      margin-bottom: 8px;
      color: #38bdf8;
      letter-spacing: 1px;
    }
    .subtitle {
      font-size: 13px;
      color: #94a3b8;
      margin-bottom: 20px;
    }
    .dashboard {
      display: flex;
      gap: 24px;
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .screen-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
    }
    #screen {
      width: 384px;
      height: 384px;
      border-radius: 8px;
      border: 2px solid #38bdf8;
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
      image-rendering: pixelated;
    }
    #episode-banner {
      display: none;
      width: 384px;
      padding: 8px 12px;
      border-radius: 6px;
      background: #7f1d1d;
      border: 1px solid #ef4444;
      color: #fecaca;
      font-size: 12px;
      text-align: center;
      font-weight: bold;
    }
    .telemetry {
      width: 260px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .stat-card {
      background: #1e293b;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid #334155;
    }
    .stat-label {
      font-size: 11px;
      text-transform: uppercase;
      color: #94a3b8;
      letter-spacing: 0.5px;
    }
    .stat-value {
      font-size: 20px;
      font-weight: bold;
      color: #f8fafc;
      margin-top: 4px;
    }
    .controls {
      margin-top: 20px;
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 12px;
    }
    button {
      background: #0284c7;
      color: white;
      border: none;
      padding: 10px 18px;
      border-radius: 6px;
      font-weight: bold;
      cursor: pointer;
      transition: background 0.2s;
    }
    button:hover { background: #0369a1; }
    .instructions {
      margin-top: 20px;
      font-size: 13px;
      color: #64748b;
      text-align: center;
      line-height: 1.6;
    }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      background: #0369a1;
      border-radius: 4px;
      color: #e0f2fe;
      font-size: 11px;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <h1>🌌 __MODEL_NAME__</h1>
  <div class="subtitle">Autonomous Neural Game Engine Simulator &bull; <span class="badge">__PARAMS_M__ Transformer + VQ-VAE</span></div>

  <div class="dashboard">
    <div class="screen-container">
      <img id="screen" src="" alt="Neural Game Frame" />
      <div id="episode-banner">Episode Finished &mdash; Press R or any key to restart.</div>
    </div>

    <div class="telemetry">
      <div class="stat-card">
        <div class="stat-label">Engine Mode</div>
        <div class="stat-value" id="mode-val" style="color: #38bdf8;">100% Neural</div>
      </div>
      <div class="stat-card">
        <div class="stat-label" id="reward-label">Predicted Reward</div>
        <div class="stat-value" id="reward-val">+0.00</div>
      </div>
      <div class="stat-card">
        <div class="stat-label" id="score-label">Predicted Score</div>
        <div class="stat-value" id="score-val">0.0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label" id="done-label">Done Probability</div>
        <div class="stat-value" id="done-val">0.00</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Step Latency</div>
        <div class="stat-value" id="latency-val">0.0 ms</div>
      </div>
    </div>
  </div>

  <div class="controls" id="env-control-bar" style="margin-bottom: 12px;">
    <label style="font-size: 13px; color: #94a3b8; font-weight: bold; margin-right: 8px;">Active Game:</label>
    <select id="env-select" onchange="changeEnv(this.value); this.blur();" style="background: #1e293b; color: #f8fafc; border: 1px solid #38bdf8; border-radius: 6px; padding: 6px 12px; font-weight: bold; cursor: pointer;">
      <option value="doom" selected>🔴 Doom 3D (First-Person Raycaster)</option>
      <option value="platformer">🟢 Mundus Platformer (2D Mario Physics)</option>
      <option value="dungeon">🗡️ Mundus Dungeon (Zelda Action RPG)</option>
      <option value="racer">🏎️ Mundus Grand Prix (High-Speed Racer)</option>
    </select>
  </div>

  <div class="controls" id="game-controls"></div>
  <div class="instructions" id="instructions-txt"></div>

  <script>
    let isBusy = false;
    let currentMode = "neural";
    let activeEnv = "__ACTIVE_ENV__";

    const GAME_META = {
      doom: {
        controls: "<b>Doom 3D Controls:</b> <b>W / S</b> = Move Fwd/Back &bull; <b>A / D</b> = Turn Left/Right &bull; <b>Space</b> = Fire Shotgun &bull; <b>R</b> = Reset",
        buttons: [
          { act: 1, label: "&larr; Turn L (A)" },
          { act: 3, label: "&uarr; Fwd (W)" },
          { act: 4, label: "&darr; Back (S)" },
          { act: 2, label: "Turn R (D) &rarr;" },
          { act: 5, label: "🔥 FIRE (Space)", color: "#ef4444" }
        ]
      },
      platformer: {
        controls: "<b>Platformer Controls:</b> <b>A / D</b> or <b>Left/Right</b> = Run &bull; <b>Space / W / Up</b> = Jump &bull; <b>S</b> = Crouch &bull; <b>R</b> = Reset",
        buttons: [
          { act: 1, label: "&larr; Run L (A)" },
          { act: 0, label: "Idle" },
          { act: 2, label: "Run R (D) &rarr;" },
          { act: 3, label: "⬆ JUMP (Space/W)", color: "#10b981" },
          { act: 4, label: "&darr; Crouch (S)" }
        ]
      },
      dungeon: {
        controls: "<b>Dungeon RPG Controls:</b> <b>W / S / A / D</b> = Move Cardinal &bull; <b>Space</b> = Sword Attack &bull; <b>R</b> = Reset",
        buttons: [
          { act: 1, label: "&uarr; Up (W)" },
          { act: 2, label: "&darr; Down (S)" },
          { act: 3, label: "&larr; Left (A)" },
          { act: 4, label: "Right (D) &rarr;" },
          { act: 5, label: "⚔️ ATTACK (Space)", color: "#f59e0b" }
        ]
      },
      racer: {
        controls: "<b>Grand Prix Controls:</b> <b>W / Up</b> = Gas &bull; <b>S / Down</b> = Brake &bull; <b>A / D</b> = Steer &bull; <b>Space</b> = Turbo Boost &bull; <b>R</b> = Reset",
        buttons: [
          { act: 1, label: "&larr; Steer L (A)" },
          { act: 3, label: "🏎️ GAS (W)", color: "#0284c7" },
          { act: 4, label: "🛑 BRAKE (S)", color: "#e11d48" },
          { act: 2, label: "Steer R (D) &rarr;" },
          { act: 5, label: "⚡ BOOST (Space)", color: "#8b5cf6" }
        ]
      }
    };

    function updateModeTelemetryLabels() {
      const modeBtn = document.getElementById('mode-btn');
      const modeVal = document.getElementById('mode-val');
      const rewardLabel = document.getElementById('reward-label');
      const scoreLabel = document.getElementById('score-label');
      const doneLabel = document.getElementById('done-label');

      if (currentMode === "neural") {
        if (modeBtn) modeBtn.innerText = "Mode: Neural";
        modeVal.innerText = "100% Neural";
        modeVal.style.color = "#38bdf8";
        rewardLabel.innerText = "Predicted Reward";
        scoreLabel.innerText = "Predicted Score";
        doneLabel.innerText = "Done Probability";
      } else {
        if (modeBtn) modeBtn.innerText = "Mode: Ground Truth";
        modeVal.innerText = "Ground Truth";
        modeVal.style.color = "#10b981";
        rewardLabel.innerText = "Ground Truth Reward";
        scoreLabel.innerText = "Ground Truth Score";
        doneLabel.innerText = "Episode Status";
      }
    }

    function updateInstructions() {
      const meta = GAME_META[activeEnv] || GAME_META.doom;
      const modeDesc = currentMode === "neural"
        ? "<br>Autoregressive neural dynamics by <b>__MODEL_NAME__</b>."
        : "<br>Direct Ground Truth simulator physics.";
      document.getElementById('instructions-txt').innerHTML = meta.controls + modeDesc;
    }

    function toggleMode() {
      currentMode = currentMode === "neural" ? "ground_truth" : "neural";
      updateModeTelemetryLabels();
      updateInstructions();
      sendAction(0);
    }

    async function sendAction(action) {
      if (isBusy) return;
      isBusy = true;
      try {
        const res = await fetch(`/step?action=${action}&mode=${currentMode}`);
        if (!res.ok) {
          const err = await res.json();
          console.error("Step error:", err);
          return;
        }
        const data = await res.json();
        document.getElementById('screen').src = "data:image/jpeg;base64," + data.image;

        if (currentMode === "neural") {
          document.getElementById('reward-val').innerText = (data.predicted_reward >= 0 ? "+" : "") + data.predicted_reward;
          document.getElementById('score-val').innerText = data.total_score;
          document.getElementById('done-val').innerText = data.done_prob;
        } else {
          document.getElementById('reward-val').innerText = (data.gt_reward >= 0 ? "+" : "") + data.gt_reward;
          document.getElementById('score-val').innerText = data.gt_score;
          document.getElementById('done-val').innerText = data.gt_done ? "Ended" : "Active";
        }
        document.getElementById('latency-val').innerText = data.latency_ms + " ms";

        const banner = document.getElementById('episode-banner');
        banner.style.display = data.gt_done ? "block" : "none";
      } catch (e) {
        console.error(e);
      } finally {
        isBusy = false;
      }
    }

    function updateControlButtons() {
      const controlsDiv = document.getElementById('game-controls');
      const meta = GAME_META[activeEnv] || GAME_META.doom;
      const modeBtnText = currentMode === "neural" ? "Mode: Neural" : "Mode: Ground Truth";

      let btnHtml = "";
      for (const btn of meta.buttons) {
        const colorStyle = btn.color ? `background: ${btn.color};` : "";
        btnHtml += `<button onclick="sendAction(${btn.act})" style="${colorStyle}">${btn.label}</button> `;
      }
      btnHtml += `<button onclick="toggleMode()" id="mode-btn" style="background: #8b5cf6;">${modeBtnText}</button> `;
      btnHtml += `<button onclick="resetEngine()" style="background: #e11d48;">Reset</button>`;

      controlsDiv.innerHTML = btnHtml;
      updateModeTelemetryLabels();
      updateInstructions();
    }

    async function changeEnv(envChoice) {
      activeEnv = envChoice;
      document.getElementById('env-select').value = envChoice;
      updateControlButtons();
      await fetch(`/set_env?env=${envChoice}`);
      document.getElementById('episode-banner').style.display = "none";
      sendAction(0);
    }

    async function resetEngine() {
      await fetch('/reset');
      document.getElementById('episode-banner').style.display = "none";
      sendAction(0);
    }

    // Keyboard bindings per active game
    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === "SELECT" || e.target.tagName === "INPUT")) return;

      if (e.key === "r" || e.key === "R") {
        resetEngine();
        return;
      }

      if (activeEnv === "doom") {
        if (e.key === "w" || e.key === "W" || e.key === "ArrowUp") sendAction(3);
        else if (e.key === "s" || e.key === "S" || e.key === "ArrowDown") sendAction(4);
        else if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") sendAction(1);
        else if (e.key === "d" || e.key === "D" || e.key === "ArrowRight") sendAction(2);
        else if (e.key === " " || e.code === "Space") { e.preventDefault(); sendAction(5); }
      }
      else if (activeEnv === "platformer") {
        if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") sendAction(1);
        else if (e.key === "d" || e.key === "D" || e.key === "ArrowRight") sendAction(2);
        else if (e.key === "w" || e.key === "W" || e.key === "ArrowUp" || e.key === " " || e.code === "Space") { e.preventDefault(); sendAction(3); }
        else if (e.key === "s" || e.key === "S" || e.key === "ArrowDown") sendAction(4);
      }
      else if (activeEnv === "dungeon") {
        if (e.key === "w" || e.key === "W" || e.key === "ArrowUp") sendAction(1);
        else if (e.key === "s" || e.key === "S" || e.key === "ArrowDown") sendAction(2);
        else if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") sendAction(3);
        else if (e.key === "d" || e.key === "D" || e.key === "ArrowRight") sendAction(4);
        else if (e.key === " " || e.code === "Space") { e.preventDefault(); sendAction(5); }
      }
      else if (activeEnv === "racer") {
        if (e.key === "w" || e.key === "W" || e.key === "ArrowUp") sendAction(3);
        else if (e.key === "s" || e.key === "S" || e.key === "ArrowDown") sendAction(4);
        else if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") sendAction(1);
        else if (e.key === "d" || e.key === "D" || e.key === "ArrowRight") sendAction(2);
        else if (e.key === " " || e.code === "Space") { e.preventDefault(); sendAction(5); }
      }
    });

    // Initialize UI on load
    updateControlButtons();
    sendAction(0);
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Mundus Live Neural Game Engine")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to world model checkpoint")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(DEFAULT_TOKENIZER_PATH),
        help="Path to standalone VQ-VAE tokenizer checkpoint",
    )
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "85m", "130m", "230m", "3b"], help="Model preset")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device ('auto', 'cpu', 'cuda')")
    playable = [g for g in list_games() if g != "all"]
    parser.add_argument("--env", type=str, default="doom", choices=playable, help=f"Default game environment ({playable})")
    args = parser.parse_args()

    global ENGINE
    ENGINE = NeuralEngineState(
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        device=args.device,
        env_name=args.env,
        tokenizer_path=args.tokenizer,
    )

    server = HTTPServer(("0.0.0.0", args.port), NeuralEngineHandler)
    print("\n=======================================================")
    print("🌌 Mundus 0.1 Live Neural Engine running!")
    print(f"👉 Open in your browser: http://localhost:{args.port}")
    print(f"🎮 Games available: {', '.join(list_games())}")
    print("=======================================================\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down engine.")


if __name__ == "__main__":
    main()
