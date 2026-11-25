"""
Mundus 0.1 - Live Interactive Neural Game Engine Player.
Streams real-time neural rollouts and ground-truth simulations to a browser canvas
with zero external GUI dependencies.
"""

import argparse
import base64
import io
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from config import DEFAULT_TOKENIZER_PATH, TokenizerConfig, get_config, get_config_230m
from dataset import PhysicsGameEnv
from doom_env import DoomMiniEnv
from tokenizer import VQVAE
from world_model import WorldModel

BASE_DIR = Path(__file__).resolve().parent

VALID_ENVS = ("arcade", "doom")
VALID_MODES = ("neural", "hybrid", "ground_truth")
ENV_ACTION_RANGES: dict[str, range] = {
    "arcade": range(3),  # 0=Stay, 1=Left, 2=Right
    "doom": range(6),  # 0=Idle, 1=Turn L, 2=Turn R, 3=Fwd, 4=Back, 5=Fire
}


def get_safe_device(requested: str = "auto") -> str:
    """Safely determines if CUDA is actually usable without kernel mismatch crashes."""
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        try:
            # Test a basic tensor operation to verify kernel availability
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

        # Check for standalone permanent tokenizer checkpoint (relative to repo root by default)
        tok_file = Path(tokenizer_path) if tokenizer_path is not None else DEFAULT_TOKENIZER_PATH
        tok_loaded = False
        if tok_file.exists():
            print(f"[NeuralEngine] Loading permanent visual codebook from {tok_file}...")
            self.tokenizer.load_state_dict(torch.load(tok_file, map_location=self.device, weights_only=False))
            tok_loaded = True

        ckpt_file = Path(checkpoint_path) if checkpoint_path is not None else None
        # Auto-detect config from checkpoint if available
        if ckpt_file is not None and ckpt_file.exists():
            print(f"[NeuralEngine] Loading weights from {ckpt_file}...")
            ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            if "config" in ckpt:
                self.wm_config = ckpt["config"]
                print(
                    f"[NeuralEngine] Detected model config: {self.wm_config.model_name} "
                    f"(Layers: {self.wm_config.n_layers}, Dim: {self.wm_config.d_model})"
                )
            else:
                self.wm_config = get_config(model_type) if model_type != "auto" else get_config_230m()

            self.world_model = WorldModel(self.wm_config).to(self.device)
            if "world_model" in ckpt:
                self.world_model.load_state_dict(ckpt["world_model"])
            if "tokenizer" in ckpt and not tok_loaded:
                self.tokenizer.load_state_dict(ckpt["tokenizer"])
            print("[NeuralEngine] Loaded checkpoint successfully!")
        else:
            self.wm_config = get_config(model_type) if model_type != "auto" else get_config_230m()
            self.world_model = WorldModel(self.wm_config).to(self.device)
            print(f"[NeuralEngine] Running in live zero-shot mode ({self.wm_config.model_name}).")

        self.world_model.eval()
        self.tokenizer.eval()

        self.env_name = "doom" if env_name == "doom" else "arcade"
        self.env = DoomMiniEnv() if self.env_name == "doom" else PhysicsGameEnv()
        self.reset()

    def valid_actions(self) -> range:
        """Returns the valid discrete action range for the currently active environment."""
        env_range = ENV_ACTION_RANGES[self.env_name]
        upper = min(env_range.stop, self.world_model.config.num_actions)
        return range(0, upper)

    def validate_action(self, action: int) -> int:
        """Validates that action is an integer within the active environment's valid action space."""
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
        """Validates that mode is one of the supported engine modes."""
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of: {', '.join(VALID_MODES)}.")
        return mode

    def set_env(self, env_name: str):
        if env_name not in VALID_ENVS:
            raise ValueError(f"Invalid environment '{env_name}'. Expected one of: {', '.join(VALID_ENVS)}.")
        self.env_name = env_name
        self.env = DoomMiniEnv() if self.env_name == "doom" else PhysicsGameEnv()
        self.reset()

    def _prime_kv_cache(self, frame: np.ndarray):
        """Primes the world model KV-cache with the provided RGB frame."""
        frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            prompt_tokens = self.tokenizer.encode(frame_t)
            out = self.world_model(prompt_tokens, use_cache=True)
            self.past_kvs = out["new_kvs"]
            self._kv_synced = True

    def reset(self):
        self.current_frame = self.env.reset()  # (64, 64, 3)
        self.past_kvs = None
        self.step_count = 0
        self.predicted_score = 0.0
        self.gt_score = 0.0
        self.done = False
        self._kv_synced = False
        self._prime_kv_cache(self.current_frame)

    def step(self, action: int, mode: str = "neural") -> dict:
        """
        Takes an action and returns the next frame and telemetry metadata.
        mode:
          - "neural": pure neural imagination frame + neural predictions
          - "hybrid": ground truth frame + neural predictions
          - "ground_truth": ground truth simulator frame only (skips autoregressive rollout)
        """
        action = self.validate_action(action)
        mode = self.validate_mode(mode)

        # If the previous step terminated the episode, cleanly reset before stepping again
        episode_reset = False
        if self.done:
            self.reset()
            episode_reset = True

        self.step_count += 1
        t0 = time.time()

        gt_frame, gt_reward, gt_done = self.env.step(action)
        self.gt_score += float(gt_reward)
        self.done = bool(gt_done)

        if mode == "ground_truth":
            pred_reward = 0.0
            done_prob_val = 1.0 if gt_done else 0.0
            display_frame = gt_frame
            self.current_frame = gt_frame
            self._kv_synced = False
        else:
            # Re-prime KV-cache if switching back from ground_truth mode or nearing context limit
            tokens_needed = self.world_model.config.tokens_per_frame + 1
            if (
                not self._kv_synced
                or self.past_kvs is None
                or self.past_kvs[0][0].shape[2] + tokens_needed > self.world_model.config.max_seq_len
            ):
                self._prime_kv_cache(self.current_frame)

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

            pred_reward = float(rew.item())
            done_prob_val = float(done_prob.item())
            self.predicted_score += pred_reward
            display_frame = neural_frame if mode == "neural" else gt_frame
            self.current_frame = display_frame

        latency_ms = (time.time() - t0) * 1000.0

        # Convert frame to Base64 JPEG
        img_np = (np.clip(display_frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_np).resize((384, 384), Image.NEAREST)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "image": img_b64,
            "step": self.step_count,
            "env": self.env_name,
            "mode": mode,
            "predicted_reward": round(pred_reward, 3),
            "total_score": round(self.predicted_score, 2),
            "done_prob": round(done_prob_val, 3),
            "latency_ms": round(latency_ms, 1),
            "gt_reward": round(float(gt_reward), 3),
            "gt_score": round(self.gt_score, 2),
            "gt_done": bool(gt_done),
            "episode_reset": episode_reset,
        }


# Global engine instance for HTTP request handler
ENGINE: NeuralEngineState | None = None


def render_html_ui(engine: NeuralEngineState | None) -> str:
    """Injects active engine metadata (model name, parameter count, active env) into the HTML UI."""
    if engine is not None:
        model_name = engine.wm_config.model_name
        total_params_m = engine.world_model.count_parameters()["total"] / 1e6
        active_env = engine.env_name
    else:
        model_name = "Mundus-0.1"
        total_params_m = 0.0
        active_env = "arcade"

    arcade_selected = " selected" if active_env == "arcade" else ""
    doom_selected = " selected" if active_env == "doom" else ""

    return (
        HTML_UI.replace("__MODEL_NAME__", model_name)
        .replace("__PARAMS_M__", f"{total_params_m:.1f}M")
        .replace("__ACTIVE_ENV__", active_env)
        .replace("__ARCADE_SELECTED__", arcade_selected)
        .replace("__DOOM_SELECTED__", doom_selected)
    )


class NeuralEngineHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_html_ui(ENGINE).encode("utf-8"))

        elif parsed.path in ("/state", "/step", "/reset", "/set_env") and ENGINE is None:
            self._send_json(503, {"error": "Engine is not initialized."})

        elif parsed.path == "/state":
            self._send_json(
                200,
                {
                    "env": ENGINE.env_name,
                    "model_name": ENGINE.wm_config.model_name,
                    "valid_actions": list(ENGINE.valid_actions()),
                    "done": ENGINE.done,
                    "step": ENGINE.step_count,
                },
            )

        elif parsed.path == "/step":
            params = parse_qs(parsed.query, keep_blank_values=True)
            raw_action = params.get("action", ["0"])[0]
            try:
                action = int(raw_action)
            except (TypeError, ValueError):
                self._send_json(400, {"error": f"Invalid action '{raw_action}': must be an integer."})
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
            params = parse_qs(parsed.query, keep_blank_values=True)
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
        # Silence verbose HTTP logging
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
      <div id="episode-banner">Episode Ended &mdash; Press R / Reset or any action key to start a new episode.</div>
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
      <option value="arcade"__ARCADE_SELECTED__>🔵 Arcade (2D Breakout)</option>
      <option value="doom"__DOOM_SELECTED__>🔴 Doom 3D (First-Person Raycaster)</option>
    </select>
  </div>

  <div class="controls" id="game-controls"></div>

  <div class="instructions" id="instructions-txt"></div>

  <script>
    let isBusy = false;
    let currentMode = "neural";
    let activeEnv = "__ACTIVE_ENV__";

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
      const modeDesc = currentMode === "neural"
        ? "Autoregressive neural dynamics by <b>__MODEL_NAME__</b>."
        : "Direct Ground Truth simulator physics (real environment).";

      if (activeEnv === "doom") {
        document.getElementById('instructions-txt').innerHTML =
          "<b>Doom 3D Controls:</b> <b>W / S</b> = Move Fwd/Back &bull; <b>A / D</b> = Turn Left/Right &bull; <b>Space</b> = Fire Weapon &bull; <b>R</b> = Reset<br>" + modeDesc;
      } else {
        document.getElementById('instructions-txt').innerHTML =
          "<b>Arcade 2D Controls:</b> <b>A / D</b> or <b>Left/Right</b> = Move Paddle &bull; <b>S</b> = Stay &bull; <b>R</b> = Reset<br>" + modeDesc;
      }
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
      const modeBtnText = currentMode === "neural" ? "Mode: Neural" : "Mode: Ground Truth";
      if (activeEnv === "doom") {
        controlsDiv.innerHTML = `
          <button onclick="sendAction(1)">&larr; Turn L (A)</button>
          <button onclick="sendAction(3)">&uarr; Fwd (W)</button>
          <button onclick="sendAction(4)">&darr; Back (S)</button>
          <button onclick="sendAction(2)">Turn R (D) &rarr;</button>
          <button onclick="sendAction(5)" style="background: #ef4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);">🔥 FIRE (Space)</button>
          <button onclick="toggleMode()" id="mode-btn" style="background: #8b5cf6;">${modeBtnText}</button>
          <button onclick="resetEngine()" style="background: #e11d48;">Reset</button>
        `;
      } else {
        controlsDiv.innerHTML = `
          <button onclick="sendAction(1)">&larr; Left (A)</button>
          <button onclick="sendAction(0)">Stay (S)</button>
          <button onclick="sendAction(2)">Right (D) &rarr;</button>
          <button onclick="toggleMode()" id="mode-btn" style="background: #8b5cf6;">${modeBtnText}</button>
          <button onclick="resetEngine()" style="background: #e11d48;">Reset</button>
        `;
      }
      updateModeTelemetryLabels();
      updateInstructions();
    }

    async function changeEnv(envChoice) {
      activeEnv = envChoice;
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

    // Keyboard bindings for Doom 3D and Arcade
    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === "SELECT" || e.target.tagName === "INPUT")) {
        return;
      }
      if (activeEnv === "doom") {
        if (e.key === "w" || e.key === "W" || e.key === "ArrowUp") {
          sendAction(3); // Forward
        } else if (e.key === "s" || e.key === "S" || e.key === "ArrowDown") {
          sendAction(4); // Backward
        } else if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") {
          sendAction(1); // Turn Left
        } else if (e.key === "d" || e.key === "D" || e.key === "ArrowRight") {
          sendAction(2); // Turn Right
        } else if (e.key === " " || e.code === "Space") {
          e.preventDefault();
          sendAction(5); // Shoot / Fire
        } else if (e.key === "r" || e.key === "R") {
          resetEngine();
        }
      } else {
        if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A") {
          sendAction(1);
        } else if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") {
          sendAction(2);
        } else if (e.key === "ArrowDown" || e.key === "s" || e.key === "S") {
          sendAction(0);
        } else if (e.key === "r" || e.key === "R") {
          resetEngine();
        }
      }
    });

    // Initial frame & controls setup
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
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "85m", "130m", "230m"], help="Model preset")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--env", type=str, default="arcade", choices=["doom", "arcade"], help="Default game environment ('doom' or 'arcade')")
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
    print("=======================================================\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down engine.")


if __name__ == "__main__":
    main()
