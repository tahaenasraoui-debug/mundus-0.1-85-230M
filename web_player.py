"""
Mundus 0.1 (230M) - Live Interactive Neural Game Engine Player.
Streams real-time neural rollouts to a browser canvas with zero external GUI dependencies.
Controls: Left Arrow / A (Move Left), Right Arrow / D (Move Right), S / Down (Stay).
"""

import os
import io
import json
import base64
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from config import get_config_85m, get_config_130m, get_config_230m, TokenizerConfig
from tokenizer import VQVAE
from world_model import WorldModel
from dataset import PhysicsGameEnv


def get_safe_device(requested: str = "auto") -> str:
    """Safely determines if CUDA is actually usable without kernel mismatch crashes."""
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        try:
            # Test a basic tensor operation to verify kernel availability
            t = torch.zeros(1, device="cuda") + 1.0
            return "cuda"
        except Exception:
            print("[NeuralEngine] CUDA detected, but current PyTorch build lacks sm_61 kernels. Falling back to CPU smoothly.")
            return "cpu"
    return "cpu"


class NeuralEngineState:
    def __init__(self, checkpoint_path=None, model_type="auto", device="auto"):
        self.device = get_safe_device(device)
        print(f"[NeuralEngine] Running on device: {self.device.upper()}")
        if self.device == "cpu":
            torch.set_num_threads(8)

        print("[NeuralEngine] Initializing Mundus Engine and VQ-VAE Tokenizer...")
        self.tok_config = TokenizerConfig()
        self.tokenizer = VQVAE(self.tok_config).to(self.device)

        # Check for standalone permanent tokenizer checkpoint
        tok_file = "/home/tahary/.gemini/antigravity/scratch/world_model_230m/checkpoints/tokenizer.pt"
        if os.path.exists(tok_file):
            print(f"[NeuralEngine] Loading permanent visual codebook from {tok_file}...")
            self.tokenizer.load_state_dict(torch.load(tok_file, map_location=self.device, weights_only=False))

        # Auto-detect config from checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[NeuralEngine] Loading weights from {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "config" in ckpt:
                self.wm_config = ckpt["config"]
                print(f"[NeuralEngine] Detected model config: {self.wm_config.model_name} (Layers: {self.wm_config.n_layers}, Dim: {self.wm_config.d_model})")
            else:
                self.wm_config = get_config_85m() if model_type == "85m" else get_config_230m()

            self.world_model = WorldModel(self.wm_config).to(self.device)
            if "world_model" in ckpt:
                self.world_model.load_state_dict(ckpt["world_model"])
            if "tokenizer" in ckpt and not os.path.exists(tok_file):
                self.tokenizer.load_state_dict(ckpt["tokenizer"])
            print("[NeuralEngine] Loaded checkpoint successfully!")
        else:
            self.wm_config = get_config_85m() if model_type == "85m" else get_config_230m()
            self.world_model = WorldModel(self.wm_config).to(self.device)
            print(f"[NeuralEngine] Running in live zero-shot mode ({self.wm_config.model_name}).")

        self.world_model.eval()
        self.tokenizer.eval()

        self.env = PhysicsGameEnv()
        self.reset()

    def reset(self):
        self.current_frame = self.env.reset() # (64, 64, 3)
        self.past_kvs = None
        self.step_count = 0
        self.predicted_score = 0.0
        self.done = False

        # Prime the world model with the initial frame
        frame_t = torch.from_numpy(self.current_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            prompt_tokens = self.tokenizer.encode(frame_t)
            out = self.world_model(prompt_tokens, use_cache=True)
            self.past_kvs = out["new_kvs"]

    def step(self, action: int, mode="neural"):
        """
        Takes an action and returns the next frame and metadata.
        mode: "neural" (pure imagination) or "hybrid" (ground truth alongside prediction)
        """
        self.step_count += 1
        t0 = time.time()

        gt_frame, gt_reward, gt_done = self.env.step(action)

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

        latency_ms = (time.time() - t0) * 1000.0
        self.predicted_score += rew.item()

        display_frame = neural_frame if mode == "neural" else gt_frame

        # Convert frame to Base64 JPEG
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
            "gt_reward": gt_reward,
        }


# Global engine instance for HTTP request handler
ENGINE = None


class NeuralEngineHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_UI.encode("utf-8"))

        elif parsed.path == "/step":
            params = parse_qs(parsed.query)
            action = int(params.get("action", [0])[0])
            mode = params.get("mode", ["neural"])[0]

            result = ENGINE.step(action, mode=mode)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif parsed.path == "/reset":
            ENGINE.reset()
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "reset_complete"}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence verbose HTTP logging
        return


HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>🌌 Mundus 0.1 (230M) - Neural Game Engine</title>
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
    }
    #screen {
      width: 384px;
      height: 384px;
      border-radius: 8px;
      border: 2px solid #38bdf8;
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
      image-rendering: pixelated;
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
  <h1>🌌 Mundus 0.1 (230M)</h1>
  <div class="subtitle">Autonomous Neural Game Engine Simulator &bull; <span class="badge">232M Transformers + VQ-VAE</span></div>

  <div class="dashboard">
    <div class="screen-container">
      <img id="screen" src="" alt="Neural Game Frame" />
    </div>

    <div class="telemetry">
      <div class="stat-card">
        <div class="stat-label">Engine Mode</div>
        <div class="stat-value" id="mode-val" style="color: #38bdf8;">100% Neural</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Predicted Reward</div>
        <div class="stat-value" id="reward-val">+0.00</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Total Score</div>
        <div class="stat-value" id="score-val">0.0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Done Probability</div>
        <div class="stat-value" id="done-val">0.00</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Step Latency</div>
        <div class="stat-value" id="latency-val">0.0 ms</div>
      </div>
    </div>
  </div>

  <div class="controls">
    <button onclick="sendAction(1)">&larr; Left (A)</button>
    <button onclick="sendAction(0)">Stay (S)</button>
    <button onclick="sendAction(2)">Right (D) &rarr;</button>
    <button onclick="toggleMode()" id="mode-btn" style="background: #8b5cf6;">Mode: Neural</button>
    <button onclick="resetEngine()" style="background: #e11d48;">Reset</button>
  </div>

  <div class="instructions">
    Use <b>A / D</b> or <b>Arrow Keys &larr; &rarr;</b> on your keyboard to play.<br>
    Every pixel and reward is simulated by the <b>Mundus 0.1</b> neural network in real-time.
  </div>

  <script>
    let isBusy = false;
    let currentMode = "neural";

    function toggleMode() {
      currentMode = currentMode === "neural" ? "hybrid" : "neural";
      document.getElementById('mode-btn').innerText = currentMode === "neural" ? "Mode: Neural" : "Mode: Ground Truth";
      document.getElementById('mode-val').innerText = currentMode === "neural" ? "100% Neural" : "Ground Truth";
      document.getElementById('mode-val').style.color = currentMode === "neural" ? "#38bdf8" : "#10b981";
      sendAction(0);
    }

    async function sendAction(action) {
      if (isBusy) return;
      isBusy = true;
      try {
        const res = await fetch(`/step?action=${action}&mode=${currentMode}`);
        const data = await res.json();
        document.getElementById('screen').src = "data:image/jpeg;base64," + data.image;
        document.getElementById('reward-val').innerText = (data.predicted_reward >= 0 ? "+" : "") + data.predicted_reward;
        document.getElementById('score-val').innerText = data.total_score;
        document.getElementById('done-val').innerText = data.done_prob;
        document.getElementById('latency-val').innerText = data.latency_ms + " ms";
      } catch (e) {
        console.error(e);
      } finally {
        isBusy = false;
      }
    }

    async function resetEngine() {
      await fetch('/reset');
      sendAction(0);
    }

    // Keyboard bindings
    window.addEventListener('keydown', (e) => {
      if (e.key === "ArrowLeft" || e.key === "a" || e.key === "A") {
        sendAction(1);
      } else if (e.key === "ArrowRight" || e.key === "d" || e.key === "D") {
        sendAction(2);
      } else if (e.key === "ArrowDown" || e.key === "s" || e.key === "S") {
        sendAction(0);
      }
    });

    // Initial frame
    sendAction(0);
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Mundus Live Neural Game Engine")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "85m", "130m", "230m"], help="Model preset")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use ('auto', 'cpu', 'cuda')")
    args = parser.parse_args()

    global ENGINE
    ENGINE = NeuralEngineState(checkpoint_path=args.checkpoint, model_type=args.model, device=args.device)

    server = HTTPServer(("0.0.0.0", args.port), NeuralEngineHandler)
    print(f"\n=======================================================")
    print(f"🌌 Mundus 0.1 Live Neural Engine running!")
    print(f"👉 Open in your browser: http://localhost:{args.port}")
    print(f"=======================================================\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down engine.")


if __name__ == "__main__":
    main()
