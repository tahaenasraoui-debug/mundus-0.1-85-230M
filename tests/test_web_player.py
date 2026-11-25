"""Tests for web_player.py validation, episode termination handling, and HTTP API."""

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

import web_player
from config import WorldModelConfig


def _tiny_config() -> WorldModelConfig:
    return WorldModelConfig(
        model_name="Mundus-0.1-TinyTest",
        d_model=64,
        n_layers=2,
        n_heads=4,
        hidden_dim=128,
        vocab_size=2048,
        num_actions=16,
        max_seq_len=256,
        tokens_per_frame=64,
    )


@pytest.fixture
def fast_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(web_player, "get_config_230m", _tiny_config)
    monkeypatch.setattr(web_player, "get_config", lambda _name: _tiny_config())
    engine = web_player.NeuralEngineState(
        checkpoint_path=None,
        model_type="auto",
        device="cpu",
        env_name="arcade",
        tokenizer_path=tmp_path / "missing_tokenizer.pt",
    )
    return engine


def _http_get_json(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


def _http_get_text(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url) as resp:
        return resp.status, resp.read().decode("utf-8")


def test_engine_action_validation_and_episode_reset(fast_engine):
    assert list(fast_engine.valid_actions()) == [0, 1, 2]
    with pytest.raises(ValueError, match="Invalid action"):
        fast_engine.validate_action(5)
    with pytest.raises(ValueError, match="Invalid mode"):
        fast_engine.validate_mode("invalid")

    # Force episode termination on the next step
    fast_engine.env.steps = fast_engine.env.max_steps - 1
    step_done = fast_engine.step(0, mode="ground_truth")
    assert step_done["gt_done"] is True
    assert step_done["episode_reset"] is False
    assert fast_engine.done is True

    # Next action automatically resets before stepping
    step_after = fast_engine.step(1, mode="ground_truth")
    assert step_after["episode_reset"] is True
    assert step_after["step"] == 1
    assert step_after["gt_done"] is False

    # Switching from ground_truth to neural mode re-primes KV cache safely
    neural_step = fast_engine.step(0, mode="neural")
    assert neural_step["mode"] == "neural"
    assert neural_step["step"] == 2
    assert fast_engine._kv_synced is True


def test_web_player_http_endpoints_and_controls(fast_engine, monkeypatch):
    monkeypatch.setattr(web_player, "ENGINE", fast_engine)
    server = HTTPServer(("127.0.0.1", 0), web_player.NeuralEngineHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    try:
        status, html = _http_get_text(f"{base}/")
        assert status == 200
        assert "Mundus-0.1-TinyTest" in html
        assert 'let activeEnv = "arcade";' in html
        assert '<option value="arcade" selected>' in html

        status, state = _http_get_json(f"{base}/state")
        assert status == 200
        assert state["env"] == "arcade"
        assert state["valid_actions"] == [0, 1, 2]

        # Invalid action inputs return HTTP 400 JSON
        for bad_query in (
            "/step?action=abc",
            "/step?action=",
            "/step?action=99",
            "/step?action=-1",
            "/step?action=5",
            "/step?action=0&mode=bogus",
        ):
            status, payload = _http_get_json(f"{base}{bad_query}")
            assert status == 400, f"Expected 400 for {bad_query}, got {status}"
            assert "error" in payload

        # Valid ground_truth step
        status, ok = _http_get_json(f"{base}/step?action=1&mode=ground_truth")
        assert status == 200
        assert ok["env"] == "arcade"
        assert ok["mode"] == "ground_truth"
        assert "image" in ok

        # Switch to Doom and verify Doom-specific controls and actions work
        status, env_resp = _http_get_json(f"{base}/set_env?env=doom")
        assert status == 200
        assert env_resp["env"] == "doom"

        status, doom_html = _http_get_text(f"{base}/")
        assert status == 200
        assert 'let activeEnv = "doom";' in doom_html
        assert '<option value="doom" selected>' in doom_html

        status, fire_resp = _http_get_json(f"{base}/step?action=5&mode=ground_truth")
        assert status == 200
        assert fire_resp["env"] == "doom"

        status, reset_resp = _http_get_json(f"{base}/reset")
        assert status == 200
        assert reset_resp["status"] == "reset_complete"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
