"""Tests for kvbridge.config."""

from pathlib import Path

from kvbridge.config import PROJECT_ROOT, Settings


def test_defaults_are_valid():
    assert Settings().validate() == []


def test_from_env_casts_types():
    env = {
        "KVB_CHUNK_SIZE": "128", "KVB_WRITE_THROUGH": "false", "KVB_TEMPERATURE": "0.7",
        "KVB_OLLAMA_MODEL": "granite4.2:latest", "KVB_STORE_DIR": "~/kv", "KVB_GUI_PORT": "9000",
    }
    s = Settings.from_env(env)
    assert s.chunk_size == 128 and isinstance(s.chunk_size, int)
    assert s.write_through is False
    assert s.temperature == 0.7
    assert s.ollama_model == "granite4.2:latest"
    assert s.store_dir == Path("~/kv").expanduser()
    assert s.gui_port == 9000


def test_from_env_empty_values_use_defaults():
    s = Settings.from_env({"KVB_CHUNK_SIZE": "", "KVB_LLAMA_URL": ""})
    assert s.chunk_size == Settings().chunk_size
    assert s.llama_url == ""


def test_default_paths_under_project_root():
    s = Settings()
    assert s.input_dir == PROJECT_ROOT / "input"
    assert s.store_dir.parent == PROJECT_ROOT / "output"


def test_validate_reports_problems():
    s = Settings(chunk_size=0, l1_mb=0, l2_mb=0, managed_port=11434)
    problems = " | ".join(s.validate())
    assert "chunk_size" in problems
    assert any("ctx_size" in p for p in Settings(chunk_size=512, ctx_size=256).validate())
    assert "at least one cache tier" in problems
    assert "reserved" in problems


def test_validate_negative_and_port_range():
    assert any("tier capacities" in p for p in Settings(l1_mb=-1).validate())
    assert any("valid TCP port" in p for p in Settings(managed_port=70000).validate())


def test_byte_helpers():
    s = Settings(l1_mb=2, l2_mb=3)
    assert s.l1_bytes == 2 * 1024 * 1024
    assert s.l2_bytes == 3 * 1024 * 1024


# ------------------------------------------------------- engine / cloud settings
import json
import os
import stat

import pytest

from kvbridge.config import (DEFAULT_LLAMA_SEARCH_PATHS, PROVIDER_PRESETS, Settings as _S,
                             update_env_file)


def test_new_settings_come_from_env():
    s = Settings.from_env({
        "KVB_ENGINE": "cloud", "KVB_CLOUD_PROVIDER": "groq", "KVB_CLOUD_MODEL": "m",
        "KVB_CLOUD_API_KEY": "k-12345", "KVB_CLOUD_VERIFY_TLS": "false", "KVB_CLOUD_TOKEN_BUDGET": "5000",
        "KVB_SEED": "9", "KVB_OLLAMA_KEEP_ALIVE": "1h", "KVB_LLAMA_START_TIMEOUT": "12.5",
        "KVB_RESERVED_PORTS": "1234; 5678", "KVB_LLAMA_SEARCH_PATHS": "/x:/y",
    })
    assert s.engine == "cloud" and s.cloud_model == "m" and s.cloud_verify_tls is False
    assert s.cloud_token_budget == 5000 and s.seed == 9 and s.ollama_keep_alive == "1h"
    assert s.llama_start_timeout == 12.5 and s.reserved_port_set == {1234, 5678}
    assert s.resolved_cloud_url() == PROVIDER_PRESETS["groq"]
    assert s.validate() == []


def test_default_search_paths_constant():
    assert "/opt/homebrew/bin" in DEFAULT_LLAMA_SEARCH_PATHS


def test_unknown_engine_is_a_validation_problem():
    assert any("unknown engine" in p for p in Settings(engine="vllm").validate())


def test_validate_is_engine_specific():
    assert Settings(engine="ollama", managed_port=11434).validate() == []       # llama-only rule
    assert any("reserved" in p for p in Settings(engine="llamacpp", managed_port=11434).validate())
    assert any("cloud_base_url" in p for p in Settings(engine="cloud").validate())
    ok = Settings(engine="cloud", cloud_base_url="https://h/v1", cloud_model="m")
    assert ok.validate() == []


def test_reserved_ports_are_configurable():
    s = Settings(managed_port=8089, reserved_ports="8089")
    assert any("reserved" in p for p in s.validate())
    assert Settings(managed_port=11434, reserved_ports="").validate() == []


def test_apply_success_coerces_types():
    s = Settings()
    assert s.apply({"chunk_size": "128", "write_through": "false", "temperature": "0.5",
                    "engine": " ollama ", "store_dir": "~/x"}) == []
    assert (s.chunk_size, s.write_through, s.temperature, s.engine) == (128, False, 0.5, "ollama")
    assert s.store_dir == Path("~/x").expanduser()


def test_apply_is_atomic_on_failure():
    s = Settings()
    before = dict(s.__dict__)
    problems = s.apply({"chunk_size": 64, "cloud_token_budget": -5, "engine": "cloud"})
    assert problems and s.__dict__ == before
    assert any("unknown setting" in p for p in s.apply({"nope": 1}))
    assert any("chunk_size" in p for p in s.apply({"chunk_size": ""}))
    assert any("chunk_size" in p for p in s.apply({"chunk_size": "abc"}))
    assert s.__dict__ == before


def test_cloud_key_resolution_order(monkeypatch):
    monkeypatch.setenv("PROVIDER_KEY", "env-key")
    assert Settings(cloud_api_key_env="PROVIDER_KEY").resolved_cloud_key() == "env-key"
    assert Settings(cloud_api_key="direct", cloud_api_key_env="PROVIDER_KEY").resolved_cloud_key() == "direct"
    assert Settings().resolved_cloud_key() == ""
    assert Settings(cloud_api_key_env="MISSING_VAR_XYZ").resolved_cloud_key() == ""


def test_cloud_url_resolution_and_headers():
    assert Settings(cloud_provider="openai").resolved_cloud_url() == "https://api.openai.com/v1"
    assert Settings(cloud_provider="openai", cloud_base_url="https://p/v1/").resolved_cloud_url() == "https://p/v1"
    assert Settings(cloud_provider="nope").resolved_cloud_url() == ""
    assert Settings().cloud_headers() == {}
    assert Settings(cloud_extra_headers='{"A": 1}').cloud_headers() == {"A": "1"}
    with pytest.raises(ValueError):
        Settings(cloud_extra_headers="[1]").cloud_headers()


def test_to_display_masks_secrets():
    s = Settings(cloud_api_key="sk-very-secret")
    d = s.to_display()
    assert d["cloud_api_key"] == "********" and "sk-very-secret" not in json.dumps(d)
    assert Settings().to_display()["cloud_api_key"] == ""
    assert isinstance(d["store_dir"], str)
    assert "sk-very-secret" not in repr(s)


def test_env_values_skip_secrets_and_paths():
    s = Settings(cloud_api_key="sk-secret", write_through=False)
    env = s.env_values()
    assert "KVB_CLOUD_API_KEY" not in env and "KVB_STORE_DIR" not in env
    assert env["KVB_WRITE_THROUGH"] == "false" and env["KVB_ENGINE"] == "llamacpp"
    assert s.env_values(include_secrets=True)["KVB_CLOUD_API_KEY"] == "sk-secret"
    assert Settings.from_env(env).write_through is False       # round-trip


def test_update_env_file_preserves_comments_and_appends(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# my comment\nKVB_ENGINE=ollama\n\nOTHER=keep # inline\nKVB_SEED=1\n")
    update_env_file(p, {"KVB_ENGINE": "cloud", "KVB_NEW": "x y"})
    text = p.read_text()
    assert text.startswith("# my comment\nKVB_ENGINE=cloud\n")
    assert "OTHER=keep # inline" in text and "KVB_SEED=1" in text
    assert text.rstrip().endswith('KVB_NEW="x y"')
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600


def test_update_env_file_creates_and_quotes(tmp_path):
    p = tmp_path / "sub" / ".env"
    update_env_file(p, {"KVB_A": 'he said "hi" $HOME', "KVB_EMPTY": ""})
    lines = p.read_text().splitlines()
    assert 'KVB_A="he said \\"hi\\" $HOME"' in lines and 'KVB_EMPTY=""' in lines
    from dotenv import dotenv_values
    assert dotenv_values(p)["KVB_A"] == 'he said "hi" $HOME'


def test_update_env_file_rejects_newlines(tmp_path):
    with pytest.raises(ValueError):
        update_env_file(tmp_path / ".env", {"KVB_A": "a\nb"})
    assert not (tmp_path / ".env").exists()


def test_env_example_covers_every_setting_and_is_valid():
    from dataclasses import fields as dc_fields
    from dotenv import dotenv_values
    vals = dotenv_values(PROJECT_ROOT / ".env.example")
    documented = set(vals)
    text = (PROJECT_ROOT / ".env.example").read_text()
    for f in dc_fields(Settings):
        assert f"KVB_{f.name.upper()}" in text, f"{f.name} missing from .env.example"
    s = Settings.from_env({k: v for k, v in vals.items() if v is not None})
    assert s.validate() == [] and s.cloud_api_key == "" and "KVB_ENGINE" in documented
