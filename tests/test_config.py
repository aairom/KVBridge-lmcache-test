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
