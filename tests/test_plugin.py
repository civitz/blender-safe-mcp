"""The Claude Code plugin in plugin/: manifests, the auth-header helper and the guide skill."""

import json
import os
import re
import subprocess

import pytest

from blender_safe_mcp import keystore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = os.path.join(ROOT, "plugin")
HELPER = os.path.join(PLUGIN, "bin", "auth-header")
SKILL = os.path.join(PLUGIN, "skills", "blender", "SKILL.md")


def load(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


def test_marketplace_lists_the_plugin():
    market = load(ROOT, ".claude-plugin", "marketplace.json")
    plugin = load(PLUGIN, ".claude-plugin", "plugin.json")
    [entry] = market["plugins"]
    assert entry["name"] == plugin["name"] == "blender-safe-mcp"
    assert os.path.realpath(os.path.join(ROOT, entry["source"])) == os.path.realpath(PLUGIN)


def test_mcp_server_uses_helper_and_default_url():
    server = load(PLUGIN, ".mcp.json")["mcpServers"]["blender"]
    assert server["type"] == "http"
    assert server["url"] == "${user_config.url}"
    assert server["headersHelper"] == "${CLAUDE_PLUGIN_ROOT}/bin/auth-header"
    assert "headers" not in server  # the key only ever comes from the helper
    with open(os.path.join(ROOT, "blender_safe_mcp", "blender.py"), encoding="utf-8") as f:
        port = re.search(r"^DEFAULT_PORT = (\d+)$", f.read(), re.M).group(1)
    url = load(PLUGIN, ".claude-plugin", "plugin.json")["userConfig"]["url"]
    assert url["default"] == f"http://127.0.0.1:{port}/mcp"


def run_helper(config_home):
    env = {"PATH": os.environ["PATH"], "HOME": str(config_home), "XDG_CONFIG_HOME": str(config_home)}
    return subprocess.run([HELPER], env=env, capture_output=True, text=True, timeout=10)


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_helper_sends_the_addon_key(config_home):
    key, _ = keystore.load_or_create()  # the addon's own path logic decides where the file goes
    r = run_helper(config_home)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == {"Authorization": f"Bearer {key}"}


def test_helper_without_key_file_fails(config_home):
    r = run_helper(config_home)
    assert r.returncode != 0 and r.stdout == ""
    assert "Start server" in r.stderr


@pytest.mark.parametrize("content", ["short", "x" * 40 + '"},{"a', "a b" * 20])
def test_helper_rejects_malformed_key(config_home, content):
    path = keystore.default_path()
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(content + "\n")
    r = run_helper(config_home)
    assert r.returncode != 0 and r.stdout == ""
    assert content not in r.stderr


def test_skill_names_only_real_names():
    """Backticked snake_case names in the skill must be tools, parameters or error codes that
    exist, so a renamed or removed op breaks this test instead of silently misleading Claude."""
    with open(os.path.join(ROOT, "TOOLS.md"), encoding="utf-8") as f:
        tools_md = f.read()
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    tools = set(re.findall(r"^### (\w+)$", tools_md, re.M))
    params = set(re.findall(r"^\| `([\w.]+)`", tools_md, re.M))
    codes = {c for pair in re.findall(r"^\| `(\w+)`(?: / `(\w+)`)?", readme, re.M) for c in pair if c}
    with open(SKILL, encoding="utf-8") as f:
        text = f.read()
    assert text.startswith("---\nname: blender\ndescription: ")
    named = set(re.findall(r"`([a-z]+(?:_[a-z]+)+)`", text))
    unknown = sorted(named - tools - params - codes)
    assert not unknown, f"SKILL.md names things that don't exist: {unknown}"
    assert len(named & tools) >= 15
