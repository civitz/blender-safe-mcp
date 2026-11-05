import os
import stat

import pytest

from blender_safe_mcp import keystore


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "cfg" / "blender-safe-mcp" / "api_key")


def mode(p):
    return stat.S_IMODE(os.stat(p).st_mode)


def test_create_then_reload_survives(path):
    key, created = keystore.load_or_create(path)
    assert created and keystore.KEY_RE.match(key)
    assert keystore.load_or_create(path) == (key, False)  # e.g. after an addon reinstall
    assert mode(path) == 0o600 and mode(os.path.dirname(path)) == 0o700


def test_regenerate_replaces(path):
    key, _ = keystore.load_or_create(path)
    new = keystore.regenerate(path)
    assert new != key and keystore.load(path) == new


def test_malformed_or_missing_is_ignored(path):
    assert keystore.load(path) is None
    keystore.save("a" * 40, path)
    with open(path, "w") as f:
        f.write("not a key; rm -rf /")
    assert keystore.load(path) is None
    key, created = keystore.load_or_create(path)
    assert created and keystore.load(path) == key


def test_loose_permissions_are_tightened(path):
    keystore.save("b" * 40, path)
    os.chmod(path, 0o644)
    assert keystore.load(path) == "b" * 40
    assert mode(path) == 0o600


def test_symlink_is_not_followed(tmp_path, path):
    target = tmp_path / "elsewhere"
    target.write_text("c" * 40)
    os.makedirs(os.path.dirname(path))
    os.symlink(target, path)
    assert keystore.load(path) is None


def test_default_path_respects_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert keystore.default_path() == str(tmp_path / "blender-safe-mcp" / "api_key")


def test_save_rejects_bad_key(path):
    with pytest.raises(ValueError):
        keystore.save("short", path)
