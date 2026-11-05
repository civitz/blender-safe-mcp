"""API key persisted in a private file, outside Blender's preferences and versioned
config dirs, so it survives addon reinstalls and Blender upgrades. Pure Python.

Location: $XDG_CONFIG_HOME/blender-safe-mcp/api_key (default ~/.config), or
%APPDATA%\\blender-safe-mcp\\api_key on Windows.
"""

import logging
import os
import re
import stat

import secrets

log = logging.getLogger("blender_safe_mcp.keystore")

KEY_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
DIR_NAME = "blender-safe-mcp"
FILE_NAME = "api_key"


def _config_base():
    if os.name == "nt":
        return os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")


def default_path():
    return os.path.join(_config_base(), DIR_NAME, FILE_NAME)


def load(path=None):
    """The stored key, or None if there's no usable file."""
    path = path or default_path()
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode):
        log.warning("ignoring key file %s: not a regular file", path)
        return None
    if os.name != "nt" and st.st_mode & 0o077:
        log.warning("key file %s is readable by other users; tightening to 0600", path)
        os.chmod(path, 0o600)
    with open(path, encoding="ascii", errors="replace") as f:
        key = f.read(256).strip()
    if not KEY_RE.match(key):
        log.warning("ignoring key file %s: malformed content", path)
        return None
    return key


def save(key, path=None):
    if not KEY_RE.match(key):
        raise ValueError("malformed key")
    path = path or default_path()
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, (key + "\n").encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written key
    return path


def load_or_create(path=None):
    """Return (key, created)."""
    key = load(path)
    if key:
        return key, False
    key = secrets.token_urlsafe(32)
    where = save(key, path)
    log.info("generated a new API key, stored in %s", where)
    return key, True


def regenerate(path=None):
    key = secrets.token_urlsafe(32)
    save(key, path)
    return key
