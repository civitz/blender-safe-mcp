"""In-memory ring buffer of log records, readable through the get_debug_logs op.
Pure Python, no bpy.

Never log secrets: the API key must not reach a log call.
"""

import collections
import itertools
import logging
import sys
import threading
import time
import traceback

MAX_BYTES = 10 * 1024 * 1024     # whole buffer, oldest records evicted first
MAX_MESSAGE = 4000
MAX_TRACEBACK = 16000
MAX_RESPONSE_BYTES = 768 * 1024  # keeps one get_debug_logs result small
_OVERHEAD = 160                  # rough per-record cost of the fixed fields

LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
          "ERROR": logging.ERROR}


class RingHandler(logging.Handler):
    def __init__(self, max_bytes=MAX_BYTES):
        super().__init__(logging.DEBUG)
        self.max_bytes = max_bytes
        self._records = collections.deque()   # (entry, size)
        self._bytes = 0
        self._ids = itertools.count(1)
        self._lock_ = threading.Lock()

    @staticmethod
    def _size(entry):
        return _OVERHEAD + sum(len(v) for v in entry.values() if isinstance(v, str))

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not kill logging
            msg = str(record.msg)
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                    + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "thread": record.threadName,
            "message": msg[:MAX_MESSAGE],
        }
        if record.exc_info and record.exc_info[0] is not None:
            tb = "".join(traceback.format_exception(*record.exc_info))
            entry["exception"] = tb[-MAX_TRACEBACK:]
        size = self._size(entry)
        with self._lock_:
            entry["id"] = next(self._ids)
            self._records.append((entry, size))
            self._bytes += size
            while self._bytes > self.max_bytes and len(self._records) > 1:
                _, old = self._records.popleft()
                self._bytes -= old

    def stats(self):
        with self._lock_:
            return {"records": len(self._records), "bytes": self._bytes,
                    "max_bytes": self.max_bytes}

    def query(self, limit=100, min_level="DEBUG", since_id=0, errors_only=False):
        """Newest matching records (returned oldest-first), at most `limit` and
        MAX_RESPONSE_BYTES. `truncated` says older matches were left out."""
        threshold = LEVELS.get(min_level, logging.DEBUG)
        with self._lock_:
            items = list(self._records)
            total = self._bytes
        out, used, truncated = [], 0, False
        for entry, size in reversed(items):
            if entry["id"] <= since_id:
                break
            if logging.getLevelName(entry["level"]) < threshold:
                continue
            if errors_only and "exception" not in entry:
                continue
            if len(out) >= limit or used + size > MAX_RESPONSE_BYTES:
                truncated = True
                break
            out.append(entry)
            used += size
        out.reverse()
        return {"records": out, "truncated": truncated,
                "last_id": items[-1][0]["id"] if items else 0,
                "oldest_id": items[0][0]["id"] if items else 0,
                "buffer_bytes": total}


def setup(name, console=True):
    """Return (logger, ring). Idempotent across addon reloads."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    ring = next((h for h in logger.handlers if isinstance(h, RingHandler)), None)
    if ring is None:
        ring = RingHandler()
        logger.addHandler(ring)
    if console and not any(getattr(h, "_blender_safe_mcp_console", False) for h in logger.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
        h._blender_safe_mcp_console = True
        logger.addHandler(h)
    return logger, ring
