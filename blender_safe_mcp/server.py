"""Pieces shared by the transport and the ops. Pure Python, no bpy.

- OpError: an expected, user-facing failure raised by ops and dispatch;
- WorkQueue: hands requests from network threads to the thread that calls pump()
  (Blender's main thread, via bpy.app.timers), which runs dispatch(op, params, blobs);
- _PeerLimiter: per-address exponential back-off after failed logins.
"""

import json
import queue
import threading
import time
from concurrent.futures import CancelledError, Future


class OpError(Exception):
    """An expected, user-facing failure (bad params, not found, denied...)."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class WorkQueue:
    """Hands requests from network threads to the thread that calls pump()."""

    def __init__(self, dispatch):
        self._dispatch = dispatch
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False

    def call(self, op, params, blobs=()):
        """Block until the main thread has run `op`; returns (result, blobs) or raises
        what dispatch raised (CancelledError if the queue was closed or cancelled)."""
        fut = Future()
        with self._lock:
            if self._closed:
                raise CancelledError()
            self._queue.put((op, params, blobs, fut))
        return fut.result()

    def close(self):
        """Refuse new work and cancel what's queued (nothing will pump it any more)."""
        with self._lock:
            self._closed = True
        self.cancel_pending()

    def pump(self, budget_s=0.05):
        """Run queued ops on the calling thread. Returns number executed."""
        deadline = time.monotonic() + budget_s
        n = 0
        while True:
            try:
                op, params, blobs, fut = self._queue.get_nowait()
            except queue.Empty:
                return n
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(self._dispatch(op, params, blobs))
            except BaseException as e:  # noqa: BLE001 - handed to the client thread
                fut.set_exception(e)
            n += 1
            if time.monotonic() > deadline:
                return n

    def cancel_pending(self):
        while True:
            try:
                *_, fut = self._queue.get_nowait()
            except queue.Empty:
                return
            fut.cancel()


class _PeerLimiter:
    """Exponential back-off per peer address after failed logins."""

    def __init__(self, base=0.5, cap=60.0):
        self.base, self.cap = base, cap
        self._state = {}
        self._lock = threading.Lock()

    def blocked(self, peer, now=None):
        """Seconds of back-off left for `peer` (0.0 = not blocked)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            failures, until = self._state.get(peer, (0, 0.0))
            return max(0.0, until - now)

    def fail(self, peer, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            failures, _ = self._state.get(peer, (0, 0.0))
            failures += 1
            delay = min(self.base * (2 ** (failures - 1)), self.cap)
            self._state[peer] = (failures, now + delay)

    def succeed(self, peer):
        with self._lock:
            self._state.pop(peer, None)


def _summary(params, limit=400):
    try:
        text = json.dumps(params, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(params)
    return text if len(text) <= limit else text[:limit] + f"... ({len(text)} chars)"
