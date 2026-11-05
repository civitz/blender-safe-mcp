"""MCP over Streamable HTTP, served by the addon itself. Pure Python, no bpy.

This is how MCP clients (Claude Code: `"type": "http"`) talk to Blender: there is no helper
process. Both protocol eras are served on one endpoint, `/mcp`:

- **2026-07-28** (stateless): each POST is one self-contained JSON-RPC request whose
  protocol version travels in `params._meta` and in the `MCP-Protocol-Version` header.
  `server/discover` advertises the supported versions, capabilities and instructions.
  No sessions.
- **2025-03-26 .. 2025-11-25** (handshake): `initialize` negotiates a version and opens a
  session (`Mcp-Session-Id`); later requests must carry it.

Every response is a single `application/json` body (no SSE streams, so GET is 405).

This module only parses HTTP/JSON-RPC and checks the key. Ops run on Blender's main thread through a `WorkQueue` and `registry.dispatch`,
so schemas, the guard, undo steps and tripwires apply unchanged.

Hardening, because a localhost HTTP port is reachable from any web page the user opens:
- `Host` must be an IP literal or `localhost` (defeats DNS rebinding);
- `Origin`, when present, must be a loopback `http://` origin (browsers always send it);
- the key (`Authorization: Bearer ...`) is required on every request, compared in constant
  time, with per-address exponential back-off after wrong keys (a missing key is not
  penalized, so a port probe can't lock the real client out);
- `Content-Length` required and capped, JSON depth capped, no batches;
- per-socket timeouts and a cap on concurrent connections.
"""

import base64
import binascii
import hmac
import ipaddress
import json
import logging
import re
import secrets
import socketserver
import threading
import time
from collections import OrderedDict
from concurrent.futures import CancelledError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .server import OpError, WorkQueue, _PeerLimiter, _summary

log = logging.getLogger("blender_safe_mcp.http")

ENDPOINT = "/mcp"
DEFAULT_PORT = 9877

# Protocol versions, as advertised: server/discover lists the stateless ones; initialize
# accepts the handshake ones (newest first; an unknown request gets the newest).
# 2024-11-05 is left out: it used the older HTTP+SSE transport, not this one.
MODERN_VERSIONS = ("2026-07-28",)
HANDSHAKE_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26")
SUPPORTED_VERSIONS = MODERN_VERSIONS + HANDSHAKE_VERSIONS

META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CLIENT_CAPS = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
HEADER_MISMATCH = -32020
UNSUPPORTED_PROTOCOL_VERSION = -32022

# HTTP status for JSON-RPC errors on the stateless path (2026-07-28 transport table).
# The handshake path answers JSON-RPC errors with 200, as before.
MODERN_ERROR_STATUS = {PARSE_ERROR: 400, INVALID_REQUEST: 400, INVALID_PARAMS: 400,
                       HEADER_MISMATCH: 400, UNSUPPORTED_PROTOCOL_VERSION: 400,
                       METHOD_NOT_FOUND: 404}

CAPABILITIES = {"tools": {"listChanged": False}}

MAX_BODY_BYTES = 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_CONNECTIONS = 16
SOCKET_TIMEOUT = 15.0
MAX_SESSIONS = 32
SESSION_IDLE_S = 24 * 3600
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

ROUTING_HEADERS = ("MCP-Protocol-Version", "Mcp-Method", "Mcp-Name", "Mcp-Session-Id")
_LOOPBACK_ORIGIN = re.compile(r"^http://(127\.0\.0\.1|localhost|\[::1\])(:\d{1,5})?$")
_B64_SENTINEL = re.compile(r"^=\?base64\?(.*)\?=$")

INSTRUCTIONS = """\
Drives the user's live Blender through a fixed set of safe operations (no Python).
Start with get_scene_info. Use capture_viewport often to see your work (camera=true shows
the scene camera's framing). get_property with an empty path lists what a datablock exposes;
set_property changes one value. Shader nodes: get_shader_graph, add_shader_node,
link_shader_nodes, set_node_input. Each change is one undo step (undo tool).
If a call is refused with read_only or unsafe_blender, tell the user what to change in
the Blender Safe MCP sidebar panel; don't try to work around it.
If something behaves oddly, call get_debug_logs (errors_only=true for crashes)."""


def _reject_constant(name):
    raise ValueError(f"non-finite number {name} not allowed")


def _too_deep(value, depth=0):
    if depth > MAX_JSON_DEPTH:
        return True
    if isinstance(value, dict):
        return any(_too_deep(v, depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(_too_deep(v, depth + 1) for v in value)
    return False


class RpcError(Exception):
    def __init__(self, code, message, data=None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data

    def as_error(self):
        err = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


# --------------------------------------------------------------------------- MCP mapping

def tool_from_op(o):
    """An addon op (a list_ops entry) as an MCP Tool."""
    ann = {"readOnlyHint": not o["mutates"], "openWorldHint": False}
    if o["mutates"]:
        # The spec's default for a non-read-only tool is destructive; say so explicitly.
        ann["destructiveHint"] = bool(o.get("destructive"))
    return {"name": o["name"], "description": o["description"],
            "inputSchema": o["input_schema"], "annotations": ann}


def to_content(result, blobs):
    """Op result as MCP content: the JSON as text, with {"$blob": i} PNGs as images."""
    images = []

    def walk(v):
        if isinstance(v, dict):
            if set(v) == {"$blob"} and isinstance(v["$blob"], int) and 0 <= v["$blob"] < len(blobs):
                blob = blobs[v["$blob"]]
                if not bytes(blob[:8]) == PNG_MAGIC:
                    log.warning("dropping non-PNG blob (%d bytes)", len(blob))
                    return "<non-PNG attachment dropped>"
                images.append({"type": "image", "mimeType": "image/png",
                               "data": base64.b64encode(blob).decode("ascii")})
                return f"<image {len(images)} attached>"
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    text = json.dumps(walk(result), indent=1, ensure_ascii=False, allow_nan=False)
    return [{"type": "text", "text": text}, *images]


def _tool_error(text):
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _decode_header_value(value):
    """Undo the `=?base64?...?=` wrapping clients use for non-ASCII header values."""
    if value is None:
        return None
    m = _B64_SENTINEL.match(value)
    if m is None:
        return value
    try:
        return base64.b64decode(m.group(1), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None


class _Sessions:
    """Handshake-era sessions: id -> negotiated version. Bounded and expiring."""

    def __init__(self):
        self._lock = threading.Lock()
        self._s = OrderedDict()

    def create(self, version):
        sid = secrets.token_hex(16)
        with self._lock:
            self._s[sid] = [version, time.monotonic()]
            while len(self._s) > MAX_SESSIONS:
                old, _ = self._s.popitem(last=False)
                log.info("session %s… evicted (too many sessions)", old[:8])
        return sid

    def get(self, sid):
        now = time.monotonic()
        with self._lock:
            entry = self._s.get(sid)
            if entry is None:
                return None
            if now - entry[1] > SESSION_IDLE_S:
                del self._s[sid]
                return None
            entry[1] = now
            self._s.move_to_end(sid)
            return entry[0]

    def remove(self, sid):
        with self._lock:
            return self._s.pop(sid, None) is not None

    def clear(self):
        with self._lock:
            self._s.clear()

    def __len__(self):
        with self._lock:
            return len(self._s)


# --------------------------------------------------------------------------- HTTP

class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, addr, mcp, family):
        self.address_family = family
        self.mcp = mcp
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        self._conns = set()
        self._conns_lock = threading.Lock()
        super().__init__(addr, _Handler)

    def server_bind(self):
        # HTTPServer.server_bind does a reverse-DNS lookup (getfqdn); skip it.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            log.warning("too many connections; dropping one from %s", client_address[0])
            self.shutdown_request(request)
            return
        with self._conns_lock:
            self._conns.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release(request)
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release(request)

    def _release(self, request):
        with self._conns_lock:
            if request not in self._conns:
                return
            self._conns.discard(request)
        self._slots.release()

    def close_connections(self):
        with self._conns_lock:
            conns = list(self._conns)
        for c in conns:
            try:
                c.shutdown(2)
            except OSError:
                pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "BlenderSafeMCP"
    sys_version = ""
    timeout = SOCKET_TIMEOUT

    # The request line has no secrets (the key is a header); keep it at DEBUG.
    def log_message(self, fmt, *args):
        log.debug("http %s " + fmt, self.client_address[0], *args)

    @property
    def mcp(self):
        return self.server.mcp

    @property
    def peer(self):
        return self.client_address[0]

    # ------------------------------------------------------------------ responses
    def _send(self, status, body=None, headers=()):
        data = b""
        if body is not None:
            try:
                data = json.dumps(body, separators=(",", ":"), ensure_ascii=False,
                                  allow_nan=False).encode("utf-8")
            except ValueError as e:  # NaN/inf in a result
                log.error("unserializable response: %s", e)
                status = 500
                data = json.dumps({"jsonrpc": "2.0", "id": body.get("id"),
                                   "error": {"code": INTERNAL_ERROR,
                                             "message": "result not serializable"}}).encode()
        self.send_response(status)
        if body is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in headers:
            self.send_header(k, v)
        if not getattr(self, "_body_read", False):
            # Unread request body left on the socket: don't reuse the connection.
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _rpc_error(self, status, rid, code, message, data=None, headers=()):
        err = RpcError(code, message, data).as_error()
        self._send(status, {"jsonrpc": "2.0", "id": rid, "error": err}, headers)

    # ------------------------------------------------------------------ gatekeeping
    def _host_ok(self):
        hosts = self.headers.get_all("Host") or []
        if len(hosts) != 1:
            return False
        host = hosts[0].strip().lower()
        if host.startswith("["):
            name = host[1:host.find("]")] if "]" in host else ""
        else:
            name = host.split(":", 1)[0]
        if name == "localhost":
            return True
        try:
            ipaddress.ip_address(name)
            return True
        except ValueError:
            return False

    def _origin_ok(self):
        origins = self.headers.get_all("Origin") or []
        if not origins:
            return True
        return len(origins) == 1 and bool(_LOOPBACK_ORIGIN.match(origins[0].strip().lower()))

    def _admit(self):
        """Common checks for every method. Returns True, or sends the refusal."""
        self._body_read = False
        if not self._host_ok():
            log.warning("refused request from %s: Host %r is not an IP or localhost "
                        "(DNS rebinding?)", self.peer, self.headers.get("Host"))
            self._send(421, {"error": "misdirected request"})
            return False
        if not self._origin_ok():
            log.warning("refused request from %s: Origin %r", self.peer, self.headers.get("Origin"))
            self._send(403, {"error": "origin not allowed"})
            return False
        if self.path.split("?", 1)[0] != ENDPOINT:
            self._send(404, {"error": "not found"})
            return False
        wait = self.mcp.limiter.blocked(self.peer)
        if wait:
            log.info("refusing %s: back-off after failed auth, %.1f s left", self.peer, wait)
            self._send(429, {"error": "too many failed logins"},
                       [("Retry-After", str(max(1, int(wait + 0.999))))])
            return False
        auth = self.headers.get_all("Authorization") or []
        if not auth:
            self._send(401, {"error": "missing API key"}, [("WWW-Authenticate", "Bearer")])
            return False
        key = self.mcp.key_provider()
        scheme, _, given = auth[0].strip().partition(" ")
        if (len(auth) != 1 or scheme.lower() != "bearer" or not key
                or not hmac.compare_digest(given.strip().encode("utf-8", "replace"),
                                           key.encode("utf-8"))):
            self.mcp.limiter.fail(self.peer)
            log.warning("rejected request from %s: authentication failed", self.peer)
            self._send(401, {"error": "invalid API key"}, [("WWW-Authenticate", "Bearer")])
            return False
        self.mcp.limiter.succeed(self.peer)
        self.mcp.last_client = (self.peer, time.monotonic())
        for h in ROUTING_HEADERS:
            if len(self.headers.get_all(h) or []) > 1:
                self._rpc_error(400, None, HEADER_MISMATCH, f"{h} header appears more than once")
                return False
        return True

    # ------------------------------------------------------------------ methods
    def do_GET(self):
        if self._admit():
            # No server-initiated stream: every response is a plain JSON body.
            self._send(405, None, [("Allow", "POST, DELETE")])

    def do_DELETE(self):
        if not self._admit():
            return
        sid = self.headers.get("Mcp-Session-Id")
        if sid is None:
            self._send(400, {"error": "missing Mcp-Session-Id"})
        elif self.mcp.sessions.remove(sid):
            log.info("session %s… closed by %s", sid[:8], self.peer)
            self._send(200)
        else:
            self._send(404, {"error": "unknown session"})

    def do_POST(self):
        if not self._admit():
            return
        if self.headers.get("Transfer-Encoding"):
            self._send(411, {"error": "Content-Length required"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send(411, {"error": "Content-Length required"})
            return
        # The cap is looked up here, after _admit authenticated the caller: an
        # unauthenticated client can never get a large body read.
        cap = self.mcp.max_body_bytes
        if not 0 <= length <= cap:
            self._send(413, {"error": f"body over {cap} bytes"})
            return
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            self._send(415, {"error": "Content-Type must be application/json"})
            return
        accept = (self.headers.get("Accept") or "*/*").lower()
        if not any(t in accept for t in ("application/json", "application/*", "*/*")):
            self._send(406, {"error": "this server only answers application/json"})
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self.close_connection = True
            return
        self._body_read = True

        try:
            msg = json.loads(body.decode("utf-8"), parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError, RecursionError):
            self._rpc_error(400, None, PARSE_ERROR, "Parse error")
            return
        if _too_deep(msg):
            self._rpc_error(400, None, INVALID_REQUEST, "JSON nested too deeply")
            return
        if isinstance(msg, list):
            self._rpc_error(400, None, INVALID_REQUEST, "batches are not supported")
            return
        rid = msg.get("id") if isinstance(msg, dict) else None
        if (not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0"
                or not isinstance(msg.get("method"), str)
                or ("id" in msg and (isinstance(rid, bool) or not isinstance(rid, (int, str))))
                or ("params" in msg and not isinstance(msg["params"], dict))):
            self._rpc_error(400, rid if isinstance(rid, (int, str)) else None, INVALID_REQUEST,
                            "body must be one JSON-RPC 2.0 request or notification object")
            return

        sid = self.headers.get("Mcp-Session-Id")
        version = self.headers.get("MCP-Protocol-Version")
        if sid is None and msg["method"] != "initialize" and version not in HANDSHAKE_VERSIONS:
            self._modern(msg)
        else:
            self._handshake_era(msg, sid, version)

    # ------------------------------------------------------------------ 2026-07-28
    def _modern(self, msg):
        method, header_version = msg["method"], self.headers.get("MCP-Protocol-Version")
        if "id" not in msg:
            # The stateless era has no client notifications to act on: acknowledge and drop.
            if header_version not in MODERN_VERSIONS:
                self._rpc_error(400, None, *self._unsupported(header_version))
            else:
                self._send(202)
            return
        rid = msg["id"]
        try:
            params = msg.get("params") or {}
            self._check_envelope(method, params, header_version)
            if method == "server/discover":
                result = {"supportedVersions": list(MODERN_VERSIONS), "capabilities": CAPABILITIES,
                          "instructions": self.mcp.instructions,
                          "cacheScope": "private", "ttlMs": 0}
            elif method == "tools/list":
                result = {"tools": self._tools(), "cacheScope": "private", "ttlMs": 0}
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")
        except RpcError as e:
            self._send(MODERN_ERROR_STATUS.get(e.code, 200),
                       {"jsonrpc": "2.0", "id": rid, "error": e.as_error()})
            return
        result["resultType"] = "complete"
        result.setdefault("_meta", {})[META_SERVER_INFO] = self.mcp.server_info
        self._send(200, {"jsonrpc": "2.0", "id": rid, "result": result})

    def _unsupported(self, requested):
        return (UNSUPPORTED_PROTOCOL_VERSION, "Unsupported protocol version",
                {"supported": list(SUPPORTED_VERSIONS), "requested": requested})

    def _check_envelope(self, method, params, header_version):
        meta = params.get("_meta")
        if not isinstance(meta, dict):
            raise RpcError(INVALID_PARAMS, f"params._meta must be an object carrying "
                                           f"{META_VERSION!r} and {META_CLIENT_CAPS!r}")
        missing = [k for k in (META_VERSION, META_CLIENT_CAPS) if k not in meta]
        if missing:
            raise RpcError(INVALID_PARAMS, f"params._meta is missing: {', '.join(missing)}")
        version = meta[META_VERSION]
        if header_version is None or header_version != version:
            raise RpcError(HEADER_MISMATCH, "MCP-Protocol-Version header does not match "
                                            "the request envelope's protocol version")
        if self.headers.get("Mcp-Method") != method:
            raise RpcError(HEADER_MISMATCH, "Mcp-Method header does not match the method")
        if method == "tools/call" and params.get("name") is not None:
            if _decode_header_value(self.headers.get("Mcp-Name")) != params.get("name"):
                raise RpcError(HEADER_MISMATCH, "Mcp-Name header does not match the tool name")
        if version not in MODERN_VERSIONS:
            raise RpcError(*self._unsupported(version))

    # ------------------------------------------------------------------ handshake era
    def _handshake_era(self, msg, sid, header_version):
        method, rid, params = msg["method"], msg.get("id"), msg.get("params") or {}
        if method == "initialize":
            if "id" not in msg:
                self._rpc_error(400, None, INVALID_REQUEST, "initialize must be a request")
                return
            requested = params.get("protocolVersion")
            version = requested if requested in HANDSHAKE_VERSIONS else HANDSHAKE_VERSIONS[0]
            sid = self.mcp.sessions.create(version)
            client = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
            log.info("session %s… opened by %s (%s %s, protocol %s)", sid[:8], self.peer,
                     str(client.get("name", "?"))[:64], str(client.get("version", ""))[:32], version)
            result = {"protocolVersion": version, "capabilities": CAPABILITIES,
                      "serverInfo": self.mcp.server_info, "instructions": self.mcp.instructions}
            self._send(200, {"jsonrpc": "2.0", "id": rid, "result": result},
                       [("Mcp-Session-Id", sid)])
            return
        if sid is None:
            self._rpc_error(400, rid, INVALID_REQUEST, "missing Mcp-Session-Id header")
            return
        if self.mcp.sessions.get(sid) is None:
            # The spec's signal for "start a new session" (e.g. Blender restarted).
            self._rpc_error(404, rid, INVALID_REQUEST, "session not found")
            return
        if header_version is not None and header_version not in HANDSHAKE_VERSIONS:
            self._rpc_error(400, rid, *self._unsupported(header_version))
            return
        session = [("Mcp-Session-Id", sid)]
        if "id" not in msg:
            self._send(202, None, session)
            return
        try:
            if method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self._tools()}
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")
        except RpcError as e:
            self._send(200, {"jsonrpc": "2.0", "id": rid, "error": e.as_error()}, session)
            return
        self._send(200, {"jsonrpc": "2.0", "id": rid, "result": result}, session)

    # ------------------------------------------------------------------ shared methods
    def _tools(self):
        try:
            result, _ = self.mcp.work.call("list_ops", {})
        except CancelledError:
            raise RpcError(INTERNAL_ERROR, "server is stopping") from None
        except Exception as e:  # noqa: BLE001
            log.error("list_ops failed", exc_info=e)
            raise RpcError(INTERNAL_ERROR, f"list_ops failed: {type(e).__name__}") from None
        tools = [tool_from_op(o) for o in result["ops"] if o["name"] != "list_ops"]
        log.debug("listed %d tools for %s", len(tools), self.peer)
        return tools

    def _call_tool(self, params):
        name, args = params.get("name"), params.get("arguments")
        if not isinstance(name, str) or not 0 < len(name) <= 64:
            raise RpcError(INVALID_PARAMS, "tools/call needs a tool name")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise RpcError(INVALID_PARAMS, "arguments must be an object")
        if name == "list_ops":
            raise RpcError(INVALID_PARAMS, f"Unknown tool: {name}")
        if log.isEnabledFor(logging.DEBUG):
            log.debug("op %s params %s (http)", name, _summary(args))
        t0 = time.monotonic()
        try:
            result, blobs = self.mcp.work.call(name, args)
            content = to_content(result, blobs)
        except OpError as e:
            if e.code == "unknown_op":
                raise RpcError(INVALID_PARAMS, f"Unknown tool: {name}") from None
            log.info("op %s refused: %s: %s", name, e.code, e.message)
            return _tool_error(f"{e.code}: {e.message}")
        except CancelledError:
            return _tool_error("disconnected: the Blender Safe MCP server was stopped")
        except Exception as e:  # noqa: BLE001
            log.error("op %s crashed", name, exc_info=e)
            return _tool_error(f"internal: {type(e).__name__}: {e}")
        log.info("op %s ok (%.0f ms, %d images, http)", name,
                 (time.monotonic() - t0) * 1000, len(content) - 1)
        return {"content": content, "isError": False}


class McpHttpServer:
    """start()/stop() from the UI; pump() on Blender's main thread runs the queued ops."""

    def __init__(self, key_provider, dispatch, host="127.0.0.1", port=DEFAULT_PORT,
                 server_version="0.1.0", instructions=INSTRUCTIONS):
        self.key_provider = key_provider
        self.host, self.port = host, port
        self.server_info = {"name": "blender-safe-mcp", "version": server_version}
        self.instructions = instructions
        self.work = WorkQueue(dispatch)
        self.sessions = _Sessions()
        self.limiter = _PeerLimiter()
        self.max_body_bytes = MAX_BODY_BYTES  # settable at runtime (images raise it)
        self.last_client = None  # (peer, monotonic time) of the last authenticated request
        self._httpd = None
        self._thread = None

    @property
    def running(self):
        return self._httpd is not None

    @property
    def url(self):
        host = self.host
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}{ENDPOINT}"

    def start(self):
        if self._httpd is not None:
            return
        import socket
        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        httpd = _HTTPServer((self.host, self.port), self, family)
        self.port = httpd.server_address[1]
        self._thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.5},
                                        name="blender-safe-mcp-http", daemon=True)
        self._thread.start()
        self._httpd = httpd
        log.info("MCP over HTTP on %s (protocols %s)", self.url, ", ".join(SUPPORTED_VERSIONS))

    def stop(self):
        httpd, self._httpd = self._httpd, None
        # Refuse new work first, so a request on a kept-alive connection can't block forever
        # on a queue nobody pumps any more.
        self.work.close()
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
            httpd.close_connections()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.sessions.clear()

    def pump(self, budget_s=0.05):
        return self.work.pump(budget_s)

    def reset_sessions(self):
        """After a key change: drop sessions (each request is re-authenticated anyway)."""
        self.sessions.clear()
