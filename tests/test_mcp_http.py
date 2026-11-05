"""The addon's own MCP-over-HTTP endpoint, driven by the official MCP SDK client
(both protocol eras) and by raw HTTP for the security checks."""
import http.client
import json
import threading
import time

import anyio
import httpx2
import mcp
import pytest
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from fixtures import KEY, make_registry
from blender_safe_mcp import mcp_http as H


class RunningHttp:
    def __init__(self, key=KEY):
        self.key = key
        self.registry = make_registry()
        self.server = H.McpHttpServer(lambda: self.key, self.registry.dispatch, port=0)
        self._stop = threading.Event()

    def __enter__(self):
        self.server.start()
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()
        return self

    def _pump(self):
        while not self._stop.is_set():
            self.server.pump()
            time.sleep(0.005)

    def __exit__(self, *exc):
        self._stop.set()
        self.server.stop()
        self._t.join(1)

    @property
    def url(self):
        return self.server.url


@pytest.fixture
def srv():
    with RunningHttp() as s:
        yield s


def sdk(srv, mode, key=KEY):
    hc = httpx2.AsyncClient(headers={"Authorization": f"Bearer {key}"})
    return hc, mcp.Client(streamable_http_client(srv.url, http_client=hc), mode=mode)


def text_of(result):
    return result.content[0].text


# --------------------------------------------------------------------------- SDK client

@pytest.mark.parametrize("mode", ["auto", "2026-07-28", "legacy"])
def test_sdk_client_both_eras(srv, mode):
    async def go():
        hc, client = sdk(srv, mode)
        async with hc, client as c:
            if mode == "legacy":
                assert c.protocol_version == H.HANDSHAKE_VERSIONS[0]
            else:
                assert c.protocol_version == "2026-07-28"
            if mode == "auto":  # the probe really went through server/discover
                d = c.session.discover_result
                assert d.supported_versions == ["2026-07-28"]
                assert d.capabilities.tools is not None
            if mode != "2026-07-28":  # a version pin skips discover, so no info/instructions
                assert c.server_info.name == "blender-safe-mcp"
                assert "get_scene_info" in c.instructions

            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert "list_ops" not in tools
            assert {"echo", "png", "refuse", "crash", "get_debug_logs"} <= set(tools)
            assert tools["echo"].annotations.read_only_hint is True
            assert tools["refuse"].annotations.read_only_hint is False
            assert tools["refuse"].annotations.destructive_hint is False
            assert tools["echo"].input_schema["additionalProperties"] is False

            r = await c.call_tool("echo", {"text": "ab", "n": 2})
            assert not r.is_error and json.loads(text_of(r)) == {"text": "abab"}

            r = await c.call_tool("png", {"size": 3})
            assert not r.is_error and r.content[1].type == "image"
            assert r.content[1].mime_type == "image/png"
            assert json.loads(text_of(r))["image"] == "<image 1 attached>"

            r = await c.call_tool("echo", {"text": "x", "bogus": 1})
            assert r.is_error and text_of(r).startswith("invalid_params:")

            r = await c.call_tool("refuse", {})
            assert r.is_error and text_of(r) == "read_only: nope"

            r = await c.call_tool("crash", {})
            assert r.is_error and "internal: KeyError" in text_of(r)

            with pytest.raises(MCPError, match="Unknown tool"):
                await c.call_tool("no_such_op", {})
            with pytest.raises(MCPError, match="Unknown tool"):
                await c.call_tool("list_ops", {})

    anyio.run(go)


def test_sdk_client_wrong_key_fails(srv):
    async def go():
        hc, client = sdk(srv, "auto", key="wrong")
        async with hc:
            with pytest.raises(Exception):
                async with client:
                    pass
    anyio.run(go)
    assert srv.server.limiter.blocked("127.0.0.1") > 0


def test_session_survives_until_server_restart():
    """A restarted Blender forgets sessions: the client gets 404, the spec's cue to re-init."""
    with RunningHttp() as s:
        conn = Raw(s)
        st, hdr, body = conn.legacy_init()
        sid = hdr["Mcp-Session-Id"]
        st, _, body = conn.post({"jsonrpc": "2.0", "id": 2, "method": "ping"},
                                {"Mcp-Session-Id": sid})
        assert st == 200 and body["result"] == {}
        s.server.reset_sessions()
        st, _, body = conn.post({"jsonrpc": "2.0", "id": 3, "method": "ping"},
                                {"Mcp-Session-Id": sid})
        assert st == 404


# --------------------------------------------------------------------------- raw HTTP

class Raw:
    def __init__(self, srv):
        self.port = srv.server.port

    def request(self, method, body=None, headers=None, path="/mcp", auth=True, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if auth:
            h["Authorization"] = f"Bearer {KEY}"
        h.update(headers or {})
        h = {k: v for k, v in h.items() if v is not None}
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        conn.request(method, path, body=data, headers=h)
        r = conn.getresponse()
        payload = r.read()
        conn.close()
        try:
            parsed = json.loads(payload) if payload else None
        except ValueError:
            parsed = payload
        return r.status, r.headers, parsed

    def post(self, body, headers=None, **kw):
        return self.request("POST", body, headers, **kw)

    def modern(self, method, params=None, version="2026-07-28", headers=None, rid=1):
        params = dict(params or {})
        params["_meta"] = {H.META_VERSION: version, H.META_CLIENT_CAPS: {}}
        h = {"MCP-Protocol-Version": version, "Mcp-Method": method}
        if method == "tools/call":
            h["Mcp-Name"] = params.get("name")
        h.update(headers or {})
        return self.post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}, h)

    def legacy_init(self, version="2025-06-18"):
        return self.post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": {"protocolVersion": version, "capabilities": {},
                                     "clientInfo": {"name": "t", "version": "1"}}})


@pytest.fixture
def raw(srv):
    return Raw(srv)


def test_discover_advertises_protocols(raw):
    st, _, body = raw.modern("server/discover")
    assert st == 200
    res = body["result"]
    assert res["supportedVersions"] == ["2026-07-28"]
    assert res["capabilities"] == {"tools": {"listChanged": False}}
    assert res["resultType"] == "complete" and res["cacheScope"] == "private"
    assert res["_meta"][H.META_SERVER_INFO]["name"] == "blender-safe-mcp"


def test_unsupported_version_lists_every_supported_one(raw):
    st, _, body = raw.modern("server/discover", version="2099-01-01")
    assert st == 400
    err = body["error"]
    assert err["code"] == H.UNSUPPORTED_PROTOCOL_VERSION
    assert err["data"] == {"supported": list(H.SUPPORTED_VERSIONS), "requested": "2099-01-01"}


def test_initialize_negotiates_version(raw):
    st, hdr, body = raw.legacy_init("2025-03-26")
    assert st == 200 and body["result"]["protocolVersion"] == "2025-03-26"
    assert len(hdr["Mcp-Session-Id"]) == 32
    st, _, body = raw.legacy_init("1999-01-01")  # unknown: server counter-offers its newest
    assert body["result"]["protocolVersion"] == H.HANDSHAKE_VERSIONS[0]


def test_envelope_checks(raw):
    # header and body disagree on the version / method / tool name
    st, _, body = raw.modern("tools/list", headers={"MCP-Protocol-Version": "2025-11-25",
                                                    "Mcp-Session-Id": None})
    assert st == 400  # routed to the handshake era: no session
    st, _, body = raw.modern("tools/list", headers={"Mcp-Method": "tools/call"})
    assert st == 400 and body["error"]["code"] == H.HEADER_MISMATCH
    st, _, body = raw.modern("tools/call", {"name": "echo", "arguments": {"text": "a"}},
                             headers={"Mcp-Name": "png"})
    assert st == 400 and body["error"]["code"] == H.HEADER_MISMATCH
    # no _meta envelope at all
    st, _, body = raw.post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                           {"MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/list"})
    assert st == 400 and body["error"]["code"] == H.INVALID_PARAMS
    # removed / unknown methods
    st, _, body = raw.modern("ping")
    assert st == 404 and body["error"]["code"] == H.METHOD_NOT_FOUND
    # notifications are acknowledged and dropped
    st, _, _ = raw.post({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}},
                        {"MCP-Protocol-Version": "2026-07-28"})
    assert st == 202
    # duplicated routing header
    conn = http.client.HTTPConnection("127.0.0.1", raw.port, timeout=5)
    conn.putrequest("POST", "/mcp")
    for k, v in [("Authorization", f"Bearer {KEY}"), ("Content-Type", "application/json"),
                 ("MCP-Protocol-Version", "2026-07-28"), ("MCP-Protocol-Version", "2026-07-28")]:
        conn.putheader(k, v)
    conn.putheader("Content-Length", "2")
    conn.endheaders(b"{}")
    r = conn.getresponse()
    assert r.status == 400 and json.loads(r.read())["error"]["code"] == H.HEADER_MISMATCH


def test_modern_tools_call_via_raw(raw):
    st, _, body = raw.modern("tools/call", {"name": "echo", "arguments": {"text": "é"}})
    assert st == 200 and not body["result"]["isError"]
    assert json.loads(body["result"]["content"][0]["text"]) == {"text": "é"}
    # non-ASCII tool names travel base64-wrapped in Mcp-Name; a wrong one is caught
    st, _, body = raw.modern("tools/call", {"name": "echo", "arguments": {"text": "a"}},
                             headers={"Mcp-Name": "=?base64?ZWNobw==?="})
    assert st == 200


def test_handshake_session_rules(raw):
    st, _, body = raw.post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                           {"MCP-Protocol-Version": "2025-11-25"})
    assert st == 400 and "Mcp-Session-Id" in body["error"]["message"]
    st, _, _ = raw.post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                        {"Mcp-Session-Id": "0" * 32})
    assert st == 404
    _, hdr, _ = raw.legacy_init()
    sid = hdr["Mcp-Session-Id"]
    st, _, _ = raw.post({"jsonrpc": "2.0", "method": "notifications/initialized"},
                        {"Mcp-Session-Id": sid})
    assert st == 202
    st, _, body = raw.post({"jsonrpc": "2.0", "id": 2, "method": "resources/list"},
                           {"Mcp-Session-Id": sid})
    assert st == 200 and body["error"]["code"] == H.METHOD_NOT_FOUND
    st, _, _ = raw.request("GET", headers={"Mcp-Session-Id": sid})
    assert st == 405
    st, _, _ = raw.request("DELETE", headers={"Mcp-Session-Id": sid})
    assert st == 200
    st, _, _ = raw.post({"jsonrpc": "2.0", "id": 3, "method": "ping"}, {"Mcp-Session-Id": sid})
    assert st == 404


def test_malformed_bodies(raw):
    st, _, body = raw.post(None, raw=b"{not json")
    assert st == 400 and body["error"]["code"] == H.PARSE_ERROR
    st, _, body = raw.post(None, raw=b'{"a": NaN}')
    assert st == 400 and body["error"]["code"] == H.PARSE_ERROR
    st, _, body = raw.post([{"jsonrpc": "2.0", "id": 1, "method": "ping"}])
    assert st == 400 and body["error"]["code"] == H.INVALID_REQUEST
    st, _, body = raw.post({"jsonrpc": "1.0", "id": 1, "method": "ping"})
    assert st == 400 and body["error"]["code"] == H.INVALID_REQUEST
    deep = "[" * 100 + "]" * 100
    st, _, body = raw.post(None, raw=('{"jsonrpc":"2.0","id":1,"method":"x","params":{"a":%s}}'
                                      % deep).encode())
    assert st == 400 and body["error"]["code"] == H.INVALID_REQUEST
    # oversized: refused from the declared length, before reading (and then the socket closes,
    # so only the headers are sent here; streaming the body would race the close)
    conn = http.client.HTTPConnection("127.0.0.1", raw.port, timeout=5)
    conn.putrequest("POST", "/mcp")
    for k, v in [("Authorization", f"Bearer {KEY}"), ("Content-Type", "application/json"),
                 ("Content-Length", str(H.MAX_BODY_BYTES + 1))]:
        conn.putheader(k, v)
    conn.endheaders()
    r = conn.getresponse()
    assert r.status == 413 and r.getheader("Connection") == "close"
    st, _, _ = raw.post({"jsonrpc": "2.0", "id": 1, "method": "ping"},
                        {"Content-Type": "text/plain"})
    assert st == 415
    st, _, _ = raw.post({"jsonrpc": "2.0", "id": 1, "method": "ping"},
                        {"Accept": "text/event-stream"})
    assert st == 406


def _declare_length(port, length, auth=True):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest("POST", "/mcp")
    hdrs = [("Content-Type", "application/json"), ("Content-Length", str(length))]
    if auth:
        hdrs.append(("Authorization", f"Bearer {KEY}"))
    for k, v in hdrs:
        conn.putheader(k, v)
    conn.endheaders()
    r = conn.getresponse()
    conn.close()
    return r.status


def test_body_cap_is_a_server_attribute_checked_after_auth(srv):
    port = srv.server.port
    big = 3 * 1024 * 1024
    # default cap (1 MB): over -> 413 for an authenticated client, 401 (not 413) otherwise
    assert srv.server.max_body_bytes == H.MAX_BODY_BYTES
    assert _declare_length(port, big) == 413
    assert _declare_length(port, big, auth=False) == 401
    # raised cap: the same length is now accepted past the size check (then fails later, on
    # the missing body / media checks, but never with 413), and a larger one still is refused
    srv.server.max_body_bytes = 64 * 1024 * 1024
    assert _declare_length(port, 64 * 1024 * 1024 + 1) == 413
    assert _declare_length(port, big, auth=False) == 401  # unauthenticated never gets there
    srv.server.max_body_bytes = H.MAX_BODY_BYTES
    assert _declare_length(port, big) == 413


def test_large_body_accepted_when_cap_raised(srv):
    srv.server.max_body_bytes = 8 * 1024 * 1024
    raw = Raw(srv)
    pad = "x" * (3 * 1024 * 1024)
    st, _, body = raw.modern("tools/list", {"pad": pad})
    assert st == 200 and "result" in body, (st, body)
    srv.server.max_body_bytes = H.MAX_BODY_BYTES
    assert _declare_length(srv.server.port, 3 * 1024 * 1024) == 413


def test_chunked_body_refused(raw):
    conn = http.client.HTTPConnection("127.0.0.1", raw.port, timeout=5)
    conn.request("POST", "/mcp", body=iter([b"{}"]), encode_chunked=True,
                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                          "Transfer-Encoding": "chunked"})
    assert conn.getresponse().status == 411


# --------------------------------------------------------------------------- security

def test_missing_key_is_401_without_backoff(srv, raw):
    for _ in range(5):
        st, hdr, _ = raw.modern("server/discover", headers={"Authorization": None})
        assert st == 401 and hdr["WWW-Authenticate"] == "Bearer"
    assert srv.server.limiter.blocked("127.0.0.1") == 0
    st, _, _ = raw.modern("server/discover")
    assert st == 200


def test_wrong_key_then_backoff(srv, raw):
    st, _, _ = raw.modern("server/discover", headers={"Authorization": "Bearer nope"})
    assert st == 401
    st, hdr, _ = raw.modern("server/discover")  # even the right key waits out the back-off
    assert st == 429 and int(hdr["Retry-After"]) >= 1
    time.sleep(srv.server.limiter.blocked("127.0.0.1") + 0.05)
    st, _, _ = raw.modern("server/discover")
    assert st == 200
    # non-Bearer schemes and duplicate Authorization headers count as wrong keys
    st, _, _ = raw.modern("server/discover", headers={"Authorization": f"Basic {KEY}"})
    assert st == 401


def test_key_rotation_takes_effect_immediately(srv, raw):
    assert raw.modern("server/discover")[0] == 200
    srv.key = "rotated-" + KEY
    assert raw.modern("server/discover")[0] == 401


@pytest.mark.parametrize("host", ["evil.example:9877", "evil.example", "localhost.evil:1",
                                  "127.0.0.1.nip.io:9877"])
def test_dns_rebinding_hosts_refused(raw, host):
    st, _, _ = raw.modern("server/discover", headers={"Host": host})
    assert st == 421


@pytest.mark.parametrize("host", ["127.0.0.1:9877", "localhost:1234", "[::1]:9877",
                                  "10.0.2.2:9877", "LOCALHOST"])
def test_ip_and_localhost_hosts_allowed(raw, host):
    assert raw.modern("server/discover", headers={"Host": host})[0] == 200


@pytest.mark.parametrize("origin,ok", [("https://evil.example", False), ("null", False),
                                       ("http://localhost.evil.example", False),
                                       ("http://127.0.0.1:3000", True),
                                       ("http://localhost", True)])
def test_origin_check(raw, origin, ok):
    st, _, _ = raw.modern("server/discover", headers={"Origin": origin})
    assert (st == 200) == ok and (ok or st == 403)


def test_unknown_path_is_404(raw):
    assert raw.request("GET", path="/.well-known/oauth-protected-resource")[0] == 404
    assert raw.post({}, path="/mcp/../x")[0] == 404


def test_stop_unblocks_waiting_requests():
    """A request whose op never gets pumped is cancelled when the server stops."""
    reg = make_registry()
    server = H.McpHttpServer(lambda: KEY, reg.dispatch, port=0)  # nobody pumps
    server.start()
    out = {}

    def call():
        try:
            out["r"] = Raw(type("S", (), {"server": server})()).modern(
                "tools/call", {"name": "echo", "arguments": {"text": "a"}})
        except Exception as e:  # noqa: BLE001
            out["r"] = e

    t = threading.Thread(target=call)
    t.start()
    time.sleep(0.3)
    server.stop()
    t.join(5)
    assert not t.is_alive()
    st, _, body = out["r"]
    assert st == 200 and body["result"]["isError"]
    assert "stopped" in body["result"]["content"][0]["text"]


def test_key_never_logged(srv, raw):
    from fixtures import ADDON_RING
    raw.modern("tools/call", {"name": "echo", "arguments": {"text": "a"}})
    raw.modern("server/discover", headers={"Authorization": "Bearer wrong-secret-xyz"})
    dump = json.dumps(ADDON_RING.query(limit=500))
    assert KEY not in dump and "wrong-secret-xyz" not in dump
