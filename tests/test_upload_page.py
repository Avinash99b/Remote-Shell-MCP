"""Tests for the browser upload page at "/" and its JSON upload API.

These start a real uvicorn server (like ``test_http_integration.py``) and drive
the page/API with ``httpx``. Nothing here touches ``/kaggle``; the default
folder is overridden per test.
"""

from __future__ import annotations

import hashlib
import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn

import mcp_shell_server as srv


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(cfg):
    port = _free_port()
    cfg.host = "127.0.0.1"
    cfg.port = port
    saved_cfg, saved_service = srv.CFG, srv.SERVICE
    app = srv.build_app(cfg)
    service: srv.ShellService = app.state.shell_service
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn-upload-test")
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    return f"http://127.0.0.1:{port}", server, thread, service, (saved_cfg, saved_service)


def _stop(server, thread, service, saved):
    server.should_exit = True
    thread.join(timeout=10)
    service.shutdown()
    srv.CFG, srv.SERVICE = saved


@pytest.fixture()
def upload_server(tmp_path):
    cfg = srv.Config(
        no_auth=True,
        cwd=str(tmp_path),
        upload_dir=str(tmp_path / "kaggle" / "working" / "uploads"),
        cleanup_interval_seconds=0.0,
    )
    base, server, thread, service, saved = _serve(cfg)
    try:
        yield base, service
    finally:
        _stop(server, thread, service, saved)


@pytest.fixture()
def authed_upload_server(tmp_path):
    cfg = srv.Config(
        no_auth=False,
        token="secret-token",
        cwd=str(tmp_path),
        upload_dir=str(tmp_path / "uploads"),
        cleanup_interval_seconds=0.0,
    )
    base, server, thread, service, saved = _serve(cfg)
    try:
        yield base
    finally:
        _stop(server, thread, service, saved)


# ---------------------------------------------------------------------------
# Page + health
# ---------------------------------------------------------------------------
def test_index_page_is_served(upload_server):
    base, _ = upload_server
    r = httpx.get(f"{base}/", timeout=5.0)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Upload files" in r.text
    assert "/api/upload" in r.text
    # The page links to the MCP endpoint it shares the origin with.
    assert "/mcp" in r.text


def test_health_exposes_upload_defaults(upload_server):
    base, service = upload_server
    data = httpx.get(f"{base}/health", timeout=5.0).json()
    assert data["upload_dir"] == service.cfg.upload_dir
    assert data["auth_required"] is False


def test_auth_required_flag_reported(authed_upload_server):
    data = httpx.get(f"{authed_upload_server}/health", timeout=5.0).json()
    assert data["auth_required"] is True


# ---------------------------------------------------------------------------
# Destination inspection
# ---------------------------------------------------------------------------
def test_check_reports_missing_parent(upload_server):
    base, service = upload_server
    target_dir = str(service.cfg.upload_dir)
    r = httpx.post(
        f"{base}/api/check",
        json={"directory": target_dir, "filename": "hello.txt"},
        timeout=5.0,
    )
    assert r.status_code == 200
    info = r.json()
    assert info["ok"] is True
    assert info["parent_exists"] is False
    assert info["target_exists"] is False
    assert info["path"] == f"{target_dir}/hello.txt"
    assert info["directory"] == target_dir


def test_check_rejects_path_in_filename(upload_server):
    base, service = upload_server
    for bad in ("", "  ", "../escape.txt", "sub/file.txt", "sub\\file.txt", ".", "..", "bad\x00name"):
        r = httpx.post(
            f"{base}/api/check",
            json={"directory": service.cfg.upload_dir, "filename": bad},
            timeout=5.0,
        )
        assert r.status_code == 400, bad
        assert r.json()["ok"] is False


# ---------------------------------------------------------------------------
# Upload: missing parent confirmation, mkdir, overwrite
# ---------------------------------------------------------------------------
def test_upload_without_create_dirs_reports_missing_parent(upload_server):
    base, service = upload_server
    target_dir = str(service.cfg.upload_dir)
    r = httpx.post(
        f"{base}/api/upload",
        params={"directory": target_dir, "filename": "hello.txt"},
        content=b"hello",
        timeout=5.0,
    )
    assert r.status_code == 409
    body = r.json()
    assert body["ok"] is False
    assert body["error"] == "parent_missing"
    assert target_dir in body["message"]
    assert not os.path.exists(os.path.join(target_dir, "hello.txt"))


def test_upload_creates_parents_recursively(upload_server):
    base, service = upload_server
    target_dir = str(service.cfg.upload_dir)
    payload = b"some binary\x00content"
    r = httpx.post(
        f"{base}/api/upload",
        params={"directory": target_dir, "filename": "nested.txt", "create_dirs": "1"},
        content=payload,
        timeout=5.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["bytes"] == len(payload)
    assert body["sha256"] == hashlib.sha256(payload).hexdigest()
    assert body["overwritten"] is False
    with open(body["path"], "rb") as fh:
        assert fh.read() == payload


def test_upload_refuses_then_allows_overwrite(upload_server):
    base, service = upload_server
    target_dir = str(service.cfg.upload_dir)
    os.makedirs(target_dir)
    params = {"directory": target_dir, "filename": "dup.txt", "create_dirs": "1"}

    first = httpx.post(f"{base}/api/upload", params=params, content=b"first", timeout=5.0)
    assert first.status_code == 200

    second = httpx.post(f"{base}/api/upload", params=params, content=b"second", timeout=5.0)
    assert second.status_code == 409
    assert second.json()["error"] == "file_exists"

    third = httpx.post(
        f"{base}/api/upload", params={**params, "overwrite": "1"}, content=b"second", timeout=5.0
    )
    assert third.status_code == 200
    assert third.json()["overwritten"] is True
    with open(third.json()["path"], "rb") as fh:
        assert fh.read() == b"second"


def test_upload_streams_larger_payload(upload_server):
    base, service = upload_server
    target_dir = str(service.cfg.upload_dir)
    payload = bytes(range(256)) * (2 * 1024 * 4)  # 2 MiB
    r = httpx.post(
        f"{base}/api/upload",
        params={"directory": target_dir, "filename": "big.bin", "create_dirs": "1"},
        content=payload,
        timeout=30.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["bytes"] == len(payload)
    assert body["sha256"] == hashlib.sha256(payload).hexdigest()
    with open(body["path"], "rb") as fh:
        assert fh.read() == payload


def test_upload_defaults_to_configured_folder(upload_server):
    base, service = upload_server
    r = httpx.post(
        f"{base}/api/upload",
        params={"filename": "defaulted.txt", "create_dirs": "1"},
        content=b"default",
        timeout=5.0,
    )
    assert r.status_code == 200
    assert r.json()["path"].startswith(str(service.cfg.upload_dir))


def test_upload_parent_is_a_file(upload_server, tmp_path):
    base, _ = upload_server
    not_a_dir = tmp_path / "blocker"
    not_a_dir.write_text("x")
    r = httpx.post(
        f"{base}/api/upload",
        params={"directory": str(not_a_dir), "filename": "x.txt"},
        content=b"x",
        timeout=5.0,
    )
    assert r.status_code == 409
    assert r.json()["error"] == "parent_not_directory"


# ---------------------------------------------------------------------------
# Auth: page is public, writes are protected
# ---------------------------------------------------------------------------
def test_auth_page_public_but_upload_protected(authed_upload_server):
    base = authed_upload_server

    assert httpx.get(f"{base}/", timeout=5.0).status_code == 200

    params = {"directory": "uploads", "filename": "auth.txt", "create_dirs": "1"}
    denied = httpx.post(f"{base}/api/upload", params=params, content=b"nope", timeout=5.0)
    assert denied.status_code == 401

    allowed = httpx.post(
        f"{base}/api/upload",
        params=params,
        content=b"yes",
        headers={"Authorization": "Bearer secret-token"},
        timeout=5.0,
    )
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["ok"] is True
