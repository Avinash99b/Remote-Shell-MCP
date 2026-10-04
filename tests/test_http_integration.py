"""End-to-end tests over the real Streamable HTTP transport + uvicorn server.

These start an actual uvicorn server (in a background thread) and connect a real
MCP client over HTTP. They verify the property that matters most: the server
answers unrelated requests while a long-running command is still executing, and
background jobs survive client disconnects.
"""

from __future__ import annotations

import socket
import threading
import time

import anyio
import pytest
import uvicorn

import mcp_shell_server as srv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def http_server(tmp_path):
    port = _free_port()
    cfg = srv.Config(
        host="127.0.0.1",
        port=port,
        no_auth=True,
        cwd=str(tmp_path),
        wait_seconds=60.0,
        max_runtime_seconds=0.0,
        max_output_chars=20_000,
        job_retention_seconds=900.0,
        max_jobs=256,
        cleanup_interval_seconds=0.0,
    )

    saved_cfg, saved_service = srv.CFG, srv.SERVICE
    app = srv.build_app(cfg)
    service: srv.ShellService = app.state.shell_service

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn-test")
    thread.start()

    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"

    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        service.shutdown()
        srv.CFG, srv.SERVICE = saved_cfg, saved_service


def _auth_config(tmp_path, token):
    return srv.Config(host="127.0.0.1", no_auth=False, token=token, cwd=str(tmp_path), cleanup_interval_seconds=0.0)


@pytest.fixture()
def authed_server(tmp_path):
    port = _free_port()
    cfg = _auth_config(tmp_path, "secret-token")
    saved_cfg, saved_service = srv.CFG, srv.SERVICE
    app = srv.build_app(cfg)
    service: srv.ShellService = app.state.shell_service
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn-auth-test")
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        service.shutdown()
        srv.CFG, srv.SERVICE = saved_cfg, saved_service


def test_http_short_command(http_server):
    async def scenario():
        async with streamablehttp_client(http_server) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()
                res = await client.call_tool("run_shell_command", {"command": "echo http-ok"})
                assert res.structuredContent["exit_code"] == 0
                assert res.structuredContent["stdout"].strip() == "http-ok"

    anyio.run(scenario)


def test_http_auth_required(authed_server):
    import httpx

    # Missing / wrong bearer token must be rejected before reaching MCP.
    r = httpx.post(authed_server, json={}, timeout=5.0)
    assert r.status_code == 401
    r2 = httpx.post(authed_server, json={}, headers={"Authorization": "Bearer wrong"}, timeout=5.0)
    assert r2.status_code == 401


def test_http_server_responsive_while_long_job_runs(http_server):
    async def scenario():
        async with streamablehttp_client(http_server) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()

                res = await client.call_tool(
                    "run_shell_command",
                    {"command": "sleep 30", "wait_seconds": 0.2},
                )
                payload = res.structuredContent
                assert payload["running"] is True
                job_id = payload["job_id"]

                t0 = time.monotonic()
                other = await client.call_tool("run_shell_command", {"command": "echo responsive"})
                elapsed = time.monotonic() - t0
                assert elapsed < 5.0, f"server was blocked for {elapsed:.1f}s"
                assert other.structuredContent["stdout"].strip() == "responsive"

                got = await client.call_tool("get_job", {"job_id": job_id})
                assert got.structuredContent["running"] is True
                term = await client.call_tool(
                    "terminate_job", {"job_id": job_id, "grace_seconds": 3.0}
                )
                assert term.structuredContent["running"] is False

    anyio.run(scenario)


def test_http_jobs_survive_reconnect(http_server):
    """A job started in one MCP session is queryable from a new session."""

    async def start_job():
        async with streamablehttp_client(http_server) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()
                res = await client.call_tool(
                    "run_shell_command",
                    {"command": "sleep 30", "wait_seconds": 0.1},
                )
                return res.structuredContent["job_id"]

    async def query_job(job_id):
        async with streamablehttp_client(http_server) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()
                got = await client.call_tool("get_job", {"job_id": job_id})
                assert got.structuredContent["found"] is True
                return got.structuredContent

    job_id = anyio.run(start_job)
    snap = anyio.run(lambda: query_job(job_id))
    assert snap["running"] is True
    assert snap["job_id"] == job_id

    srv.SERVICE.manager.get(srv.CFG, job_id).terminate()


def test_http_disconnect_does_not_kill_job(http_server):
    """Closing the client must not cancel the background process."""

    async def start_and_disconnect():
        async with streamablehttp_client(http_server) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()
                res = await client.call_tool(
                    "run_shell_command",
                    {"command": "sleep 30", "wait_seconds": 0.1},
                )
                return res.structuredContent["job_id"]

    job_id = anyio.run(start_and_disconnect)
    time.sleep(0.3)
    job = srv.SERVICE.manager.get(srv.CFG, job_id)
    assert job is not None and job.running
    job.terminate()
    job.wait_done(3.0)


@pytest.mark.slow
def test_http_60s_transition_default(http_server):
    """Default (no wait_seconds) must cap at 60s and hand back a job id."""

    async def scenario():
        async with streamablehttp_client(http_server) as (read, write, _):
            async with ClientSession(read, write) as client:
                await client.initialize()
                t0 = time.monotonic()
                res = await client.call_tool("run_shell_command", {"command": "sleep 90"})
                elapsed = time.monotonic() - t0
                payload = res.structuredContent
                assert payload["running"] is True, payload
                assert payload["job_id"]
                assert 59.0 <= elapsed <= 68.0, f"unexpected wait {elapsed:.1f}s"
                await client.call_tool(
                    "terminate_job", {"job_id": payload["job_id"], "grace_seconds": 2.0}
                )

    anyio.run(scenario)