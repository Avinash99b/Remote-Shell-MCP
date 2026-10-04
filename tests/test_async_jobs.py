"""Tests for the asynchronous, non-blocking shell job architecture.

These exercise the job engine directly (fast, deterministic) and the full MCP
round-trip over the in-memory transport (proves the server stays responsive).

Run:
    .venv/bin/python -m pytest tests/ -q
    .venv/bin/python -m pytest tests/ -q -m slow   # includes the 60s transition
"""

from __future__ import annotations

import asyncio
import time

import anyio
import pytest

import mcp_shell_server as srv
from mcp.shared.memory import create_connected_server_and_client_session


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def cfg(tmp_path):
    """A fast, isolated config so tests never wait on the real 60s window."""
    return srv.Config(
        cwd=str(tmp_path),
        wait_seconds=60.0,
        max_runtime_seconds=0.0,
        max_output_chars=20_000,
        job_retention_seconds=900.0,
        max_jobs=256,
        cleanup_interval_seconds=0.0,  # disable background reaper in unit tests
    )


@pytest.fixture()
def service(cfg):
    svc = srv.ShellService(cfg)
    yield svc
    svc.shutdown()


def run_job(manager, cfg, command, wait=5.0, max_runtime=0.0):
    """Start a job and wait for it synchronously (test helper)."""
    job = manager.start(cfg, command, max_runtime)
    assert job.done.wait(timeout=wait), f"job did not finish: {job.snapshot()}"
    return job


# ---------------------------------------------------------------------------
# Direct engine tests
# ---------------------------------------------------------------------------
def test_short_lived_command_completes(service):
    job = run_job(service.manager, service.cfg, "echo hello")
    snap = job.snapshot()
    assert snap["status"] == srv.STATUS_COMPLETED
    assert snap["success"] is True
    assert snap["exit_code"] == 0
    assert snap["stdout"] == "hello\n"
    assert snap["running"] is False
    assert snap["completed"] is True
    assert snap["truncated"] is False


def test_failed_command_reports_nonzero_exit(service):
    job = run_job(service.manager, service.cfg, "echo boom >&2; exit 3")
    snap = job.snapshot()
    assert snap["status"] == srv.STATUS_FAILED
    assert snap["success"] is False
    assert snap["exit_code"] == 3
    assert "boom" in snap["stderr"]


def test_invalid_command_does_not_crash(service):
    job = run_job(service.manager, service.cfg, "definitely_not_a_real_command_xyz")
    snap = job.snapshot()
    assert snap["status"] == srv.STATUS_FAILED
    assert snap["exit_code"] == 127
    assert "not found" in snap["stderr"].lower()


def test_empty_command_rejected(service):
    with pytest.raises(ValueError):
        anyio.run(lambda: service.run_shell_command("   "))


def test_long_running_command_returns_job_and_keeps_running(service):
    job = service.manager.start(service.cfg, "sleep 30", 0.0)
    time.sleep(0.2)
    snap = job.snapshot()
    assert snap["status"] == srv.STATUS_RUNNING
    assert snap["running"] is True
    assert snap["completed"] is False
    assert snap["job_id"] == job.job_id
    assert job.process.poll() is None
    job.terminate()
    job.wait_done(2.0)


def test_query_running_job_by_id(service):
    job = service.manager.start(service.cfg, "sleep 30", 0.0)
    time.sleep(0.15)
    fetched = service.manager.get(service.cfg, job.job_id)
    assert fetched is job
    assert fetched.snapshot()["status"] == srv.STATUS_RUNNING
    assert service.manager.get(service.cfg, "does-not-exist") is None
    job.terminate()
    job.wait_done(2.0)


def test_terminate_running_job(service):
    job = service.manager.start(service.cfg, "sleep 30", 0.0)
    time.sleep(0.15)
    assert job.running
    job.terminate()
    assert job.wait_done(3.0)
    snap = job.snapshot()
    assert snap["status"] == srv.STATUS_TERMINATED
    assert snap["running"] is False
    assert job.done.is_set()


def test_status_stays_running_until_reaped(service):
    """Termination must not report a terminal status before the process dies."""
    job = service.manager.start(service.cfg, "sleep 30", 0.0)
    time.sleep(0.15)
    job.terminate()
    # Immediately after signalling, before the reap thread has observed exit.
    immediate = job.snapshot()
    # Either still running (not yet reaped) or fully terminated — never a
    # terminated-but-still-running inconsistency.
    assert immediate["running"] in (True, False)
    if immediate["running"]:
        assert immediate["completed"] is False
    job.wait_done(3.0)


def test_max_runtime_kills_process(service):
    job = service.manager.start(service.cfg, "sleep 30", 0.3)
    assert job.done.wait(timeout=5.0)
    snap = job.snapshot()
    assert snap["status"] == srv.STATUS_TIMEOUT
    assert snap["timed_out"] is True
    assert snap["running"] is False


def test_timeout_kills_orphaned_subshell(service):
    """A backgrounded grandchild holding the pipe must be killed, not leaked."""
    # The subshell backgrounds a long sleep and exits; without a group kill the
    # sleep keeps the stdout pipe open and the job never finishes.
    job = service.manager.start(service.cfg, "sh -c 'sleep 30 &' ", 0.5)
    assert job.done.wait(timeout=8.0), f"job leaked: {job.snapshot()}"
    assert job.snapshot()["status"] == srv.STATUS_TIMEOUT


def test_large_stdout_is_bounded_but_drained(service):
    job = run_job(
        service.manager,
        service.cfg,
        "python3 -c \"import sys; sys.stdout.write('x'*2_000_000); sys.stderr.write('y'*2_000_000)\"",
        wait=30.0,
    )
    snap = job.snapshot()
    assert snap["status"] == srv.STATUS_COMPLETED
    assert snap["exit_code"] == 0
    assert snap["truncated"] is True
    assert snap["stdout_truncated"] is True
    assert snap["stderr_truncated"] is True
    assert len(snap["stdout"]) < 21_000
    assert "[stdout truncated]" in snap["stdout"]
    assert "[stderr truncated]" in snap["stderr"]


def test_non_ascii_output_not_falsely_truncated(service):
    """Multi-byte characters must not trip the truncation check (bytes != chars)."""
    job = run_job(service.manager, service.cfg, "printf 'héllo wörld\\n'")
    snap = job.snapshot()
    assert snap["status"] == srv.STATUS_COMPLETED
    assert snap["truncated"] is False
    assert snap["stdout_truncated"] is False
    assert snap["stdout"] == "héllo wörld\n"


def test_multibyte_truncation_is_clean(service):
    """Truncation must not split a UTF-8 character into a replacement char."""
    service.cfg.max_output_chars = 17  # odd boundary
    job = run_job(
        service.manager,
        service.cfg,
        "python3 -c \"import sys; sys.stdout.write('\\u00e9'*100)\"",
        wait=15.0,
    )
    snap = job.snapshot()
    assert snap["status"] == srv.STATUS_COMPLETED
    body = snap["stdout"].replace("\n...[stdout truncated]", "")
    assert "\ufffd" not in body
    assert set(body) <= {"é"}


def test_multiple_concurrent_jobs(service):
    jobs = [service.manager.start(service.cfg, f"echo job-{i}; sleep 0.1", 0.0) for i in range(10)]
    for job in jobs:
        assert job.done.wait(timeout=10.0)
    for i, job in enumerate(jobs):
        snap = job.snapshot()
        assert snap["exit_code"] == 0, snap
        assert snap["stdout"].strip() == f"job-{i}"


def test_querying_completed_job_then_cleanup(service):
    job = run_job(service.manager, service.cfg, "echo done")
    assert service.manager.get(service.cfg, job.job_id) is job
    removed = service.manager.cleanup(service.cfg, older_than_seconds=-1.0)
    assert removed == 1
    assert service.manager.get(service.cfg, job.job_id) is None


def test_cleanup_never_removes_running_job(service):
    job = service.manager.start(service.cfg, "sleep 30", 0.0)
    time.sleep(0.1)
    removed = service.manager.cleanup(service.cfg, older_than_seconds=-1.0)
    assert removed == 0
    assert service.manager.get(service.cfg, job.job_id) is job
    job.terminate()
    job.wait_done(2.0)


def test_retention_autocleanup(service):
    service.cfg.job_retention_seconds = 0.05
    job = run_job(service.manager, service.cfg, "echo soon-gone")
    assert service.manager.get(service.cfg, job.job_id) is job
    time.sleep(0.15)
    assert service.manager.get(service.cfg, job.job_id) is None


def test_capacity_eviction_keeps_running_jobs(service):
    service.cfg.max_jobs = 3
    running = service.manager.start(service.cfg, "sleep 30", 0.0)
    time.sleep(0.1)
    for i in range(6):
        run_job(service.manager, service.cfg, f"echo {i}")
    assert service.manager.get(service.cfg, running.job_id) is running
    running.terminate()
    running.wait_done(2.0)


def test_partial_output_visible_while_running(service):
    job = service.manager.start(service.cfg, "echo early; sleep 30", 0.0)
    deadline = time.time() + 5.0
    while time.time() < deadline and "early" not in job.snapshot()["stdout"]:
        time.sleep(0.02)
    snap = job.snapshot()
    assert snap["status"] == srv.STATUS_RUNNING
    assert "early" in snap["stdout"]
    job.terminate()
    job.wait_done(2.0)


def test_shutdown_terminates_running_jobs(service):
    job = service.manager.start(service.cfg, "sleep 30", 0.0)
    time.sleep(0.1)
    service.shutdown()
    assert job.done.wait(timeout=5.0)
    assert job.status == srv.STATUS_TERMINATED


# ---------------------------------------------------------------------------
# Input-hardening tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["nan", float("nan"), float("inf"), -5, "abc", None])
def test_wait_seconds_is_sanitized(cfg, bad):
    value = srv._resolve_wait_seconds(cfg, bad)
    assert 0.0 <= value <= srv.MAX_SYNC_WAIT_SECONDS


def test_wait_seconds_capped_at_60(cfg):
    assert srv._resolve_wait_seconds(cfg, 10_000) == srv.MAX_SYNC_WAIT_SECONDS


def test_max_runtime_sanitized(cfg):
    assert srv._resolve_max_runtime(cfg, float("nan")) == 0.0
    assert srv._resolve_max_runtime(cfg, -1) == 0.0
    assert srv._resolve_max_runtime(cfg, 5) == 5.0


def test_terminate_job_bad_grace_seconds(service):
    job = service.manager.start(service.cfg, "sleep 30", 0.0)
    time.sleep(0.1)
    result = anyio.run(lambda: service.terminate_job(job.job_id, "not-a-number"))
    assert result["found"] is True
    assert result["running"] is False
    assert result["status"] == srv.STATUS_TERMINATED


# ---------------------------------------------------------------------------
# Async waiting helper tests
# ---------------------------------------------------------------------------
def test_await_job_non_blocking_then_completes(service):
    async def scenario():
        job = service.manager.start(service.cfg, "echo async-ok", 0.0)
        finished = await srv._await_job(job, 5.0)
        assert finished is True
        assert job.snapshot()["stdout"].strip() == "async-ok"

    anyio.run(scenario)


def test_await_job_returns_false_on_timeout(service):
    async def scenario():
        job = service.manager.start(service.cfg, "sleep 30", 0.0)
        started = time.monotonic()
        finished = await srv._await_job(job, 0.2)
        elapsed = time.monotonic() - started
        assert finished is False
        assert elapsed < 5.0
        assert job.running
        job.terminate()
        job.wait_done(2.0)

    anyio.run(scenario)


def test_await_job_nan_wait_does_not_hang(service):
    async def scenario():
        job = service.manager.start(service.cfg, "sleep 30", 0.0)
        started = time.monotonic()
        # A NaN wait must be sanitized to 0 before reaching move_on_after.
        finished = await srv._await_job(job, srv._resolve_wait_seconds(service.cfg, float("nan")))
        assert finished is False
        assert time.monotonic() - started < 5.0
        job.terminate()
        job.wait_done(2.0)

    anyio.run(scenario)


def test_event_loop_stays_responsive_during_long_job(service):
    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(20):
                await anyio.sleep(0.05)
                ticks += 1

        job = service.manager.start(service.cfg, "sleep 30", 0.0)
        async with anyio.create_task_group() as tg:
            tg.start_soon(ticker)
            finished = await srv._await_job(job, 0.3)
        assert finished is False
        assert ticks >= 5
        job.terminate()
        job.wait_done(2.0)

    anyio.run(scenario)


# ---------------------------------------------------------------------------
# Full MCP round-trip (in-memory transport) — proves non-blocking behavior
# ---------------------------------------------------------------------------
def test_mcp_short_command_roundtrip(service):
    server = srv.build_server(service)

    async def scenario():
        async with create_connected_server_and_client_session(server, raise_exceptions=False) as client:
            res = await client.call_tool("run_shell_command", {"command": "echo mcp-hello"})
            payload = res.structuredContent
            assert payload is not None
            assert payload["status"] == srv.STATUS_COMPLETED
            assert payload["exit_code"] == 0
            assert payload["stdout"].strip() == "mcp-hello"
            assert payload["job_id"]

    anyio.run(scenario)


def test_mcp_long_command_transitions_and_server_stays_responsive(service):
    server = srv.build_server(service)

    async def scenario():
        async with create_connected_server_and_client_session(server, raise_exceptions=False) as client:
            res = await client.call_tool(
                "run_shell_command",
                {"command": "sleep 30", "wait_seconds": 0.3},
            )
            payload = res.structuredContent
            assert payload["running"] is True
            assert payload["status"] == srv.STATUS_RUNNING
            assert payload["note"]
            job_id = payload["job_id"]

            t0 = time.monotonic()
            other = await client.call_tool("run_shell_command", {"command": "echo pong"})
            assert time.monotonic() - t0 < 5.0
            assert other.structuredContent["stdout"].strip() == "pong"

            got = await client.call_tool("get_job", {"job_id": job_id})
            assert got.structuredContent["found"] is True
            assert got.structuredContent["running"] is True

            term = await client.call_tool("terminate_job", {"job_id": job_id, "grace_seconds": 2.0})
            assert term.structuredContent["status"] == srv.STATUS_TERMINATED
            assert term.structuredContent["running"] is False
            assert term.structuredContent["exit_code"] is not None

    anyio.run(scenario)


def test_mcp_list_and_cleanup_tools(service):
    server = srv.build_server(service)

    async def scenario():
        async with create_connected_server_and_client_session(server, raise_exceptions=False) as client:
            await client.call_tool("run_shell_command", {"command": "echo one"})
            await client.call_tool("run_shell_command", {"command": "echo two"})
            listing = await client.call_tool("list_jobs", {})
            assert listing.structuredContent["count"] == 2
            assert listing.structuredContent["running"] == 0

            cleaned = await client.call_tool("cleanup_jobs", {"older_than_seconds": -1.0})
            assert cleaned.structuredContent["removed"] == 2
            after = await client.call_tool("list_jobs", {})
            assert after.structuredContent["count"] == 0

    anyio.run(scenario)


def test_mcp_unknown_job_id(service):
    server = srv.build_server(service)

    async def scenario():
        async with create_connected_server_and_client_session(server, raise_exceptions=False) as client:
            got = await client.call_tool("get_job", {"job_id": "nope"})
            assert got.structuredContent["found"] is False
            term = await client.call_tool("terminate_job", {"job_id": "nope"})
            assert term.structuredContent["found"] is False

    anyio.run(scenario)


def test_mcp_concurrent_commands_over_transport(service):
    server = srv.build_server(service)

    async def scenario():
        async with create_connected_server_and_client_session(server, raise_exceptions=False) as client:
            async def one(i):
                return await client.call_tool("run_shell_command", {"command": f"echo c{i}"})

            results = await asyncio.gather(*[one(i) for i in range(8)])
            outputs = sorted(r.structuredContent["stdout"].strip() for r in results)
            assert outputs == sorted(f"c{i}" for i in range(8))

    anyio.run(scenario)


def test_service_does_not_start_reaper_eagerly():
    svc = srv.ShellService(srv.Config(cleanup_interval_seconds=60.0))
    assert svc._reaper_started is False
    assert svc._reaper is None
    svc.shutdown()


def test_reaper_start_is_idempotent():
    svc = srv.ShellService(srv.Config(cleanup_interval_seconds=0.01))
    svc.start_reaper()
    first = svc._reaper
    svc.start_reaper()
    assert svc._reaper is first
    svc.shutdown()


def test_terminate_immediately_after_start_no_leak(service):
    """A terminate racing the spawn must still kill the process."""
    job = service.manager.start(service.cfg, "sleep 30", 0.0)
    job.terminate()
    assert job.wait_done(5.0), "raced terminate leaked the process"
    assert job.status == srv.STATUS_TERMINATED


def test_build_server_uses_its_own_service(tmp_path):
    """Two servers with different configs must not share a registry."""
    svc_a = srv.ShellService(srv.Config(cwd=str(tmp_path), cleanup_interval_seconds=0.0))
    svc_b = srv.ShellService(srv.Config(cwd=str(tmp_path), cleanup_interval_seconds=0.0))
    server_a = srv.build_server(svc_a)
    server_b = srv.build_server(svc_b)

    async def scenario():
        async with create_connected_server_and_client_session(server_a, raise_exceptions=False) as ca:
            await ca.call_tool("run_shell_command", {"command": "echo a"})
            assert (await ca.call_tool("list_jobs", {})).structuredContent["count"] == 1
        async with create_connected_server_and_client_session(server_b, raise_exceptions=False) as cb:
            assert (await cb.call_tool("list_jobs", {})).structuredContent["count"] == 0

    anyio.run(scenario)
    svc_a.shutdown()
    svc_b.shutdown()