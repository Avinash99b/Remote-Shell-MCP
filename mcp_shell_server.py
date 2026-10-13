#!/usr/bin/env python3
"""
mcp_shell_server.py — Standalone MCP server (Streamable HTTP transport) that
gives a connected agent the ability to execute shell commands on this host.

WHY THIS MATTERS (read before running):
  This server lets ANY client that can reach it (and pass the auth check)
  run arbitrary shell commands as the user running this process. That is
  extremely powerful and extremely dangerous if exposed beyond a trusted
  network. Defaults below are deliberately conservative:
    - Binds to 127.0.0.1 (localhost) unless you pass --host.
    - Requires a bearer token unless you pass --no-auth.
  If you expose this on a network (0.0.0.0, a tunnel, a public IP), put it
  behind a VPN/firewall and keep auth enabled. Don't run this as root.

ASYNC EXECUTION MODEL:
  Shell commands are *never* executed inline on the MCP request path. Every
  command is launched as a background job owned by a process-wide
  ``JobManager`` (independent of any MCP session). The tool handler then waits
  for at most ``wait_seconds`` (default/hard cap: 60s):

    * Commands that finish quickly (or within the wait window) return their
      stdout/stderr/exit code *as if* they were synchronous.
    * Commands still running when the wait window expires return a ``job_id``
      plus partial output/status. The process keeps running in the background
      and can be queried or terminated later.

  The handler only ever awaits a non-blocking event, and subprocess I/O is
  drained by dedicated threads, so the MCP server stays responsive no matter
  how long a command runs, and jobs survive MCP client disconnects / session
  teardown.

Requires:
    pip install "mcp[cli]"

Run:
    python mcp_shell_server.py                       # localhost:8765, random token printed
    python mcp_shell_server.py --token mysecret       # fixed token
    python mcp_shell_server.py --host 0.0.0.0 --port 8765 --token mysecret
    python mcp_shell_server.py --no-auth              # NOT recommended
    python mcp_shell_server.py --allowed-host shell-mcp-proxy.example.com   # behind a proxy/tunnel

The agent connects to:  http://<host>:<port>/mcp
with header:            Authorization: Bearer <token>   (unless --no-auth)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import codecs
import hashlib
import logging
import logging.handlers
import math
import os
import queue
import secrets
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import anyio

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# --------------------------------------------------------------------------
# Logging — every incoming tool call is logged to the console.
# --------------------------------------------------------------------------
#
# Log records are handed to a background thread through a bounded queue instead of
# being written to stderr directly. Writing straight to stderr from the event loop
# means that if whoever launched this process never reads the pipe (for example
# ``subprocess.Popen(..., stderr=PIPE)`` without draining it), the OS pipe buffer
# (~64 KB) fills up, the write blocks, and the whole server freezes. With the queue
# only the logging thread can block, and once the queue is full new records are
# dropped, so the server keeps serving requests no matter what.
class _DropOnFullQueueHandler(logging.handlers.QueueHandler):
    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass  # a stalled log consumer must never block the event loop


_LOG_QUEUE_SIZE = 10_000
_log_queue: queue.Queue = queue.Queue(maxsize=_LOG_QUEUE_SIZE)
_log_stream_handler = logging.StreamHandler()
_log_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
_log_listener = logging.handlers.QueueListener(_log_queue, _log_stream_handler)
_log_listener.start()  # daemon thread
logging.basicConfig(level=logging.INFO, handlers=[_DropOnFullQueueHandler(_log_queue)])
log = logging.getLogger("mcp-shell-server")

# Commands can be huge (whole scripts); log only a prefix so logs stay small.
_LOG_COMMAND_MAX_CHARS = 300


def _clip(text: str, limit: int = _LOG_COMMAND_MAX_CHARS) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... [+{len(text) - limit} chars]"

# How long a tool call may block before it hands back a background job. This is
# a hard cap and intentionally matches the "60 second" requirement.
MAX_SYNC_WAIT_SECONDS = 60.0

# There is no total file-size limit. Files move as a series of chunks, one MCP call
# each, so no single message holds the whole file. These caps bound each *call*
# (memory per request), not the file. The real ceiling is the disk.
UPLOAD_CHUNK_MAX_BYTES = 8 * 1024 * 1024
DOWNLOAD_CHUNK_MAX_BYTES = 8 * 1024 * 1024
# An upload that receives no chunk for this long is aborted and its temp file removed.
UPLOAD_IDLE_TIMEOUT_SECONDS = 24 * 3600.0
# Largest HTTP request body accepted: one base64 chunk plus JSON framing, with headroom.
MCP_MAX_REQUEST_BODY_BYTES = 16 * 1024 * 1024
# Hard limit on any single git command run by git_push.
GIT_TIMEOUT_SECONDS = 300.0

# How often the async handler re-checks a job when the event-loop bridge is
# unavailable (non-asyncio backends). Only a fallback.
_POLL_INTERVAL_SECONDS = 0.02

# Status strings for a job.
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_TERMINATED = "terminated"

_TERMINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_FAILED, STATUS_TIMEOUT, STATUS_TERMINATED})


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8765
    token: str | None = None
    no_auth: bool = False
    # How long a request waits for completion before returning a job handle.
    wait_seconds: float = MAX_SYNC_WAIT_SECONDS
    # Hard per-command execution limit. 0 == unlimited (do not kill long jobs).
    max_runtime_seconds: float = 0.0
    max_output_chars: int = 20_000
    # Completed jobs stay queryable for this long before being reaped.
    job_retention_seconds: float = 900.0
    # Upper bound on retained *completed* jobs (running jobs are always kept).
    max_jobs: int = 256
    # After the top-level shell exits, allow its pipes this long to drain before
    # force-abandoning any orphaned grandchildren that still hold them open.
    orphan_drain_grace: float = 1.0
    # Background retention-reaper interval.
    cleanup_interval_seconds: float = 60.0
    cwd: str = field(default_factory=os.getcwd)
    allowed_hosts: tuple = ()


def _positive_float(value: Any, default: float) -> float:
    """Coerce ``value`` to a finite, non-negative float or return ``default``."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f) or f < 0:
        return default
    return f


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, localhost only)")
    p.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    p.add_argument("--token", default=None, help="Bearer token required in Authorization header")
    p.add_argument("--no-auth", action="store_true", help="Disable auth (NOT recommended)")
    p.add_argument(
        "--timeout",
        type=float,
        default=MAX_SYNC_WAIT_SECONDS,
        help=(
            "Seconds a command may run before the request returns a background "
            f"job handle instead of waiting (default: {MAX_SYNC_WAIT_SECONDS:g}, hard cap)."
        ),
    )
    p.add_argument(
        "--max-runtime",
        type=float,
        default=0.0,
        help="Hard execution limit in seconds; the process is killed when exceeded. 0 = unlimited (default: 0)",
    )
    p.add_argument(
        "--job-retention",
        type=float,
        default=900.0,
        help="Seconds a completed job remains queryable before cleanup (default: 900)",
    )
    p.add_argument("--max-jobs", type=int, default=256, help="Maximum completed jobs retained (default: 256)")
    p.add_argument("--cwd", default=os.getcwd(), help="Working directory for executed commands")
    p.add_argument("--allowed-host", action="append", default=[],
                   help="Extra Host header to accept when behind a proxy/tunnel, e.g. shell-mcp-proxy.example.com (repeatable)")
    a = p.parse_args()

    cfg = Config(
        host=a.host,
        port=a.port,
        token=a.token,
        no_auth=a.no_auth,
        wait_seconds=_positive_float(a.timeout, MAX_SYNC_WAIT_SECONDS),
        max_runtime_seconds=_positive_float(a.max_runtime, 0.0),
        job_retention_seconds=_positive_float(a.job_retention, 900.0),
        max_jobs=a.max_jobs,
        cwd=a.cwd,
        allowed_hosts=tuple(a.allowed_host),
    )
    if not cfg.no_auth and not cfg.token:
        cfg.token = secrets.token_urlsafe(24)
    return cfg


# --------------------------------------------------------------------------
# Job registry — one Job per background shell command.
# --------------------------------------------------------------------------
class Job:
    """A single background shell command.

    Thread-safety: all mutable state (``status``, buffers, totals, waiters) is
    guarded by ``self.lock``. The subprocess is drained by dedicated daemon
    threads so a child writing lots of output can never deadlock on a full pipe,
    and so that nothing blocks the asyncio event loop.

    Status only becomes terminal inside :meth:`_maybe_finalize_locked`, which
    runs once the process has been reaped *and* both output pipes have been
    drained. This keeps ``running`` truthful and prevents half-collected
    snapshots from looking "completed".
    """

    _READ_CHUNK = 65536
    _SELECT_TIMEOUT = 0.2

    def __init__(self, job_id: str, command: str, max_runtime: float, max_output_chars: int, drain_grace: float) -> None:
        self.job_id = job_id
        self.command = command
        self.max_runtime = max_runtime
        self.max_output_chars = max_output_chars
        self.drain_grace = max(0.0, drain_grace)

        self.lock = threading.RLock()
        self.done = threading.Event()
        self._stop_reads = threading.Event()
        self.status = STATUS_RUNNING
        self.exit_code: int | None = None
        self.error: str | None = None
        self.pid: int | None = None

        self.started_at: float = time.monotonic()
        self.started_wall: float = time.time()
        self.ended_at: float | None = None

        # Output is decoded incrementally so truncation never splits a UTF-8
        # character; ``_kept`` counts characters, ``_total`` counts bytes read.
        self._stdout_parts: list[str] = []
        self._stderr_parts: list[str] = []
        self._stdout_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._stdout_total = 0
        self._stderr_total = 0
        self._stdout_kept = 0
        self._stderr_kept = 0

        self._readers_left = 0
        self._reaped = False
        self._terminate_requested = False
        self._timeout_requested = False
        self._waiters: list[tuple[asyncio.AbstractEventLoop, anyio.Event]] = []
        self._timer: threading.Timer | None = None
        self._drain_timer: threading.Timer | None = None

        self.process: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []

    # -- lifecycle -----------------------------------------------------------
    def start(self, cwd: str) -> None:
        """Launch the subprocess and its I/O/wait threads. Never raises."""
        try:
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as e:  # noqa: BLE001 - subprocess launch failure
            with self.lock:
                self.error = f"[server error] failed to start command: {e}"
                self.exit_code = None
                self._reaped = True
                self._readers_left = 0
                self._stop_reads.set()
                self._maybe_finalize_locked()
            return

        with self.lock:
            self.pid = self.process.pid
            self._readers_left = 2
            # Close the race where terminate()/shutdown() was called while the
            # process was still being spawned (pid/process were not yet set).
            killed_early = self._terminate_requested or self._timeout_requested

        if killed_early:
            self._kill()

        self._threads = [
            threading.Thread(target=self._pump, args=(self.process.stdout, True), daemon=True, name=f"jobs-{self.job_id[:8]}-out"),
            threading.Thread(target=self._pump, args=(self.process.stderr, False), daemon=True, name=f"jobs-{self.job_id[:8]}-err"),
            threading.Thread(target=self._reap, daemon=True, name=f"jobs-{self.job_id[:8]}-wait"),
        ]
        for t in self._threads:
            t.start()

        if self.max_runtime and self.max_runtime > 0:
            self._timer = threading.Timer(self.max_runtime, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()

        if killed_early:
            # Ensure the timer cannot later re-kill or log a bogus timeout.
            with self.lock:
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None

        log.info("JOB START   | id=%s | pid=%s | command=%s", self.job_id, self.pid, _clip(self.command))

    def _pump(self, stream, is_stdout: bool) -> None:
        """Drain one pipe. Keeps a bounded character prefix but reads to EOF.

        Uses ``select`` with a short timeout so the thread can also notice
        ``_stop_reads`` (set once the process is reaped and orphaned
        grandchildren would otherwise hold the pipe open forever).
        """
        try:
            fd = stream.fileno()
            while not self._stop_reads.is_set():
                try:
                    ready, _, _ = select.select([fd], [], [], self._SELECT_TIMEOUT)
                except (OSError, ValueError):
                    break
                if not ready:
                    continue
                try:
                    chunk = os.read(fd, self._READ_CHUNK)
                except (OSError, ValueError):
                    break
                if not chunk:
                    break
                self._store_output(chunk, is_stdout)
            # Flush any bytes the incremental decoder is holding (e.g. a
            # truncated trailing multibyte character).
            self._flush_decoder(is_stdout)
        except Exception as e:  # noqa: BLE001 - pipe read failure must not crash
            with self.lock:
                if self.error is None:
                    self.error = f"[server error] output read failed: {e}"
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
            with self.lock:
                self._readers_left -= 1
                self._maybe_finalize_locked()

    def _store_output(self, chunk: bytes, is_stdout: bool) -> None:
        decoder = self._stdout_decoder if is_stdout else self._stderr_decoder
        # Decode first, then account in *characters* so the byte size of a
        # multi-byte character never causes a false "truncated" report.
        text = decoder.decode(chunk)
        if not text:
            return
        with self.lock:
            if is_stdout:
                self._stdout_total += len(text)
                remaining = self.max_output_chars - self._stdout_kept
                if remaining > 0:
                    kept = text[:remaining]
                    self._stdout_parts.append(kept)
                    self._stdout_kept += len(kept)
            else:
                self._stderr_total += len(text)
                remaining = self.max_output_chars - self._stderr_kept
                if remaining > 0:
                    kept = text[:remaining]
                    self._stderr_parts.append(kept)
                    self._stderr_kept += len(kept)

    def _flush_decoder(self, is_stdout: bool) -> None:
        try:
            tail = self._stdout_decoder.decode(b"", final=True) if is_stdout else self._stderr_decoder.decode(b"", final=True)
        except Exception:  # noqa: BLE001
            tail = ""
        if not tail:
            return
        with self.lock:
            if is_stdout:
                self._stdout_total += len(tail)
                remaining = self.max_output_chars - self._stdout_kept
                if remaining > 0:
                    kept = tail[:remaining]
                    self._stdout_parts.append(kept)
                    self._stdout_kept += len(kept)
            else:
                self._stderr_total += len(tail)
                remaining = self.max_output_chars - self._stderr_kept
                if remaining > 0:
                    kept = tail[:remaining]
                    self._stderr_parts.append(kept)
                    self._stderr_kept += len(kept)

    def _reap(self) -> None:
        """Wait for the process to exit, capturing its exit code."""
        rc: int | None
        try:
            rc = self.process.wait()
        except Exception as e:  # noqa: BLE001
            rc = None
            with self.lock:
                self.error = f"[server error] wait failed: {e}"
        with self.lock:
            self.exit_code = rc
            self._reaped = True
            self._maybe_finalize_locked()
            needs_drain_guard = not self.done.is_set()
        if needs_drain_guard:
            # The top-level shell is gone but a pipe is still open (an orphaned
            # grandchild inherited it). Bound how long we wait for EOF.
            timer = threading.Timer(self.drain_grace, self._stop_reads.set)
            timer.daemon = True
            with self.lock:
                self._drain_timer = timer
            timer.start()

    def _maybe_finalize_locked(self) -> None:
        """Transition to a terminal status once the process and readers finish.

        ``done`` fires only after the process has been reaped *and* both output
        pipes have hit EOF, so the snapshot is complete when callers observe it.
        """
        if self.done.is_set():
            return
        if not self._reaped or self._readers_left > 0:
            return

        if self._timeout_requested:
            self.status = STATUS_TIMEOUT
            self.error = f"[process killed after {self.max_runtime:g}s max runtime]"
        elif self._terminate_requested:
            self.status = STATUS_TERMINATED
            self.error = "[process terminated by request]"
        elif self.exit_code is None or self.exit_code != 0:
            self.status = STATUS_FAILED
        else:
            self.status = STATUS_COMPLETED

        if self.ended_at is None:
            self.ended_at = time.monotonic()

        for timer_attr in ("_timer", "_drain_timer"):
            timer = getattr(self, timer_attr)
            if timer is not None:
                timer.cancel()
                setattr(self, timer_attr, None)

        waiters = self._waiters
        self._waiters = []
        self.done.set()
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # Event loop already closed (server shutting down) — ignore.
                pass

    def _on_timeout(self) -> None:
        with self.lock:
            if self.done.is_set() or self._timeout_requested:
                return
            self._timeout_requested = True
        log.warning("JOB TIMEOUT | id=%s | pid=%s", self.job_id, self.pid)
        self._kill()

    def terminate(self) -> None:
        """Request termination of the process group (non-blocking, idempotent).

        The process is signalled immediately but the status stays ``running``
        until it is actually reaped, so callers never observe a "terminated"
        job that is still alive. Await ``done`` (or use ``terminate_job``) to
        observe the terminal state.
        """
        with self.lock:
            if self.done.is_set() or self._terminate_requested:
                return
            self._terminate_requested = True
        log.warning("JOB TERMINATE | id=%s | pid=%s", self.job_id, self.pid)
        self._kill()

    def wait_done(self, timeout: float) -> bool:
        """Block (for non-async callers) until the job is terminal."""
        return self.done.wait(timeout=max(0.0, timeout))

    def _kill(self) -> None:
        """SIGKILL the whole process group.

        ``start_new_session=True`` makes the direct child a session/group leader,
        so its pid *is* the pgid. We always signal the group — even if the
        top-level shell has already exited — so surviving subshells/background
        children (which otherwise hold the output pipes open and leak threads)
        are reaped too.
        """
        proc = self.process
        pid = self.pid
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
                return
            except ProcessLookupError:
                # Group already gone; fall through to a direct kill attempt.
                pass
            except Exception:  # noqa: BLE001 - fall back to direct kill
                pass
        if proc is not None:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass

    # -- waiting -------------------------------------------------------------
    @property
    def running(self) -> bool:
        return not self.done.is_set()

    def add_waiter(self, loop: asyncio.AbstractEventLoop, event: anyio.Event) -> None:
        with self.lock:
            if self.done.is_set():
                try:
                    loop.call_soon_threadsafe(event.set)
                except RuntimeError:
                    pass
                return
            self._waiters.append((loop, event))

    def remove_waiter(self, event: anyio.Event) -> None:
        with self.lock:
            self._waiters = [(l, e) for (l, e) in self._waiters if e is not event]

    # -- inspection ----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            stdout = "".join(self._stdout_parts)
            stderr = "".join(self._stderr_parts)
            stdout_truncated = self._stdout_total > self._stdout_kept
            stderr_truncated = self._stderr_total > self._stderr_kept
            ended = self.ended_at if self.ended_at is not None else time.monotonic()
            duration = round(ended - self.started_at, 3)
            status = self.status
            exit_code = self.exit_code
            error = self.error
            pid = self.pid
            started_wall = self.started_wall

        if stdout_truncated:
            stdout += "\n...[stdout truncated]"
        if stderr_truncated:
            stderr += "\n...[stderr truncated]"
        if error:
            stderr = (stderr + "\n" + error) if stderr else error

        return {
            "job_id": self.job_id,
            "command": self.command,
            "status": status,
            "running": status == STATUS_RUNNING,
            "completed": status in _TERMINAL_STATUSES,
            "success": status == STATUS_COMPLETED,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": stdout_truncated or stderr_truncated,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "duration_seconds": duration,
            "pid": pid,
            "started_at": started_wall,
            "timed_out": status == STATUS_TIMEOUT,
        }

    def summary(self) -> dict[str, Any]:
        with self.lock:
            ended = self.ended_at if self.ended_at is not None else time.monotonic()
            return {
                "job_id": self.job_id,
                "command": self.command,
                "status": self.status,
                "running": self.status == STATUS_RUNNING,
                "exit_code": self.exit_code,
                "pid": self.pid,
                "duration_seconds": round(ended - self.started_at, 3),
            }

    def ended_age(self) -> float:
        with self.lock:
            if self.ended_at is None:
                return 0.0
            return time.monotonic() - self.ended_at


class JobManager:
    """Registry of active and completed jobs, shared across MCP sessions."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    # -- creation ------------------------------------------------------------
    def start(self, cfg: Config, command: str, max_runtime: float) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(job_id, command, max_runtime, cfg.max_output_chars, cfg.orphan_drain_grace)
        # Launch first, then publish: nothing can terminate a job before it is
        # visible, and a job is never visible before it has a pid.
        try:
            job.start(cfg.cwd)
        except Exception as e:  # noqa: BLE001 - defensive; Job.start already guards
            log.exception("Unexpected error starting job %s: %s", job_id, e)
        with self._lock:
            self._reap_locked(cfg)
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._enforce_capacity_locked(cfg)
        return job

    # -- lookup --------------------------------------------------------------
    def get(self, cfg: Config, job_id: str) -> Job | None:
        with self._lock:
            self._reap_locked(cfg)
            return self._jobs.get(job_id)

    def list(self, cfg: Config) -> list[Job]:
        with self._lock:
            self._reap_locked(cfg)
            return [self._jobs[jid] for jid in self._order if jid in self._jobs]

    # -- cleanup -------------------------------------------------------------
    def _reap_locked(self, cfg: Config) -> int:
        """Drop completed jobs older than the retention window."""
        retention = cfg.job_retention_seconds
        if retention is None or retention < 0:
            return 0
        removed = 0
        for jid in list(self._jobs.keys()):
            job = self._jobs[jid]
            if job.done.is_set() and job.ended_age() > retention:
                del self._jobs[jid]
                removed += 1
        if removed:
            self._order = [jid for jid in self._order if jid in self._jobs]
        return removed

    def _enforce_capacity_locked(self, cfg: Config) -> None:
        """Bound the retained *completed* job history.

        Only finished jobs (``done`` set) are eligible for eviction; a running
        job is never dropped, because that would make its process unreachable.
        """
        max_jobs = cfg.max_jobs
        if not max_jobs or max_jobs <= 0:
            return
        self._order = [jid for jid in self._order if jid in self._jobs]
        terminal = [jid for jid in self._order if self._jobs[jid].done.is_set()]
        excess = len(terminal) - max_jobs
        if excess <= 0:
            return
        for jid in terminal[:excess]:
            self._jobs.pop(jid, None)
        self._order = [jid for jid in self._order if jid in self._jobs]

    def cleanup(self, cfg: Config, older_than_seconds: float | None = None) -> int:
        """Force cleanup. Returns number of jobs removed."""
        with self._lock:
            if older_than_seconds is None:
                return self._reap_locked(cfg)
            age = _positive_float(older_than_seconds, 0.0)
            removed = 0
            for jid in list(self._jobs.keys()):
                job = self._jobs[jid]
                if job.done.is_set() and job.ended_age() > age:
                    del self._jobs[jid]
                    removed += 1
            if removed:
                self._order = [jid for jid in self._order if jid in self._jobs]
            return removed

    def shutdown(self, wait_seconds: float = 3.0) -> None:
        """Terminate every running job and give it a bounded time to die."""
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.running]
        for job in jobs:
            job.terminate()
        deadline = time.monotonic() + max(0.0, wait_seconds)
        for job in jobs:
            job.wait_done(max(0.0, deadline - time.monotonic()))
        log.info("JobManager shutdown: terminated %d running job(s)", len(jobs))


# --------------------------------------------------------------------------
# Async waiting helpers
# --------------------------------------------------------------------------
def _resolve_wait_seconds(cfg: Config, requested: float | None) -> float:
    """Clamp a caller-supplied wait to [0, MAX_SYNC_WAIT_SECONDS]; reject NaN."""
    if requested is None:
        value = cfg.wait_seconds
    else:
        try:
            value = float(requested)
        except (TypeError, ValueError):
            value = cfg.wait_seconds
    if math.isnan(value) or math.isinf(value) or value < 0:
        value = 0.0
    return min(value, MAX_SYNC_WAIT_SECONDS)


def _resolve_max_runtime(cfg: Config, timeout_seconds: float | None) -> float:
    if timeout_seconds is None:
        value = cfg.max_runtime_seconds
    else:
        value = _positive_float(timeout_seconds, 0.0)
    return value if value and value > 0 else 0.0


async def _await_job(job: Job, wait_seconds: float) -> bool:
    """Wait (without blocking the event loop) up to ``wait_seconds``.

    Returns True if the job reached a terminal state, False on timeout.
    """
    if job.done.is_set() or wait_seconds <= 0:
        return job.done.is_set()

    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - non-asyncio backend / no loop
        loop = None

    if loop is not None:
        event = anyio.Event()
        job.add_waiter(loop, event)
        try:
            with anyio.move_on_after(wait_seconds):
                await event.wait()
        finally:
            job.remove_waiter(event)
        return job.done.is_set()

    # Fallback for non-asyncio event loops: cooperative polling.
    with anyio.move_on_after(wait_seconds):  # pragma: no cover
        while not job.done.is_set():
            await anyio.sleep(_POLL_INTERVAL_SECONDS)
    return job.done.is_set()


# --------------------------------------------------------------------------
# Shell service — owns configuration, the job registry and the tool methods.
# --------------------------------------------------------------------------
@dataclass
class _UploadSession:
    """State of one chunked upload. Bytes go to a temp file in the target directory."""

    target: str
    tmp: str
    overwrite: bool
    executable: bool
    received: int = 0
    sha: Any = field(default_factory=hashlib.sha256)
    last_active: float = field(default_factory=time.monotonic)
    closed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def _decode_chunk(content_base64: str) -> bytes:
    try:
        return base64.b64decode(content_base64, validate=True)
    except (ValueError, TypeError) as e:
        raise ValueError(f"content_base64 is not valid base64: {e}") from e


class ShellService:
    """Binds one :class:`Config` and :class:`JobManager` to the MCP tool methods.

    Tool methods read ``self.cfg``/``self.manager`` instead of module globals, so
    multiple independently-configured servers can coexist (tests, embedders).
    A lightweight daemon thread periodically reaps expired completed jobs even
    when no requests are arriving.
    """

    def __init__(self, cfg: Config, manager: JobManager | None = None) -> None:
        self.cfg = cfg
        self.manager = manager if manager is not None else JobManager()
        self._stop_cleanup = threading.Event()
        self._reaper: threading.Thread | None = None
        self._reaper_started = False
        self._uploads: dict[str, _UploadSession] = {}
        self._uploads_lock = threading.Lock()

    def start_reaper(self) -> None:
        """Start the periodic cleanup thread (idempotent).

        Not started at construction time so merely importing the module or
        building a throwaway service never spawns background threads.
        """
        interval = self.cfg.cleanup_interval_seconds
        if self._reaper_started or not interval or interval <= 0:
            return
        self._reaper_started = True
        self._reaper = threading.Thread(target=self._cleanup_loop, args=(interval,), daemon=True, name="jobs-reaper")
        self._reaper.start()

    def _cleanup_loop(self, interval: float) -> None:
        while not self._stop_cleanup.wait(interval):
            try:
                removed = self.manager.cleanup(self.cfg)
                self.sweep_uploads()
                if removed:
                    log.info("CLEANUP     | reaped %d expired job(s)", removed)
            except Exception:  # noqa: BLE001 - reaper must never die
                log.exception("Job cleanup reaper error")

    # -- tools ---------------------------------------------------------------
    async def run_shell_command(
        self,
        command: str,
        timeout_seconds: float | None = None,
        wait_seconds: float | None = None,
    ) -> dict[str, Any]:
        self.validate_command(command)
        wait = _resolve_wait_seconds(self.cfg, wait_seconds)
        max_runtime = _resolve_max_runtime(self.cfg, timeout_seconds)

        log.info(
            "EXEC REQUEST | cwd=%s | wait=%ss | max_runtime=%ss | command=%s",
            self.cfg.cwd, wait, max_runtime or "unlimited", _clip(command),
        )

        job = self.manager.start(self.cfg, command, max_runtime)
        finished = await _await_job(job, wait)
        result = job.snapshot()
        if not finished:
            result["note"] = self._running_note(job)

        log.info(
            "EXEC RESULT  | id=%s | status=%s | exit_code=%s | duration=%ss | stdout_len=%d | stderr_len=%d",
            job.job_id, result["status"], result["exit_code"], result["duration_seconds"],
            len(result["stdout"]), len(result["stderr"]),
        )
        return result

    async def get_job(self, job_id: str, wait_seconds: float = 0.0) -> dict[str, Any]:
        job = self.manager.get(self.cfg, job_id)
        if job is None:
            return {"found": False, "error": f"unknown job_id: {job_id}", "job_id": job_id}

        wait = _resolve_wait_seconds(self.cfg, wait_seconds)
        if job.running and wait > 0:
            await _await_job(job, wait)
        result = job.snapshot()
        result["found"] = True
        return result

    async def list_jobs(self) -> dict[str, Any]:
        jobs = self.manager.list(self.cfg)
        summaries = [j.summary() for j in jobs]
        return {
            "jobs": summaries,
            "count": len(summaries),
            "running": sum(1 for s in summaries if s["running"]),
        }

    async def terminate_job(self, job_id: str, grace_seconds: float = 2.0) -> dict[str, Any]:
        job = self.manager.get(self.cfg, job_id)
        if job is None:
            return {"found": False, "error": f"unknown job_id: {job_id}", "job_id": job_id}
        if not job.done.is_set():
            job.terminate()
            grace = _positive_float(grace_seconds, 2.0)
            if grace > 0:
                await _await_job(job, grace)
        result = job.snapshot()
        result["found"] = True
        return result

    async def cleanup_jobs(self, older_than_seconds: float | None = None) -> dict[str, Any]:
        removed = self.manager.cleanup(self.cfg, older_than_seconds)
        remaining = len(self.manager.list(self.cfg))
        return {"removed": removed, "count": remaining}

    def shutdown(self) -> None:
        self._stop_cleanup.set()
        self.manager.shutdown()

    # -- file transfer -------------------------------------------------------
    def _resolve_path(self, path: str) -> str:
        """Resolve a user-supplied path against the server's working directory."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")
        return os.path.abspath(os.path.join(self.cfg.cwd, os.path.expanduser(path)))

    # Single-call upload: the whole file in one message. Fine for small files; use the
    # chunked tools (begin_upload / upload_chunk / finish_upload) for anything large.
    async def upload_file(
        self,
        path: str,
        content_base64: str,
        overwrite: bool = False,
        executable: bool = False,
    ) -> dict[str, Any]:
        target = self._resolve_path(path)
        return await asyncio.to_thread(self._write_file, target, content_base64, overwrite, executable)

    @staticmethod
    def _write_file(target: str, content_base64: str, overwrite: bool, executable: bool) -> dict[str, Any]:
        data = _decode_chunk(content_base64)
        if os.path.isdir(target):
            raise IsADirectoryError(f"{target} is a directory")
        existed = os.path.exists(target)
        if existed and not overwrite:
            raise FileExistsError(f"{target} already exists; pass overwrite=true to replace it")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = f"{target}.part-{uuid.uuid4().hex[:8]}"
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            if executable:
                os.chmod(tmp, 0o755)
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        log.info("UPLOAD      | path=%s | bytes=%d | overwritten=%s", target, len(data), existed)
        return {
            "path": target,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "overwritten": existed,
        }

    # Chunked upload: begin -> upload_chunk (repeat) -> finish. Any size.
    async def begin_upload(self, path: str, overwrite: bool = False, executable: bool = False) -> dict[str, Any]:
        target = self._resolve_path(path)
        return await asyncio.to_thread(self._begin_upload, target, overwrite, executable)

    def _begin_upload(self, target: str, overwrite: bool, executable: bool) -> dict[str, Any]:
        if os.path.isdir(target):
            raise IsADirectoryError(f"{target} is a directory")
        if os.path.exists(target) and not overwrite:
            raise FileExistsError(f"{target} already exists; pass overwrite=true to replace it")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        upload_id = uuid.uuid4().hex
        tmp = f"{target}.part-{upload_id[:8]}"
        open(tmp, "wb").close()
        with self._uploads_lock:
            self._uploads[upload_id] = _UploadSession(
                target=target, tmp=tmp, overwrite=overwrite, executable=executable
            )
        log.info("UPLOAD BEGIN| id=%s | path=%s", upload_id, target)
        return {"upload_id": upload_id, "path": target, "chunk_max_bytes": UPLOAD_CHUNK_MAX_BYTES}

    def _get_upload(self, upload_id: str) -> _UploadSession:
        with self._uploads_lock:
            session = self._uploads.get(upload_id)
        if session is None:
            raise ValueError(f"unknown upload_id: {upload_id} (already finished, aborted, or expired)")
        return session

    async def upload_chunk(self, upload_id: str, offset: int, content_base64: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._upload_chunk, upload_id, offset, content_base64)

    def _upload_chunk(self, upload_id: str, offset: int, content_base64: str) -> dict[str, Any]:
        session = self._get_upload(upload_id)
        data = _decode_chunk(content_base64)
        if len(data) > UPLOAD_CHUNK_MAX_BYTES:
            raise ValueError(f"chunk is {len(data)} bytes; the per-call maximum is {UPLOAD_CHUNK_MAX_BYTES}")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        with session.lock:
            if session.closed:
                raise ValueError(f"upload {upload_id} is already finished or aborted")
            if offset > session.received:
                raise ValueError(
                    f"gap in upload: expected offset {session.received}, got {offset}. "
                    "Send chunks in order (or resend from the expected offset)."
                )
            end = offset + len(data)
            # Bytes below `received` were already accepted. A retry must match them exactly.
            overlap_end = min(end, session.received)
            if overlap_end > offset:
                with open(session.tmp, "rb") as fh:
                    fh.seek(offset)
                    existing = fh.read(overlap_end - offset)
                if existing != data[: overlap_end - offset]:
                    raise ValueError(f"chunk at offset {offset} conflicts with data already received")
            if end > session.received:
                new = data[session.received - offset:]
                with open(session.tmp, "ab") as fh:
                    fh.write(new)
                session.sha.update(new)
                session.received = end
            session.last_active = time.monotonic()
            return {"upload_id": upload_id, "received_bytes": session.received}

    async def finish_upload(self, upload_id: str, expected_sha256: str | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._finish_upload, upload_id, expected_sha256)

    def _finish_upload(self, upload_id: str, expected_sha256: str | None) -> dict[str, Any]:
        session = self._get_upload(upload_id)
        with session.lock:
            if session.closed:
                raise ValueError(f"upload {upload_id} is already finished or aborted")
            digest = session.sha.hexdigest()
            if expected_sha256 and expected_sha256.lower() != digest:
                raise ValueError(
                    f"sha256 mismatch: expected {expected_sha256}, received {digest}. "
                    "The upload is still open; resend or abort it."
                )
            existed = os.path.exists(session.target)
            if existed and not session.overwrite:
                raise FileExistsError(f"{session.target} already exists; the upload is still open and can be aborted")
            with open(session.tmp, "rb+") as fh:
                fh.flush()
                os.fsync(fh.fileno())
            if session.executable:
                os.chmod(session.tmp, 0o755)
            os.replace(session.tmp, session.target)
            session.closed = True
            size = session.received
        with self._uploads_lock:
            self._uploads.pop(upload_id, None)
        log.info("UPLOAD DONE | id=%s | path=%s | bytes=%d | overwritten=%s", upload_id, session.target, size, existed)
        return {
            "path": session.target,
            "bytes": size,
            "sha256": digest,
            "overwritten": existed,
        }

    async def abort_upload(self, upload_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._abort_upload, upload_id)

    def _abort_upload(self, upload_id: str) -> dict[str, Any]:
        with self._uploads_lock:
            session = self._uploads.pop(upload_id, None)
        if session is None:
            return {"upload_id": upload_id, "aborted": False, "note": "unknown or already finished"}
        with session.lock:
            session.closed = True
            if os.path.exists(session.tmp):
                os.remove(session.tmp)
        log.info("UPLOAD ABORT| id=%s | path=%s", upload_id, session.target)
        return {"upload_id": upload_id, "aborted": True}

    def sweep_uploads(self, max_idle: float = UPLOAD_IDLE_TIMEOUT_SECONDS) -> int:
        """Abort uploads that have gone idle, so abandoned temp files do not pile up."""
        now = time.monotonic()
        with self._uploads_lock:
            stale = [uid for uid, s in self._uploads.items() if now - s.last_active > max_idle]
        for uid in stale:
            self._abort_upload(uid)
        return len(stale)

    # Chunked download: read `length` bytes starting at `offset`. Loop until eof.
    async def download_file(self, path: str, offset: int = 0, length: int | None = None) -> dict[str, Any]:
        target = self._resolve_path(path)
        if length is None:
            length = DOWNLOAD_CHUNK_MAX_BYTES
        if not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer")
        if length > DOWNLOAD_CHUNK_MAX_BYTES:
            raise ValueError(f"length {length} exceeds the per-call maximum of {DOWNLOAD_CHUNK_MAX_BYTES}; read in more calls")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        return await asyncio.to_thread(self._read_chunk, target, offset, length)

    @staticmethod
    def _read_chunk(target: str, offset: int, length: int) -> dict[str, Any]:
        if not os.path.isfile(target):
            raise FileNotFoundError(f"no such file: {target}")
        with open(target, "rb") as fh:
            total = os.fstat(fh.fileno()).st_size
            if offset > total:
                raise ValueError(f"offset {offset} is past the end of the file ({total} bytes)")
            fh.seek(offset)
            data = fh.read(length)
        return {
            "path": target,
            "offset": offset,
            "bytes": len(data),
            "total_bytes": total,
            "eof": offset + len(data) >= total,
            "content_base64": base64.b64encode(data).decode("ascii"),
        }

    async def file_sha256(self, path: str) -> dict[str, Any]:
        target = self._resolve_path(path)
        return await asyncio.to_thread(self._file_sha256, target)

    @staticmethod
    def _file_sha256(target: str) -> dict[str, Any]:
        if not os.path.isfile(target):
            raise FileNotFoundError(f"no such file: {target}")
        h = hashlib.sha256()
        size = 0
        with open(target, "rb") as fh:
            while block := fh.read(1024 * 1024):
                h.update(block)
                size += len(block)
        return {"path": target, "bytes": size, "sha256": h.hexdigest()}

    # -- git -----------------------------------------------------------------
    async def git_push(
        self,
        repo_path: str,
        commit_message: str | None = None,
        add_paths: list[str] | None = None,
        branch: str | None = None,
        remote: str = "origin",
    ) -> dict[str, Any]:
        repo = self._resolve_path(repo_path)
        return await asyncio.to_thread(self._git_push, repo, commit_message, add_paths, branch, remote)

    @staticmethod
    def _run_git(repo: str, *args: str) -> tuple[int, str, str]:
        # GIT_TERMINAL_PROMPT=0 makes git fail instead of hanging on a credentials prompt.
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        try:
            proc = subprocess.run(
                ["git", "-C", repo, *args],
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_SECONDS,
                stdin=subprocess.DEVNULL,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"git {' '.join(args)} timed out after {GIT_TIMEOUT_SECONDS:g}s") from e
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    @classmethod
    def _git_push(
        cls,
        repo: str,
        commit_message: str | None,
        add_paths: list[str] | None,
        branch: str | None,
        remote: str,
    ) -> dict[str, Any]:
        rc, top, err = cls._run_git(repo, "rev-parse", "--show-toplevel")
        if rc != 0:
            raise ValueError(f"not a git repository: {repo} ({err})")
        repo = top

        if branch is None:
            # symbolic-ref works on an unborn branch (no commits yet), unlike rev-parse --abbrev-ref.
            rc, branch_out, _ = cls._run_git(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
            if rc != 0 or not branch_out:
                raise ValueError("repository is in detached HEAD state; pass branch explicitly")
            branch = branch_out

        # Stage either the requested paths or everything (.gitignore still applies).
        if add_paths:
            rc, _, err = cls._run_git(repo, "add", "--", *add_paths)
        else:
            rc, _, err = cls._run_git(repo, "add", "-A")
        if rc != 0:
            raise RuntimeError(f"git add failed: {err}")

        # `git diff --cached --quiet` exits 1 when there are staged changes, 0 when there are none.
        rc, _, err = cls._run_git(repo, "diff", "--cached", "--quiet")
        if rc not in (0, 1):
            raise RuntimeError(f"git diff failed: {err}")
        has_changes = rc == 1

        if has_changes:
            message = commit_message or "Update from remote shell MCP"
            rc, _, err = cls._run_git(repo, "commit", "-m", message)
            if rc != 0:
                raise RuntimeError(f"git commit failed: {err}")

        rc, head, _ = cls._run_git(repo, "rev-parse", "HEAD")
        commit_sha = head if rc == 0 else None

        rc, out, err = cls._run_git(repo, "push", "-u", remote, f"HEAD:refs/heads/{branch}")
        if rc != 0:
            raise RuntimeError(f"git push failed: {err or out}")

        log.info("GIT PUSH    | repo=%s | branch=%s | committed=%s | commit=%s", repo, branch, has_changes, commit_sha)
        return {
            "repo": repo,
            "branch": branch,
            "remote": remote,
            "committed": has_changes,
            "commit": commit_sha,
            "pushed": True,
            "push_output": err or out,
        }

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def validate_command(command: str) -> str:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        return command

    @staticmethod
    def _running_note(job: Job) -> str:
        return (
            f"Command is still running (status={job.status!r}). "
            f"Poll with get_job(job_id={job.job_id!r}) and terminate with "
            f"terminate_job(job_id={job.job_id!r})."
        )


# Human-readable descriptions attached to the bound tool methods.
run_shell_command_doc = """
Execute a shell command on the host system as a background job and return its result.

The command is always launched asynchronously and can never block the server.
If it finishes within the wait window the response looks synchronous; otherwise
a ``job_id`` is returned so the caller can poll ``get_job`` / terminate it.

Args:
    command: The full shell command to run (e.g. "ls -la /tmp").
    timeout_seconds: Optional hard execution limit; the process is killed when
        exceeded. Omit for no limit (long commands keep running in background).
    wait_seconds: Optional seconds to wait for completion before returning a
        background job handle. Capped at 60s.

Returns:
    A dict including: command, exit_code, stdout, stderr, truncated,
    duration_seconds, status, job_id, running, completed, timed_out.
"""

get_job_doc = """
Query a background job by id.

Args:
    job_id: The id returned by run_shell_command or list_jobs.
    wait_seconds: Optionally wait up to this many seconds for completion (capped at 60s).

Returns:
    The job snapshot (status, stdout, stderr, exit_code, running, ...), or
    {"error": ..., "found": False} if the id is unknown.
"""

list_jobs_doc = """
List known jobs (active and recently completed).

Returns:
    {"jobs": [ {job_id, command, status, running, exit_code, pid, duration_seconds}, ... ],
     "count": N, "running": N}
"""

terminate_job_doc = """
Terminate a running job's process group.

Args:
    job_id: The job to terminate.
    grace_seconds: How long to wait after SIGKILL before returning.

Returns:
    The job snapshot, or {"error": ..., "found": False} if unknown.
"""

cleanup_jobs_doc = """
Remove completed jobs from the registry.

Args:
    older_than_seconds: Only remove jobs that finished more than this long ago.
        Omit to use the configured retention window.

Returns:
    {"removed": N, "count": remaining_jobs}
"""

upload_file_doc = """
Upload a whole file in one call (base64-encoded). Good for small files.

For large files use begin_upload, then upload_chunk repeatedly, then finish_upload.

Args:
    path: Destination path. Relative paths resolve against the server's working directory.
    content_base64: The file's bytes, base64-encoded.
    overwrite: Replace the file if it already exists (default false).
    executable: Mark the file executable (mode 755).

Returns:
    {"path", "bytes", "sha256", "overwritten"}.
"""

begin_upload_doc = """
Start a chunked upload. Returns an upload_id to use with upload_chunk and finish_upload.

Args:
    path: Destination path. Relative paths resolve against the server's working directory.
    overwrite: Allow replacing an existing file when the upload finishes (default false).
    executable: Mark the file executable when it finishes (mode 755).

Returns:
    {"upload_id", "path", "chunk_max_bytes"}.
"""

upload_chunk_doc = """
Send one chunk of an open upload.

Chunks must be sent in order. Resending a chunk you already sent is safe and is checked
against the data already received. Use a chunk size up to chunk_max_bytes from begin_upload.

Args:
    upload_id: From begin_upload.
    offset: Byte offset of this chunk in the file (0 for the first chunk).
    content_base64: The chunk's bytes, base64-encoded.

Returns:
    {"upload_id", "received_bytes"}.
"""

finish_upload_doc = """
Complete a chunked upload: verify, fsync, and move it into place.

Args:
    upload_id: From begin_upload.
    expected_sha256: Optional sha256 of the whole file. If given, the upload is rejected on mismatch.

Returns:
    {"path", "bytes", "sha256", "overwritten"}.
"""

abort_upload_doc = """
Cancel an open upload and delete its temp file. Uploads idle for 24 hours are also aborted.

Args:
    upload_id: From begin_upload.
"""

download_file_doc = """
Read part of a file, base64-encoded. Read a large file by looping: start at offset 0 and
advance offset by the returned bytes until eof is true.

Args:
    path: File to read. Relative paths resolve against the server's working directory.
    offset: Byte offset to start from (default 0).
    length: Bytes to read in this call (default and maximum: DOWNLOAD_CHUNK_MAX_BYTES).

Returns:
    {"path", "offset", "bytes", "total_bytes", "eof", "content_base64"}.
"""

file_sha256_doc = """
Compute the sha256 of a file on the server, streaming it, so it works for any size.

Args:
    path: File to hash. Relative paths resolve against the server's working directory.

Returns:
    {"path", "bytes", "sha256"}.
"""

git_push_doc = """
Commit and push a git repository on the server to its remote (for example GitHub).

Stages changes, commits them if anything is staged, then pushes the current branch. Uses the
server's existing git credentials. Never prompts for a password, so it fails instead of hanging.

Args:
    repo_path: Path to the repository (a subdirectory of it is fine).
    commit_message: Commit message. Default: "Update from remote shell MCP". Ignored if there are no changes.
    add_paths: Specific paths to stage. Omit to stage everything (.gitignore still applies).
    branch: Branch to push to. Default: the current branch.
    remote: Remote name. Default: origin.

Returns:
    {"repo", "branch", "remote", "committed", "commit", "pushed", "push_output"}.
"""


# --------------------------------------------------------------------------
# Server construction
# --------------------------------------------------------------------------
def _transport_security(cfg: Config) -> TransportSecuritySettings:
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"]
        + [h for host in cfg.allowed_hosts for h in (host, f"{host}:*")],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
        + [o for host in cfg.allowed_hosts for o in (f"https://{host}", f"http://{host}")],
    )


def create_service(cfg: Config | None = None) -> ShellService:
    """Create a shell service (config + registry + reaper) for a server."""
    return ShellService(cfg if cfg is not None else Config())


def build_server(service: ShellService) -> FastMCP:
    """Create an MCP server exposing the shell + job-management tools."""
    server = FastMCP(
        name="shell-exec-server",
        instructions=(
            "Executes shell commands as non-blocking background jobs. "
            "run_shell_command returns quickly for short commands and a job_id for "
            "long-running ones; use get_job / list_jobs / terminate_job to manage jobs."
        ),
        host=service.cfg.host,
        port=service.cfg.port,
        transport_security=_transport_security(service.cfg),
        # The SDK default is 4 MiB per HTTP request body. A chunk of UPLOAD_CHUNK_MAX_BYTES
        # is about 11 MiB once base64-encoded, so raise the limit to fit one chunk plus JSON
        # framing. It stays finite, so one request still cannot exhaust memory.
        max_request_body_size=MCP_MAX_REQUEST_BODY_BYTES,
    )
    server.add_tool(service.run_shell_command, description=run_shell_command_doc)
    server.add_tool(service.get_job, description=get_job_doc)
    server.add_tool(service.list_jobs, description=list_jobs_doc)
    server.add_tool(service.terminate_job, description=terminate_job_doc)
    server.add_tool(service.cleanup_jobs, description=cleanup_jobs_doc)
    server.add_tool(service.upload_file, description=upload_file_doc)
    server.add_tool(service.download_file, description=download_file_doc)
    server.add_tool(service.begin_upload, description=begin_upload_doc)
    server.add_tool(service.upload_chunk, description=upload_chunk_doc)
    server.add_tool(service.finish_upload, description=finish_upload_doc)
    server.add_tool(service.abort_upload, description=abort_upload_doc)
    server.add_tool(service.file_sha256, description=file_sha256_doc)
    server.add_tool(service.git_push, description=git_push_doc)
    server.shell_service = service  # type: ignore[attr-defined]
    return server


# --------------------------------------------------------------------------
# Bearer-token auth middleware for the HTTP transport
# --------------------------------------------------------------------------
class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Bearer-token gate bound to the *app's own* config, not a module global."""

    def __init__(self, app, cfg: Config) -> None:
        super().__init__(app)
        self._cfg = cfg

    async def dispatch(self, request: Request, call_next):
        cfg = self._cfg
        if cfg.no_auth or request.method == "OPTIONS":
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        expected = f"Bearer {cfg.token}"
        if not secrets.compare_digest(auth, expected):
            log.warning("AUTH REJECTED | path=%s | client=%s", request.url.path, request.client)
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def build_app(cfg: Config):
    """Build the Starlette ASGI app (MCP transport + auth) for ``cfg``."""
    global CFG, SERVICE
    CFG = cfg
    SERVICE = create_service(cfg)
    SERVICE.start_reaper()
    app = build_server(SERVICE).streamable_http_app()
    app.state.shell_service = SERVICE
    app.add_middleware(BearerAuthMiddleware, cfg)
    return app


def main() -> None:
    cfg = parse_args()
    log.info("Starting MCP shell server on http://%s:%s/mcp", cfg.host, cfg.port)
    log.info("Working directory for commands: %s", cfg.cwd)
    log.info(
        "Sync wait window: %ss | max runtime: %s | retention: %ss",
        _resolve_wait_seconds(cfg, None), cfg.max_runtime_seconds or "unlimited", cfg.job_retention_seconds,
    )
    if cfg.no_auth:
        log.warning("AUTH DISABLED — anyone who can reach this server can run shell commands!")
    else:
        log.info("Auth token (send as 'Authorization: Bearer <token>'): %s", cfg.token)
    if cfg.host not in ("127.0.0.1", "localhost"):
        log.warning(
            "Bound to %s (not localhost). Make sure this is on a trusted network/firewall.",
            cfg.host,
        )

    app = build_app(cfg)
    service: ShellService = app.state.shell_service  # type: ignore[attr-defined]

    try:
        import uvicorn

        uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")
    finally:
        service.shutdown()


# --------------------------------------------------------------------------
# Module-level defaults (backward compatible surface).
# `build_app`/`main` replace these with the CLI-configured equivalents.
# --------------------------------------------------------------------------
CFG = Config()
SERVICE = create_service(CFG)
mcp = build_server(SERVICE)


async def run_shell_command(command: str, timeout_seconds: float | None = None, wait_seconds: float | None = None) -> dict[str, Any]:
    """Backward-compatible wrapper around ``mcp_shell_server.SERVICE``."""
    return await SERVICE.run_shell_command(command, timeout_seconds=timeout_seconds, wait_seconds=wait_seconds)


async def get_job(job_id: str, wait_seconds: float = 0.0) -> dict[str, Any]:
    return await SERVICE.get_job(job_id, wait_seconds=wait_seconds)


async def list_jobs() -> dict[str, Any]:
    return await SERVICE.list_jobs()


async def terminate_job(job_id: str, grace_seconds: float = 2.0) -> dict[str, Any]:
    return await SERVICE.terminate_job(job_id, grace_seconds=grace_seconds)


async def cleanup_jobs(older_than_seconds: float | None = None) -> dict[str, Any]:
    return await SERVICE.cleanup_jobs(older_than_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        sys.exit(0)