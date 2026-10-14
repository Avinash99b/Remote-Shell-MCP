"""Tests for the logging setup that keeps an unconsumed stderr pipe from filling.

Context: the server is often launched with ``subprocess.Popen(..., stderr=PIPE)``
and nobody drains that pipe. If routine per-request logging is emitted, the ~64 KB
OS pipe buffer eventually fills and the server appears hung. These tests pin the
two guarantees:

  * by default, routine request/job and third-party (MCP SDK / uvicorn / httpx)
    logging is *not* emitted, while startup lines, warnings and errors still are;
  * even when verbose logging is enabled, a stalled consumer cannot block the
    request path because records go through a bounded dropping queue.

Run:
    .venv/bin/python -m pytest tests/test_logging.py -q
"""

from __future__ import annotations

import logging
import queue

import pytest

import mcp_shell_server as srv


@pytest.fixture(autouse=True)
def _restore_logging():
    """Give each test a clean, deterministic logging state (and no stderr noise)."""
    yield
    srv.configure_logging(srv.DEFAULT_LOG_LEVEL)


def _record(name: str, level: int, msg: str = "x") -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, None, None)


# ---------------------------------------------------------------------------
# Default (safe) behavior
# ---------------------------------------------------------------------------
def test_handler_does_not_filter_startup_lines(tmp_path):
    """Startup INFO lines from LOG_NAME survive the default (warning) gate."""
    log_path = tmp_path / "startup.log"
    listener = srv.configure_logging(srv.DEFAULT_LOG_LEVEL, str(log_path))
    try:
        logging.getLogger(srv.LOG_NAME).info("startup-token-visible")
    finally:
        listener.stop()
    assert "startup-token-visible" in log_path.read_text()


def test_default_level_is_warning():
    assert srv.DEFAULT_LOG_LEVEL == "warning"
    assert srv.Config().log_level == "warning"
    assert srv.Config().log_file is None


def test_routine_request_logging_suppressed_by_default(tmp_path):
    srv.configure_logging(log_file=str(tmp_path / "log.txt"))

    # Routine per-request/job lines must not be emitted at the default level.
    assert not logging.getLogger(srv.REQUEST_LOG_NAME).isEnabledFor(logging.INFO)


def test_startup_warnings_and_errors_still_emitted_by_default(tmp_path):
    srv.configure_logging(log_file=str(tmp_path / "log.txt"))

    lifecycle = logging.getLogger(srv.LOG_NAME)
    assert lifecycle.isEnabledFor(logging.INFO)   # startup lines
    assert lifecycle.isEnabledFor(logging.WARNING)
    assert lifecycle.isEnabledFor(logging.ERROR)


def test_third_party_loggers_gated_by_default(tmp_path):
    srv.configure_logging(log_file=str(tmp_path / "log.txt"))

    for name in ("mcp.server.lowlevel.server", "mcp.server.streamable_http_manager",
                 "uvicorn.access", "httpx"):
        logger = logging.getLogger(name)
        assert not logger.isEnabledFor(logging.INFO), name
        assert logger.isEnabledFor(logging.WARNING), name


# ---------------------------------------------------------------------------
# Opt-in verbosity
# ---------------------------------------------------------------------------
def test_log_level_info_enables_request_logging(tmp_path):
    srv.configure_logging("info", str(tmp_path / "log.txt"))

    assert logging.getLogger(srv.REQUEST_LOG_NAME).isEnabledFor(logging.INFO)
    assert logging.getLogger("mcp.server.lowlevel.server").isEnabledFor(logging.INFO)


def test_log_level_debug_is_most_verbose(tmp_path):
    srv.configure_logging("debug", str(tmp_path / "log.txt"))
    assert logging.getLogger(srv.REQUEST_LOG_NAME).isEnabledFor(logging.DEBUG)


def test_numeric_and_name_levels_resolve():
    assert srv._resolve_log_level("info") == logging.INFO
    assert srv._resolve_log_level("DEBUG") == logging.DEBUG
    assert srv._resolve_log_level("warning") == logging.WARNING
    assert srv._resolve_log_level(10) == logging.DEBUG
    assert srv._resolve_log_level(True) == logging.WARNING      # bool is not a level
    assert srv._resolve_log_level(None) == logging.WARNING
    assert srv._resolve_log_level("not-a-level") == logging.WARNING
    assert srv._log_level_name("INFO") == "info"
    assert srv._log_level_name(None) == "warning"


def test_configure_logging_is_idempotent(tmp_path):
    srv.configure_logging("info", str(tmp_path / "a.txt"))
    srv.configure_logging("warning", str(tmp_path / "b.txt"))
    srv.configure_logging("warning", str(tmp_path / "b.txt"))

    root = logging.getLogger()
    queue_handlers = [h for h in root.handlers if isinstance(h, srv._DropOnFullQueueHandler)]
    assert len(queue_handlers) == 1


# ---------------------------------------------------------------------------
# Queue behavior — a stalled consumer must never block
# ---------------------------------------------------------------------------
def test_drop_on_full_queue_handler_never_blocks():
    small: queue.Queue = queue.Queue(maxsize=1)
    handler = srv._DropOnFullQueueHandler(small)
    handler.enqueue(_record("t", logging.INFO, "first"))
    handler.enqueue(_record("t", logging.INFO, "second"))  # must be dropped, not raise
    assert small.qsize() == 1
    assert small.get_nowait().getMessage() == "first"


def test_emit_through_queue_reaches_destination(tmp_path):
    log_path = tmp_path / "out.log"
    listener = srv.configure_logging("info", str(log_path))
    try:
        logging.getLogger(srv.LOG_NAME).info("hello-queue")
    finally:
        listener.stop()  # joins the writer thread and flushes
    assert "hello-queue" in log_path.read_text()


# ---------------------------------------------------------------------------
# Suppression holds end-to-end across a real (in-memory) MCP round-trip
# ---------------------------------------------------------------------------
class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_default_level_emits_no_routine_records_during_requests(tmp_path):
    import anyio
    from mcp.shared.memory import create_connected_server_and_client_session

    srv.configure_logging(srv.DEFAULT_LOG_LEVEL, str(tmp_path / "log.txt"))

    cfg = srv.Config(cwd=str(tmp_path), cleanup_interval_seconds=0.0)
    service = srv.ShellService(cfg)
    server = srv.build_server(service)

    async def scenario():
        async with create_connected_server_and_client_session(server, raise_exceptions=False) as client:
            await client.call_tool("run_shell_command", {"command": "echo hi"})
            await client.call_tool("list_jobs", {})

    capture = _Capture()
    logging.getLogger().addHandler(capture)
    try:
        anyio.run(scenario)  # capture the per-request window only
    finally:
        logging.getLogger().removeHandler(capture)
        service.shutdown()

    routine = [
        r for r in capture.records
        if r.levelno < logging.WARNING
        and (r.name.startswith("mcp-shell-server") or r.name.startswith("mcp") or r.name.startswith("httpx"))
    ]
    assert routine == [], [f"{r.name}:{r.getMessage()}" for r in routine]


def test_parse_args_log_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["mcp_shell_server.py"])
    cfg = srv.parse_args()
    assert cfg.log_level == "warning"
    assert cfg.log_file is None


def test_parse_args_log_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["mcp_shell_server.py", "--log-level", "info", "--log-file", str(tmp_path / "s.log")],
    )
    cfg = srv.parse_args()
    assert cfg.log_level == "info"
    assert cfg.log_file == str(tmp_path / "s.log")