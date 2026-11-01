"""Tests for the ``screenshot`` MCP tool.

These are fully portable: the actual capture is injected (a fake backend writes a
known tiny PNG), so nothing here needs a real display or any specific screenshot
tool installed. One optional test exercises the real ``xwd``+``ffmpeg`` backend
and is skipped unless that tooling and a display are present.

Run:
    .venv/bin/python -m pytest tests/test_screenshot.py -q
"""

from __future__ import annotations

import base64
import os
import shutil
import struct

import anyio
import pytest

import mcp_shell_server as srv
from mcp.shared.memory import create_connected_server_and_client_session


# A real, valid 1x1 PNG (used only as fixture bytes in these tests).
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
TINY_PNG_B64 = base64.b64encode(TINY_PNG).decode("ascii")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def cfg(tmp_path):
    return srv.Config(
        cwd=str(tmp_path),
        cleanup_interval_seconds=0.0,  # disable the background reaper in unit tests
    )


@pytest.fixture()
def service(cfg):
    svc = srv.ShellService(cfg)
    yield svc
    svc.shutdown()


def install_fake_capture(monkeypatch, service, name="fake", payload=TINY_PNG):
    """Replace the blocking capture with one that writes known bytes."""

    def fake_capture(target, region=None, backend=None, display=None, timeout_seconds=srv.SCREENSHOT_DEFAULT_TIMEOUT):
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(payload)
        return name

    monkeypatch.setattr(service, "_capture_screenshot", fake_capture)


# ---------------------------------------------------------------------------
# Region validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "region",
    [
        [1, 2, 3],                       # wrong length (too short)
        [1, 2, 3, 4, 5],                 # wrong length (too long)
        [1.5, 2, 3, 4],                  # non-int
        ["1", 2, 3, 4],                  # non-int
        [-1, 0, 10, 10],                 # negative x
        [0, -1, 10, 10],                 # negative y
        [0, 0, 0, 10],                   # zero width
        [0, 0, 10, 0],                   # zero height
        [0, 0, -5, 10],                  # negative width
        [0, 0, 10, -5],                  # negative height
        "0,0,10,10",                     # not a list
    ],
)
def test_region_validation_rejects_bad_input(service, region):
    with pytest.raises(ValueError):
        service._validate_region(region)


def test_region_validation_accepts_good_input(service):
    assert service._validate_region(None) is None
    assert service._validate_region([0, 0, 1, 1]) == [0, 0, 1, 1]
    assert service._validate_region((5, 6, 7, 8)) == [5, 6, 7, 8]


# ---------------------------------------------------------------------------
# PNG dimension parsing
# ---------------------------------------------------------------------------
def test_png_dimensions_parsed():
    assert srv._png_dimensions(TINY_PNG) == (1, 1)


def test_png_dimensions_rejects_non_png():
    assert srv._png_dimensions(b"not a png at all, definitely") is None
    assert srv._png_dimensions(b"") is None


def test_png_dimensions_reads_arbitrary_size():
    header = srv._PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 1920, 1080)
    assert srv._png_dimensions(header + b"\x00" * 8) == (1920, 1080)


# ---------------------------------------------------------------------------
# Timeout sanitization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 0, "abc", None])
def test_screenshot_timeout_sanitized(bad):
    assert srv._resolve_screenshot_timeout(bad) == srv.SCREENSHOT_DEFAULT_TIMEOUT
    assert srv._resolve_screenshot_timeout(5) == 5.0


# ---------------------------------------------------------------------------
# Default path generation + metadata (no transport)
# ---------------------------------------------------------------------------
def test_default_path_is_absolute_under_cwd(service, monkeypatch):
    install_fake_capture(monkeypatch, service, name="fake")

    async def scenario():
        return await service.screenshot(inline=False)

    result = anyio.run(scenario)
    meta = result.structuredContent

    assert meta["path"].startswith(service.cfg.cwd)
    assert os.path.isabs(meta["path"])
    assert os.path.basename(meta["path"]).startswith("screenshot-")
    assert os.path.basename(meta["path"]).endswith(".png")
    assert os.path.isfile(meta["path"])
    assert meta["bytes"] == len(TINY_PNG)
    assert meta["format"] == "png"
    assert meta["mime_type"] == "image/png"
    assert meta["backend"] == "fake"
    assert meta["width"] == 1
    assert meta["height"] == 1
    assert meta["inline"] is False
    assert meta["region"] is None
    assert isinstance(meta["duration_seconds"], float)
    # inline=False must not add an image content block.
    assert [c.type for c in result.content] == ["text"]


def test_explicit_relative_path_resolved_and_region_echoed(service, monkeypatch):
    install_fake_capture(monkeypatch, service, name="fake")

    async def scenario():
        return await service.screenshot(path="shots/a.png", region=[1, 2, 3, 4], inline=False)

    meta = anyio.run(scenario).structuredContent
    assert meta["path"] == os.path.join(service.cfg.cwd, "shots", "a.png")
    assert os.path.isfile(meta["path"])
    assert meta["region"] == [1, 2, 3, 4]


def test_missing_extension_defaults_to_png(service, monkeypatch):
    install_fake_capture(monkeypatch, service, name="fake")

    async def scenario():
        return await service.screenshot(path="noext", inline=False)

    meta = anyio.run(scenario).structuredContent
    assert meta["path"].endswith(".png")
    assert meta["format"] == "png"


# ---------------------------------------------------------------------------
# Forced backend errors
# ---------------------------------------------------------------------------
def test_unknown_backend_raises(service):
    async def scenario():
        await service.screenshot(path="x.png", backend="definitely-not-a-backend")

    with pytest.raises(ValueError):
        anyio.run(scenario)


def test_forced_missing_backend_raises_runtime_error(service):
    # A backend that does not exist on this host: forcing it must fail clearly.
    if shutil.which("scrot") is not None:
        pytest.skip("scrot is installed here; cannot test the missing-backend path")

    async def scenario():
        await service.screenshot(path="x.png", backend="scrot", inline=False)

    with pytest.raises(RuntimeError):
        anyio.run(scenario)


# ---------------------------------------------------------------------------
# Full MCP round-trip over the in-memory transport
# ---------------------------------------------------------------------------
def test_mcp_screenshot_returns_native_image(service, monkeypatch):
    install_fake_capture(monkeypatch, service, name="fake")
    server = srv.build_server(service)

    async def scenario():
        async with create_connected_server_and_client_session(server, raise_exceptions=False) as client:
            res = await client.call_tool("screenshot", {"path": "roundtrip.png"})
            return res

    res = anyio.run(scenario)

    types = [c.type for c in res.content]
    assert "image" in types, types
    image_blocks = [c for c in res.content if c.type == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0].mimeType == "image/png"
    assert image_blocks[0].data == TINY_PNG_B64
    assert base64.b64decode(image_blocks[0].data) == TINY_PNG

    meta = res.structuredContent
    assert meta is not None
    assert meta["path"] == os.path.join(service.cfg.cwd, "roundtrip.png")
    assert meta["bytes"] == len(TINY_PNG)
    assert meta["width"] == 1
    assert meta["height"] == 1
    assert meta["backend"] == "fake"
    assert meta["mime_type"] == "image/png"
    assert meta["inline"] is True
    # The base64 payload must NOT be duplicated into structured content.
    assert TINY_PNG_B64 not in str(meta)


def test_mcp_screenshot_inline_false_omits_image(service, monkeypatch):
    install_fake_capture(monkeypatch, service, name="fake")
    server = srv.build_server(service)

    async def scenario():
        async with create_connected_server_and_client_session(server, raise_exceptions=False) as client:
            return await client.call_tool("screenshot", {"path": "meta-only.png", "inline": False})

    res = anyio.run(scenario)
    assert [c.type for c in res.content] == ["text"]
    meta = res.structuredContent
    assert meta is not None
    assert meta["inline"] is False
    assert meta["path"].endswith("meta-only.png")
    assert meta["bytes"] == len(TINY_PNG)


# ---------------------------------------------------------------------------
# Optional real backend test (skipped without the needed host tooling)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("xwd") is None or shutil.which("ffmpeg") is None,
                    reason="xwd and ffmpeg are required for the real fallback capture")
@pytest.mark.skipif(not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
                    reason="no DISPLAY/WAYLAND_DISPLAY set")
def test_real_xwd_ffmpeg_backend(service):
    """Exercise the real xwd+ffmpeg fallback end-to-end through the MCP client."""
    server = srv.build_server(service)

    async def scenario():
        async with create_connected_server_and_client_session(server, raise_exceptions=False) as client:
            return await client.call_tool("screenshot", {"path": "real.png", "backend": "xwd+ffmpeg"})

    res = anyio.run(scenario)
    image_blocks = [c for c in res.content if c.type == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0].mimeType == "image/png"
    decoded = base64.b64decode(image_blocks[0].data)
    assert decoded[:8] == srv._PNG_SIGNATURE
    assert len(decoded) > 100
    assert res.structuredContent["backend"] == "xwd+ffmpeg"
