"""H.264 screen capture for Comets without a JPEG snapshot (Comet Q / GL-RMQ1).

Observed on a Comet Q (firmware V1.8.4, kvmd 4.82, encoder RV1126-H264):
``GET /api/streamer/snapshot`` returns ~80 raw H.264 slice bytes labelled
``image/jpeg`` and HTTP 500 with ``preview=true``. Frames have to come from the
media websocket instead. These tests drive the real client and decoder against
a loopback fake of that websocket; the video is real H.264 made with libx264.
"""
from __future__ import annotations

import asyncio
import io
import json
import time
from fractions import Fraction
from typing import Any, Callable, Optional

import av
import httpx
import pytest
from aiohttp import web
from PIL import Image as PILImage

from src.kvm_core.comet.client import (
    CometCaptureError,
    CometClient,
    CometNoSignalError,
    needs_h264_capture,
)
from src.kvm_core.comet.h264_stream import (
    CometH264Stream,
    H264Decoder,
    H264NoSignal,
    H264StreamError,
    encode_jpeg,
    source_online,
)
from src.kvm_core.tools_core import _capture_state
from tests.bios_test_helpers import jpeg_bytes
from tests.local_services import CometProtocolService

RED = (200, 20, 20)
GREEN = (20, 180, 20)
BLUE = (20, 20, 200)
RAW_H264_SLICE = b"\x00\x00\x00\x01\x41" + bytes(75)  # NAL type 1, like the Comet Q


def run(coro):
    return asyncio.run(coro)


async def wait_until(predicate: Callable[[], Any], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition was not met in time")
        await asyncio.sleep(0.01)


class ScreenEncoder:
    """Real H.264 encoder: one Annex-B access unit per frame, no B-frames."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self._ctx = av.CodecContext.create("libx264", "w")
        self._ctx.width, self._ctx.height = size
        self._ctx.pix_fmt = "yuv420p"
        self._ctx.time_base = Fraction(1, 30)
        self._ctx.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "g": "100000",
        }
        self._pts = 0

    def encode(self, color: tuple[int, int, int]) -> tuple[bool, bytes]:
        frame = av.VideoFrame.from_image(PILImage.new("RGB", self.size, color))
        frame = frame.reformat(format="yuv420p")
        frame.pts = self._pts
        self._pts += 1
        packets = self._ctx.encode(frame)
        assert len(packets) == 1
        return bool(packets[0].is_keyframe), bytes(packets[0])


def color_of(jpeg: bytes) -> tuple[int, int, int]:
    image = PILImage.open(io.BytesIO(jpeg)).convert("RGB")
    return image.getpixel((image.width // 2, image.height // 2))


def assert_color(jpeg: bytes, expected: tuple[int, int, int]) -> None:
    assert jpeg.startswith(b"\xff\xd8")
    actual = color_of(jpeg)
    assert all(abs(a - e) <= 30 for a, e in zip(actual, expected)), (actual, expected)


def size_of(jpeg: bytes) -> tuple[int, int]:
    return PILImage.open(io.BytesIO(jpeg)).size


class MediaSession:
    def __init__(self, ws: web.WebSocketResponse, headers: dict[str, str]) -> None:
        self.ws = ws
        self.headers = headers
        self.started = False
        self.start_event: Any = None
        self.heartbeats = 0
        self.key_requests = 0
        self.encoder: Optional[ScreenEncoder] = None


class CometQService(CometProtocolService):
    """Loopback Comet Q: non-JPEG snapshot plus the kvmd media websocket."""

    def __init__(self, model: str = "RMQ1", size: tuple[int, int] = (320, 180)) -> None:
        super().__init__()
        self.model = model
        self.screen_color = RED
        self.screen_size = size
        self.online = True
        self.offer_h264 = True
        self.send_video = True
        self.honor_key_requests = True
        self.answer_heartbeats = True
        self.sessions: list[MediaSession] = []
        self.respond("/api/upgrade/version", 200, {
            "ok": True,
            "result": {"model": model, "version": "V1.8.4"},
        })

    def live_sessions(self) -> list[MediaSession]:
        return [s for s in self.sessions if not s.ws.closed and s.started]

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/api/media/ws":
            return await self._media(request)
        if request.path == "/api/streamer":
            self.requests.append({"method": request.method, "path": request.path})
            return web.json_response({"ok": True, "result": {"streamer": {
                "source": {"online": self.online, "resolution": {
                    "width": self.screen_size[0], "height": self.screen_size[1],
                }},
            }}})
        if request.path == "/api/streamer/snapshot":
            self.requests.append({
                "method": request.method, "path": request.path, "query": dict(request.query),
            })
            if request.query.get("preview") == "true":
                return web.json_response({"ok": False, "result": {}}, status=500)
            return web.Response(body=RAW_H264_SLICE, content_type="image/jpeg")
        return await super()._handle(request)

    async def _send_frame(self, session: MediaSession, key: bool) -> None:
        if not self.send_video:
            return
        if key or session.encoder is None or session.encoder.size != self.screen_size:
            session.encoder = ScreenEncoder(self.screen_size)
        is_key, data = session.encoder.encode(self.screen_color)
        await session.ws.send_bytes(bytes([1, int(is_key)]) + data)

    async def show(self, color: tuple[int, int, int], size: tuple[int, int] | None = None) -> None:
        """Change the screen; live viewers get a delta frame (key frame on resize)."""
        self.screen_color = color
        if size is not None:
            self.screen_size = size
        for session in self.live_sessions():
            await self._send_frame(session, key=False)

    async def send_delta_only(self, color: tuple[int, int, int]) -> None:
        """Send a delta frame that references a frame the client never saw."""
        encoder = ScreenEncoder(self.screen_size)
        encoder.encode(BLUE)
        _, data = encoder.encode(color)
        for session in self.live_sessions():
            await session.ws.send_bytes(bytes([1, 0]) + data)

    async def drop_viewers(self) -> None:
        for session in list(self.sessions):
            await session.ws.close()

    async def _media(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        session = MediaSession(ws, dict(request.headers))
        self.sessions.append(session)
        video = {"h264": {"profile_level_id": "42E01F"}} if self.offer_h264 else {}
        await ws.send_json({"event_type": "media", "event": {"video": video}})
        async for message in ws:
            if message.type == web.WSMsgType.TEXT:
                payload = json.loads(message.data)
                if payload.get("event_type") == "start":
                    session.started = True
                    session.start_event = payload.get("event")
                    await self._send_frame(session, key=True)
            elif message.type == web.WSMsgType.BINARY:
                if message.data == b"\x00":
                    session.heartbeats += 1
                    if self.answer_heartbeats:
                        await ws.send_bytes(b"\xff")
                elif message.data == b"\x01":
                    session.key_requests += 1
                    if self.honor_key_requests and session.started:
                        await self._send_frame(session, key=True)
        return ws


def make_stream(service: CometQService, **overrides: Any) -> CometH264Stream:
    async def probe() -> Optional[bool]:
        return service.online

    stream = CometH264Stream(
        service.base_url.replace("http://", "ws://") + "/api/media/ws",
        {"Token": "token-123", "Cookie": "auth_token=token-123"},
        online_probe=probe,
        name="test",
    )
    stream.heartbeat_period_s = 0.05
    stream.idle_keyframe_s = 0.15
    stream.online_poll_s = 0.02
    stream.reconnect_backoff_s = (0.02,)
    stream.frame_timeout_s = 3.0
    for name, value in overrides.items():
        setattr(stream, name, value)
    return stream


async def connect_q(service: CometQService) -> CometClient:
    client = CometClient(service.base_url, password="secret")
    await client.connect()
    return client


# ── decoder and encoder units ─────────────────────────────────────────


def test_decoder_returns_each_access_unit_without_a_frame_of_delay():
    encoder = ScreenEncoder((320, 180))
    decoder = H264Decoder()

    key, first = encoder.encode(RED)
    assert key
    frame = decoder.decode(first, key)
    assert (frame.width, frame.height) == (320, 180)

    key, second = encoder.encode(GREEN)
    assert not key
    frame = decoder.decode(second, key)
    assert frame is not None, "the last frame of a static screen must not be held back"
    assert_color(encode_jpeg(frame, False, 0, 0), GREEN)


def test_decoder_waits_for_a_key_frame_and_follows_a_resolution_change():
    decoder = H264Decoder()
    encoder = ScreenEncoder((320, 180))
    _, key_frame = encoder.encode(RED)
    _, delta = encoder.encode(GREEN)

    assert decoder.decode(delta, False) is None
    assert decoder.decode(key_frame, True) is not None

    resized = ScreenEncoder((640, 360))
    key, data = resized.encode(BLUE)
    frame = decoder.decode(data, key)
    assert (frame.width, frame.height) == (640, 360)

    decoder.reset()
    assert decoder.decode(delta, False) is None


def test_encode_jpeg_scales_previews_down_only_and_keeps_full_frames():
    encoder = ScreenEncoder((640, 360))
    key, data = encoder.encode(GREEN)
    frame = H264Decoder().decode(data, key)

    preview = encode_jpeg(frame, True, 320, 60)
    assert size_of(preview) == (320, 180)
    assert_color(preview, GREEN)

    assert size_of(encode_jpeg(frame, True, 4096, 60)) == (640, 360)
    assert size_of(encode_jpeg(frame, False, 320, 10)) == (640, 360)

    noisy = av.VideoFrame.from_image(PILImage.effect_noise((640, 360), 80).convert("RGB"))
    assert len(encode_jpeg(noisy, True, 640, 20)) < len(encode_jpeg(noisy, True, 640, 95))


def test_source_online_reads_kvmd_streamer_state():
    assert source_online({"source": {"online": True}}) is True
    assert source_online({"source": {"online": False}}) is False
    assert source_online(None) is None
    assert source_online({"source": {}}) is None


def test_model_detection_matches_comet_q_identity_only():
    assert needs_h264_capture("RMQ1")
    assert needs_h264_capture(None, {"platform": {"base": "AX620QP ... GL-RMQ1"}})
    assert not needs_h264_capture("RM10", {"model": "rm4pe"}, None)


# ── stream ────────────────────────────────────────────────────────────


def test_stream_follows_the_media_protocol_and_serves_the_latest_frame():
    async def scenario():
        async with CometQService() as service:
            stream = make_stream(service)
            stream.start()
            try:
                first = await stream.snapshot_jpeg(preview=False)
                session = service.sessions[0]
                assert session.headers["Token"] == "token-123"
                assert "auth_token=token-123" in session.headers["Cookie"]
                assert session.start_event == {"type": "video", "format": "h264"}
                assert_color(first, RED)
                assert size_of(first) == (320, 180)

                seq = stream._latest.seq
                await service.show(GREEN)
                await wait_until(lambda: stream._latest.seq > seq)
                assert_color(await stream.snapshot_jpeg(preview=False), GREEN)

                await wait_until(lambda: session.heartbeats >= 2)
                assert len(service.sessions) == 1, "one persistent socket, not one per capture"
                assert stream.status()["frame_current"] is True
            finally:
                await stream.stop()
            assert not stream.is_running()
            assert stream._thread is None
            await wait_until(lambda: all(s.ws.closed for s in service.sessions))

    run(scenario())


def test_warm_screenshot_is_a_fast_cache_read():
    async def scenario():
        async with CometQService(size=(1920, 1080)) as service:
            stream = make_stream(service)
            stream.start()
            try:
                await stream.snapshot_jpeg()
                seq = stream._latest.seq
                await service.show(GREEN)
                await wait_until(lambda: stream._latest.seq > seq)

                started = time.perf_counter()
                data = await stream.snapshot_jpeg(preview=True, max_width=1024, quality=60)
                elapsed = time.perf_counter() - started
                assert size_of(data) == (1024, 576)
                assert elapsed < 0.2, f"warm capture took {elapsed * 1000:.0f} ms"

                # The same frame and parameters are not encoded twice.
                assert await stream.snapshot_jpeg(preview=True, max_width=1024, quality=60) is data
            finally:
                await stream.stop()

    run(scenario())


def test_stream_reconnects_and_never_serves_a_frame_from_the_old_connection():
    async def scenario():
        async with CometQService() as service:
            stream = make_stream(service)
            stream.start()
            try:
                assert_color(await stream.snapshot_jpeg(), RED)
                service.screen_color = GREEN
                await service.drop_viewers()
                await wait_until(lambda: not stream.connected or stream.generation > 1)
                assert_color(await stream.snapshot_jpeg(), GREEN)
                assert len(service.sessions) == 2
                assert stream.reconnects >= 1
            finally:
                await stream.stop()

    run(scenario())


def test_stream_follows_a_resolution_change():
    async def scenario():
        async with CometQService() as service:
            stream = make_stream(service)
            stream.start()
            try:
                assert size_of(await stream.snapshot_jpeg(preview=False)) == (320, 180)
                await service.show(BLUE, size=(640, 360))
                await wait_until(lambda: stream._latest.width == 640)
                data = await stream.snapshot_jpeg(preview=False)
                assert size_of(data) == (640, 360)
                assert_color(data, BLUE)
            finally:
                await stream.stop()

    run(scenario())


def test_no_signal_is_reported_instead_of_the_cached_frame():
    async def scenario():
        async with CometQService() as service:
            stream = make_stream(service)
            stream.start()
            try:
                assert_color(await stream.snapshot_jpeg(), RED)

                service.online = False
                await wait_until(lambda: stream.status()["source_online"] is False)
                started = time.perf_counter()
                with pytest.raises(H264NoSignal) as caught:
                    await stream.snapshot_jpeg()
                assert time.perf_counter() - started < 0.5, "no signal must not wait for a timeout"
                assert "no signal" in str(caught.value)
                assert "source.online=false" in str(caught.value)

                service.online = True
                await wait_until(lambda: stream.status()["source_online"] is True)
                assert_color(await stream.snapshot_jpeg(), RED)
            finally:
                await stream.stop()

    run(scenario())


def test_pushed_state_marks_the_source_offline_at_once():
    async def scenario():
        async with CometQService() as service:
            stream = make_stream(service, online_poll_s=30.0)
            stream.start()
            try:
                await stream.snapshot_jpeg()
                stream.note_online(False)
                with pytest.raises(H264NoSignal):
                    await stream.snapshot_jpeg()
            finally:
                await stream.stop()

    run(scenario())


def test_idle_screen_is_revalidated_with_key_frame_requests():
    async def scenario():
        async with CometQService() as service:
            stream = make_stream(service)
            stream.start()
            try:
                await stream.snapshot_jpeg()
                first_seq = stream._latest.seq
                # Nothing changes on screen: the stream asks for key frames itself.
                await wait_until(lambda: service.sessions[0].key_requests >= 2)
                await wait_until(lambda: stream._latest.seq > first_seq)
                assert stream._latest.key is True
                assert stream.status()["frame_age_s"] < 1.0
                assert len(service.sessions) == 1
            finally:
                await stream.stop()

    run(scenario())


def test_stale_frame_forces_a_reconnect_for_a_fresh_key_frame():
    async def scenario():
        async with CometQService() as service:
            service.honor_key_requests = False
            stream = make_stream(service, max_frame_age_s=0.2)
            stream.start()
            try:
                assert_color(await stream.snapshot_jpeg(), RED)
                service.screen_color = GREEN  # changed, but the encoder went silent
                await asyncio.sleep(0.3)
                assert_color(await stream.snapshot_jpeg(), GREEN)
                assert stream.forced_reconnects == 1
                assert len(service.sessions) == 2
            finally:
                await stream.stop()

    run(scenario())


def test_stalled_stream_raises_instead_of_returning_a_stale_frame():
    async def scenario():
        async with CometQService() as service:
            service.honor_key_requests = False
            stream = make_stream(service, max_frame_age_s=0.2, frame_timeout_s=0.6)
            stream.start()
            try:
                await stream.snapshot_jpeg()
                service.send_video = False
                await asyncio.sleep(0.3)
                with pytest.raises(H264StreamError) as caught:
                    await stream.snapshot_jpeg()
                assert "capture failed" in str(caught.value)
                assert "source.online=true" in str(caught.value)
            finally:
                await stream.stop()

    run(scenario())


def test_lost_heartbeats_reconnect_the_stream():
    async def scenario():
        async with CometQService() as service:
            service.answer_heartbeats = False
            stream = make_stream(service, heartbeat_period_s=0.02)
            stream.start()
            try:
                await wait_until(lambda: len(service.sessions) >= 2)
                assert "heartbeat" in (stream.last_error or "")
            finally:
                await stream.stop()

    run(scenario())


def test_corrupt_video_requests_a_key_frame_and_recovers():
    async def scenario():
        async with CometQService() as service:
            stream = make_stream(service, idle_keyframe_s=60.0)
            stream.start()
            try:
                await stream.snapshot_jpeg()
                seq = stream._latest.seq
                service.screen_color = GREEN
                for session in service.live_sessions():
                    await session.ws.send_bytes(b"\x01\x00" + b"\x00\x00\x00\x01\x65" + bytes(64))
                await wait_until(lambda: service.sessions[0].key_requests >= 1)
                await wait_until(lambda: stream._latest.seq > seq)
                assert stream.decode_errors >= 1
                assert_color(await stream.snapshot_jpeg(), GREEN)
            finally:
                await stream.stop()

    run(scenario())


def test_decoder_backlog_is_dropped_and_resynchronised_on_a_key_frame():
    async def scenario():
        async with CometQService() as service:
            stream = make_stream(service, idle_keyframe_s=60.0, max_decode_backlog=0)
            stream.start()
            try:
                # Every frame overflows the zero-length backlog, so each one
                # drops the queue, resets the decoder and asks for a key frame.
                await wait_until(lambda: stream.backlog_drops >= 1)
                await wait_until(lambda: service.sessions[0].key_requests >= 1)
                stream.max_decode_backlog = 120
                assert_color(await stream.snapshot_jpeg(), RED)
            finally:
                await stream.stop()

    run(scenario())


def test_media_socket_without_h264_fails_fast_for_a_probe():
    async def scenario():
        async with CometQService() as service:
            service.offer_h264 = False
            stream = make_stream(service)
            stream.start()
            try:
                started = time.perf_counter()
                with pytest.raises(H264StreamError):
                    await stream.snapshot_jpeg(timeout=3.0, fail_fast=True)
                assert time.perf_counter() - started < 1.0
                assert stream.h264_offered is False
            finally:
                await stream.stop()

    run(scenario())


# ── client routing ────────────────────────────────────────────────────


def test_comet_q_is_detected_at_connect_and_captures_from_the_stream():
    async def scenario():
        async with CometQService() as service:
            client = await connect_q(service)
            try:
                assert client.capture_mode == "h264"
                assert client.capabilities["capture_mode"] == "h264"
                # The decoder is warmed at connect, before any screenshot.
                await wait_until(lambda: service.live_sessions())

                full = await client.get_screenshot(preview=False)
                assert_color(full, RED)
                assert size_of(full) == (320, 180)
                preview = await client.get_screenshot(preview=True, max_width=160, quality=50)
                assert size_of(preview) == (160, 90)

                assert client.last_capture_ok is True
                snapshot_calls = [r for r in service.requests if r["path"] == "/api/streamer/snapshot"]
                assert snapshot_calls == [], "the broken snapshot endpoint is never called"

                state = _capture_state(client)
                assert state["mode"] == "h264"
                assert state["stream"]["frame_current"] is True
                assert (state["stream"]["width"], state["stream"]["height"]) == (320, 180)
            finally:
                stream = client._h264_stream
                await client.disconnect()
            assert client._h264_stream is None
            assert not stream.is_running()

    run(scenario())


def test_comet_q_no_signal_is_a_capture_error_naming_the_cause():
    async def scenario():
        async with CometQService() as service:
            client = await connect_q(service)
            try:
                await client.get_screenshot()
                # kvmd pushes partial streamer state on the control websocket.
                client._note_streamer_event({"streamer": {"source": {"online": False}}})
                with pytest.raises(CometNoSignalError) as caught:
                    await client.get_screenshot()
                assert isinstance(caught.value, CometCaptureError)
                assert "no signal" in str(caught.value)
                assert client.last_capture_ok is False
                assert client.is_connected(), "no signal must not tear the session down"

                client._note_streamer_event({"params": {"quality": 80}})
                with pytest.raises(CometNoSignalError):
                    await client.get_screenshot()

                client._note_streamer_event({"streamer": {"source": {"online": True}}})
                assert_color(await client.get_screenshot(), RED)
            finally:
                await client.disconnect()

    run(scenario())


def test_pushed_streamer_state_seeds_the_stream_and_slows_the_poll():
    async def scenario():
        async with CometQService() as service:
            client = await connect_q(service)
            try:
                await client.get_screenshot()
                stream = client._h264_stream
                assert stream._online_pushed is False

                # State pushed before a (re)created stream exists is not lost.
                await client._stop_h264_stream()
                client._note_streamer_event({"streamer": {"source": {"online": False}}})
                with pytest.raises(CometNoSignalError):
                    await client.get_screenshot()
                assert client._h264_stream._online_pushed is True

                # A pushed value stands without HTTP polling on the capture path.
                client._note_streamer_event({"streamer": {"source": {"online": True}}})
                await client.get_screenshot()
                client._h264_stream._online_at -= 60
                polls = len([r for r in service.requests if r["path"] == "/api/streamer"])
                await client.get_screenshot()
                assert polls == len([r for r in service.requests if r["path"] == "/api/streamer"])
            finally:
                await client.disconnect()

    run(scenario())


def test_unlisted_model_falls_back_when_the_snapshot_is_not_a_jpeg():
    async def scenario():
        async with CometQService(model="FUTURE1") as service:
            client = await connect_q(service)
            try:
                assert client.capture_mode == "jpeg"
                assert service.sessions == []

                data = await client.get_screenshot(preview=False)
                assert_color(data, RED)
                assert client.capture_mode == "h264"
                assert client.last_capture_ok is True

                calls = len([r for r in service.requests if r["path"] == "/api/streamer/snapshot"])
                await client.get_screenshot()
                assert calls == len(
                    [r for r in service.requests if r["path"] == "/api/streamer/snapshot"]
                ), "after the fallback the client stays on the stream"
            finally:
                await client.disconnect()

    run(scenario())


def test_unlisted_model_falls_back_when_the_preview_snapshot_returns_500():
    async def scenario():
        async with CometQService(model="FUTURE1") as service:
            client = await connect_q(service)
            try:
                data = await client.get_screenshot(preview=True, max_width=160, quality=50)
                assert size_of(data) == (160, 90)
                assert client.capture_mode == "h264"
            finally:
                await client.disconnect()

    run(scenario())


class JpegCometService(CometQService):
    """A Comet Pro/X: the snapshot endpoint returns a real JPEG."""

    image = jpeg_bytes()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        if request.path == "/api/streamer/snapshot":
            self.requests.append({"path": request.path, "query": dict(request.query)})
            return web.Response(body=self.image, content_type="image/jpeg")
        return await super()._handle(request)


def test_jpeg_comets_keep_the_snapshot_path_and_open_no_media_socket():
    async def scenario():
        async with JpegCometService(model="RM10") as service:
            client = await connect_q(service)
            try:
                assert client.capture_mode == "jpeg"
                data = await client.get_screenshot(preview=True, max_width=320, quality=40)
                assert data == service.image, "device JPEG bytes are returned unmodified"
                query = [r for r in service.requests if r["path"] == "/api/streamer/snapshot"][0]["query"]
                assert query == {
                    "allow_offline": "true",
                    "preview": "true",
                    "preview_max_width": "320",
                    "preview_quality": "40",
                }
                assert service.sessions == []
                assert client._h264_stream is None
                assert "mode" not in _capture_state(client)
            finally:
                await client.disconnect()

    run(scenario())


def test_non_jpeg_snapshot_without_a_media_stream_keeps_the_old_result(monkeypatch):
    async def scenario():
        async with CometProtocolService() as service:
            service.respond("/api/streamer/snapshot", 200, "not a jpeg")
            client = CometClient(service.base_url)
            client.http = httpx.AsyncClient()
            client.auth_token = "token-123"
            try:
                assert await client.get_screenshot(preview=False) == b"not a jpeg"
                assert client.capture_mode == "jpeg"
                assert client._h264_stream is None
                probes = [r for r in service.requests if r["path"] == "/api/media/ws"]
                assert len(probes) == 1

                # A failed probe is not repeated on every capture.
                await client.get_screenshot(preview=False)
                assert len([r for r in service.requests if r["path"] == "/api/media/ws"]) == 1
            finally:
                await client.http.aclose()

    run(scenario())


def test_capture_mode_can_be_pinned_by_environment(monkeypatch):
    async def scenario(mode: str, model: str) -> str:
        monkeypatch.setenv("COMET_CAPTURE_MODE", mode)
        async with CometQService(model=model) as service:
            client = await connect_q(service)
            try:
                if mode == "jpeg":
                    assert await client.get_screenshot(preview=False) == RAW_H264_SLICE
                    assert service.sessions == []
                return client.capture_mode
            finally:
                await client.disconnect()

    assert run(scenario("jpeg", "RMQ1")) == "jpeg"
    assert run(scenario("h264", "RM10")) == "h264"
