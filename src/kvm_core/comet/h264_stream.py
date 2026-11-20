"""Persistent H.264 screen capture for Comets without a JPEG snapshot.

The Comet Q (GL-RMQ1, RV1126 encoder) answers ``GET /api/streamer/snapshot``
with raw H.264 slice bytes, so a screenshot has to come from the video stream.
kvmd's media server (``kvmd/apps/media/server.py``) speaks this protocol on
``/api/media/ws``:

* server -> ``{"event_type": "media", "event": {"video": {"h264": {...}}}}``
* client -> ``{"event_type": "start", "event": {"type": "video", "format": "h264"}}``
* client -> binary ``0x00`` heartbeat, server answers binary ``0xFF``
* client -> binary ``0x01`` asks the encoder for a key frame
* server -> binary ``0x01 <key flag> <Annex-B access unit>``, one frame per
  message; the first frame for a new client is a key frame

``CometH264Stream`` keeps that socket open for the life of a Comet session and
decodes every frame on a worker thread, so a screenshot is a cache read plus a
local scale and JPEG encode. Nothing here blocks the event loop that also runs
the HID watchdog and pinger.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import av
import websockets

LOG = logging.getLogger("kvm_core.comet.h264")

# kvmd's media server is a separate app from the kvmd API server, so this route
# is not part of the generated kvmd endpoint inventory under docs/reference.
MEDIA_WS_PATH = "/api/media/ws"

MSG_HEARTBEAT = b"\x00"
MSG_KEY_REQUEST = b"\x01"
TYPE_VIDEO = 0x01
TYPE_HEARTBEAT_ACK = 0xFF

HEARTBEAT_PERIOD_S = 1.0
MISSED_HEARTBEAT_LIMIT = 5
# A static screen can leave the encoder silent. After this much silence the
# stream asks for a key frame, which keeps the cached frame provably current.
IDLE_KEYFRAME_S = 2.0
# A cached frame older than this is not served: the stream is reconnected
# (a new client always gets a key frame) and the capture waits for that frame.
MAX_FRAME_AGE_S = 5.0
# Upper bound for one capture to obtain a usable frame (cold start, reconnect).
FRAME_TIMEOUT_S = 5.0
# GET /api/streamer poll for source.online. kvmd also pushes streamer state on
# the control websocket; once a pushed value has been seen it is trusted and the
# poll drops to a slow safety net.
ONLINE_POLL_S = 2.0
ONLINE_POLL_PUSHED_S = 30.0
ONLINE_MAX_AGE_S = 5.0
ONLINE_PROBE_TIMEOUT_S = 2.0
RECONNECT_BACKOFF_S = (0.2, 0.5, 1.0, 2.0)
# The decoder cannot skip delta frames. If it falls this far behind, drop the
# backlog and resynchronise on a requested key frame.
MAX_DECODE_BACKLOG = 120
FULL_FRAME_QUALITY = 90
WS_MAX_SIZE = 32 * 1024 * 1024

_STOP = object()
_RESET = object()


class H264StreamError(RuntimeError):
    """The H.264 capture path could not supply a frame."""


class H264NoSignal(H264StreamError):
    """The Comet reports that its video source is offline."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def source_online(streamer_state: Any) -> Optional[bool]:
    """``source.online`` from a kvmd ``streamer`` state object, or None if absent."""
    if not isinstance(streamer_state, dict):
        return None
    source = streamer_state.get("source")
    if not isinstance(source, dict) or "online" not in source:
        return None
    return bool(source["online"])


@dataclass(frozen=True)
class CachedFrame:
    frame: Any  # av.VideoFrame
    seq: int
    generation: int
    received_at: float  # time.monotonic() when the websocket delivered it
    key: bool

    @property
    def width(self) -> int:
        return self.frame.width

    @property
    def height(self) -> int:
        return self.frame.height


class H264Decoder:
    """Synchronous decoder. One websocket message is one complete access unit.

    No bitstream parser is used: a parser releases a frame only when the next
    one starts, which would hold back the last frame of a screen that stopped
    changing. ``low_delay`` likewise stops the decoder from holding frames.
    """

    def __init__(self) -> None:
        self._ctx: Any = None
        self._need_key = True

    def reset(self) -> None:
        self._ctx = None
        self._need_key = True

    def decode(self, data: bytes, key: bool) -> Any:
        """Return the decoded ``av.VideoFrame`` or None. Raises on corrupt data."""
        if self._need_key:
            if not key:
                return None
            ctx = av.CodecContext.create("h264", "r")
            ctx.thread_type = "SLICE"
            ctx.flags |= av.codec.context.Flags.low_delay
            self._ctx = ctx
            self._need_key = False
        try:
            frames = self._ctx.decode(av.Packet(data))
        except Exception:
            self.reset()
            raise
        return frames[-1] if frames else None


def encode_jpeg(frame: Any, preview: bool, max_width: int, quality: int) -> bytes:
    """Scale (preview only, never up) and JPEG-encode a decoded frame."""
    width, height = frame.width, frame.height
    if preview:
        max_width = int(max_width)
        if 0 < max_width < width:
            height = max(1, round(height * max_width / width))
            width = max_width
        quality = max(1, min(100, int(quality)))
    else:
        quality = FULL_FRAME_QUALITY
    if (width, height) != (frame.width, frame.height):
        image = frame.to_image(width=width, height=height, interpolation="AREA")
    else:
        image = frame.to_image()
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


OnlineProbe = Callable[[], Awaitable[Optional[bool]]]


class CometH264Stream:
    """Background media-websocket reader that always holds the latest frame."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        ssl: Any = None,
        online_probe: Optional[OnlineProbe] = None,
        *,
        name: str = "default",
    ) -> None:
        self.url = url
        self.headers = dict(headers)
        self.ssl = ssl
        self.online_probe = online_probe
        self.name = name

        self.heartbeat_period_s = HEARTBEAT_PERIOD_S
        self.idle_keyframe_s = IDLE_KEYFRAME_S
        self.max_frame_age_s = _env_float("COMET_H264_MAX_FRAME_AGE_S", MAX_FRAME_AGE_S)
        self.frame_timeout_s = _env_float("COMET_H264_FRAME_TIMEOUT_S", FRAME_TIMEOUT_S)
        self.online_poll_s = ONLINE_POLL_S
        self.online_poll_pushed_s = ONLINE_POLL_PUSHED_S
        self.online_max_age_s = ONLINE_MAX_AGE_S
        self.allow_offline = _env_flag("COMET_H264_ALLOW_OFFLINE")
        self.reconnect_backoff_s = RECONNECT_BACKOFF_S
        self.max_decode_backlog = MAX_DECODE_BACKLOG

        self.generation = 0
        self.connected = False
        self.streaming = False
        self.h264_offered = False
        self.last_error: Optional[str] = None
        self.reconnects = 0
        self.frames_decoded = 0
        self.decode_errors = 0
        self.backlog_drops = 0
        self.keyframe_requests = 0
        self.forced_reconnects = 0
        self.session_failures = 0

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._ws: Any = None
        self._stopping = False
        self._missed_heartbeats = 0
        self._last_video_at = 0.0
        self._key_wanted: Optional[asyncio.Event] = None
        self._skip_backoff = False
        self._waiters: list[asyncio.Future] = []

        self._online: Optional[bool] = None
        self._online_at: Optional[float] = None
        self._online_pushed = False

        self._lock = threading.Lock()
        self._latest: Optional[CachedFrame] = None
        self._seq = 0
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._jpeg_cache: tuple[tuple, bytes] | None = None

    # ── lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the reader. Must be called from a running event loop."""
        if self._task is not None and not self._task.done():
            return
        self._loop = asyncio.get_running_loop()
        self._stopping = False
        self._key_wanted = asyncio.Event()
        self._thread = threading.Thread(
            target=self._decode_worker, name=f"comet-h264-decode-{self.name}", daemon=True
        )
        self._thread.start()
        self._task = asyncio.create_task(self._run(), name=f"comet-h264-{self.name}")
        if self.online_probe is not None:
            self._poll_task = asyncio.create_task(
                self._online_poll_loop(), name=f"comet-h264-online-{self.name}"
            )

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def stop(self) -> None:
        self._stopping = True
        tasks = [t for t in (self._task, self._poll_task) if t is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
        self._poll_task = None
        self.connected = False
        self.streaming = False
        thread, self._thread = self._thread, None
        if thread is not None:
            self._queue.put(_STOP)
            await asyncio.to_thread(thread.join, 2.0)
        with self._lock:
            self._latest = None
        self._jpeg_cache = None
        self._notify()

    # ── public capture API ─────────────────────────────────────────

    def note_online(self, online: Optional[bool], pushed: bool = False) -> None:
        """Record ``source.online``. ``pushed`` marks a control-websocket state event."""
        self._online = online
        self._online_at = time.monotonic()
        if pushed:
            self._online_pushed = True
        if online is False:
            self._notify()

    async def get_frame(
        self, timeout: Optional[float] = None, fail_fast: bool = False
    ) -> CachedFrame:
        """Return a current decoded frame or raise ``H264StreamError``.

        ``fail_fast`` gives up at the first failed connection instead of waiting
        out ``timeout`` across reconnects (used to probe for stream support).
        """
        if not self.is_running():
            raise H264StreamError("H.264 capture stream is not running")
        deadline = time.monotonic() + (self.frame_timeout_s if timeout is None else timeout)
        forced = False
        failures_at_start = self.session_failures
        while True:
            online = await self._current_online()
            if online is False and not self.allow_offline:
                raise H264NoSignal(self._no_signal_message())
            now = time.monotonic()
            frame = self._latest
            if frame is not None and self.connected and frame.generation == self.generation:
                age = now - frame.received_at
                if age <= self.max_frame_age_s:
                    return frame
                if not forced:
                    forced = True
                    self._force_reconnect(f"cached frame is {age:.1f}s old")
            remaining = deadline - now
            gave_up = fail_fast and not self.connected and self.session_failures > failures_at_start
            if remaining <= 0 or gave_up or not self.is_running():
                raise H264StreamError(self._timeout_message())
            await self._wait_change(min(remaining, 0.5))

    async def snapshot_jpeg(
        self,
        preview: bool = True,
        max_width: int = 1024,
        quality: int = 60,
        timeout: Optional[float] = None,
        fail_fast: bool = False,
    ) -> bytes:
        frame = await self.get_frame(timeout, fail_fast)
        key = (frame.generation, frame.seq, bool(preview), int(max_width), int(quality))
        cached = self._jpeg_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        data = await asyncio.to_thread(encode_jpeg, frame.frame, preview, max_width, quality)
        self._jpeg_cache = (key, data)
        return data

    def status(self) -> dict[str, Any]:
        frame = self._latest
        current = frame is not None and frame.generation == self.generation
        return {
            "connected": self.connected,
            "streaming": self.streaming,
            "source_online": self._online,
            "width": frame.width if frame is not None else None,
            "height": frame.height if frame is not None else None,
            "frame_age_s": round(time.monotonic() - frame.received_at, 3) if frame is not None else None,
            "frame_current": bool(current and self.connected),
            "frames_decoded": self.frames_decoded,
            "decode_errors": self.decode_errors,
            "backlog_drops": self.backlog_drops,
            "keyframe_requests": self.keyframe_requests,
            "reconnects": self.reconnects,
            "forced_reconnects": self.forced_reconnects,
            "last_error": self.last_error,
        }

    # ── messages ───────────────────────────────────────────────────

    def _no_signal_message(self) -> str:
        frame = self._latest
        detail = ""
        if frame is not None:
            detail = (
                f" The last decoded frame ({frame.width}x{frame.height}) is "
                f"{time.monotonic() - frame.received_at:.1f}s old and was not returned."
            )
        return (
            "no signal: the Comet reports streamer source.online=false, so the "
            "target is sending no video (powered off, asleep, or cable unplugged)."
            + detail
        )

    def _timeout_message(self) -> str:
        if not self.connected:
            state = "the media websocket is not connected"
        elif not self.h264_offered:
            state = "the Comet has not offered an H.264 stream"
        elif self._latest is None or self._latest.generation != self.generation:
            state = "no key frame has arrived on the current connection"
        else:
            state = "the stream delivered no new frame"
        online = {True: "true", False: "false", None: "unknown"}[self._online]
        message = (
            f"capture failed: no current H.264 frame ({state}; source.online={online})."
        )
        if self.last_error:
            message += f" Last stream error: {self.last_error}"
        return message

    # ── event-loop side ────────────────────────────────────────────

    def _notify(self) -> None:
        waiters, self._waiters = self._waiters, []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    async def _wait_change(self, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self._waiters.append(waiter)
        try:
            await asyncio.wait_for(waiter, timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            if waiter in self._waiters:
                self._waiters.remove(waiter)

    def _call_soon(self, callback: Callable[[], None]) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(callback)
        except RuntimeError:
            pass  # the loop closed during shutdown

    def _want_key(self) -> None:
        if self._key_wanted is not None:
            self._key_wanted.set()

    def _force_reconnect(self, reason: str) -> None:
        ws = self._ws
        if ws is None:
            return
        self.forced_reconnects += 1
        self._skip_backoff = True
        self.last_error = f"reconnect forced: {reason}"
        LOG.warning("H.264 stream %s: %s; reconnecting", self.name, reason)
        self.connected = False
        asyncio.ensure_future(self._close_quietly(ws))

    @staticmethod
    async def _close_quietly(ws: Any) -> None:
        try:
            await ws.close()
        except Exception:
            pass

    async def _current_online(self) -> Optional[bool]:
        now = time.monotonic()
        if self._online_at is not None and now - self._online_at <= self.online_max_age_s:
            return self._online
        # A pushed state stands until the Comet pushes a change.
        if self.online_probe is None or self._online_pushed:
            return self._online
        await self._probe_online()
        return self._online

    async def _probe_online(self) -> None:
        try:
            online = await asyncio.wait_for(self.online_probe(), ONLINE_PROBE_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Unknown is not offline: the frame checks decide.
            if not self._online_pushed:
                self._online = None
                self._online_at = time.monotonic()
            return
        self.note_online(online)

    async def _online_poll_loop(self) -> None:
        while not self._stopping:
            await self._probe_online()
            await asyncio.sleep(
                self.online_poll_pushed_s if self._online_pushed else self.online_poll_s
            )

    async def _run(self) -> None:
        failures = 0
        while not self._stopping:
            frames_before = self.frames_decoded
            try:
                await self._session()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._skip_backoff:
                    self.session_failures += 1
                    error = f"{type(exc).__name__}: {exc}"
                    # A Comet that stays unreachable repeats the same error
                    # every backoff period; log it once.
                    level = logging.DEBUG if error == self.last_error else logging.WARNING
                    self.last_error = error
                    LOG.log(level, "H.264 stream %s: %s", self.name, error)
            self._notify()
            if self._stopping:
                return
            self.reconnects += 1
            if self._skip_backoff:
                self._skip_backoff = False
                failures = 0
                continue
            failures = 0 if self.frames_decoded > frames_before else failures + 1
            index = min(failures, len(self.reconnect_backoff_s) - 1)
            await asyncio.sleep(self.reconnect_backoff_s[index])

    async def _session(self) -> None:
        ws = await websockets.connect(
            self.url,
            ssl=self.ssl,
            max_size=WS_MAX_SIZE,
            open_timeout=5.0,
            close_timeout=1.0,
            ping_interval=None,
            additional_headers=self.headers,
        )
        self._ws = ws
        self.generation += 1
        generation = self.generation
        self._queue.put(_RESET)
        self._missed_heartbeats = 0
        self._last_video_at = time.monotonic()
        self._key_wanted.clear()
        self.streaming = False
        self.connected = True
        tasks = {
            asyncio.create_task(self._heartbeat(ws)),
            asyncio.create_task(self._receive(ws, generation)),
        }
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            raise ConnectionError("media websocket closed")
        finally:
            self.connected = False
            self.streaming = False
            self._ws = None
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            await self._close_quietly(ws)

    async def _heartbeat(self, ws: Any) -> None:
        next_ping = time.monotonic()
        while True:
            now = time.monotonic()
            if now >= next_ping:
                self._missed_heartbeats += 1
                if self._missed_heartbeats >= MISSED_HEARTBEAT_LIMIT:
                    raise ConnectionError("media websocket heartbeat lost")
                await ws.send(MSG_HEARTBEAT)
                next_ping = now + self.heartbeat_period_s
                if self.streaming and now - self._last_video_at >= self.idle_keyframe_s:
                    self._key_wanted.set()
            if self._key_wanted.is_set():
                self._key_wanted.clear()
                if self.streaming:
                    await ws.send(MSG_KEY_REQUEST)
                    self.keyframe_requests += 1
            try:
                await asyncio.wait_for(
                    self._key_wanted.wait(), max(0.0, next_ping - time.monotonic())
                )
            except asyncio.TimeoutError:
                pass

    async def _receive(self, ws: Any, generation: int) -> None:
        async for message in ws:
            if isinstance(message, str):
                await self._on_json(ws, message)
                continue
            if not message:
                continue
            kind = message[0]
            if kind == TYPE_HEARTBEAT_ACK:
                self._missed_heartbeats = 0
            elif kind == TYPE_VIDEO and len(message) > 2:
                now = time.monotonic()
                self._last_video_at = now
                if self._queue.qsize() >= self.max_decode_backlog:
                    self._drop_backlog()
                self._queue.put((generation, bool(message[1]), bytes(message[2:]), now))

    async def _on_json(self, ws: Any, message: str) -> None:
        try:
            payload = json.loads(message)
        except ValueError:
            return
        if not isinstance(payload, dict) or payload.get("event_type") != "media":
            return
        event = payload.get("event")
        video = event.get("video") if isinstance(event, dict) else None
        if not isinstance(video, dict) or "h264" not in video:
            self.h264_offered = False
            raise H264StreamError("the Comet media websocket offers no H.264 video")
        self.h264_offered = True
        if not self.streaming:
            await ws.send(json.dumps(
                {"event_type": "start", "event": {"type": "video", "format": "h264"}}
            ))
            self.streaming = True

    def _drop_backlog(self) -> None:
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            dropped += 1
        self.backlog_drops += 1
        self._queue.put(_RESET)
        self._want_key()
        LOG.warning(
            "H.264 stream %s: decoder fell behind; dropped %d frames and requested a key frame",
            self.name, dropped,
        )

    # ── decode thread ──────────────────────────────────────────────

    def _decode_worker(self) -> None:
        decoder = H264Decoder()
        work = self._queue
        while True:
            item = work.get()
            if item is _STOP:
                return
            if item is _RESET:
                decoder.reset()
                continue
            generation, key, data, received_at = item
            if generation != self.generation:
                continue
            try:
                frame = decoder.decode(data, key)
            except Exception as exc:
                self.decode_errors += 1
                self.last_error = f"decode failed: {type(exc).__name__}: {exc}"
                self._call_soon(self._want_key)
                continue
            if frame is None:
                continue
            with self._lock:
                self._seq += 1
                self._latest = CachedFrame(frame, self._seq, generation, received_at, key)
            self.frames_decoded += 1
            self._call_soon(self._notify)
