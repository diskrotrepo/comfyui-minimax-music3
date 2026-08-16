"""Progressive delivery for the MiniMax Music 3 nodes: buffer the chunks, announce them, serve them.

The fork's `stream_song` hands finished ~4 s sections of audio to a callback on ComfyUI's prompt-worker thread
while the rest of the song is still being sampled. This module is the plumbing that gets those sections to a
browser, and nothing more — it never decides *what* to generate.

  * `streaming_run(...)` opens one logical STREAM per node execution and closes it when that execution ends.
  * Each chunk is held in RAM as raw little-endian float32 PLANAR PCM, keyed `(stream_id, index)`.
  * Arrival is announced as a small JSON websocket event. The *bytes* never go over the websocket: `protocol.py`
    defines exactly four binary opcodes (PREVIEW_IMAGE=1, UNENCODED_PREVIEW_IMAGE=2, TEXT=3,
    PREVIEW_IMAGE_WITH_METADATA=4), none of them audio, and the pinned frontend throws on a fifth.
  * `GET /mm3/chunk?stream=..&index=..` serves one chunk's bytes.
    `GET /mm3/streams` lists what is buffered, so a player opened mid-run can catch up.
    `GET /mm3/player` serves the page.

=============================================================================================================
THE ONE VOCABULARY (every file in this pack agrees on these five words)
=============================================================================================================
  run       one queued prompt = one graph execution. `run_id` IS ComfyUI's `prompt_id`.
  stream    one node execution inside that run. `stream_id` == "{run_id}:{node_id}:{seq}".
  seq       the stream's ordinal within the run: 0 for the Generate, 1..8 for the Extends of the Song
            Builder graph. GLOBAL PLAY ORDER IS `(seq, index)` — this is what makes an eight-stage chain
            play as one continuous song.
  index     the chunk's ordinal within its stream. `stream_song` restarts it at 0 on every call, which is
            exactly why a per-stream id exists.
  chunk id  the pair `(stream_id, index)`. Nothing else identifies a chunk.

Runs never overlap in one process: `main.py:538` starts a single `prompt_worker` daemon thread which does
`q.get()` then `e.execute()`, and `execution.py` runs a prompt's nodes one at a time. So stream `seq+1` cannot
open before stream `seq` has closed, and announcements arrive in play order.

=============================================================================================================
WIRE FORMAT — planar float32, little-endian
=============================================================================================================
A chunk's body is `waveform.contiguous().numpy().tobytes()` on the `[channels, samples]` producer tensor: ALL
of channel 0, then ALL of channel 1. `format:"f32le"`, `layout:"planar"`. Planar (not interleaved) because it
costs the producer zero transposes and lets the browser build two `Float32Array` views straight onto the
response ArrayBuffer with no de-interleave loop — 176,640 frames stereo is 1,413,120 bytes, viewed as
`Float32Array(ab, 0, 176640)` and `Float32Array(ab, 706560, 176640)`.

`samples` is FRAMES PER CHANNEL (`waveform.shape[-1]`), never `numel()`. `bytes == samples * channels * 4`;
a consumer that disagrees should trust the body length and say so.

=============================================================================================================
TWO THINGS A CONSUMER MUST GET RIGHT — both fail silently
=============================================================================================================
  * THE PLAYER PAGE MUST BE SERVED FROM `/mm3/player`, never opened off disk. ComfyUI installs an origin-only
    middleware (`server.py:159-166`) whose first act is to return 403 for any request carrying
    `Sec-Fetch-Site: cross-site`. Chromium stamps that on every fetch made from a `file://` page, so a
    double-clicked player.html gets a bare 403 on every chunk — no CORS message, no log line — while the
    websocket still connects and events still arrive, which reads as a buffer bug rather than an origin one.
  * SUBSCRIBE BEFORE SNAPSHOTTING. Open `/ws` first, WITHOUT a clientId (`server.py:274-276` pops any socket
    whose clientId matches, so reusing the main UI's would silently disconnect the ComfyUI tab), and only then
    `GET /mm3/streams`. A chunk pushed between the snapshot being taken under the lock and the socket being
    registered is announced to nobody and listed in no manifest — one silent four-second hole, no error. The
    other order costs nothing: dedupe by `(stream_id, index)`.

Threading: pushes run on the prompt-worker thread, so `send_sync` (`server.py:1392`, a
`loop.call_soon_threadsafe`) flushes while the GPU is busy; reads run on the aiohttp loop thread. One lock
guards the whole store, and every websocket send happens outside it. Bytes are stored BEFORE the announcement
goes out, so a fetch triggered by the announcement can always be served.

Routes are registered at import time, which is custom-node import time: `main.py:521` calls `init_extra_nodes`
before `main.py:535` calls `prompt_server.add_routes()`, so they are picked up. `add_routes` also re-exports
every RouteDef under `/api`, so `/api/mm3/chunk` works too.

Nothing here is allowed to break a generation: a delivery failure logs and is swallowed.
"""

import asyncio
import itertools
import logging
import math
import os
import threading
import time
import urllib.parse
import uuid
from collections import OrderedDict

import torch


# Websocket event names. Two of them, so the main ComfyUI tab logs at most two "Unhandled message" warnings
# when web/mm3.js is not loaded — and none at all when it is, because mm3.js registers both names and the
# frontend only dispatches unknown JSON types that something has registered.
CHUNK_EVENT = "mm3.chunk"
STREAM_EVENT = "mm3.stream"

# One route naming scheme: everything under /mm3/, GET only, chunk addressed by ?stream=&index=.
CHUNK_PATH = "/mm3/chunk"
STREAMS_PATH = "/mm3/streams"
PLAYER_PATH = "/mm3/player"
_PLAYER_FILE = os.path.join("web", "player.html")

WIRE_FORMAT = "f32le"
WIRE_LAYOUT = "planar"

# The checkpoint's rate. Replaced by the pipeline's real value the first time a node calls set_sample_rate.
_DEFAULT_SAMPLE_RATE = 44100
_LAST_SAMPLE_RATE = _DEFAULT_SAMPLE_RATE


def _env_float(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


# Buffer lifecycle. Chunks are freed by whichever of these bites first:
#   1. AGE  — `_RETAIN_SECONDS` after the stream that produced them was closed (a stream closes when the node
#      execution returns or raises). Open streams are never age-freed. A timer enforces this (see
#      `_install_sweeper`); without it the sweep would only ever run when something else touched the store,
#      which for the last run of a session may be never.
#   2. CAP  — total buffered bytes never exceed `_MAX_BUFFER_BYTES`; storing past it drops the OLDEST chunks
#      first, across all streams, including the open one. A player consumes from the front, so the oldest
#      chunk is the one it has already played; a player that never connected loses the head of the song.
#
# 192 MiB is ~9.5 minutes of 44.1 kHz stereo float32. Note this is only ONE of the copies in flight:
# `stream_song` also accumulates every chunk on the vocoder's device and concatenates them
# (streaming.py:147/177), and `_run` then materialises the ComfyUI AUDIO tensor. A full 6-minute song is
# ~127 MB here, ~254 MB in VRAM, and ~127 MB again in the AUDIO dict. Raise this only if scrollback matters
# more than host RAM; the VRAM copies are the binding ones.
_MAX_BUFFER_BYTES = int(_env_float("MINIMAX_MM3_STREAM_BUFFER_MB", 192.0) * 1024 * 1024)
_RETAIN_SECONDS = _env_float("MINIMAX_MM3_STREAM_RETAIN_S", 120.0)
# How many recent run_ids keep their stage bookkeeping (seq counter, cumulative seconds).
_MAX_RUNS_TRACKED = 8
# How many freed stream ids stay tombstoned so a late fetch gets 410 (gone) rather than 404 (not yet).
_MAX_TOMBSTONES = 256


_RUN_TOKEN = itertools.count(1)


def next_run_token():
    """A value that never repeats in this process — what `IS_CHANGED` returns for a streaming node.

    ComfyUI replays a cached `executed` for a byte-identical prompt and never calls the node, so a streamed
    stage would push nothing; the in-tree `websocket_image_save.py` dodges that with `IS_CHANGED = time.time()`.
    A counter is used here instead because Windows' wall clock ticks at ~15.6 ms, so two `time.time()` calls in
    the same tick compare equal — and the output cache lives only as long as the process, so a per-process
    counter is all the uniqueness that is needed.
    """
    return next(_RUN_TOKEN)


def _chunk_geometry():
    """The fork's window geometry, so the chunk-count estimate tracks the fork instead of hard-coding it."""
    try:
        from diffusers.modular_pipelines.minimax_music3.before_denoise import _CHUNK_FRAMES, _CHUNK_HOP

        return int(_CHUNK_FRAMES), int(_CHUNK_HOP)
    except Exception:
        return 200, 100


def estimate_chunk_count(audio_duration, frame_rate=25.0):
    """How many chunks `stream_song` will emit for `audio_duration` seconds of NEW audio.

    `stream_song` renders one chunk per window start (`_window_starts`, streaming.py:51-55) and then one final
    tail chunk on the drain (streaming.py:151-158), so the count is `len(starts) + 1`. This is an ESTIMATE
    only because the language model may end the song early; it is exact when the song runs the full duration.
    Checked against all three measured runs: 60 s -> 15, 16 s -> 4, 12 s -> 3.
    """
    chunk_frames, hop = _chunk_geometry()
    try:
        frames = int(round(float(audio_duration) * float(frame_rate)))
    except (TypeError, ValueError):
        return None
    if frames <= 0:
        return None
    starts = 1 if frames <= chunk_frames else int(math.ceil((frames - hop) / float(hop)))
    return starts + 1


def _server():
    try:
        from server import PromptServer

        return PromptServer.instance
    except Exception:
        return None


def _send(event, payload):
    """Broadcast a JSON event to every connected socket (`sid=None`), including the standalone player page."""
    srv = _server()
    if srv is None:
        return
    try:
        srv.send_sync(event, payload, sid=None)
    except Exception:
        logging.exception("MiniMax Music 3: failed to announce %s", event)


def _executing():
    """(run_id, node_id) for the node currently executing.

    The run is identified by ComfyUI's `prompt_id`. `comfy_execution.utils` publishes both ids in a contextvar
    that `execution.py:305` sets around every synchronous node call, so nothing has to be threaded through
    INPUT_TYPES — which is what keeps saved workflows' widget layout untouched. (A `hidden` UNIQUE_ID would
    also work and would not add a widget, but it does fold the node id into the cache signature at
    `caching.py:117` via `include_unique_id_in_input`; the contextvar costs nothing and changes nothing.)
    """
    run_id = node_id = None
    try:
        from comfy_execution.utils import get_executing_context

        ctx = get_executing_context()
        if ctx is not None:
            run_id, node_id = ctx.prompt_id, ctx.node_id
    except Exception:
        pass
    if run_id is None:  # older cores, or called outside the executor
        run_id = getattr(_server(), "last_prompt_id", None)  # main.py:364 sets this per queue item
    if run_id is None:
        run_id = "local-" + uuid.uuid4().hex[:8]
    if node_id is None:
        node_id = uuid.uuid4().hex[:8]
    return str(run_id), str(node_id)


def stream_sample_rate():
    """Output rate without forcing a model load: whatever the last loaded pipeline reported, else 44100."""
    return int(_LAST_SAMPLE_RATE)


def _pcm_bytes(waveform):
    """`[channels, samples]` float tensor -> (planar little-endian float32 bytes, channels, samples).

    C-order `tobytes()` on a `[channels, samples]` array IS the planar layout: row 0 (all of channel 0) then
    row 1 (all of channel 1). `.contiguous()` matters — `stream_song` hands us a slice of the vocoder output
    (`waveform[..., left:right]`, streaming.py:146), which is a strided view.
    """
    t = waveform.detach().to("cpu", torch.float32)
    if t.ndim == 3 and t.shape[0] == 1:  # tolerate a stray batch dim
        t = t[0]
    if t.ndim == 1:
        t = t.unsqueeze(0)
    t = t.contiguous()
    channels, samples = int(t.shape[0]), int(t.shape[-1])
    # astype("<f4", copy=False) is a no-op on a little-endian host and an explicit statement of the wire
    # format everywhere else.
    return t.numpy().astype("<f4", copy=False).tobytes(), channels, samples


class _Stream:
    """One node execution's worth of audio. Chunk indices are `stream_song`'s, restarting at 0 per stream."""

    def __init__(self, store, run_id, node_id, seq, kind, sample_rate, rewind_seconds, run_offset,
                 expected_chunks, expected_seconds):
        self.store = store
        self.run_id = run_id
        self.node_id = node_id
        self.seq = int(seq)
        # Unique per node EXECUTION: `seq` is a per-run counter, so even a node that somehow runs twice in one
        # prompt gets two streams instead of clobbering itself.
        self.stream_id = "{}:{}:{}".format(run_id, node_id, self.seq)
        self.kind = kind
        self.sample_rate = int(sample_rate)
        self.rewind_seconds = float(rewind_seconds)
        self.run_offset = float(run_offset)
        self.expected_chunks = expected_chunks
        self.expected_seconds = expected_seconds
        self.channels = 0  # 0 = not measured yet; announced as null, never guessed
        self.samples = 0
        self.first_index = 0  # lowest index still buffered; rises as chunks are evicted
        self.next_index = 0  # one past the highest index pushed
        self.chunk_meta = []  # metadata for the chunks still buffered, oldest first
        self.opened_at = time.time()
        self.closed_at = None
        self.status = None

    # -- producer side (prompt-worker thread) --------------------------------------------------------------
    def push(self, waveform, index):
        """`stream_song`'s `on_audio_chunk`. Never raises: delivery must not be able to kill a generation."""
        try:
            self.store._push(self, waveform, index)
        except Exception:
            logging.exception("MiniMax Music 3: dropping streamed chunk %s of %s", index, self.stream_id)

    def set_sample_rate(self, sample_rate):
        """Correct the announced rate once the pipeline is loaded. A no-op in practice (44.1 kHz checkpoint)."""
        try:
            self.store._set_sample_rate(self, int(sample_rate))
        except Exception:
            logging.exception("MiniMax Music 3: could not set stream sample rate")

    def close(self, status="ok", message=None):
        try:
            self.store._close(self, status, message)
        except Exception:
            logging.exception("MiniMax Music 3: could not close stream %s", self.stream_id)


class _ChunkStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._streams = OrderedDict()  # stream_id -> _Stream
        self._chunks = OrderedDict()  # (stream_id, index) -> bytes, in insertion order (oldest first)
        self._bytes = 0
        self._run_seq = OrderedDict()  # run_id -> next stage ordinal
        self._run_seconds = OrderedDict()  # run_id -> seconds of audio streamed by finished stages
        # stream_id -> chunk_count, for streams the gc has freed. A player that went away for longer than the
        # retain window must be told GONE (410), not NOT-YET (404), or it polls a dead URL forever.
        self._gone = OrderedDict()

    # -- open / close --------------------------------------------------------------------------------------
    def open(self, kind, sample_rate=None, rewind_seconds=0.0, expected_chunks=None, expected_seconds=None):
        run_id, node_id = _executing()
        rate = int(sample_rate or stream_sample_rate())
        with self._lock:
            seq = self._run_seq.get(run_id, 0)
            self._run_seq[run_id] = seq + 1
            self._run_seq.move_to_end(run_id)
            while len(self._run_seq) > _MAX_RUNS_TRACKED:
                old, _ = self._run_seq.popitem(last=False)
                self._run_seconds.pop(old, None)
            st = _Stream(self, run_id, node_id, seq, kind, rate, rewind_seconds,
                         self._run_seconds.get(run_id, 0.0), expected_chunks, expected_seconds)
            self._streams[st.stream_id] = st
            payload = self._payload_locked(st, "start", None)
            self._gc_locked()
        _send(STREAM_EVENT, payload)
        return st

    def _close(self, st, status, message):
        with self._lock:
            if st.closed_at is None:
                st.closed_at = time.time()
                # A stage that produced nothing (Extend's "song already ended" pass-through) is reported as
                # "empty", not "ok", so a player can skip it instead of waiting for chunks that never come.
                st.status = "empty" if (status == "ok" and st.next_index == 0) else status
                self._run_seconds[st.run_id] = st.run_offset + st.samples / float(st.sample_rate)
                self._run_seconds.move_to_end(st.run_id)
                while len(self._run_seconds) > _MAX_RUNS_TRACKED:
                    self._run_seconds.popitem(last=False)
            payload = self._payload_locked(st, "end", message)
            self._gc_locked()
        _send(STREAM_EVENT, payload)

    def _set_sample_rate(self, st, sample_rate):
        global _LAST_SAMPLE_RATE
        if sample_rate > 0:
            _LAST_SAMPLE_RATE = int(sample_rate)
        with self._lock:
            if sample_rate == st.sample_rate or st.next_index:
                return
            st.sample_rate = int(sample_rate)
            payload = self._payload_locked(st, "start", None)
        # Re-announce so a player uses the real rate. Consumers must treat a repeated "start" for a stream
        # they already know as an UPDATE, never as a reset.
        _send(STREAM_EVENT, payload)

    # -- push ----------------------------------------------------------------------------------------------
    def _push(self, st, waveform, index):
        data, channels, samples = _pcm_bytes(waveform)
        index = int(index)
        with self._lock:
            if st.stream_id not in self._streams:
                return  # already dropped by the gc; the run is over as far as delivery is concerned
            if channels:
                st.channels = channels
            rate = float(st.sample_rate)
            stream_time = st.samples / rate
            st.samples += samples
            st.next_index = max(st.next_index, index + 1)
            meta = {
                "index": index,
                "samples": samples,
                "bytes": len(data),
                "stream_time": round(stream_time, 6),
                "duration": round(samples / rate, 6),
            }
            st.chunk_meta.append(meta)
            # Stored BEFORE the announcement leaves this method, so the fetch it triggers can always be
            # served. The gc runs after the insert so the cap is never transiently exceeded; the new chunk is
            # last in insertion order, so an eviction round (oldest-first) cannot take it while anything
            # older survives.
            self._chunks[(st.stream_id, index)] = data
            self._bytes += len(data)
            payload = {
                "event": "chunk",
                "run_id": st.run_id,
                "stream_id": st.stream_id,
                "node_id": st.node_id,
                "seq": st.seq,
                "kind": st.kind,
                "index": index,
                "sample_rate": st.sample_rate,
                "channels": st.channels or channels,
                "format": WIRE_FORMAT,
                "layout": WIRE_LAYOUT,
                "samples": samples,  # FRAMES PER CHANNEL
                "bytes": len(data),  # == samples * channels * 4
                "duration": meta["duration"],
                "stream_time": meta["stream_time"],
                "stream_seconds": round(stream_time + samples / rate, 6),
                # Seconds of this RUN before this chunk. Advisory when a stage rewinds
                # (`rewind_seconds` != 0): a rewind trims the assembled song, but a live listen still plays
                # the stages back to back.
                "run_time": round(st.run_offset + stream_time, 6),
                "expected_chunks": st.expected_chunks,
                "expected_seconds": st.expected_seconds,
                "path": CHUNK_PATH,
                "url": "{}?stream={}&index={}".format(
                    CHUNK_PATH, urllib.parse.quote(st.stream_id, safe=""), index
                ),
                "player_url": PLAYER_PATH,
            }
            self._gc_locked()
        _send(CHUNK_EVENT, payload)

    # -- consumer side (aiohttp loop thread) ---------------------------------------------------------------
    def get(self, stream_id, index):
        """-> (bytes, headers, None) or (None, None, (http_status, reason))."""
        with self._lock:
            self._gc_locked()
            st = self._streams.get(stream_id)
            data = self._chunks.get((stream_id, index))
            if data is not None:
                headers = {
                    "X-MM3-Stream": stream_id,
                    "X-MM3-Index": str(index),
                    "X-MM3-Channels": str(st.channels if st is not None and st.channels else 2),
                    "X-MM3-Sample-Rate": str(st.sample_rate if st is not None else _DEFAULT_SAMPLE_RATE),
                    "X-MM3-Format": WIRE_FORMAT,
                    "X-MM3-Layout": WIRE_LAYOUT,
                }
                return data, headers, None
            if st is None:
                count = self._gone.get(stream_id)
                if count is not None:
                    # Existed, ran to completion, then aged out. 410 so a player that was paused or
                    # backgrounded past the retain window stops waiting and re-reads /mm3/streams.
                    return None, None, (410, "stream %s was freed %.0fs after it closed (it had %d chunks); "
                                             "re-read %s for what is still buffered"
                                             % (stream_id, _RETAIN_SECONDS, count, STREAMS_PATH))
                return None, None, (404, "no such stream %s" % stream_id)
            if index >= st.next_index:
                state = "still rendering" if st.closed_at is None else "finished with %d chunks" % st.next_index
                return None, None, (404, "chunk %d not produced yet (stream %s)" % (index, state))
            return None, None, (410,
                                "chunk %d has been freed; earliest still buffered is %d" % (index, st.first_index))

    def snapshot(self):
        """Everything currently fetchable, so a player opened mid-run can catch up without waiting."""
        with self._lock:
            self._gc_locked()
            streams = []
            for st in self._streams.values():
                streams.append({
                    "run_id": st.run_id,
                    "stream_id": st.stream_id,
                    "node_id": st.node_id,
                    "seq": st.seq,
                    "kind": st.kind,
                    "sample_rate": st.sample_rate,
                    "channels": st.channels or None,  # null until a chunk measured it
                    "format": WIRE_FORMAT,
                    "layout": WIRE_LAYOUT,
                    "open": st.closed_at is None,
                    "status": st.status,
                    "rewind_seconds": st.rewind_seconds,
                    "run_time": round(st.run_offset, 6),
                    "expected_chunks": st.expected_chunks,
                    "expected_seconds": st.expected_seconds,
                    "first_index": st.first_index,
                    "chunk_count": st.next_index,
                    "samples": st.samples,
                    "duration": round(st.samples / float(st.sample_rate), 6),
                    "opened_at": st.opened_at,
                    "closed_at": st.closed_at,
                    # Only the chunks still fetchable, oldest first — fetch these to catch up, then follow
                    # the websocket for the rest.
                    "chunks": list(st.chunk_meta),
                })
            streams.sort(key=lambda s: (s["opened_at"], s["seq"]))
            return {
                "now": time.time(),
                "buffered_bytes": self._bytes,
                "buffer_limit_bytes": _MAX_BUFFER_BYTES,
                "retain_seconds": _RETAIN_SECONDS,
                "path": CHUNK_PATH,
                "player_url": PLAYER_PATH,
                "format": WIRE_FORMAT,
                "layout": WIRE_LAYOUT,
                "streams": streams,
            }

    def sweep(self):
        """Age out whatever the retain window has expired. Driven by a timer, not just by traffic."""
        with self._lock:
            self._gc_locked()

    # -- gc ------------------------------------------------------------------------------------------------
    def _gc_locked(self):
        now = time.time()
        for stream_id, st in list(self._streams.items()):
            if st.closed_at is not None and now - st.closed_at > _RETAIN_SECONDS:
                self._drop_stream_locked(stream_id)
        while self._bytes > _MAX_BUFFER_BYTES and self._chunks:
            (stream_id, index), data = self._chunks.popitem(last=False)  # oldest first
            self._bytes -= len(data)
            st = self._streams.get(stream_id)
            if st is not None:
                st.first_index = max(st.first_index, index + 1)
                st.chunk_meta = [m for m in st.chunk_meta if m["index"] >= st.first_index]

    def _drop_stream_locked(self, stream_id):
        st = self._streams.pop(stream_id, None)
        if st is None:
            return
        for index in range(st.first_index, st.next_index):
            data = self._chunks.pop((stream_id, index), None)
            if data is not None:
                self._bytes -= len(data)
        self._gone[stream_id] = st.next_index
        self._gone.move_to_end(stream_id)
        while len(self._gone) > _MAX_TOMBSTONES:
            self._gone.popitem(last=False)

    # -- payloads ------------------------------------------------------------------------------------------
    def _payload_locked(self, st, state, message):
        return {
            "event": "stream",
            "state": state,  # "start" | "end"; a repeat of "start" is an UPDATE, not a reset
            "run_id": st.run_id,
            "stream_id": st.stream_id,
            "node_id": st.node_id,
            "seq": st.seq,  # stage ordinal within the run; play stages in this order
            "kind": st.kind,  # "generate" | "extend"
            "sample_rate": st.sample_rate,
            # null until the first chunk MEASURES it. Never guess 2 here: a consumer that sizes its
            # de-interleave from the start event would silently mis-render a mono or 5.1 vocoder.
            "channels": st.channels or None,
            "format": WIRE_FORMAT,
            "layout": WIRE_LAYOUT,
            "rewind_seconds": st.rewind_seconds,
            "run_time": round(st.run_offset, 6),
            "expected_chunks": st.expected_chunks,
            "expected_seconds": st.expected_seconds,
            "first_index": st.first_index,  # lowest index still buffered; > 0 means chunks were freed
            "chunk_count": st.next_index,  # one past the highest index pushed
            "samples": st.samples,
            "duration": round(st.samples / float(st.sample_rate), 6),
            "status": st.status,  # None while open, then "ok" | "empty" | "error"
            "message": message,
            "path": CHUNK_PATH,
            "player_url": PLAYER_PATH,
        }


STORE = _ChunkStore()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _StreamingRun:
    """Opens a stream for one node execution and closes it however that execution ends."""

    def __init__(self, kind, rewind_seconds, expected_chunks, expected_seconds):
        self.kind = kind
        self.rewind_seconds = rewind_seconds
        self.expected_chunks = expected_chunks
        self.expected_seconds = expected_seconds
        self.stream = None

    def __enter__(self):
        try:
            self.stream = STORE.open(self.kind, rewind_seconds=self.rewind_seconds,
                                     expected_chunks=self.expected_chunks,
                                     expected_seconds=self.expected_seconds)
        except Exception:
            logging.exception("MiniMax Music 3: could not open an audio stream; generating without streaming")
            self.stream = None
        return self.stream

    def __exit__(self, exc_type, exc, tb):
        if self.stream is not None:
            if exc is None:
                self.stream.close("ok")
            else:
                self.stream.close("error", "{}: {}".format(type(exc).__name__, exc))
        return False


def streaming_run(kind, enabled, audio_duration=None, rewind_seconds=0.0):
    """Context manager yielding a stream sink, or None when `enabled` is False (the untouched batch path)."""
    if not enabled:
        return _NullContext()
    expected_seconds = None
    try:
        expected_seconds = round(float(audio_duration), 3) if audio_duration is not None else None
    except (TypeError, ValueError):
        expected_seconds = None
    expected_chunks = estimate_chunk_count(audio_duration) if audio_duration is not None else None
    return _StreamingRun(kind, float(rewind_seconds), expected_chunks, expected_seconds)


# ----------------------------------------------------------------------------------------------------------
# HTTP routes + the retain-window timer
# ----------------------------------------------------------------------------------------------------------
_ROUTES_REGISTERED = False
_SWEEP_KEY = "mm3_stream_sweeper"


def _install_sweeper(srv):
    """Free closed streams on a timer instead of only when something else touches the store.

    Without this the gc is purely lazy: `_close` runs it when the stream's age is 0, so the last run of a
    session would stay fully resident until the next generation or the next HTTP hit — which may be never.
    `srv.app` is built at server.py:248 inside PromptServer.__init__, which main.py runs before
    init_extra_nodes at main.py:521, so the app exists and its on_startup list is still open here.
    """
    interval = max(5.0, _RETAIN_SECONDS / 4.0)

    async def _on_startup(app):
        async def _loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    STORE.sweep()
                except Exception:
                    logging.exception("MiniMax Music 3: chunk-buffer sweep failed")

        app[_SWEEP_KEY] = asyncio.create_task(_loop())

    async def _on_cleanup(app):
        task = app.pop(_SWEEP_KEY, None)
        if task is not None:
            task.cancel()

    srv.app.on_startup.append(_on_startup)
    srv.app.on_cleanup.append(_on_cleanup)


def register_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    srv = _server()
    if srv is None or not hasattr(srv, "routes"):
        logging.warning("MiniMax Music 3: no PromptServer at import time; /mm3 streaming routes not registered.")
        return
    if getattr(srv, "_mm3_routes_registered", False):
        # A second import of this package builds a second STORE, but add_routes() (server.py:1220) already
        # consumed srv.routes at startup and the live handlers still close over the FIRST store. Registering
        # again would leave the nodes pushing where nothing serves — silently, with every fetch 404ing.
        logging.warning("MiniMax Music 3: /mm3 routes were registered by an earlier import of this package; "
                        "the running handlers serve the previous chunk store. Restart ComfyUI for streaming "
                        "to work.")
        _ROUTES_REGISTERED = True
        return

    from aiohttp import web

    routes = srv.routes

    @routes.get(PLAYER_PATH)
    async def mm3_player(request):
        # The page MUST be reached through this route. Opening web/player.html off disk makes Chromium stamp
        # Sec-Fetch-Site: cross-site on its fetches, and server.py:159-166 returns a bare 403 for those — no
        # CORS message, no log line, while the websocket still connects.
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PLAYER_FILE)
        if not os.path.isfile(path):
            return web.Response(status=404, text="{} is missing from the comfyui-minimax-music3 folder".format(
                _PLAYER_FILE))
        # A complete static file, so FileResponse is safe here — unlike ComfyUI's /view, which snapshots
        # Content-Length at request time and would truncate a file that is still growing.
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    @routes.get(STREAMS_PATH)
    async def mm3_streams(request):
        # Catch-up manifest. Subscribe to /ws BEFORE calling this (see the module docstring): a chunk pushed
        # in the gap between this snapshot and your socket registering is announced to nobody. Dedupe by
        # (stream_id, index); anything below a stream's `chunk_count` and at or above its `first_index` that
        # you have not seen is a backfill target.
        return web.json_response(STORE.snapshot(), headers={"Cache-Control": "no-store"})

    @routes.get(CHUNK_PATH)
    async def mm3_chunk(request):
        stream_id = request.rel_url.query.get("stream", "")
        try:
            index = int(request.rel_url.query.get("index", ""))
        except ValueError:
            return web.Response(status=400, text="index must be an integer")
        if not stream_id or index < 0:
            return web.Response(status=400, text="stream and a non-negative index are required")
        data, headers, err = STORE.get(stream_id, index)
        if data is None:
            # 404 = never produced (yet), keep asking. 410 = produced then freed, stop asking and skip
            # forward. A consumer must branch on the status code, not on the English.
            return web.Response(status=err[0], text=err[1])
        headers["Cache-Control"] = "no-store"
        return web.Response(body=data, content_type="application/octet-stream", headers=headers)

    _install_sweeper(srv)
    srv._mm3_routes_registered = True
    _ROUTES_REGISTERED = True
    logging.info("MiniMax Music 3: streaming routes registered at %s, %s, %s (also under /api)",
                 PLAYER_PATH, STREAMS_PATH, CHUNK_PATH)


try:
    register_routes()
except Exception:
    logging.exception("MiniMax Music 3: failed to register streaming routes; the nodes still work without them")
