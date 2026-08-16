"""MiniMax Music 3 nodes for ComfyUI.

Drives the diffusers `ModularPipeline` for MiniMaxAI/MiniMax-Music3. Requires the
audio-continuation fork installed into ComfyUI's Python:

    pip install -e /path/to/diffusers   # diskrotrepo/diffusers, branch minimax-music3-continuation

Requires a CUDA GPU (the model's autoregressive step needs the language model and
the RVQ depth decoder co-resident on a CUDA device).

Nodes:
  * MiniMaxMusic3Generate  — prompt + lyrics -> AUDIO + STATE.
  * MiniMaxMusic3Extend    — STATE -> more AUDIO that continues it, + new STATE.
  * MiniMaxMusic3SaveState — write a STATE to disk so it can be extended later.
  * MiniMaxMusic3LoadState — read a STATE back from disk.

Generate and Extend both take an optional `stream` toggle. With it on, that stage's audio is pushed to a
browser section by section as it renders — open /mm3/player on this ComfyUI server — so a long Generate ->
Extend chain can be listened to while it is still building. The returned AUDIO and STATE are the same either
way. See mm3_stream.py for the transport and its vocabulary (run / stream / seq / index).

STATE is a self-describing bundle:
    {"frame_codes": LongTensor[frames, 2, num_codebooks],  # the fork's prefix handle
     "prompt": str, "lyrics": str, "seed": int}            # the text it was generated under
The prompt/lyrics travel with the codes because a coherent continuation must replay
the codes under the SAME text context they were generated with.

Typical use: generate several songs, SaveState the ones you may want to grow, then
later LoadState a specific one -> Extend. The saved .wav is NOT usable for this — you
must keep the STATE (the codes), since the release ships no audio->codes encoder.
"""

import os

import torch

# Progressive delivery: the in-memory chunk buffer, the websocket announce and the /mm3 HTTP routes that sit
# behind the `stream` toggle. Importing it registers those routes (see mm3_stream.register_routes).
from .mm3_stream import next_run_token as _next_run_token, streaming_run as _streaming_run

# Cache the pipeline across executions — it is many GB and slow to build.
_PIPE = None
# HF repo id by default; set MINIMAX_MUSIC3_PATH to a local diffusers dir (e.g. a Modal Volume path) to load offline.
_MODEL_ID = os.environ.get("MINIMAX_MUSIC3_PATH", "MiniMaxAI/MiniMax-Music3")
_STATE_EXT = ".mm3state"
# The model emits 25 RVQ frames per second of audio; used to report song positions in the node UI.
_FRAME_RATE = 25.0


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        from diffusers import ModularPipeline

        pipe = ModularPipeline.from_pretrained(_MODEL_ID)
        pipe.load_components(dtype=torch.bfloat16)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MiniMax Music 3 requires a CUDA GPU; no CUDA device is available in this ComfyUI environment."
            )
        pipe.to("cuda")
        _PIPE = pipe
    return _PIPE


def _run(prompt, lyrics, audio_duration, seed, prefix_frame_codes=None, prefix_keep_seconds=0.0,
         cfg_scale=1.5, top_k=50, temperature=1.0, num_inference_steps=30, min_new_seconds=0.0,
         stream=None):
    """Generate (optionally continuing `prefix_frame_codes`). Returns (AUDIO dict, frame_codes LongTensor[cpu]).

    `prefix_keep_seconds` > 0 rewinds the prefix to that timestamp first: only its first N seconds are replayed, so
    generation branches from that point and everything after it is discarded.

    `stream` is an open delivery sink from `mm3_stream` (or None). It only swaps the driver: `stream_song`
    interleaves sampling and rendering and hands each finished window to the sink, then returns the same
    (audios, frame_codes) pair `pipe(...)` would have. Everything the caller sees is identical either way.
    """
    pipe = _get_pipe()
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(int(seed))

    if prefix_frame_codes is not None and prefix_keep_seconds != 0:
        frame_rate = float(getattr(pipe, "frame_rate", _FRAME_RATE))
        keep = int(round(prefix_keep_seconds * frame_rate))
        if prefix_keep_seconds < 0:  # negative = back up that many seconds from the end
            keep += prefix_frame_codes.shape[0]
        keep = max(1, min(keep, prefix_frame_codes.shape[0]))
        prefix_frame_codes = prefix_frame_codes[:keep]

    kwargs = dict(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=float(audio_duration),
        generator=generator,
        cfg_scale=float(cfg_scale),
        top_k=int(top_k),
        temperature=float(temperature),
        num_inference_steps=int(num_inference_steps),
        min_new_seconds=float(min_new_seconds),
    )
    if prefix_frame_codes is not None:
        kwargs["prefix_frame_codes"] = prefix_frame_codes.to(device=device, dtype=torch.long)

    if stream is None:
        # `frame_codes` is the fork's resumable continuation handle (see MiniMaxMusic3SemanticGenerationStep).
        result = pipe(**kwargs, output=["audios", "frame_codes"])

        # Modular pipelines return the requested outputs in order; be tolerant of dict/tuple/list/attr shapes.
        if isinstance(result, dict):
            audios, frame_codes = result["audios"], result["frame_codes"]
        elif isinstance(result, (list, tuple)):
            audios, frame_codes = result[0], result[1]
        else:
            audios, frame_codes = result.audios, result.frame_codes
    else:
        # Same pipeline, same inputs, different driver. `stream_song(pipe, on_audio_chunk, num_inference_steps,
        # generator, **inputs)` (streaming.py:76) filters `**inputs` through `sampling.inputs` (line 168), which
        # declares exactly prompt / lyrics / audio_duration / prefix_frame_codes / cfg_scale / top_k /
        # temperature / min_new_seconds (encoders.py:190-355). It returns (audios, frame_codes) directly, with
        # `audios` already [1, C, samples] - no `output=` selector, so the tail of this function is shared.
        from diffusers.modular_pipelines.minimax_music3.streaming import stream_song

        stream.set_sample_rate(getattr(pipe, "sampling_rate", 44100))
        audios, frame_codes = stream_song(pipe, on_audio_chunk=stream.push, **kwargs)

    # audios: [batch, channels, samples] float in [-1, 1] -> ComfyUI AUDIO wants exactly [B, C, samples].
    # Depending on the diffusers build these come back as torch tensors or numpy arrays.
    if not torch.is_tensor(audios):
        audios = torch.as_tensor(audios)
    if not torch.is_tensor(frame_codes):
        frame_codes = torch.as_tensor(frame_codes)
    waveform = audios.detach().to("cpu", torch.float32)
    if waveform.ndim == 2:  # [C, samples] -> add batch
        waveform = waveform.unsqueeze(0)
    sample_rate = int(getattr(pipe, "sampling_rate", 44100))
    audio = {"waveform": waveform, "sample_rate": sample_rate}
    # Keep codes on CPU: the state bundle stays cheap and serializable; Extend moves them back to the device.
    return audio, frame_codes.detach().to("cpu", torch.long)


def _bundle(frame_codes, prompt, lyrics, seed):
    return {"frame_codes": frame_codes, "prompt": prompt, "lyrics": lyrics, "seed": int(seed)}


def _resolve_dir():
    """Directory for state files: ComfyUI's output dir when available, else the cwd."""
    try:
        import folder_paths

        return folder_paths.get_output_directory()
    except Exception:
        return os.getcwd()


_DEFAULT_PROMPT = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)
_DEFAULT_LYRICS = "[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe"


def _sampling_widgets():
    """Autoregressive sampling knobs shared by Generate and Extend. Defaults match the reference recipe."""
    return {
        "cfg_scale": ("FLOAT", {
            "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.05,
            "tooltip": "Classifier-free guidance. Higher follows the prompt/lyrics harder (including against an "
                       "existing song's momentum when extending) at the cost of diversity/naturalness. 1.0 = off; "
                       "reference recipe = 1.5; try 2-4 to force a style change through.",
        }),
        "top_k": ("INT", {
            "default": 50, "min": 1, "max": 16384,
            "tooltip": "Top-k sampling cutoff for the semantic and residual codes. Lower = safer and more "
                       "repetitive, higher = more adventurous. Reference recipe = 50.",
        }),
        "temperature": ("FLOAT", {
            "default": 1.0, "min": 0.05, "max": 3.0, "step": 0.05,
            "tooltip": "Sampling temperature; higher = more random. Reference recipe = 1.0.",
        }),
        "num_inference_steps": ("INT", {
            "default": 30, "min": 4, "max": 100,
            "tooltip": "Flow-matching render steps per chunk. Fewer = faster rendering at slightly lower audio "
                       "quality (~15 is a usable fast draft). 30 = reference quality.",
        }),
    }


def _stream_widget():
    """The progressive-delivery toggle, shared by Generate and Extend.

    ALWAYS spread this LAST into a node's `optional` dict: widgets are positional in saved workflows, so a new
    widget may only ever be appended (same rule as the sampling widgets on Extend).
    """
    return {
        "stream": ("BOOLEAN", {
            "default": False,
            "tooltip": "Play this stage in the browser while it renders. Open /mm3/player on this ComfyUI "
                       "server and press Play; each finished ~4s section is pushed to it as soon as it "
                       "exists, so a 60s song starts playing about 7s in instead of after the whole render. "
                       "In a Generate -> Extend chain, every stage with this on plays back to back as one "
                       "continuous song. The returned and saved audio is byte-for-byte what you would get "
                       "with this off. Caveat: a streaming stage can never be served from ComfyUI's cache, "
                       "so re-queueing the graph re-runs it and every stage downstream of it.",
        }),
    }


class MiniMaxMusic3Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": _DEFAULT_PROMPT}),
                "lyrics": ("STRING", {"multiline": True, "default": _DEFAULT_LYRICS}),
                "audio_duration": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 360.0, "step": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            # `stream` is appended last so widget indices of workflows saved before it existed still align.
            "optional": {**_sampling_widgets(), **_stream_widget()},
        }

    RETURN_TYPES = ("AUDIO", "MINIMAX_MM3_STATE")
    RETURN_NAMES = ("audio", "state")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "audio/MiniMax Music 3"

    @classmethod
    def IS_CHANGED(cls, stream=False, **kwargs):
        # A streaming run must actually execute: ComfyUI replays a cached `executed` for a byte-identical
        # prompt and never calls the node (execution.py:446-450), so nothing would reach the player. With
        # `stream` off the value is a constant, so caching behaves exactly as before. **kwargs absorbs every
        # other input (including any future widget): a signature mismatch here is caught by execution.py:96-98
        # and turned into NaN, which would silently make the node re-run on every queue.
        return _next_run_token() if stream else False

    def generate(self, prompt, lyrics, audio_duration, seed, cfg_scale=1.5, top_k=50, temperature=1.0,
                 num_inference_steps=30, stream=False):
        with _streaming_run("generate", stream, audio_duration=audio_duration) as sink:
            audio, frame_codes = _run(prompt, lyrics, audio_duration, seed,
                                      cfg_scale=cfg_scale, top_k=top_k, temperature=temperature,
                                      num_inference_steps=num_inference_steps, stream=sink)
        seconds = frame_codes.shape[0] / _FRAME_RATE
        return {
            "ui": {"text": [f"song: {seconds:.1f}s"]},
            "result": (audio, _bundle(frame_codes, prompt, lyrics, seed)),
        }


class MiniMaxMusic3Extend:
    """Continue a prior generation. `audio_duration` is how much NEW audio to add.

    Prompt/lyrics default to the incoming STATE's (matching the text the codes were generated under keeps the seam
    coherent). The optional prompt/lyrics fields override that when non-empty — most usefully a longer lyric sheet
    (the original plus new verses) once the song has sung everything it was given. `seed` controls only the newly
    sampled frames.

    Chaining: connect the previous stage's `audio` to `song_audio` and this node's `audio` output is the FULL song
    (accumulated audio, rewound to `continue_from_seconds` if set, plus the new section); `new_audio` is just the new
    section. Because `song_audio`->`audio` and `state`->`state` line up by type, a BYPASSED (Ctrl+B) Extend node
    passes the song through untouched — so a long chain of Extend stages can have any subset of stages disabled.
    Without `song_audio` connected, `audio` and `new_audio` are both just the new section.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": ("MINIMAX_MM3_STATE",),
                "audio_duration": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 360.0, "step": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            # Leave prompt/lyrics empty to continue under the state's own text (the safe default — mismatched text
            # makes the seam incoherent). Fill them to steer this section, e.g. a LONGER lyric sheet (the original
            # lyrics plus new verses) when the song has already sung everything it was given.
            "optional": {
                "song_audio": ("AUDIO", {
                    "tooltip": "The full song audio so far (previous Generate/Extend `audio` output). When "
                               "connected, this node's `audio` output is the assembled full song instead of just "
                               "the new section.",
                }),
                "continue_from_seconds": ("FLOAT", {
                    "default": 0.0, "min": -360.0, "max": 360.0, "step": 0.5,
                    "tooltip": "0 = continue from the end of the song. NEGATIVE = back up that many seconds from "
                               "the end (e.g. -5 drops the last 5s and continues from there). Positive = absolute: "
                               "keep only the first N seconds of the song and continue from that point.",
                }),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                # Appended after prompt/lyrics so widget indices of workflows saved before these existed still align.
                **_sampling_widgets(),
                "force_continue": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Forbid the song from ending during this stage: the model must keep performing for "
                               "the full audio_duration. Turn on for mid-song stages (the model loves to stop right "
                               "after a chorus); leave off on the final stage so the song can end naturally.",
                }),
                # Appended last, after force_continue, for the same widget-index reason.
                **_stream_widget(),
            },
        }

    RETURN_TYPES = ("AUDIO", "MINIMAX_MM3_STATE", "AUDIO")
    RETURN_NAMES = ("audio", "state", "new_audio")
    FUNCTION = "extend"
    OUTPUT_NODE = True
    CATEGORY = "audio/MiniMax Music 3"

    @classmethod
    def IS_CHANGED(cls, stream=False, **kwargs):
        # See MiniMaxMusic3Generate.IS_CHANGED. Linked inputs (state, song_audio) arrive here as None because
        # IS_CHANGED runs before upstream nodes execute (execution.py:91 passes execution_list=None, and
        # get_input_data:177-179 then marks links missing) - which is why only the `stream` widget is consulted.
        return _next_run_token() if stream else False

    def extend(self, state, audio_duration, seed, song_audio=None, continue_from_seconds=0.0, prompt="", lyrics="",
               cfg_scale=1.5, top_k=50, temperature=1.0, num_inference_steps=30, force_continue=False,
               stream=False):
        # Empty override fields inherit the state's text; non-empty ones replace it for this and later extensions.
        prompt = prompt.strip() or state["prompt"]
        lyrics = lyrics.strip() or state["lyrics"]
        # Each stage of a graph run is its own logical STREAM, so chunk indices (which `stream_song` restarts at
        # 0 on every call) never collide; the announced `seq` is what puts the stages back in order, so a player
        # hears the whole chain as one continuous song. The `with` closes the stream on every exit path below,
        # including the pass-through returns and any raise.
        with _streaming_run("extend", stream, audio_duration=audio_duration,
                            rewind_seconds=continue_from_seconds) as sink:
            try:
                new_audio, frame_codes = _run(
                    prompt, lyrics, audio_duration, seed,
                    prefix_frame_codes=state["frame_codes"],
                    prefix_keep_seconds=continue_from_seconds,
                    cfg_scale=cfg_scale, top_k=top_k, temperature=temperature,
                    num_inference_steps=num_inference_steps,
                    min_new_seconds=float(audio_duration) if force_continue else 0.0,
                    stream=sink,
                )
            except ValueError as e:
                if "zero new audio frames" not in str(e):
                    raise
                if song_audio is not None:
                    # In a stage chain, an already-finished song shouldn't kill the run: pass it through
                    # unchanged and flag it on the node. new_audio becomes a token of silence so downstream
                    # audio nodes stay valid. The stream closes with status "empty" - it never emitted a chunk,
                    # and a player skips straight past it to the next stage.
                    silence = {
                        "waveform": torch.zeros_like(
                            song_audio["waveform"][..., : int(song_audio["sample_rate"] // 10)]),
                        "sample_rate": song_audio["sample_rate"],
                    }
                    return {
                        "ui": {"text": ["song had already ended — stage added nothing "
                                        "(enable force_continue to push past the ending)"]},
                        "result": (song_audio, state, silence),
                    }
                raise ValueError(
                    "This song already reached its natural ending, so there is nothing to continue. "
                    "Rewind into it (negative `continue_from_seconds`, e.g. -5) to branch before the ending, or "
                    "enable `force_continue` to make the model keep performing past it."
                ) from e
            if song_audio is not None:
                old = song_audio["waveform"]
                if continue_from_seconds != 0:
                    # Same clamping as the frame-code trim in `_run`: negative counts back from the end.
                    samples = old.shape[-1]
                    idx = int(round(continue_from_seconds * song_audio["sample_rate"]))
                    if continue_from_seconds < 0:
                        idx += samples
                    old = old[..., : max(1, min(idx, samples))]
                full_audio = {
                    "waveform": torch.cat((old.to(new_audio["waveform"]), new_audio["waveform"]), dim=-1),
                    "sample_rate": new_audio["sample_rate"],
                }
            else:
                full_audio = new_audio
            # New state carries the full accumulated codes (prefix + new) so it can be extended again. The
            # bundle keeps the ORIGINAL generation seed (the song's identity, e.g. for %seed% filenames);
            # `seed` here only controlled the newly sampled frames.
            total = frame_codes.shape[0] / _FRAME_RATE
            section_start = total - new_audio["waveform"].shape[-1] / new_audio["sample_rate"]
            label = f"section {section_start:.1f}s → {total:.1f}s · song is now {total:.1f}s"
            if continue_from_seconds != 0:
                label = f"rewound to {section_start:.1f}s · {label}"
            return {
                "ui": {"text": [label]},
                "result": (full_audio, _bundle(frame_codes, prompt, lyrics, state["seed"]), new_audio),
            }


class MiniMaxMusic3SaveState:
    """Persist a STATE to disk so a specific song can be extended in a later session."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": ("MINIMAX_MM3_STATE",),
                "filename": ("STRING", {"default": "song"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "audio/MiniMax Music 3"

    def save(self, state, filename):
        # `%seed%` in the filename resolves to the state's originating generation seed, so seed-named state files
        # don't need manual renaming between songs.
        filename = filename.replace("%seed%", str(state["seed"]))
        # Absolute path is honored as-is (e.g. a Modal Volume mount); otherwise resolve under the output dir.
        name = filename if filename.endswith(_STATE_EXT) else filename + _STATE_EXT
        path = name if os.path.isabs(name) else os.path.join(_resolve_dir(), name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = _bundle(state["frame_codes"].cpu(), state["prompt"], state["lyrics"], state["seed"])
        torch.save(payload, path)
        frames = int(payload["frame_codes"].shape[0])
        return {"ui": {"text": [f"saved {frames} frames -> {path}"]}, "result": (path,)}


class MiniMaxMusic3LoadState:
    """Load a STATE saved by SaveState. `path` may be absolute or relative to the output dir."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": "song" + _STATE_EXT})}}

    RETURN_TYPES = ("MINIMAX_MM3_STATE",)
    RETURN_NAMES = ("state",)
    FUNCTION = "load"
    CATEGORY = "audio/MiniMax Music 3"

    @classmethod
    def IS_CHANGED(cls, path):
        full = path if os.path.isabs(path) else os.path.join(_resolve_dir(), path)
        return os.path.getmtime(full) if os.path.exists(full) else float("nan")

    def load(self, path):
        full = path if os.path.isabs(path) else os.path.join(_resolve_dir(), path)
        if not os.path.exists(full):
            raise FileNotFoundError(f"MiniMax Music 3 state file not found: {full}")
        payload = torch.load(full, map_location="cpu", weights_only=False)
        return (_bundle(payload["frame_codes"], payload["prompt"], payload["lyrics"], payload["seed"]),)


NODE_CLASS_MAPPINGS = {
    "MiniMaxMusic3Generate": MiniMaxMusic3Generate,
    "MiniMaxMusic3Extend": MiniMaxMusic3Extend,
    "MiniMaxMusic3SaveState": MiniMaxMusic3SaveState,
    "MiniMaxMusic3LoadState": MiniMaxMusic3LoadState,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxMusic3Generate": "MiniMax Music 3 — Generate",
    "MiniMaxMusic3Extend": "MiniMax Music 3 — Extend (continue)",
    "MiniMaxMusic3SaveState": "MiniMax Music 3 — Save State",
    "MiniMaxMusic3LoadState": "MiniMax Music 3 — Load State",
}
