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

# Cache the pipeline across executions — it is many GB and slow to build.
_PIPE = None
# HF repo id by default; set MINIMAX_MUSIC3_PATH to a local diffusers dir (e.g. a Modal Volume path) to load offline.
_MODEL_ID = os.environ.get("MINIMAX_MUSIC3_PATH", "MiniMaxAI/MiniMax-Music3")
_STATE_EXT = ".mm3state"


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


def _run(prompt, lyrics, audio_duration, seed, prefix_frame_codes=None, prefix_keep_seconds=0.0):
    """Generate (optionally continuing `prefix_frame_codes`). Returns (AUDIO dict, frame_codes LongTensor[cpu]).

    `prefix_keep_seconds` > 0 rewinds the prefix to that timestamp first: only its first N seconds are replayed, so
    generation branches from that point and everything after it is discarded.
    """
    pipe = _get_pipe()
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(int(seed))

    if prefix_frame_codes is not None and prefix_keep_seconds > 0:
        frame_rate = float(getattr(pipe, "frame_rate", 25.0))
        keep = max(1, int(round(prefix_keep_seconds * frame_rate)))
        prefix_frame_codes = prefix_frame_codes[:keep]

    kwargs = dict(
        prompt=prompt,
        lyrics=lyrics,
        audio_duration=float(audio_duration),
        generator=generator,
        # `frame_codes` is the fork's resumable continuation handle (see MiniMaxMusic3SemanticGenerationStep).
        output=["audios", "frame_codes"],
    )
    if prefix_frame_codes is not None:
        kwargs["prefix_frame_codes"] = prefix_frame_codes.to(device=device, dtype=torch.long)

    result = pipe(**kwargs)

    # Modular pipelines return the requested outputs in order; be tolerant of dict/tuple/list/attr shapes.
    if isinstance(result, dict):
        audios, frame_codes = result["audios"], result["frame_codes"]
    elif isinstance(result, (list, tuple)):
        audios, frame_codes = result[0], result[1]
    else:
        audios, frame_codes = result.audios, result.frame_codes

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


class MiniMaxMusic3Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": _DEFAULT_PROMPT}),
                "lyrics": ("STRING", {"multiline": True, "default": _DEFAULT_LYRICS}),
                "audio_duration": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 360.0, "step": 1.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("AUDIO", "MINIMAX_MM3_STATE")
    RETURN_NAMES = ("audio", "state")
    FUNCTION = "generate"
    CATEGORY = "audio/MiniMax Music 3"

    def generate(self, prompt, lyrics, audio_duration, seed):
        audio, frame_codes = _run(prompt, lyrics, audio_duration, seed)
        return (audio, _bundle(frame_codes, prompt, lyrics, seed))


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
                    "default": 0.0, "min": 0.0, "max": 360.0, "step": 0.5,
                    "tooltip": "0 = continue from the end of the song. > 0 = rewind: keep only the first N seconds "
                               "and continue from there, discarding everything after that point.",
                }),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO", "MINIMAX_MM3_STATE", "AUDIO")
    RETURN_NAMES = ("audio", "state", "new_audio")
    FUNCTION = "extend"
    CATEGORY = "audio/MiniMax Music 3"

    def extend(self, state, audio_duration, seed, song_audio=None, continue_from_seconds=0.0, prompt="", lyrics=""):
        # Empty override fields inherit the state's text; non-empty ones replace it for this and later extensions.
        prompt = prompt.strip() or state["prompt"]
        lyrics = lyrics.strip() or state["lyrics"]
        try:
            new_audio, frame_codes = _run(
                prompt, lyrics, audio_duration, seed,
                prefix_frame_codes=state["frame_codes"],
                prefix_keep_seconds=continue_from_seconds,
            )
        except ValueError as e:
            if "zero new audio frames" in str(e):
                raise ValueError(
                    "This song already reached its natural ending, so there is nothing to continue. "
                    "Either rewind into it (`continue_from_seconds` > 0) and branch from mid-song, or regenerate "
                    "the base song with more lyrics and/or a shorter `audio_duration` so it gets cut off "
                    "mid-performance."
                ) from e
            raise
        if song_audio is not None:
            old = song_audio["waveform"]
            if continue_from_seconds > 0:
                old = old[..., : int(round(continue_from_seconds * song_audio["sample_rate"]))]
            full_audio = {
                "waveform": torch.cat((old.to(new_audio["waveform"]), new_audio["waveform"]), dim=-1),
                "sample_rate": new_audio["sample_rate"],
            }
        else:
            full_audio = new_audio
        # New state carries the full accumulated codes (prefix + new) so it can be extended again. The bundle keeps
        # the ORIGINAL generation seed (the song's identity, e.g. for %seed% filenames); `seed` here only controlled
        # the newly sampled frames.
        return (full_audio, _bundle(frame_codes, prompt, lyrics, state["seed"]), new_audio)


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
