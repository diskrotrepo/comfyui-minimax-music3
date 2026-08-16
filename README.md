# ComfyUI — MiniMax Music 3 (Generate + Extend/Continuation)

ComfyUI custom nodes for [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3): generate
full songs with vocals from a text description + lyrics, then **extend them** — continue a song where it left
off, rewind into it and branch, steer each new section with fresh prompt/lyrics, and resume any saved song in a
later session. Think of it as a self-hosted, node-graph take on a Suno-style song builder.

Backed by an audio-continuation fork of `diffusers`
([diskrotrepo/diffusers @ `minimax-music3-continuation`](https://github.com/diskrotrepo/diffusers/tree/minimax-music3-continuation)).

---

## Easy install (no technical experience needed)

**What you need**

- A Windows PC with an **NVIDIA graphics card with 24 GB of memory or more** (RTX 3090, 4090, 5090…).
  Without an NVIDIA card the model will not run at all.
- About **40 GB of free disk space** and a decent internet connection (the music model is a one-time ~25 GB
  download).

**Step 1 — Install ComfyUI Desktop.** Download it from [comfy.org/download](https://www.comfy.org/download),
run the installer, and pick the **NVIDIA** option when asked about your hardware. Open the app once so it
finishes setting itself up.

**Step 2 — Install this node pack.** In ComfyUI, open the **Manager** (its button is in the top toolbar).
Choose **"Install via Git URL"**, paste this address, and confirm:

```
https://github.com/diskrotrepo/comfyui-minimax-music3
```

When it finishes, let it **restart ComfyUI** (it will offer to). This also installs the music engine the nodes
need — it can take a few minutes.

**Step 3 — Get the workflow.** Download
[`MM3 - Song Builder (8 extends).json`](example_workflows/MM3%20-%20Song%20Builder%20(8%20extends).json)
from this repository's `example_workflows` folder (open the file on GitHub → the download button at the top
right of the file view). Then simply **drag the downloaded file onto the ComfyUI window**. The song-builder
graph appears.

**Step 4 — Make your first song.** Click the blue **Run** button. The very first run downloads the ~25 GB
music model — this only happens once; later runs start in seconds. When it finishes, press play on the
**PreviewAudio** boxes to listen. Finished songs are also saved as files in ComfyUI's `output/audio/mm3`
folder.

**Step 5 — Make it yours.**

- The big **Generate** box on the left is the start of the song: describe the style in *prompt* (genre, BPM,
  mood, instruments, type of singer) and write the words in *lyrics* — keep section tags like `[verse]` and
  `[chorus]` on their own lines.
- The row of **Extend** boxes each add a new section to the song. Only the first one is active to begin with —
  click a grayed-out one and press **Ctrl+B** to switch it on (or off). Each stage can have its **own**
  prompt, lyrics, length, and seed, so you can build a song section by section.
- Change a stage's *seed* to reroll just that section — earlier sections are remembered and don't regenerate.

**If something breaks:** the most common fixes are ① fully close and reopen ComfyUI, ② open Manager and use
"Install via Git URL" again (safe to repeat), ③ after a ComfyUI app update, repeat step 2 once — updates can
reset the app's Python environment.

---

## Nodes (category `audio/MiniMax Music 3`)

| Node | In | Out |
|---|---|---|
| **Generate** | prompt, lyrics, audio_duration, seed, *(cfg_scale, top_k, temperature, num_inference_steps)* | `AUDIO`, `state` |
| **Extend (continue)** | `state`, audio_duration, seed, *(song_audio, continue_from_seconds, prompt, lyrics, cfg_scale, top_k, temperature, num_inference_steps)* | `AUDIO` (full song), `state`, `new_audio` (new section only) |
| **Save State** | `state`, filename (`%seed%` → the song's original seed) | writes `<filename>.mm3state`, returns path |
| **Load State** | path | `state` |

`state` (socket `MINIMAX_MM3_STATE`) is a bundle `{frame_codes, prompt, lyrics, seed}`. Extend replays the
codes into the model's KV cache and continues — no audio is re-read. The bundle's `seed` stays the **original
generation seed** across extensions (the song's identity, used by `%seed%` filenames).

### Extend specifics

- **`continue_from_seconds`** — 0 continues from the end. **Negative backs up from the end** (`-5` drops the
  last 5 seconds and continues from there — the everyday "that section went sideways" fix, no math needed).
  Positive is absolute: keep only the first N seconds and branch from that point. After each run the node shows
  the section span and total song length right on the node, so absolute positions are always visible.
- **`prompt` / `lyrics`** — empty inherits the state's text (the safe default). Fill to steer the new section.
  Lyrics work best as a **superset** (original lyrics + new verses); the model aligns the sung audio to the
  sheet. Raise `cfg_scale` (try 2–4) to push a style change through against the song's momentum.
- **`song_audio` chaining** — feed the previous stage's `audio` in, and this node's `audio` out is the
  assembled full song. Types line up so a **bypassed (Ctrl+B) Extend passes everything through**, which is what
  makes chains of optional stages work.
- **"Song already ended" error** — a song that reached its natural ending can't be continued; rewind into it
  with `continue_from_seconds`, or give the base generation more lyrics / a shorter `audio_duration` so it gets
  cut off mid-performance.

### Sampling / speed knobs (all default to the reference recipe)

- `cfg_scale` (1.5) — prompt adherence vs. naturalness; the lever for steering extends.
- `top_k` (50), `temperature` (1.0) — sampling variety.
- `num_inference_steps` (30) — flow-matching render steps; ~15 is a usable fast draft.

Continuation prefixes replay at **prefill speed** (batched), so extending even a long song spends seconds, not
minutes, re-reading it.

### Generation speed

On an RTX 5090 a 60s song takes about **53s**. Most of that is the autoregressive stage (around three quarters of
it), which is why `num_inference_steps` is a weaker speed lever than it looks: halving it trims the flow-matching
stage only, worth about 10% of the total.

The flow-matching transformer runs in **fp8**, which is where that stage's ~1.8x came from. Its output is therefore
not bit-identical to a bf16 render — the semantic codes are untouched, so it is the same performance rendered
slightly differently. The autoregressive stage stays in bf16, where fp8 measured slower.

The **first generation after ComfyUI starts is ~12% slower** while the language model, depth decoder and
flow-matching block compile; every generation after that in the same session runs at full speed. Set `TORCHDYNAMO_DISABLE=1` to skip compilation
entirely and run eagerly.

## Requirements

- **NVIDIA GPU / CUDA** — mandatory (the autoregressive step needs the LLM + RVQ depth decoder co-resident on
  CUDA). ~24 GB VRAM for the standard path, plus a preallocated KV cache that scales with song length (~0.6 GB for a
  60s song, ~4 GB for a full six-minute one). 8 GB+ only with slow CPU offload, which skips the compiled fast path.
- Installs the diffusers fork + `transformers`, `accelerate`, `soundfile` (see `requirements.txt`).

## Manual install (into ComfyUI's environment, e.g. Desktop's bundled venv)

```
cd <ComfyUI>/custom_nodes
git clone https://github.com/diskrotrepo/comfyui-minimax-music3
python -m pip install -r comfyui-minimax-music3/requirements.txt
```

Requires Git installed for the `git+https` diffusers line. On Windows Desktop, run pip from the app's built-in
terminal so it targets the **bundled venv**, not system Python.

## Model weights

Leave `MINIMAX_MUSIC3_PATH` unset to auto-download `MiniMaxAI/MiniMax-Music3` (~25 GB) from Hugging Face on
first run, or set `MINIMAX_MUSIC3_PATH` to a local diffusers-format folder to load offline.

## Example workflows

In [`example_workflows/`](example_workflows) — drag a `.json` onto the ComfyUI canvas to load it:

- **MM3 - Song Builder (8 extends)** — the main one: Generate + a chain of 8 optional Extend stages with
  per-stage steering, per-stage previews, and seed-named outputs/state.
- **MM3 - Resume saved song** — load a `.mm3state` from an earlier session and keep growing it.

## Limitation: uploaded audio

You can extend audio **this model generated** (via the `state`). You **cannot** extend an arbitrary
**uploaded** clip — the released weights ship no audio→codes analysis encoder, so a waveform can't be turned
into the codes the model continues from. Approximations (caption-and-regenerate, or training the missing codec)
are out of scope here.
