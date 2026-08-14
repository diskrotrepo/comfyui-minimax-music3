# ComfyUI — MiniMax Music 3 (Generate + Extend/Continuation)

ComfyUI custom nodes for [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3),
with **audio continuation**: extend a song this model generated, seamlessly, across sessions.

Backed by an audio-continuation fork of `diffusers`
([diskrotrepo/diffusers @ `minimax-music3-continuation`](https://github.com/diskrotrepo/diffusers/tree/minimax-music3-continuation))
that adds a `prefix_frame_codes` input / `frame_codes` output to the MiniMax Music 3 autoregressive step.

## Nodes (category `audio/MiniMax Music 3`)

| Node | In | Out |
|---|---|---|
| **Generate** | prompt, lyrics, audio_duration, seed | `AUDIO`, `state` |
| **Extend (continue)** | `state`, audio_duration, seed | `AUDIO`, `state` |
| **Save State** | `state`, filename | writes `<filename>.mm3state`, returns path |
| **Load State** | path | `state` |

`state` (socket `MINIMAX_MM3_STATE`) is a bundle `{frame_codes, prompt, lyrics, seed}`. Extend replays the
codes into the model's KV cache and continues — no audio is re-read. Prompt/lyrics ride with the codes so the
text context can't be mismatched. Save/Load persist a song so you can extend a specific one later.

**Generate 5 songs, extend #2 later:** SaveState the ones you might grow → later LoadState the specific one →
Extend → concatenate the audio. The `.wav` is not enough — you must keep the `.mm3state` (the codes).

## Requirements

- **NVIDIA GPU / CUDA** — mandatory (the autoregressive step needs the LLM + RVQ depth decoder co-resident on CUDA).
  ~24 GB VRAM for the standard path (8 GB+ only with slow CPU offload).
- Installs the diffusers fork + `transformers`, `accelerate`, `soundfile` (see `requirements.txt`).

## Install

**ComfyUI-Manager (recommended):** *Install via Git URL* →
`https://github.com/diskrotrepo/comfyui-minimax-music3` → Restart. Manager installs `requirements.txt` into the
bundled environment.

**Manual (into ComfyUI's environment, e.g. Desktop's bundled venv):**
```
cd <ComfyUI>/custom_nodes
git clone https://github.com/diskrotrepo/comfyui-minimax-music3
python -m pip install -r comfyui-minimax-music3/requirements.txt
```
Requires Git installed for the `git+https` diffusers line. On Windows Desktop, run pip from the app's built-in
terminal so it targets the **bundled venv**, not system Python.

## Model weights

Leave `MINIMAX_MUSIC3_PATH` unset to auto-download `MiniMaxAI/MiniMax-Music3` (~25 GB) from Hugging Face on first
run, or set `MINIMAX_MUSIC3_PATH` to a local diffusers-format folder to load offline.

## Limitation: uploaded audio

You can extend audio **this model generated** (via the `state`). You **cannot** extend an arbitrary **uploaded**
clip — the released weights ship no audio→codes analysis encoder, so a waveform can't be turned into the codes the
model continues from. Approximations (caption-and-regenerate, or training the missing codec) are out of scope here.
