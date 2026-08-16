// MiniMax Music 3 — in-graph streaming status.
//
// Served at /extensions/comfyui-minimax-music3/mm3.js.
//   nodes.py:2286-2289  `if hasattr(module, "WEB_DIRECTORY")` -> EXTENSION_WEB_DIRS[<folder>] = <repo>/web
//   server.py:1243-1244 `web.static('/extensions/' + name, dir)`
//   server.py:356-368   GET /extensions globs **/*.js and returns "/extensions/<name>/<rel>"
//   settingStore bundle: `await import(api.fileURL(e))` per listed file, inside try/catch, so a syntax error
//                        here is logged and does not break the app.
// web.static is not a RouteDef, so it gets no /api twin (server.py:1237) — api.fileURL (no /api prefix) is
// the right resolver for these paths, which is what core itself uses.
//
// =============================================================================================================
// THE ONE VOCABULARY (mm3_stream.py is the authority; player.html and this file must agree word for word)
//   run      one queued prompt.  run_id  == ComfyUI prompt_id.
//   stream   one node execution. stream_id == "{run_id}:{node_id}:{seq}".
//   seq      the stream's ordinal in the run: 0 = Generate, 1..8 = the Extends.
//   index    the chunk's ordinal in its stream, restarting at 0 every stream.
//   chunk id (stream_id, index).
//   events   "mm3.chunk" per chunk; "mm3.stream" with state "start"|"end" per stream. A repeated "start" is
//            an UPDATE, never a reset.
// A NEW stream_id on the same node id is what resets that node's counters — that is how a re-queued graph
// starts from zero instead of accumulating onto the previous run.
// =============================================================================================================
//
// WHY THE LISTENERS ARE REGISTERED AT MODULE TOP LEVEL: verified in the pinned bundle
// (comfyui-frontend-package 1.48.7, static/assets/api--JY_wdaT.js) the JSON switch default is
//   if (this._registered.has(t.type)) super.dispatchEvent(new CustomEvent(t.type,{detail:t.data}));
//   else if (!this.reportedUnknownMessageTypes.has(t.type)) throw (add(t.type), Error(...))
// and `_registered` is only populated by api.addEventListener. The throw is swallowed by the enclosing
// try/catch as console.warn("Unhandled message:", ...), so an unregistered type is dropped silently rather
// than killing the socket. Registering at import time gets us into that set before any prompt runs — and it
// is also why loading this file makes the two "Unhandled message: mm3.*" warnings disappear from the console.
//
// Residual race: if a run is ALREADY in flight when the page reloads, chunks that land before this module
// evaluates are lost. Every counter below is therefore derived from the ABSOLUTE fields the producer sends
// (`index`, `stream_seconds`) and never from "how many messages have I seen".

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT_NAME = "MiniMaxMusic3.Stream";

// Matches NODE_CLASS_MAPPINGS in nodes.py.
const STREAM_NODE_CLASSES = new Set(["MiniMaxMusic3Generate", "MiniMaxMusic3Extend"]);

const STATUS_WIDGET = "mm3_status";
const STATUS_LABEL = "stream";
const PLAYER_WIDGET = "mm3_player";
const PLAYER_LABEL = "▶ Open player";
// Fallback only. The producer sends `player_url` on every event so the route name lives in Python.
const PLAYER_PATH = "/mm3/player";

// nodeId (string) -> {streamId, seq, kind, chunks, ready, expectedChunks, expectedSeconds, playerUrl, ended}
// ended: null | "done" | "empty" | "error" | "stopped"
const streams = new Map();
// Attribution fallback when an event somehow arrives without node_id: the node ComfyUI last reported as
// executing. api--JY_wdaT.js dispatches `executing` with `t.data.display_node || t.data.node`, i.e. a bare id
// string, not the {node, display_node, prompt_id} object the server sends (execution.py:496).
let executingNodeId = null;
// Newest stream seen anywhere, so the button still resolves before a given node's first event.
let latest = { streamId: null, playerUrl: null };

const finiteNum = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** Canonical snake_case contract; camelCase spellings tolerated. */
function normalize(data) {
  if (!data || typeof data !== "object") return null;
  const rawNode = data.node_id ?? data.nodeId ?? null;
  return {
    nodeId: rawNode == null ? null : String(rawNode),
    streamId: data.stream_id ?? data.streamId ?? null,
    runId: data.run_id ?? data.runId ?? null,
    seq: finiteNum(data.seq),
    kind: data.kind ?? null,
    state: data.state ?? null,
    status: data.status ?? null,
    message: data.message ?? null,
    index: finiteNum(data.index),
    chunkCount: finiteNum(data.chunk_count),
    // `stream_seconds` is cumulative for this stream INCLUDING this chunk; `duration` is the stream total on
    // an end event. Both are absolute, so a missed message never skews the readout.
    seconds: finiteNum(data.stream_seconds) ?? finiteNum(data.duration),
    expectedChunks: finiteNum(data.expected_chunks),
    expectedSeconds: finiteNum(data.expected_seconds),
    rewind: finiteNum(data.rewind_seconds),
    playerUrl: data.player_url ?? data.playerUrl ?? null,
  };
}

/**
 * Resolve a ComfyUI execution id to a litegraph node.
 *
 * Exact-id match first, across the visible graph, the root graph and its subgraph definitions — only then
 * fall back to the trailing segment of a "parent:child" subgraph execution id. Matching the tail first would
 * happily return the root-level node that happens to share the child's local id, and paint the wrong node.
 */
function findNode(nodeId) {
  const raw = String(nodeId ?? "");
  if (!raw) return null;

  const graphs = [];
  const push = (g) => { if (g && !graphs.includes(g)) graphs.push(g); };
  push(app?.canvas?.graph);
  push(app?.graph);
  push(app?.rootGraph);
  for (const g of [...graphs]) {
    const subs = g?.subgraphs?.values?.();
    if (subs) for (const sub of subs) push(sub);
  }

  const lookup = (key) => {
    for (const graph of graphs) {
      const nodes = graph?._nodes ?? graph?.nodes;
      if (Array.isArray(nodes)) {
        const hit = nodes.find((n) => String(n.id) === key);
        if (hit) return hit;
      }
      const asNumber = Number(key);
      if (Number.isFinite(asNumber) && graph?.getNodeById) {
        const hit = graph.getNodeById(asNumber);
        if (hit) return hit;
      }
    }
    return null;
  };

  return lookup(raw) ?? (raw.includes(":") ? lookup(raw.split(":").pop()) : null);
}

function playerUrl(state, node) {
  // api.fileURL prepends api_base (a PATH, not an origin), so resolve against location.href: ComfyUI
  // Desktop's window-open handler hands the URL to shell.openExternal, which needs a fully-qualified URL,
  // and an absolute string is harmless in a plain browser.
  const base = state?.playerUrl ?? latest.playerUrl ?? api.fileURL(PLAYER_PATH);
  let url;
  try { url = new URL(base, window.location.href); } catch { return null; }
  const streamId = state?.streamId ?? latest.streamId;
  if (streamId != null) url.searchParams.set("stream", String(streamId));
  if (node?.id != null) url.searchParams.set("node", String(node.id));
  return url.href;
}

function openPlayer(node) {
  const state = node ? streams.get(String(node.id)) : null;
  const url = playerUrl(state, node);
  if (!url) return;
  // In ComfyUI Desktop this does NOT open an in-app tab: the comfy view's setWindowOpenHandler falls through
  // to `shell.openExternal(url); return {action:'deny'}` for anything outside its four hard-coded popup
  // prefixes (Firebase / Stripe checkout / Google / GitHub OAuth), so the player launches in the user's
  // default browser and window.open returns null. Never treat null as failure, and never rely on this click
  // as the autoplay gesture for the player — /mm3/player carries its own Play button for exactly that reason.
  window.open(url, "_blank", "noopener");
}

function statusText(state) {
  if (!state) return "idle";
  const ready = `${state.ready.toFixed(1)}s ready`;
  if (state.ended === "empty") return `added nothing`;
  if (state.ended === "error") return `failed · ${ready}`;
  if (state.ended === "stopped") return `stopped · ${ready}`;
  if (state.ended === "done") return `done · ${ready}`;
  const total = state.expectedChunks > 0 ? `~${state.expectedChunks}` : "?";
  return `${state.chunks} of ${total} sections · ${ready}`;
}

function paint(node) {
  if (!node) return;
  const state = streams.get(String(node.id));
  const status = node.widgets?.find((w) => w.name === STATUS_WIDGET);
  if (status) status.value = statusText(state);
  const button = node.widgets?.find((w) => w.name === PLAYER_WIDGET);
  if (button) button.label = state ? PLAYER_LABEL : `${PLAYER_LABEL} (idle)`;
  node.setDirtyCanvas?.(true);
}

function repaint(nodeId) { paint(findNode(nodeId)); }

function resetAll() {
  const ids = [...streams.keys()];
  streams.clear();
  for (const id of ids) repaint(id);
}

/** Mark every still-open stream terminal. Called when the run ends, however it ends. */
function endAll(reason) {
  for (const [id, state] of streams) {
    if (state.ended) continue;
    state.ended = reason;
    repaint(id);
  }
}

/** Get (creating or resetting as required) the per-node state for an event. */
function stateFor(ev) {
  const key = String(ev.nodeId ?? executingNodeId ?? "");
  if (!key) return null;
  let state = streams.get(key);
  // A new stream_id on the same node id means a fresh execution of that node — restart its counters instead
  // of continuing the previous total.
  if (!state || (ev.streamId != null && state.streamId !== ev.streamId)) {
    state = {
      streamId: ev.streamId ?? null, seq: ev.seq, kind: ev.kind,
      chunks: 0, ready: 0, expectedChunks: 0, expectedSeconds: 0,
      playerUrl: null, ended: null,
    };
    streams.set(key, state);
  }
  if (ev.expectedChunks != null) state.expectedChunks = ev.expectedChunks;
  if (ev.expectedSeconds != null) state.expectedSeconds = ev.expectedSeconds;
  if (ev.seq != null) state.seq = ev.seq;
  if (ev.kind != null) state.kind = ev.kind;
  if (ev.playerUrl != null) state.playerUrl = ev.playerUrl;
  if (state.streamId != null) latest.streamId = state.streamId;
  if (state.playerUrl != null) latest.playerUrl = state.playerUrl;
  return { key, state };
}

function onChunk(event) {
  const ev = normalize(event?.detail);
  if (!ev) return;
  const got = stateFor(ev);
  if (!got) return;
  // Absolute, not incremental: a message missed during page load must not shift the count.
  got.state.chunks = ev.index != null ? ev.index + 1 : got.state.chunks + 1;
  if (ev.seconds != null) got.state.ready = ev.seconds;
  got.state.ended = null;
  repaint(got.key);
}

function onStream(event) {
  const ev = normalize(event?.detail);
  if (!ev) return;
  const got = stateFor(ev);
  if (!got) return;
  if (ev.state === "end") {
    // "ok" | "empty" | "error" straight from the producer; "empty" is the Extend pass-through for a song
    // that had already reached its natural ending.
    got.state.ended = ev.status === "ok" || ev.status == null ? "done" : ev.status;
    if (ev.chunkCount != null) got.state.chunks = ev.chunkCount;
    if (ev.seconds != null) got.state.ready = ev.seconds;
    if (ev.status === "error" && ev.message) console.warn("MiniMax Music 3 stream error:", ev.message);
  }
  repaint(got.key);
}

// --- Registration -------------------------------------------------------------------------------------
// Top-level, not inside setup(): these names must be in api's `_registered` set before the first mm3.*
// message, or the frontend discards it as an unknown message type.
api.addEventListener("mm3.chunk", onChunk);
api.addEventListener("mm3.stream", onStream);

// Attribution fallback. detail is a bare id string (or null between nodes).
api.addEventListener("executing", (e) => {
  const id = e?.detail;
  executingNodeId = id == null ? null : String(id);
});

// A run covers Generate + every Extend, so clear once per run, not per node.
api.addEventListener("execution_start", () => {
  executingNodeId = null;
  resetAll();
});

// Terminal states. Without these a stream that dies mid-run (OOM, interrupt, a producer that never closes)
// sits at "4 of ~15" forever and the user cannot tell live from stalled. These four are dispatched by name
// in the frontend's JSON switch, so no registration trick is needed — and unlike the mm3.* events they are
// addressed to the executing client (execution.py:684), which is this tab.
api.addEventListener("executed", (e) => {
  const d = e?.detail;
  const id = d?.display_node ?? d?.node;
  if (id == null) return;
  const state = streams.get(String(id));
  if (state && !state.ended) { state.ended = "done"; repaint(id); }
});
api.addEventListener("execution_success", () => { executingNodeId = null; endAll("done"); });
api.addEventListener("execution_error", () => { executingNodeId = null; endAll("stopped"); });
api.addEventListener("execution_interrupted", () => { executingNodeId = null; endAll("stopped"); });

app.registerExtension({
  name: EXT_NAME,
  nodeCreated(node) {
    const cls = node?.constructor?.comfyClass ?? node?.comfyClass ?? node?.type;
    if (!STREAM_NODE_CLASSES.has(cls)) return;
    if (node.widgets?.some((w) => w.name === PLAYER_WIDGET)) return;

    // Both widgets are appended at the END and marked serialize:false, and that is load-bearing.
    // litegraph's serializer writes widgets_values at the FULL widget-array index while `continue`-ing past
    // serialize===false widgets, but configure reads back with a COMPACTED counter over serialize!==false
    // widgets:
    //     save: for (const [n, w] of widgets.entries()) { if (w.serialize === false) continue;
    //                                                     widgets_values[n] = w.value }
    //     load: let t = 0; for (const w of widgets) if (w.serialize !== false) w.value = widgets_values[t++]
    // The two indexings agree only while every non-serialized widget sits after every serialized one.
    // addWidget appends, which keeps that true — so the widget-index alignment nodes.py relies on for older
    // saved workflows holds, and a workflow saved WITH these widgets still round-trips.
    //
    // addWidget always returns a widget (it returns addCustomWidget's `toConcreteWidget(...) ?? e`), so there
    // is no null case to guard and no second addWidget call: a `?? node.addWidget(...)` fallback would be
    // dead code that double-adds if it ever fired. "string" is a real widget type here — toConcreteWidget
    // maps both "string" and "text" to TextWidget.
    const status = node.addWidget("string", STATUS_WIDGET, "idle", () => {}, {
      serialize: false,
      canvasOnly: true,
    });
    status.serialize = false;
    if (status.options) status.options.serialize = false;
    // TextWidget draws `label || name` (BaseWidget.displayName), so without this the node reads
    // "mm3_status  3 of ~15 sections".
    status.label = STATUS_LABEL;
    // Genuinely read-only: TextWidget.onClick opens canvas.prompt("Value", ...) and the resulting setValue
    // bumps the graph version (an undo entry and an unsaved-changes marker for a field that carries no
    // state). Shadowing onClick on the instance suppresses the dialog outright; the callback stays as a
    // belt-and-braces snap-back, since setValue assigns this.value before invoking the callback.
    status.onClick = () => {};
    status.callback = () => { status.value = statusText(streams.get(String(node.id))); };

    const button = node.addWidget("button", PLAYER_WIDGET, "", () => openPlayer(node), {
      serialize: false,
      canvasOnly: true,
    });
    button.serialize = false;
    if (button.options) button.options.serialize = false;
    button.label = `${PLAYER_LABEL} (idle)`;

    const onRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
      streams.delete(String(node.id));
      return onRemoved?.apply(this, args);
    };

    // A node created (or a workflow loaded) while a stream for this id already exists should show that
    // stream, not "idle".
    paint(node);
  },
});
