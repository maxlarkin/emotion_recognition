// In-memory mock backend with interfaces.md shapes. State lives for the page session.
import { EMOTIONS, type Emotion } from "../../live/protocol";
import {
  DEFAULT_SETTINGS,
  DEFAULT_STREAM_SETTINGS,
  type AppSettings,
  type EmotionEvent,
  type EventFilters,
  type ExportFormat,
  type FileSession,
  type NewStream,
  type Stream,
  type StreamFace,
  type StreamLogRow,
  type StreamPatch,
  type StreamTestResult,
  type StreamsMessage,
  type SystemLoad,
  type Timeline,
} from "../types";
import type { StreamFrame } from "../streamsSocket";
import { DEMO_FILES, DEMO_STREAMS, demoStream } from "./data";
import { applyRule, makeProbs, negativeSum, newRuleState, rng, stepStream, topEmotion, type SimStream } from "./sim";

const LATENCY_MS = 150;
const delay = <T>(value: T, ms = LATENCY_MS): Promise<T> =>
  new Promise((resolve) => setTimeout(() => resolve(structuredClone(value)), ms));
const fail = (message: string, ms = LATENCY_MS): Promise<never> =>
  new Promise((_, reject) => setTimeout(() => reject(new Error(message)), ms));

const rnd = rng(20261010);
let eventSeq = 0;
const newEventId = () => `ev-${Date.now().toString(36)}-${++eventSeq}`;

const state = {
  sims: DEMO_STREAMS.map(
    (d): SimStream => ({ stream: demoStream(d), tracks: [], nextTrack: 1, frameId: 0, speech: false, faceShare: d.faceShare, maxTracks: d.tracks }),
  ),
  settings: structuredClone(DEFAULT_SETTINGS) as AppSettings,
  events: [] as EmotionEvent[],
  files: [] as FileSession[],
  timelines: new Map<string, Timeline>(),
};

/** Seeds the journal with a few past events and one new incident on cam1. */
function seedEvents(now: number) {
  const mk = (i: number, streamId: string, kind: EmotionEvent["kind"], emotion: Emotion, agoS: number, st: EmotionEvent["state"], track: number) => {
    const sim = state.sims.find((s) => s.stream.id === streamId)!;
    return {
      event_id: `ev-seed-${i}`,
      kind,
      session_id: `rtsp-${streamId}`,
      stream_id: streamId,
      camera_type: sim.stream.camera_type,
      track_id: track,
      start_ts: now - agoS * 1000 - 4200,
      emit_ts: now - agoS * 1000,
      duration_ms: 4200 + i * 900,
      emotion,
      group_prob: kind === "negative" ? 0.72 + (i % 3) * 0.07 : 0.81,
      state: st,
    } satisfies EmotionEvent;
  };
  state.events = [
    mk(1, "cam1", "negative", "anger", 40, "new", 12),
    mk(2, "cam5", "positive", "happiness", 260, null, 4),
    mk(3, "cam4", "negative", "sadness", 900, "confirmed", 2),
    mk(4, "cam2", "negative", "anger", 2400, "false_alarm", 7),
    mk(5, "cam6", "positive", "happiness", 3100, null, 1),
    mk(6, "cam1", "negative", "disgust", 5400, "confirmed", 9),
  ];
}

function seedFiles(now: number) {
  state.files = DEMO_FILES.map((f, i) => ({
    id: f.id,
    file_name: f.file_name,
    created_at: now - (i + 1) * 3600_000,
    status: f.status,
    progress: f.status === "done" ? 1 : f.status === "processing" ? 0.42 : null,
    duration_ms: f.duration_s * 1000,
    resolution: f.resolution,
    notes: "notes" in f && f.notes ? [...f.notes] : [],
    error: null,
    usable_pct: f.status === "done" ? 61 + i * 3 : null,
  }));
  for (const f of state.files) if (f.status === "done") buildFileResult(f, "confirmed");
}

/** Generates a file timeline (5 fps) and its events with the same rule as live streams. */
/** `seededState`: files analysed before this page session are shown as already reviewed, so the bell
 * starts with the one seeded live incident instead of every old file incident. */
function buildFileResult(f: FileSession, seededState: "new" | "confirmed" = "new") {
  const r = rng(f.id.length * 7919 + f.file_name.length);
  const dur = f.duration_ms ?? 60_000;
  const [w, h] = (() => {
    const [sw, sh] = f.resolution ?? [1280, 720];
    const k = 640 / Math.max(sw, sh);
    return [Math.round(sw * k), Math.round(sh * k)];
  })();
  const frames: Timeline["frames"] = [];
  const tracks = new Map<number, { x: number; neg: number; target: number; rule: ReturnType<typeof newRuleState>; lead: Emotion; negStart: number | null; sum: number; n: number }>();
  let nextId = 1;
  let gapUntil = 0;
  const cfg = state.settings.events;
  for (let t = 0; t < dur; t += 200) {
    if (t >= gapUntil && r() < 0.0015) gapUntil = t + 8000 + r() * 15000;
    if (t < gapUntil) {
      tracks.clear();
      frames.push({ ts_ms: t, width: w, height: h, faces: [] });
      continue;
    }
    if (tracks.size < 2 && r() < 0.01) {
      tracks.set(nextId++, { x: 0.1 + r() * 0.6, neg: 0.15, target: 0.15, rule: newRuleState(t), lead: (["anger", "sadness", "fear", "disgust"] as Emotion[])[Math.floor(r() * 4)], negStart: null, sum: 0, n: 0 });
    }
    const faces: StreamFace[] = [];
    for (const [id, tr] of tracks) {
      if (r() < 0.006) tr.target = r() < 0.4 ? 0.7 + r() * 0.25 : 0.1 + r() * 0.2;
      tr.neg += (tr.target - tr.neg) * 0.1 + (r() - 0.5) * 0.03;
      const probs = makeProbs(tr.neg, tr.neg < 0.3 && r() < 0.5 ? 0.62 : 0.1, tr.lead);
      const out = applyRule(tr.rule, probs, 90, 48, t, cfg);
      const neg = negativeSum(probs, cfg.negative_emotions);
      if (out.decision === "watch" || out.decision === "alert") {
        tr.negStart ??= t;
        tr.sum += neg;
        tr.n++;
      } else {
        tr.negStart = null;
        tr.sum = 0;
        tr.n = 0;
      }
      if (out.emit) {
        const ev: EmotionEvent = {
          event_id: newEventId(),
          kind: out.emit,
          session_id: f.id,
          stream_id: null,
          camera_type: null,
          track_id: id,
          start_ts: out.emit === "negative" ? (tr.negStart ?? t) : (tr.rule.posSince ?? t),
          emit_ts: t,
          duration_ms: 0,
          emotion: out.emit === "negative" ? tr.lead : "happiness",
          group_prob: out.emit === "negative" ? tr.sum / Math.max(1, tr.n) : probs.happiness,
          state: out.emit === "negative" ? seededState : null,
        };
        ev.duration_ms = 2000 + Math.round(r() * 6000);
        state.events.push(ev);
        if (out.emit === "negative") tr.rule.openNegative = ev.event_id;
      }
      const bw = 0.12 * w;
      faces.push({
        track_id: id,
        box: [Math.round(tr.x * w), Math.round(0.25 * h), Math.round(bw), Math.round(bw * 1.25)],
        face: probs,
        face_label: topEmotion(probs),
        fused: probs,
        fused_label: topEmotion(probs),
        confidence: probs[topEmotion(probs)],
        decision: out.decision,
        face_px: 90,
      });
    }
    frames.push({ ts_ms: t, width: w, height: h, faces });
  }
  const noAudio = f.notes.includes("no_audio_track");
  const audio: Timeline["audio"] = [];
  if (!noAudio) {
    for (let t = 3000; t < dur; t += 1000) {
      const speech = Math.sin(t / 9000) > 0;
      const probs = speech ? { ...makeProbs(0.2, 0.25, "anger"), disgust: 0, fear: 0, surprise: 0 } : makeProbs(0, 0, "anger");
      if (!speech) for (const e of EMOTIONS) probs[e] = 0;
      audio.push({ speech, probs, label: speech ? "neutral" : null, window_end_ts: t });
    }
  }
  state.timelines.set(f.id, { frames, audio });
}

let started = false;
let filesTimer: ReturnType<typeof setInterval> | undefined;
function ensureStarted() {
  if (started) return;
  started = true;
  const now = Date.now();
  seedEvents(now);
  seedFiles(now);
  filesTimer = setInterval(tickFiles, 1000);
}

// --- Streams ---

export async function listStreams(): Promise<Stream[]> {
  ensureStarted();
  return delay(state.sims.map((s) => s.stream));
}

function findSim(id: string): SimStream {
  const sim = state.sims.find((s) => s.stream.id === id);
  if (!sim) throw new Error("Поток не найден");
  return sim;
}

const RTSP_RE = /^rtsps?:\/\/\S+$/i;

export async function createStream(body: NewStream): Promise<Stream> {
  ensureStarted();
  if (!body.name.trim() || !RTSP_RE.test(body.url.trim())) return fail("Укажите название и адрес вида rtsp://…");
  const id = `cam${Date.now().toString(36)}`;
  const stream: Stream = {
    id,
    name: body.name.trim(),
    site: body.site.trim(),
    url: body.url.trim(),
    camera_type: body.camera_type,
    settings: { ...DEFAULT_STREAM_SETTINGS, ...body.settings },
    state: "connecting",
    fps: 0,
    has_audio: body.camera_type !== "security",
    source_resolution: null,
    session_id: null,
  };
  const sim: SimStream = { stream, tracks: [], nextTrack: 1, frameId: 0, speech: false, faceShare: body.camera_type === "webcam" ? 0.2 : 0.05, maxTracks: 2 };
  state.sims.push(sim);
  broadcast({ type: "stream_status", stream_id: id, state: "connecting" });
  setTimeout(() => {
    if (!state.sims.includes(sim) || !stream.settings.enabled) return; // deleted or disabled meanwhile
    stream.state = "running";
    stream.source_resolution = body.camera_type === "webcam" ? [1280, 960] : body.camera_type === "sector" ? [3840, 2160] : [2560, 1440];
    stream.session_id = `rtsp-${id}`;
    broadcast({ type: "stream_status", stream_id: id, state: "running" });
  }, 2500);
  return delay(stream);
}

export async function updateStream(id: string, patch: StreamPatch): Promise<Stream> {
  const sim = findSim(id);
  const { settings, ...rest } = patch;
  Object.assign(sim.stream, rest);
  if (settings) sim.stream.settings = { ...sim.stream.settings, ...settings };
  if (!sim.stream.settings.enabled) {
    sim.stream.state = "stopped";
    sim.tracks = [];
    broadcast({ type: "stream_status", stream_id: id, state: "stopped" });
  } else if (sim.stream.state === "stopped") {
    sim.stream.state = "running";
    sim.stream.source_resolution ??= sim.stream.camera_type === "webcam" ? [1280, 960] : sim.stream.camera_type === "sector" ? [3840, 2160] : [2560, 1440];
    sim.stream.session_id ??= `rtsp-${id}`;
    broadcast({ type: "stream_status", stream_id: id, state: "running" });
  }
  return delay(sim.stream);
}

export async function deleteStream(id: string): Promise<void> {
  findSim(id);
  state.sims = state.sims.filter((s) => s.stream.id !== id);
  return delay(undefined);
}

export async function reconnectStream(id: string): Promise<void> {
  const sim = findSim(id);
  if (!sim.stream.settings.enabled) return fail("Анализ этой камеры выключен");
  sim.stream.state = "reconnecting";
  broadcast({ type: "stream_status", stream_id: id, state: "reconnecting" });
  setTimeout(() => {
    if (sim.stream.state !== "reconnecting") return;
    sim.stream.state = "error";
    broadcast({ type: "stream_status", stream_id: id, state: "error", message: "Источник не отвечает (демо)" });
  }, 3000);
  return delay(undefined);
}

export async function testStream(url: string): Promise<StreamTestResult> {
  const u = url.trim();
  if (!RTSP_RE.test(u)) return delay({ ok: false, error: "Адрес должен начинаться с rtsp://" }, 600);
  if (/unreachable|0\.0\.0\.0/.test(u)) return delay({ ok: false, error: "Нет ответа от камеры за 5 с" }, 1200);
  return delay({ ok: true, resolution: [1920, 1080], has_audio: true }, 900);
}

export async function fetchStreamLog(id: string, intervalS: number, limit = 50): Promise<StreamLogRow[]> {
  const sim = findSim(id);
  if (sim.stream.state !== "running") return delay([]);
  const r = rng(id.length * 31 + intervalS);
  const now = Math.floor(Date.now() / (intervalS * 1000)) * intervalS * 1000;
  const rows: StreamLogRow[] = [];
  for (let i = 0; i < Math.min(limit, 12); i++) {
    const neg = Math.min(0.95, Math.max(0.03, 0.25 + (r() - 0.5) * 0.6 + (i === 0 && id === "cam1" ? 0.6 : 0)));
    const smallFaces = sim.faceShare < 0.02;
    const decision = smallFaces && r() < 0.6 ? "abstain" : neg >= 0.8 ? "alert" : neg >= 0.6 ? "watch" : r() < 0.15 ? "positive" : "none";
    rows.push({
      from_ts: now - (i + 1) * intervalS * 1000,
      to_ts: now - i * intervalS * 1000,
      clients: Math.max(1, sim.maxTracks - (r() < 0.3 ? 1 : 0)),
      dominant_emotion: neg >= 0.6 ? (r() < 0.6 ? "anger" : "sadness") : decision === "positive" ? "happiness" : "neutral",
      negative_prob: neg,
      decision,
    });
  }
  return delay(rows);
}

// --- Events ---

function matches(e: EmotionEvent, f: EventFilters): boolean {
  return (
    (!f.kind || e.kind === f.kind) &&
    (!f.state || e.state === f.state) &&
    (!f.stream_id || e.stream_id === f.stream_id) &&
    (!f.session_id || e.session_id === f.session_id) &&
    (f.from_ms === undefined || e.emit_ts >= f.from_ms) &&
    (f.to_ms === undefined || e.emit_ts <= f.to_ms)
  );
}

function filterEvents(f: EventFilters): EmotionEvent[] {
  const sorted = state.events.filter((e) => matches(e, f)).sort((a, b) => b.emit_ts - a.emit_ts);
  let start = 0;
  if (f.before_id) {
    const i = sorted.findIndex((e) => e.event_id === f.before_id);
    start = i < 0 ? sorted.length : i + 1;
  }
  return sorted.slice(start, start + (f.limit ?? 200));
}

export async function listEvents(f: EventFilters): Promise<EmotionEvent[]> {
  ensureStarted();
  return delay(filterEvents(f));
}

export async function ackEvent(id: string, st: "confirmed" | "false_alarm"): Promise<EmotionEvent> {
  const ev = state.events.find((e) => e.event_id === id);
  if (!ev) return fail("Событие не найдено");
  if (ev.kind !== "negative") return fail("Подтверждать можно только негативные события");
  ev.state = st;
  broadcast({ type: "event", event: structuredClone(ev) });
  return delay(ev);
}

const EVENT_COLUMNS = ["event_id", "kind", "session_id", "stream_id", "camera_type", "track_id", "start_ts", "emit_ts", "duration_ms", "emotion", "group_prob", "state"] as const;

export async function exportEvents(f: EventFilters, format: ExportFormat): Promise<{ url: string; filename: string }> {
  const rows = filterEvents({ ...f, limit: Number.MAX_SAFE_INTEGER, before_id: undefined });
  const text =
    format === "json"
      ? JSON.stringify(rows, null, 2)
      : [EVENT_COLUMNS.join(","), ...rows.map((e) => EVENT_COLUMNS.map((c) => e[c] ?? "").join(","))].join("\n");
  const blob = new Blob([text], { type: format === "json" ? "application/json" : "text/csv" });
  return delay({ url: URL.createObjectURL(blob), filename: `events-demo.${format}` });
}

// --- Settings ---

export async function fetchSettings(): Promise<AppSettings> {
  return delay(state.settings);
}

export async function saveSettings(s: AppSettings): Promise<AppSettings> {
  state.settings = structuredClone(s);
  return delay(state.settings);
}

// --- System ---

let load: SystemLoad = { server_ts: Date.now(), cpu_pct: 42, ram_used_mb: 7400, ram_total_mb: 15800, gpu_util_pct: 55, vram_used_mb: 1350, vram_total_mb: 4096 };
function nextLoad(): SystemLoad {
  const j = (v: number, d: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v + (rnd() - 0.5) * d));
  load = {
    ...load,
    server_ts: Date.now(),
    cpu_pct: j(load.cpu_pct, 6, 20, 90),
    ram_used_mb: j(load.ram_used_mb, 80, 6000, 9000),
    gpu_util_pct: j(load.gpu_util_pct ?? 50, 8, 25, 95),
    vram_used_mb: j(load.vram_used_mb ?? 1300, 20, 1200, 1600),
  };
  return load;
}

export async function fetchSystem(): Promise<SystemLoad> {
  return delay(nextLoad());
}

// --- /ws/streams emitter ---

type Listener = (msg: StreamsMessage) => void;
const listeners = new Set<Listener>();
let wsTimers: number[] = [];

function broadcast(msg: StreamsMessage) {
  for (const l of listeners) l(msg);
}

function startEmitter() {
  ensureStarted();
  wsTimers.push(
    window.setInterval(() => {
      const now = Date.now();
      for (const sim of state.sims) {
        if (sim.stream.state !== "running") continue;
        const { result, events } = stepStream(sim, rnd, now, state.settings.events, newEventId);
        sim.stream.fps = result.fps;
        broadcast(result);
        for (const ev of events) {
          state.events.push(ev);
          broadcast({ type: "event", event: structuredClone(ev) });
        }
      }
    }, 200),
    window.setInterval(() => broadcast({ type: "system", ...nextLoad() }), 1000),
  );
  // initial statuses, like the server does on connect
  setTimeout(() => {
    for (const sim of state.sims) broadcast({ type: "stream_status", stream_id: sim.stream.id, state: sim.stream.state });
  }, 50);
}

/** Mock of `/ws/streams`: same messages, no binary frames (tiles show a placeholder). */
export function connectStreams(onMessage: Listener, _onFrame?: (f: StreamFrame) => void): () => void {
  listeners.add(onMessage);
  if (listeners.size === 1) startEmitter();
  else for (const sim of state.sims) onMessage({ type: "stream_status", stream_id: sim.stream.id, state: sim.stream.state });
  return () => {
    listeners.delete(onMessage);
    if (listeners.size === 0) {
      wsTimers.forEach((t) => window.clearInterval(t));
      wsTimers = [];
    }
  };
}

// --- File analysis ---

function tickFiles() {
  const processing = state.files.find((f) => f.status === "processing");
  if (processing) {
    processing.progress = Math.min(1, (processing.progress ?? 0) + 0.04);
    if (processing.progress >= 1) {
      processing.status = "done";
      processing.usable_pct = 58;
      buildFileResult(processing);
    }
    return;
  }
  const queued = state.files.filter((f) => f.status === "queued").sort((a, b) => a.created_at - b.created_at)[0];
  if (queued) {
    queued.status = "processing";
    queued.progress = 0;
  }
}

export async function listFileSessions(): Promise<FileSession[]> {
  ensureStarted();
  return delay([...state.files].sort((a, b) => b.created_at - a.created_at), 80);
}

export async function fetchFileSession(id: string): Promise<FileSession> {
  ensureStarted();
  const f = state.files.find((x) => x.id === id);
  return f ? delay(f, 80) : fail("Сессия не найдена");
}

export const ACCEPTED_EXT = ["mp4", "webm", "mkv", "avi"];

export async function uploadFile(file: File): Promise<FileSession> {
  ensureStarted();
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!ACCEPTED_EXT.includes(ext)) return fail("Неподдерживаемый формат", 300);
  const f: FileSession = {
    id: `file-${Date.now().toString(36)}-${Math.floor(rnd() * 1e6)}`,
    file_name: file.name,
    created_at: Date.now(),
    status: "queued",
    progress: null,
    duration_ms: 60_000 + Math.round(rnd() * 240_000),
    resolution: [1920, 1080],
    notes: [],
    error: null,
    usable_pct: null,
  };
  state.files.push(f);
  return delay(f, 400);
}

export async function fetchTimeline(id: string): Promise<Timeline> {
  const t = state.timelines.get(id);
  return t ? delay(t, 200) : fail("Результат ещё не готов");
}

export const mediaUrl = (): string | null => null; // mock files have no media: the player runs a virtual clock

export function exportUrl(id: string, format: ExportFormat): string {
  const t = state.timelines.get(id);
  const rows = (t?.frames ?? []).flatMap((fr) => fr.faces.map((f) => ({ ts_ms: fr.ts_ms, track_id: f.track_id, fused_label: f.fused_label })));
  const text = format === "json" ? JSON.stringify(rows) : ["ts_ms,track_id,fused_label", ...rows.map((r) => `${r.ts_ms},${r.track_id},${r.fused_label}`)].join("\n");
  return URL.createObjectURL(new Blob([text], { type: format === "json" ? "application/json" : "text/csv" }));
}

/** Test hook: resets the mock to its seeded state. */
export function __resetMock() {
  clearInterval(filesTimer);
  started = false;
  state.sims = DEMO_STREAMS.map((d) => ({ stream: demoStream(d), tracks: [], nextTrack: 1, frameId: 0, speech: false, faceShare: d.faceShare, maxTracks: d.tracks }));
  state.settings = structuredClone(DEFAULT_SETTINGS);
  state.events = [];
  state.files = [];
  state.timelines.clear();
  seedEvents(Date.now());
}
