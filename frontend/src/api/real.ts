// Real backend calls (interfaces.md REST). Endpoints the backend does not serve yet are selected
// away in ./index.ts; their implementations here follow the contract so a later story only flips the switch.
import type {
  AppSettings,
  EmotionEvent,
  EventFilters,
  ExportFormat,
  FileSession,
  Health,
  NewStream,
  SessionInfo,
  Stream,
  StreamLogRow,
  StreamPatch,
  StreamTestResult,
  SystemLoad,
  Timeline,
} from "./types";

const HEALTH_TIMEOUT_MS = 4000;
const DEFAULT_TIMEOUT_MS = 8000;

async function errorText(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    const detail = (body as { detail?: unknown } | null)?.detail;
    if (typeof detail === "string") return detail;
  } catch {
    // not JSON
  }
  return `HTTP ${res.status}`;
}

function withTimeout(timeoutMs: number, signal?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

/** GET a JSON body; aborted by `signal` or after `timeoutMs`. Errors carry the server's `detail`. */
async function getJson(url: string, timeoutMs: number, signal?: AbortSignal): Promise<unknown> {
  const res = await fetch(url, { signal: withTimeout(timeoutMs, signal), cache: "no-store" });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

async function sendJson(method: string, url: string, body?: unknown): Promise<unknown> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.status === 204 ? null : res.json();
}

function query(params: object): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  const s = q.toString();
  return s ? `?${s}` : "";
}

const enc = encodeURIComponent;

export async function fetchHealth(signal?: AbortSignal): Promise<Health> {
  const body = await getJson("/api/health", HEALTH_TIMEOUT_MS, signal);
  const models = (body as { models?: unknown } | null)?.models;
  if (typeof models !== "object" || models === null || Array.isArray(models)) {
    throw new Error("некорректный ответ сервера");
  }
  return body as Health;
}

export async function fetchSessions(signal?: AbortSignal): Promise<SessionInfo[]> {
  const body = await getJson("/api/sessions", DEFAULT_TIMEOUT_MS, signal);
  if (!Array.isArray(body) || !body.every((s) => typeof s?.id === "string" && typeof s?.created_at === "number")) {
    throw new Error("некорректный ответ сервера");
  }
  return body as SessionInfo[];
}

export async function deleteSession(id: string): Promise<void> {
  const res = await fetch(`/api/sessions/${enc(id)}`, {
    method: "DELETE",
    signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(await errorText(res));
}

export function exportUrl(id: string, format: ExportFormat): string {
  return `/api/sessions/${enc(id)}/export?format=${format}`;
}

// --- Streams (story 1.7) ---

export const listStreams = (signal?: AbortSignal) =>
  getJson("/api/streams", DEFAULT_TIMEOUT_MS, signal) as Promise<Stream[]>;

export async function createStream(body: NewStream): Promise<Stream> {
  const { id } = (await sendJson("POST", "/api/streams", body)) as { id: string };
  const all = await listStreams();
  const created = all.find((s) => s.id === id);
  if (!created) throw new Error("некорректный ответ сервера");
  return created;
}

export const updateStream = (id: string, patch: StreamPatch) =>
  sendJson("PATCH", `/api/streams/${enc(id)}`, patch) as Promise<Stream>;
export const deleteStream = async (id: string) => void (await sendJson("DELETE", `/api/streams/${enc(id)}`));
export const reconnectStream = async (id: string) => void (await sendJson("POST", `/api/streams/${enc(id)}/reconnect`));
export const testStream = (url: string) => sendJson("POST", "/api/streams/test", { url }) as Promise<StreamTestResult>;
export const fetchStreamLog = (id: string, intervalS: number, limit = 50) =>
  getJson(`/api/streams/${enc(id)}/log${query({ interval_s: intervalS, limit })}`, DEFAULT_TIMEOUT_MS) as Promise<StreamLogRow[]>;
export const fetchSystem = (signal?: AbortSignal) => getJson("/api/system", DEFAULT_TIMEOUT_MS, signal) as Promise<SystemLoad>;

// --- Events and settings (story 1.10) ---

export const listEvents = (f: EventFilters, signal?: AbortSignal) =>
  getJson(`/api/events${query(f)}`, DEFAULT_TIMEOUT_MS, signal) as Promise<EmotionEvent[]>;
export const ackEvent = (id: string, state: "confirmed" | "false_alarm") =>
  sendJson("POST", `/api/events/${enc(id)}/ack`, { state }) as Promise<EmotionEvent>;
export async function exportEvents(f: EventFilters, format: ExportFormat): Promise<{ url: string; filename: string }> {
  const { limit: _limit, before_id: _before, ...filters } = f;
  return { url: `/api/events/export${query({ ...filters, format })}`, filename: `events.${format}` };
}
export const fetchSettings = () => getJson("/api/settings", DEFAULT_TIMEOUT_MS) as Promise<AppSettings>;
export const saveSettings = (s: AppSettings) => sendJson("PUT", "/api/settings", s) as Promise<AppSettings>;

// --- File analysis (story 1.11) ---

type RawFileSession = Omit<SessionInfo, "status"> & Partial<Omit<FileSession, "status">> & { status: string };

function toFileSession(s: RawFileSession): FileSession {
  return {
    id: s.id,
    file_name: s.file_name ?? s.id,
    created_at: s.created_at,
    status: (s.status as FileSession["status"]) ?? "queued",
    progress: s.progress ?? null,
    duration_ms: s.duration_ms ?? (s.duration ? s.duration * 1000 : null),
    resolution: s.resolution ?? null,
    notes: s.notes ?? [],
    error: s.error ?? null,
    usable_pct: s.usable_pct ?? null,
  };
}

export async function listFileSessions(signal?: AbortSignal): Promise<FileSession[]> {
  const list = (await getJson("/api/sessions?kind=file", DEFAULT_TIMEOUT_MS, signal)) as SessionInfo[];
  return list.map(toFileSession);
}

export async function fetchFileSession(id: string): Promise<FileSession> {
  const body = (await getJson(`/api/sessions/${enc(id)}`, DEFAULT_TIMEOUT_MS)) as RawFileSession & { summary?: { usable_pct?: number } };
  return toFileSession({ ...body, usable_pct: body.summary?.usable_pct ?? null });
}

export async function uploadFile(file: File): Promise<FileSession> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/files", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorText(res));
  const { session_id } = (await res.json()) as { session_id: string };
  return fetchFileSession(session_id);
}

export const fetchTimeline = (id: string) => getJson(`/api/sessions/${enc(id)}/timeline`, 30000) as Promise<Timeline>;
export const mediaUrl = (id: string): string | null => `/api/sessions/${enc(id)}/media`;
