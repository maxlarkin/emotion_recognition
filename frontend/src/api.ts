export type ModelKey = "yunet" | "fer" | "vad" | "ser";

export interface Health {
  status: "ok" | "degraded";
  device: "cpu" | "cuda";
  models: Record<ModelKey, boolean>;
  gpu_name?: string;
  warning?: string;
}

const HEALTH_TIMEOUT_MS = 4000;
const SESSIONS_TIMEOUT_MS = 8000;

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

/** GET a JSON body; aborted by `signal` or after `timeoutMs`. Errors carry the server's `detail`. */
async function getJson(url: string, timeoutMs: number, signal?: AbortSignal): Promise<unknown> {
  const timeout = AbortSignal.timeout(timeoutMs);
  const res = await fetch(url, { signal: signal ? AbortSignal.any([signal, timeout]) : timeout, cache: "no-store" });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function fetchHealth(signal?: AbortSignal): Promise<Health> {
  const body = await getJson("/api/health", HEALTH_TIMEOUT_MS, signal);
  const models = (body as { models?: unknown } | null)?.models;
  if (typeof models !== "object" || models === null || Array.isArray(models)) {
    throw new Error("некорректный ответ сервера");
  }
  return body as Health;
}

export type SessionKind = "live" | "file" | "rtsp";

/** One row of GET /api/sessions. `created_at` is server epoch ms, `duration` is in seconds. */
export interface SessionInfo {
  id: string;
  kind: SessionKind;
  stream_id: string | null;
  created_at: number;
  duration: number;
  status: string;
}

export async function fetchSessions(signal?: AbortSignal): Promise<SessionInfo[]> {
  const body = await getJson("/api/sessions", SESSIONS_TIMEOUT_MS, signal);
  if (!Array.isArray(body) || !body.every((s) => typeof s?.id === "string" && typeof s?.created_at === "number")) {
    throw new Error("некорректный ответ сервера");
  }
  return body as SessionInfo[];
}

export async function deleteSession(id: string): Promise<void> {
  const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`, {
    method: "DELETE",
    signal: AbortSignal.timeout(SESSIONS_TIMEOUT_MS),
  });
  if (!res.ok) throw new Error(await errorText(res));
}

export function exportUrl(id: string, format: "csv" | "json"): string {
  return `/api/sessions/${encodeURIComponent(id)}/export?format=${format}`;
}
