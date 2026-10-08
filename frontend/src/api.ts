export type ModelKey = "yunet" | "fer" | "vad" | "ser";

export interface Health {
  status: "ok" | "degraded";
  device: "cpu" | "cuda";
  models: Record<ModelKey, boolean>;
  gpu_name?: string;
  warning?: string;
}

const HEALTH_TIMEOUT_MS = 4000;

export async function fetchHealth(signal?: AbortSignal): Promise<Health> {
  const timeout = AbortSignal.timeout(HEALTH_TIMEOUT_MS);
  const res = await fetch("/api/health", {
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const body: unknown = await res.json();
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

export async function fetchSessions(signal?: AbortSignal): Promise<SessionInfo[]> {
  const timeout = AbortSignal.timeout(SESSIONS_TIMEOUT_MS);
  const res = await fetch("/api/sessions", {
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await errorText(res));
  const body: unknown = await res.json();
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
