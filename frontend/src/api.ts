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
