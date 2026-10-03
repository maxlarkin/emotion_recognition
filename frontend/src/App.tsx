import { useEffect, useState } from "react";
import { fetchHealth, type Health, type ModelKey } from "./api";
import LiveView from "./live/LiveView";

const MODEL_LABELS: Record<ModelKey, string> = {
  yunet: "Детектор лиц (YuNet)",
  fer: "Эмоции по лицу (EmotiEffLib)",
  vad: "Детектор речи (Silero VAD)",
  ser: "Эмоции по голосу (GigaAM-Emo)",
};

const WARNINGS: Record<string, string> = {
  cuda_unavailable:
    "Образ собран для GPU, но видеокарта недоступна. Работа идёт на процессоре (CPU).",
};

const UNKNOWN_WARNING = "Сервер сообщил о проблеме. Подробности смотрите в журнале бэкенда.";

const POLL_MS = 5000;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let timer: number | undefined;
    const ctrl = new AbortController();
    const load = async () => {
      try {
        setHealth(await fetchHealth(ctrl.signal));
        setError(null);
      } catch (e) {
        if (ctrl.signal.aborted) return;
        setError(e instanceof Error ? e.message : String(e));
      }
      timer = window.setTimeout(load, POLL_MS);
    };
    load();
    return () => {
      ctrl.abort();
      window.clearTimeout(timer);
    };
  }, []);

  return (
    <main className="page">
      <h1>Распознавание эмоций</h1>
      <div className="layout">
        <LiveView />
        <section className="card health" aria-live="polite">
          <h2>Состояние системы</h2>
          {error && !health && <p className="error">Сервер недоступен ({error}). Повторная попытка…</p>}
          {!error && !health && <p>Загрузка…</p>}
          {health && <HealthView health={health} stale={error !== null} />}
        </section>
      </div>
    </main>
  );
}

function HealthView({ health, stale }: { health: Health; stale: boolean }) {
  const missing = (Object.keys(MODEL_LABELS) as ModelKey[]).filter((k) => !health.models[k]);
  return (
    <>
      {stale && <p className="error">Связь с сервером потеряна, показаны последние данные.</p>}
      <p>
        Устройство: <strong>{health.device === "cuda" ? "GPU (CUDA)" : "CPU"}</strong>
      </p>
      {health.gpu_name && (
        <p>
          Видеокарта: <strong>{health.gpu_name}</strong>
        </p>
      )}
      {health.warning && (
        <p className="warning" role="alert" title={health.warning}>
          ⚠ {WARNINGS[health.warning] ?? UNKNOWN_WARNING}
        </p>
      )}
      <p>
        Статус:{" "}
        <strong className={health.status === "ok" ? "ok" : "error"}>
          {health.status === "ok" ? "всё готово" : "работа ограничена"}
        </strong>
      </p>
      <h3>Модели</h3>
      <ul className="models">
        {(Object.keys(MODEL_LABELS) as ModelKey[]).map((k) => (
          <li key={k} className={health.models[k] ? "ok" : "error"}>
            <span aria-hidden="true">{health.models[k] ? "✓" : "✗"}</span> {MODEL_LABELS[k]}
            {!health.models[k] && " — файл отсутствует"}
          </li>
        ))}
      </ul>
      {missing.length > 0 && (
        <p className="error">
          Отсутствуют модели: {missing.map((k) => MODEL_LABELS[k]).join(", ")}. Пересоберите образ.
        </p>
      )}
    </>
  );
}
