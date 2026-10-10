// `/streams`: all RTSP tiles with system load (CAP-10), or the empty state (mockup Empty).
import { Link } from "react-router";
import { isMocked, type Stream, type SystemLoad } from "../api";
import { appData, FEED_STALE_MS, useFeedVersion, useStreams } from "../state/appData";
import { useShell } from "../shell/context";
import { DecisionChip, DemoBadge, StatusLabel } from "../ui/controls";
import { EMOTION_LABEL, mb, percent, statusGroup, worstDecision } from "../ui/format";
import { CameraIcon } from "../ui/icons";
import FrameView from "./FrameView";

export default function Grid() {
  const { streams, error } = useStreams();
  useFeedVersion();
  const { setAddStreamOpen } = useShell();

  if (!streams) return <main className="content">{error ? <p className="error-text">Не удалось загрузить потоки ({error})</p> : <p className="empty-note">Загрузка…</p>}</main>;
  if (streams.length === 0) return <EmptyState onAdd={() => setAddStreamOpen(true)} />;

  return (
    <main className="content">
      <div className="page-head">
        <h1>Все потоки</h1>
        <span className="muted">{streams.length}</span>
        {isMocked("streams") && <DemoBadge />}
        <SystemLoadLine load={appData.system} />
      </div>
      <div className="grid">
        {streams.map((s) => (
          <StreamTile key={s.id} stream={s} />
        ))}
      </div>
    </main>
  );
}

export function SystemLoadLine({ load }: { load: SystemLoad | null }) {
  if (!load) return null;
  return (
    <dl className="sysload" aria-label="Загрузка системы">
      <div>
        <dt>CPU</dt>
        <dd>{percent(load.cpu_pct)}</dd>
      </div>
      <div>
        <dt>RAM</dt>
        <dd>
          {mb(load.ram_used_mb)} / {mb(load.ram_total_mb)}
        </dd>
      </div>
      {load.gpu_util_pct !== null && (
        <div>
          <dt>GPU</dt>
          <dd>{percent(load.gpu_util_pct)}</dd>
        </div>
      )}
      {load.vram_used_mb !== null && load.vram_total_mb !== null && (
        <div>
          <dt>VRAM</dt>
          <dd>
            {mb(load.vram_used_mb)} / {mb(load.vram_total_mb)}
          </dd>
        </div>
      )}
    </dl>
  );
}

const ms = (v: number | null) => (v === null ? "—" : `${Math.round(v)}`);

function StreamTile({ stream }: { stream: Stream }) {
  const feed = appData.feed(stream.id);
  const live = statusGroup(stream.state) === "live";
  const fresh = live && feed.result && Date.now() - feed.resultAt < FEED_STALE_MS;
  const faces = fresh ? feed.result!.faces : [];
  const decision = worstDecision(faces.map((f) => f.decision));
  const audio = fresh ? feed.result!.audio : null;
  const voice = !stream.has_audio ? "нет аудио" : !audio ? "—" : audio.speech && audio.label ? EMOTION_LABEL[audio.label] : "нет речи";
  return (
    <Link to={`/streams/${stream.id}`} className="tile-card" aria-label={`${stream.site} · ${stream.name}`}>
      <FrameView frameUrl={live ? feed.frameUrl : null} faces={faces} sourceResolution={stream.source_resolution} compact>
        {!live && (
          <span className="frame-off">
            <StatusLabel state={stream.state} />
          </span>
        )}
      </FrameView>
      <span className="tile-card-body">
        <span className="tile-card-title">
          <span className="tile-card-name">{stream.name}</span>
          {decision && <DecisionChip decision={decision} />}
        </span>
        <span className="tile-card-meta">
          <StatusLabel state={stream.state} />
          <span className="mono">{live && fresh ? `${feed.result!.fps.toFixed(1).replace(".", ",")} к/с` : "—"}</span>
          <span className="mono" title="Задержка p50 / p95, мс">
            {ms(feed.stats.p50)}/{ms(feed.stats.p95)} мс
          </span>
        </span>
        <span className="tile-card-meta">
          <span>{stream.site || "Без точки"}</span>
          <span>Голос: {voice}</span>
        </span>
      </span>
    </Link>
  );
}

function EmptyState({ onAdd }: { onAdd: () => void }) {
  return (
    <main className="content empty-state">
      <div className="empty-icon">
        <CameraIcon />
      </div>
      <h1>Потоков пока нет</h1>
      <p className="secondary-text">Добавьте поток с IP-камеры по ссылке RTSP — распознавание эмоций начнётся сразу после подключения.</p>
      <div className="empty-actions">
        <button type="button" className="btn btn-primary" onClick={onAdd}>
          Добавить поток
        </button>
        <Link to="/analyze" className="btn">
          Проанализировать запись
        </Link>
      </div>
      {isMocked("streams") && <DemoBadge />}
    </main>
  );
}

export { EmptyState };
