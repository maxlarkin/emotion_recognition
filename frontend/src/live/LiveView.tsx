import { useEffect, useRef, useState } from "react";
import { LatencyWindow, type LatencyStats } from "./latency";
import { containRect, drawOverlay } from "./overlay";
import { encodeFrame, parseServerMessage, type FrameResult } from "./protocol";

const FPS = 8;
const MAX_SIDE = 640;
const JPEG_QUALITY = 0.8;
const MAX_BUFFERED_BYTES = 64 * 1024; // skip a capture while the socket still holds unsent data
const STALE_MS = 1000; // no result for this long: clear the overlay
const STATS_MS = 500;
const RATE_WINDOW_MS = 5000;
const SIZES_MAX = 512; // frame sizes kept while awaiting results
const NOTICE_MS = 4000;

type Phase = "idle" | "starting" | "running";

interface Readout extends LatencyStats {
  rate: number; // results per second over the last 5 s
}

const EMPTY_READOUT: Readout = { count: 0, p50: null, p95: null, rate: 0 };

const nowEpochMs = () => performance.timeOrigin + performance.now();

function cameraError(e: unknown): string {
  const name = e instanceof DOMException || e instanceof Error ? e.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Доступ к камере запрещён. Разрешите его в настройках браузера и нажмите «Старт» снова.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "Камера не найдена. Подключите камеру и попробуйте снова.";
    case "NotReadableError":
    case "AbortError":
      return "Камера занята другим приложением или недоступна.";
    default:
      return `Не удалось включить камеру${name ? ` (${name})` : ""}.`;
  }
}

interface Callbacks {
  onSession: () => void;
  onFatal: (message: string) => void;
  onNotice: (message: string) => void;
  onStats: (r: Readout) => void;
}

/** One live session: socket, capture loop, overlay and latency statistics. */
class LiveRun {
  private ws: WebSocket;
  private running = false;
  private stopped = false;
  private encoding = false;
  private frameId = 0;
  private sizes = new Map<number, { w: number; h: number }>(); // frame_id -> JPEG size
  private latest: { res: FrameResult; w: number; h: number } | null = null;
  private lastResultAt = 0;
  private hasOverlay = false;
  private lastServerError: string | null = null;
  private latency = new LatencyWindow();
  private resultTimes: number[] = [];
  private captureCanvas = document.createElement("canvas");
  private timers: number[] = [];
  private raf = 0;

  constructor(
    private readonly stream: MediaStream,
    private readonly video: HTMLVideoElement,
    private readonly overlay: HTMLCanvasElement,
    private readonly cb: Callbacks,
  ) {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    for (const track of stream.getVideoTracks()) {
      // Fires when the camera is unplugged or access is revoked (not on our own track.stop()).
      track.addEventListener("ended", () => {
        if (!this.stopped) this.cb.onFatal("Камера отключилась. Подключите её и нажмите «Старт» снова.");
      });
    }
    this.ws = new WebSocket(`${proto}//${location.host}/ws/live`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = () => this.ws.send(JSON.stringify({ type: "start" }));
    this.ws.onmessage = (ev) => {
      if (typeof ev.data === "string") this.onMessage(ev.data);
    };
    this.ws.onclose = (ev) => {
      if (this.stopped) return;
      this.cb.onFatal(this.lastServerError ?? `Соединение с сервером закрыто (код ${ev.code}).`);
    };
  }

  private onMessage(text: string): void {
    if (this.stopped) return;
    const msg = parseServerMessage(text);
    if (msg === null) return;
    if (msg.type === "session") {
      this.running = true;
      this.cb.onSession();
      this.timers.push(
        window.setInterval(() => this.capture(), 1000 / FPS),
        window.setInterval(() => this.checkStale(), 200),
        window.setInterval(() => this.publishStats(), STATS_MS),
      );
    } else if (msg.type === "error") {
      this.lastServerError = msg.message;
      if (this.running) this.cb.onNotice(msg.message);
      else this.cb.onFatal(msg.message); // e.g. models still loading: no session
    } else {
      const size = this.sizes.get(msg.frame_id);
      if (!size) return;
      for (const id of this.sizes.keys()) if (id <= msg.frame_id) this.sizes.delete(id);
      if (document.hidden) {
        this.latest = null; // rAF is paused in a hidden tab: its late paint is not a latency sample
        return;
      }
      this.latest = { res: msg, ...size };
      this.lastResultAt = performance.now();
      this.resultTimes.push(this.lastResultAt);
      if (!this.raf) this.raf = requestAnimationFrame(() => this.render());
    }
  }

  private capture(): void {
    const { ws, video } = this;
    if (document.hidden || ws.readyState !== WebSocket.OPEN || this.encoding || ws.bufferedAmount > MAX_BUFFERED_BYTES) return;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || vw === 0 || vh === 0) return;
    const scale = Math.min(1, MAX_SIDE / Math.max(vw, vh));
    const w = Math.round(vw * scale);
    const h = Math.round(vh * scale);
    const c = this.captureCanvas;
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, w, h);
    const captureTs = nowEpochMs();
    this.encoding = true;
    c.toBlob(
      (blob) => {
        if (!blob || this.stopped) {
          this.encoding = false;
          return;
        }
        blob.arrayBuffer().then(
          (buf) => {
            this.encoding = false;
            if (this.stopped || this.ws.readyState !== WebSocket.OPEN) return;
            this.frameId = (this.frameId + 1) >>> 0;
            this.sizes.set(this.frameId, { w, h });
            if (this.sizes.size > SIZES_MAX) this.sizes.delete(this.sizes.keys().next().value!);
            this.ws.send(encodeFrame(captureTs, this.frameId, buf));
          },
          () => {
            this.encoding = false;
          },
        );
      },
      "image/jpeg",
      JPEG_QUALITY,
    );
  }

  /** Fits the overlay canvas to its CSS box; returns the 2D context in CSS pixels. */
  private overlayContext(): { ctx: CanvasRenderingContext2D; cssW: number; cssH: number } | null {
    const cv = this.overlay;
    const cssW = cv.clientWidth;
    const cssH = cv.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.round(cssW * dpr);
    const ph = Math.round(cssH * dpr);
    if (cv.width !== pw || cv.height !== ph) {
      cv.width = pw;
      cv.height = ph;
    }
    const ctx = cv.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, cssW, cssH };
  }

  private render(): void {
    this.raf = 0;
    const latest = this.latest;
    if (this.stopped || !latest) return;
    this.latest = null;
    const oc = this.overlayContext();
    if (!oc) return;
    const shown = containRect(this.video.videoWidth, this.video.videoHeight, oc.cssW, oc.cssH);
    drawOverlay(oc.ctx, oc.cssW, oc.cssH, latest.res.faces, latest.w, latest.h, shown);
    this.hasOverlay = latest.res.faces.length > 0;
    // Full loop: capture -> encode -> WS -> worker -> WS -> this paint.
    if (document.hidden) return;
    const renderTs = nowEpochMs();
    this.latency.add(renderTs, renderTs - latest.res.capture_ts);
  }

  private clearOverlay(): void {
    const oc = this.overlayContext();
    oc?.ctx.clearRect(0, 0, oc.cssW, oc.cssH);
    this.hasOverlay = false;
  }

  private checkStale(): void {
    if (this.hasOverlay && performance.now() - this.lastResultAt > STALE_MS) this.clearOverlay();
  }

  private publishStats(): void {
    const now = performance.now();
    while (this.resultTimes.length && this.resultTimes[0] < now - RATE_WINDOW_MS) this.resultTimes.shift();
    const elapsed = Math.min(RATE_WINDOW_MS, now - (this.resultTimes[0] ?? now)) || RATE_WINDOW_MS;
    const rate = this.resultTimes.length > 1 ? ((this.resultTimes.length - 1) * 1000) / elapsed : 0;
    this.cb.onStats({ ...this.latency.stats(nowEpochMs()), rate });
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.timers.forEach((t) => window.clearInterval(t));
    if (this.raf) cancelAnimationFrame(this.raf);
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop" }));
      this.ws.close(1000);
    } else if (this.ws.readyState === WebSocket.CONNECTING) {
      this.ws.close();
    }
    this.stream.getTracks().forEach((t) => t.stop());
    this.clearOverlay();
  }
}

function formatMs(v: number | null): string {
  return v === null ? "—" : `${Math.round(v)} мс`;
}

export default function LiveView() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const runRef = useRef<LiveRun | null>(null);
  const attemptRef = useRef(0); // bumped by Stop/unmount: a pending camera request is then discarded
  const noticeTimerRef = useRef(0);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [readout, setReadout] = useState<Readout>(EMPTY_READOUT);

  useEffect(
    () => () => {
      attemptRef.current++;
      window.clearTimeout(noticeTimerRef.current);
      runRef.current?.stop();
    },
    [],
  );

  const stop = () => {
    attemptRef.current++;
    window.clearTimeout(noticeTimerRef.current);
    runRef.current?.stop();
    runRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setPhase("idle");
  };

  const start = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || runRef.current) return;
    setError(null);
    setNotice(null);
    setReadout(EMPTY_READOUT);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Браузер не даёт доступа к камере. Откройте приложение по адресу localhost (без HTTPS камера доступна только там).");
      return;
    }
    setPhase("starting");
    const attempt = ++attemptRef.current;
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
    } catch (e) {
      if (attempt !== attemptRef.current) return;
      setError(cameraError(e));
      setPhase("idle");
      return;
    }
    if (attempt !== attemptRef.current) {
      stream.getTracks().forEach((t) => t.stop()); // stopped while the camera was opening
      return;
    }
    video.srcObject = stream;
    try {
      await video.play();
    } catch {
      // autoplay of a muted stream is allowed; a failure here is reported by the capture loop
    }
    if (attempt !== attemptRef.current) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    const run = new LiveRun(stream, video, canvas, {
      onSession: () => setPhase("running"),
      onFatal: (message) => {
        if (runRef.current !== run) return;
        stop();
        setError(message);
      },
      onNotice: (message) => {
        setNotice(message);
        window.clearTimeout(noticeTimerRef.current);
        noticeTimerRef.current = window.setTimeout(() => setNotice(null), NOTICE_MS);
      },
      onStats: setReadout,
    });
    runRef.current = run;
  };

  const p95Class = readout.p95 === null ? "" : readout.p95 < 1000 ? "ok" : "error";

  return (
    <section className="card live" aria-labelledby="live-title">
      <div className="live-head">
        <h2 id="live-title">Анализ с веб-камеры</h2>
        <div className="actions">
          <button type="button" onClick={start} disabled={phase !== "idle"}>
            {phase === "starting" ? "Подключение…" : "Старт"}
          </button>
          <button type="button" className="secondary" onClick={stop} disabled={phase === "idle"}>
            Стоп
          </button>
        </div>
      </div>
      <div className="stage">
        <video ref={videoRef} autoPlay muted playsInline />
        <canvas ref={canvasRef} aria-hidden="true" />
        {phase === "idle" && <p className="stage-hint">Нажмите «Старт», чтобы включить камеру и начать анализ</p>}
      </div>
      <dl className="readout" aria-live="off">
        <div>
          <dt>Задержка p50</dt>
          <dd>{formatMs(readout.p50)}</dd>
        </div>
        <div>
          <dt>Задержка p95</dt>
          <dd className={p95Class}>{formatMs(readout.p95)}</dd>
        </div>
        <div>
          <dt>Замеров за 60 с</dt>
          <dd>{readout.count}</dd>
        </div>
        <div>
          <dt>Ответов в секунду</dt>
          <dd>{readout.rate.toFixed(1)}</dd>
        </div>
      </dl>
      <p className="hint">
        Задержка — от захвата кадра до отрисовки рамки, за последние 60 секунд. Цель: p95 меньше 1000 мс.
      </p>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {notice && !error && (
        <p className="warning" role="status">
          {notice}
        </p>
      )}
    </section>
  );
}
