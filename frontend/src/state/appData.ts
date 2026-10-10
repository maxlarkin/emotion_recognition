// App-wide live data: stream list, per-stream feed from /ws/streams (or its mock), system load and
// new incidents. One instance for the whole app; screens read it through the hooks below.
import { api, type EmotionEvent, type Stream, type StreamResult, type StreamState, type SystemLoad } from "../api";
import type { StreamFrame } from "../api/streamsSocket";
import { RollingStats, type StatsView } from "./stats";
import { Store, useStore } from "./store";

export interface StreamFeed {
  result: StreamResult | null;
  resultAt: number;
  frameUrl: string | null;
  stats: StatsView;
}

const EMPTY_STATS: StatsView = { p50: null, p95: null, usablePct: null, medianFacePx: null };
/** No result for this long: the tile treats the stream as having no current faces. */
export const FEED_STALE_MS = 3000;
/** New incidents fetched for the bell; the badge shows «N+» at this count. */
export const INCIDENTS_LIMIT = 100;

export class AppData {
  readonly streamsStore = new Store();
  readonly feedStore = new Store();
  readonly incidentsStore = new Store();

  streams: Stream[] | null = null;
  streamsError: string | null = null;
  system: SystemLoad | null = null;
  newIncidents: EmotionEvent[] = [];
  /** Incidents whose banner the user hid on the stream screen (UI only, ui.md). */
  hidden = new Set<string>();
  private eventListeners = new Set<(e: EmotionEvent) => void>();

  private feeds = new Map<string, StreamFeed>();
  private trackSeen = new Map<string, Map<number, number>>(); // stream → track → first seen (ms)
  private rolling = new Map<string, RollingStats>();
  private clockOffset = 0; // client − server, from system messages
  private disconnect: (() => void) | null = null;
  private refs = 0;
  private streamsSeq = 0;

  start(): void {
    if (this.refs++ > 0) return;
    void this.refreshStreams();
    void this.refreshIncidents();
    this.disconnect = api.connectStreams(
      (m) => this.onMessage(m),
      (f) => this.onFrame(f),
    );
  }

  stop(): void {
    if (--this.refs > 0) return;
    this.disconnect?.();
    this.disconnect = null;
  }

  async refreshStreams(): Promise<void> {
    const seq = ++this.streamsSeq;
    try {
      const list = await api.listStreams();
      if (seq !== this.streamsSeq) return; // a newer refresh is in flight or done
      this.streams = list;
      this.streamsError = null;
    } catch (e) {
      if (seq !== this.streamsSeq) return;
      this.streamsError = e instanceof Error ? e.message : String(e);
    }
    this.streamsStore.emit();
  }

  async refreshIncidents(): Promise<void> {
    try {
      this.newIncidents = await api.listEvents({ kind: "negative", state: "new", limit: INCIDENTS_LIMIT });
      this.incidentsStore.emit();
    } catch {
      // the bell keeps its last state; the journal screen reports errors
    }
  }

  stream(id: string): Stream | undefined {
    return this.streams?.find((s) => s.id === id);
  }

  feed(id: string): StreamFeed {
    return this.feeds.get(id) ?? { result: null, resultAt: 0, frameUrl: null, stats: EMPTY_STATS };
  }

  /** Seconds the track has been in view on this stream. */
  trackAge(streamId: string, trackId: number): number {
    const t = this.trackSeen.get(streamId)?.get(trackId);
    return t === undefined ? 0 : (Date.now() - t) / 1000;
  }

  incidentFor(streamId: string): EmotionEvent | undefined {
    return this.newIncidents.find((e) => e.stream_id === streamId);
  }

  onEvent(l: (e: EmotionEvent) => void): () => void {
    this.eventListeners.add(l);
    return () => this.eventListeners.delete(l);
  }

  async ack(id: string, state: "confirmed" | "false_alarm"): Promise<void> {
    const ev = await api.ackEvent(id, state);
    this.applyEvent(ev);
  }

  hide(id: string): void {
    this.hidden.add(id);
    this.incidentsStore.emit();
  }

  private applyEvent(ev: EmotionEvent): void {
    const without = this.newIncidents.filter((e) => e.event_id !== ev.event_id);
    this.newIncidents = ev.kind === "negative" && ev.state === "new" ? [ev, ...without] : without;
    this.incidentsStore.emit();
    for (const l of this.eventListeners) l(ev);
  }

  private setState(id: string, state: StreamState): void {
    const s = this.stream(id);
    if (!s) {
      void this.refreshStreams(); // a stream we have not listed yet
      return;
    }
    if (s.state === state) return;
    this.streams = this.streams!.map((x) => (x.id === id ? { ...x, state } : x));
    if (state !== "running") {
      this.rolling.get(id)?.clear();
      const f = this.feeds.get(id);
      if (f) this.feeds.set(id, { ...f, result: null, stats: EMPTY_STATS });
    }
    this.streamsStore.emit();
    this.feedStore.emit();
  }

  private onMessage(m: Parameters<Parameters<typeof api.connectStreams>[0]>[0]): void {
    switch (m.type) {
      case "stream_result": {
        const now = Date.now();
        let r = this.rolling.get(m.stream_id);
        if (!r) this.rolling.set(m.stream_id, (r = new RollingStats()));
        r.push(now, now - (m.decode_ts + this.clockOffset), m.faces);
        const seen = this.trackSeen.get(m.stream_id) ?? new Map<number, number>();
        const present = new Map<number, number>();
        for (const f of m.faces) present.set(f.track_id, seen.get(f.track_id) ?? now);
        this.trackSeen.set(m.stream_id, present);
        const prev = this.feed(m.stream_id);
        this.feeds.set(m.stream_id, { ...prev, result: m, resultAt: now, stats: r.view() });
        this.feedStore.emitThrottled();
        break;
      }
      case "stream_status":
        this.setState(m.stream_id, m.state);
        break;
      case "event":
        this.applyEvent(m.event);
        break;
      case "system": {
        const { type: _t, ...load } = m;
        this.clockOffset = Date.now() - load.server_ts;
        this.system = load;
        this.feedStore.emitThrottled();
        break;
      }
    }
  }

  private onFrame(f: StreamFrame): void {
    const prev = this.feed(f.stream_id);
    if (prev.frameUrl) URL.revokeObjectURL(prev.frameUrl);
    const url = URL.createObjectURL(new Blob([f.jpeg as Uint8Array<ArrayBuffer>], { type: "image/jpeg" }));
    this.feeds.set(f.stream_id, { ...prev, frameUrl: url });
  }
}

export const appData = new AppData();

export function useStreams(): { streams: Stream[] | null; error: string | null } {
  useStore(appData.streamsStore);
  return { streams: appData.streams, error: appData.streamsError };
}

export function useFeedVersion(): number {
  return useStore(appData.feedStore);
}

export function useIncidents(): EmotionEvent[] {
  useStore(appData.incidentsStore);
  return appData.newIncidents;
}
