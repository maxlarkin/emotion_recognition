import { beforeEach, describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS } from "../types";
import { mergeEvent } from "../../history/EventsJournal";
import { AppData } from "../../state/appData";
import { decodeStreamFrame, STREAM_FRAME } from "../streamsSocket";
import * as mock from "./index";
import { applyRule, makeProbs, newRuleState } from "./sim";

const cfg = DEFAULT_SETTINGS.events;

describe("mock event rule (stack.md)", () => {
  it("abstains on small faces and young tracks", () => {
    const st = newRuleState(0);
    expect(applyRule(st, makeProbs(0.9, 0, "anger"), 30, 48, 5000, cfg).decision).toBe("abstain");
    expect(applyRule(newRuleState(0), makeProbs(0.9, 0, "anger"), 90, 48, 500, cfg).decision).toBe("abstain");
  });

  it("watches, then raises one negative event after the minimum duration, then honours the cooldown", () => {
    const st = newRuleState(0);
    const neg = makeProbs(0.8, 0, "anger");
    expect(applyRule(st, neg, 90, 48, 1000, cfg)).toMatchObject({ decision: "watch", emit: null });
    const fired = applyRule(st, neg, 90, 48, 3000, cfg);
    expect(fired).toMatchObject({ decision: "alert", emit: "negative" });
    st.openNegative = "e1";
    expect(applyRule(st, neg, 90, 48, 3500, cfg)).toMatchObject({ decision: "alert", emit: null });
    expect(applyRule(st, makeProbs(0.1, 0, "anger"), 90, 48, 4000, cfg)).toMatchObject({ decision: "none", closeNegative: true });
    applyRule(st, neg, 90, 48, 5000, cfg);
    expect(applyRule(st, neg, 90, 48, 8000, cfg).emit).toBeNull(); // within 30 s cooldown
  });

  it("marks sustained happiness as positive and emits a positive event", () => {
    const st = newRuleState(0);
    const pos = makeProbs(0.05, 0.8, "anger");
    expect(applyRule(st, pos, 90, 48, 1000, cfg).decision).toBe("positive");
    expect(applyRule(st, pos, 90, 48, 3100, cfg).emit).toBe("positive");
  });
});

describe("mock backend", () => {
  beforeEach(() => mock.__resetMock());

  it("serves demo streams across several sites, including connecting and offline ones", async () => {
    const list = await mock.listStreams();
    expect(new Set(list.map((s) => s.site)).size).toBeGreaterThanOrEqual(2);
    expect(list.some((s) => s.state === "connecting")).toBe(true);
    expect(list.some((s) => s.state === "error")).toBe(true);
  });

  it("confirms an incident: it leaves the bell, the journal shows the new state", async () => {
    const data = new AppData();
    await data.refreshIncidents();
    const before = data.newIncidents.length;
    expect(before).toBeGreaterThan(0);
    const inc = data.newIncidents[0];
    await data.ack(inc.event_id, "confirmed");
    expect(data.newIncidents.length).toBe(before - 1);
    const journal = await mock.listEvents({ kind: "negative" });
    expect(journal.find((e) => e.event_id === inc.event_id)?.state).toBe("confirmed");
    expect(await mock.listEvents({ state: "new" })).toHaveLength(before - 1);
  });

  it("hiding the banner keeps the incident new in the bell", async () => {
    const data = new AppData();
    await data.refreshIncidents();
    const inc = data.newIncidents[0];
    data.hide(inc.event_id);
    expect(data.newIncidents).toContainEqual(inc);
    expect(data.hidden.has(inc.event_id)).toBe(true);
  });

  it("rejects ack of a positive event", async () => {
    const pos = (await mock.listEvents({ kind: "positive" }))[0];
    await expect(mock.ackEvent(pos.event_id, "confirmed")).rejects.toThrow("только негативные");
  });

  it("reports a failed connection test with a Russian message", async () => {
    expect(await mock.testStream("http://x")).toEqual({ ok: false, error: "Адрес должен начинаться с rtsp://" });
    expect((await mock.testStream("rtsp://192.0.2.5/main")).ok).toBe(true);
  });

  it("updates only the patched camera settings", async () => {
    const [first] = await mock.listStreams();
    const updated = await mock.updateStream(first.id, { settings: { analysis_fps: 8 } });
    expect(updated.settings).toEqual({ ...first.settings, analysis_fps: 8 });
  });

  it("queues accepted uploads and rejects other formats", async () => {
    const ok = await mock.uploadFile(new File(["x"], "clip.mp4"));
    expect(ok.status).toBe("queued");
    await expect(mock.uploadFile(new File(["x"], "notes.txt"))).rejects.toThrow("Неподдерживаемый формат");
  });

  it("filters and pages the journal", async () => {
    const all = await mock.listEvents({});
    const page = await mock.listEvents({ limit: 2 });
    expect(page).toEqual(all.slice(0, 2));
    expect(await mock.listEvents({ limit: 2, before_id: page[1].event_id })).toEqual(all.slice(2, 4));
    expect((await mock.listEvents({ kind: "positive" })).every((e) => e.kind === "positive")).toBe(true);
  });
});

describe("journal live merge", () => {
  it("inserts by time, replaces on update and drops events that stop matching", async () => {
    mock.__resetMock();
    const list = await mock.listEvents({ state: "new" });
    const ev = { ...list[0] };
    expect(mergeEvent(list, ev, { state: "new" })).toHaveLength(list.length);
    expect(mergeEvent(list, { ...ev, state: "confirmed" }, { state: "new" })).toHaveLength(list.length - 1);
    const newer = { ...ev, event_id: "x", emit_ts: ev.emit_ts + 1 };
    expect(mergeEvent(list, newer, {})[0].event_id).toBe("x");
  });
});

describe("/ws/streams binary frame", () => {
  it("decodes stream id, frame id and JPEG bytes", () => {
    const id = new TextEncoder().encode("cam7");
    const buf = new ArrayBuffer(3 + id.length + 4 + 3);
    const v = new DataView(buf);
    v.setUint8(0, STREAM_FRAME);
    v.setUint16(1, id.length, true);
    new Uint8Array(buf, 3, id.length).set(id);
    v.setUint32(3 + id.length, 42, true);
    new Uint8Array(buf, 7 + id.length).set([0xff, 0xd8, 0xff]);
    const f = decodeStreamFrame(buf)!;
    expect(f.stream_id).toBe("cam7");
    expect(f.frame_id).toBe(42);
    expect([...f.jpeg]).toEqual([0xff, 0xd8, 0xff]);
    expect(decodeStreamFrame(new ArrayBuffer(2))).toBeNull();
  });
});
