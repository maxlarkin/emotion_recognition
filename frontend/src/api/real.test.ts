import { afterEach, describe, expect, it, vi } from "vitest";
import * as real from "./real";
import { deleteSession, fetchSessions } from "./real";

function stubFetch(status: number, body: string, contentType = "application/json") {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(body, { status, headers: { "Content-Type": contentType } })));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("sessions API errors", () => {
  it("throws the Russian detail of a JSON error", async () => {
    stubFetch(409, JSON.stringify({ detail: "Сессия ещё идёт" }));
    await expect(deleteSession("abc")).rejects.toThrow("Сессия ещё идёт");
    stubFetch(404, JSON.stringify({ detail: "Сессия не найдена" }));
    await expect(deleteSession("abc")).rejects.toThrow("Сессия не найдена");
    stubFetch(503, JSON.stringify({ detail: "Хранилище сессий недоступно" }));
    await expect(fetchSessions()).rejects.toThrow("Хранилище сессий недоступно");
  });

  it("falls back to HTTP <status> for a non-JSON error", async () => {
    stubFetch(502, "<html>Bad Gateway</html>", "text/html");
    await expect(fetchSessions()).rejects.toThrow("HTTP 502");
    stubFetch(500, "oops", "text/plain");
    await expect(deleteSession("abc")).rejects.toThrow("HTTP 500");
  });

  it("rejects a malformed list body", async () => {
    stubFetch(200, JSON.stringify({ sessions: [] }));
    await expect(fetchSessions()).rejects.toThrow("некорректный ответ сервера");
    stubFetch(200, JSON.stringify([{ id: 1, created_at: "x" }]));
    await expect(fetchSessions()).rejects.toThrow("некорректный ответ сервера");
  });

  it("returns a well-formed list", async () => {
    const list = [{ id: "a", kind: "live", stream_id: null, created_at: 1, duration: 2, status: "finished" }];
    stubFetch(200, JSON.stringify(list));
    await expect(fetchSessions()).resolves.toEqual(list);
  });
});

/** Records fetch calls and answers each with the next queued response. */
function recordFetch(...responses: [number, unknown][]) {
  const calls: { url: string; method: string; body: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, method: init?.method ?? "GET", body: init?.body ? JSON.parse(String(init.body)) : undefined });
      const [status, body] = responses.shift() ?? [200, null];
      return new Response(status === 204 ? null : JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
    }),
  );
  return calls;
}

describe("streams and system adapters (interfaces.md)", () => {
  const stream = {
    id: "s1", name: "Вход", site: "", url: "rtsp://u:***@h/x", camera_type: "sector",
    settings: { enabled: true, analysis_fps: 5, detection_size: "auto", min_face_px: 48 },
    state: "running", fps: 5, has_audio: false, source_resolution: [1920, 1080], decode: "cpu", session_id: "x",
  };

  it("lists, creates (then reads the row back), patches, deletes and reconnects", async () => {
    const calls = recordFetch([200, [stream]], [200, { id: "s1" }], [200, [stream]], [200, stream], [204, null], [204, null]);
    expect(await real.listStreams()).toEqual([stream]);
    expect(await real.createStream({ name: "Вход", site: "", url: "rtsp://h/x", camera_type: "sector" })).toEqual(stream);
    expect(await real.updateStream("s1", { settings: { analysis_fps: 3 } })).toEqual(stream);
    await real.deleteStream("s1");
    await real.reconnectStream("s1");
    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual([
      "GET /api/streams",
      "POST /api/streams",
      "GET /api/streams",
      "PATCH /api/streams/s1",
      "DELETE /api/streams/s1",
      "POST /api/streams/s1/reconnect",
    ]);
    expect(calls[1].body).toEqual({ name: "Вход", site: "", url: "rtsp://h/x", camera_type: "sector" });
    expect(calls[3].body).toEqual({ settings: { analysis_fps: 3 } });
  });

  it("tests a URL and reads system load", async () => {
    const calls = recordFetch([200, { ok: false, error: "Нет ответа от камеры" }], [200, { server_ts: 1, cpu_pct: 3 }]);
    expect(await real.testStream("rtsp://h/x")).toEqual({ ok: false, error: "Нет ответа от камеры" });
    expect((await real.fetchSystem()).cpu_pct).toBe(3);
    expect(calls[0]).toMatchObject({ method: "POST", url: "/api/streams/test", body: { url: "rtsp://h/x" } });
    expect(calls[1].url).toBe("/api/system");
  });

  it("surfaces the Russian detail of a rejected stream", async () => {
    recordFetch([422, { detail: "Адрес должен начинаться с rtsp://" }]);
    await expect(real.createStream({ name: "A", site: "", url: "http://x", camera_type: "sector" })).rejects.toThrow("Адрес должен начинаться с rtsp://");
  });
});
