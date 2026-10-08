import { afterEach, describe, expect, it, vi } from "vitest";
import { deleteSession, fetchSessions } from "./api";

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
