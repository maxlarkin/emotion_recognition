import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router";
import { EmptyState } from "./Grid";
import { describe, expect, it } from "vitest";
import { DEFAULT_STREAM_SETTINGS, type Stream, type StreamFace } from "../api/types";
import { EMOTIONS } from "../live/protocol";
import { DecisionChip, SettingRow, StatusLabel } from "../ui/controls";
import { canAddStream } from "./AddStreamDialog";
import { boxLabel, tileSize } from "./FrameView";
import { groupStreams, matchesQuery, NO_SITE, WEBCAM_ENTRY } from "./grouping";
import { incidentTitle, JournalTable, smallFaces } from "./StreamDetail";

function stream(over: Partial<Stream>): Stream {
  return {
    id: "s",
    name: "Камера",
    site: "",
    url: "rtsp://x",
    camera_type: "sector",
    settings: DEFAULT_STREAM_SETTINGS,
    state: "running",
    fps: 5,
    has_audio: true,
    source_resolution: [1920, 1080],
    session_id: null,
    ...over,
  };
}

const LIST = [
  stream({ id: "a", name: "Ресепшн", site: "Тверская" }),
  stream({ id: "b", name: "Вход", site: "Ленинский" }),
  stream({ id: "c", name: "Касса", site: "Тверская" }),
  stream({ id: "d", name: "Склад", site: " " }),
];

describe("sidebar grouping and search", () => {
  it("groups by site in first-seen order with «Без точки» for an empty site", () => {
    const g = groupStreams(LIST, "");
    expect(g.map((x) => x.site)).toEqual(["Тверская", "Ленинский", NO_SITE]);
    expect(g[0].streams.map((s) => s.id)).toEqual(["a", "c"]);
  });

  it("filters by camera and site, case-insensitive, and hides empty groups", () => {
    expect(groupStreams(LIST, "твер").map((x) => x.site)).toEqual(["Тверская"]);
    const g = groupStreams(LIST, "ВХОД");
    expect(g).toHaveLength(1);
    expect(g[0].streams[0].id).toBe("b");
    expect(groupStreams(LIST, "нет такого")).toEqual([]);
  });

  it("hides the webcam entry unless it matches", () => {
    expect(matchesQuery(WEBCAM_ENTRY.name, WEBCAM_ENTRY.site, "")).toBe(true);
    expect(matchesQuery(WEBCAM_ENTRY.name, WEBCAM_ENTRY.site, "твер")).toBe(false);
    expect(matchesQuery(WEBCAM_ENTRY.name, WEBCAM_ENTRY.site, "веб")).toBe(true);
  });
});

function face(decision: StreamFace["decision"], neg: number): StreamFace {
  const p = Object.fromEntries(EMOTIONS.map((e) => [e, 0])) as StreamFace["fused"];
  p.anger = neg;
  p.neutral = 1 - neg;
  return { track_id: 12, box: [0, 0, 10, 10], face: p, face_label: "anger", fused: p, fused_label: "anger", confidence: neg, decision, face_px: 60 };
}

describe("stream detail pieces", () => {
  it("labels boxes with the negative probability, «?» when abstained", () => {
    expect(boxLabel(face("alert", 0.91))).toBe("#12 · 0,91");
    expect(boxLabel(face("abstain", 0.4))).toBe("#12 · ?");
  });

  it("sizes analysed frames to a 640 px longest side", () => {
    expect(tileSize([3840, 2160])).toEqual([640, 360]);
    expect(tileSize([1280, 960])).toEqual([640, 480]);
  });

  it("titles the incident banner like the mockup", () => {
    expect(incidentTitle({ track_id: 12, group_prob: 0.91, duration_ms: 80_000 })).toBe("Клиент #12 · негатив 0,91 · 1:20");
  });

  it("warns about small faces only below the stream minimum", () => {
    expect(smallFaces(44, 56)).toBe(true);
    expect(smallFaces(56, 56)).toBe(false);
    expect(smallFaces(null, 56)).toBe(false);
  });

  it("highlights negative values for alert and watch rows", () => {
    const html = renderToStaticMarkup(
      <JournalTable
        timeZone="UTC"
        rows={[
          { from_ts: Date.UTC(2026, 9, 10, 14, 32, 0), to_ts: Date.UTC(2026, 9, 10, 14, 32, 10), clients: 3, dominant_emotion: "anger", negative_prob: 0.91, decision: "alert" },
          { from_ts: Date.UTC(2026, 9, 10, 14, 31, 50), to_ts: Date.UTC(2026, 9, 10, 14, 32, 0), clients: 2, dominant_emotion: "neutral", negative_prob: 0.08, decision: "none" },
        ]}
      />,
    );
    expect(html).toContain("14:32:00–32:10");
    expect(html).toContain('class="num hi">0,91');
    expect(html).toContain('class="num">0,08');
    expect(html).toContain("Злость");
    expect(html).toContain("Инцидент");
    expect(html).toContain("Норма");
  });

  it("renders chips, status and setting rows", () => {
    expect(renderToStaticMarkup(<DecisionChip decision="watch" />)).toBe('<span class="chip chip-watch">Наблюдение</span>');
    expect(renderToStaticMarkup(<StatusLabel state="error" />)).toContain("Нет сигнала");
    const row = renderToStaticMarkup(
      <SettingRow label="Порог негатива">
        <span>x</span>
      </SettingRow>,
    );
    expect(row).toContain('class="setting-row"');
    expect(row).toContain("Порог негатива");
  });
});

describe("add-stream form", () => {
  it("needs a name and an address", () => {
    expect(canAddStream("", "rtsp://x")).toBe(false);
    expect(canAddStream("Вход", "  ")).toBe(false);
    expect(canAddStream("Вход", "rtsp://x")).toBe(true);
  });
});

describe("empty streams state", () => {
  it("offers adding a stream and analysing a recording", () => {
    const html = renderToStaticMarkup(
      <MemoryRouter>
        <EmptyState onAdd={() => {}} />
      </MemoryRouter>,
    );
    expect(html).toContain("Потоков пока нет");
    expect(html).toContain("Добавить поток");
    expect(html).toContain('href="/analyze"');
    expect(html).toContain("Проанализировать запись");
    expect(html).toContain("Демо-данные");
  });
});
