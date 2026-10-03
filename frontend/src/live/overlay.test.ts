import { describe, expect, it } from "vitest";
import { containRect, EMOTION_RU, faceLabel, scaleBox } from "./overlay";
import { EMOTIONS } from "./protocol";

describe("containRect", () => {
  it("letterboxes a 4:3 video in a 16:9 box", () => {
    expect(containRect(640, 480, 1600, 900)).toEqual({ x: 200, y: 0, w: 1200, h: 900 });
  });
  it("pillarboxes nothing for a matching aspect", () => {
    expect(containRect(1280, 720, 640, 360)).toEqual({ x: 0, y: 0, w: 640, h: 360 });
  });
  it("is empty for unknown sizes", () => {
    expect(containRect(0, 0, 640, 360)).toEqual({ x: 0, y: 0, w: 0, h: 0 });
  });
});

describe("scaleBox", () => {
  it("maps JPEG pixels to the displayed video", () => {
    // 640x360 JPEG of a 1280x720 camera shown 960x540 at offset (20, 10)
    const shown = { x: 20, y: 10, w: 960, h: 540 };
    expect(scaleBox([100, 50, 64, 36], 640, 360, shown)).toEqual({ x: 170, y: 85, w: 96, h: 54 });
  });
});

describe("faceLabel", () => {
  it("shows the Russian emotion and a rounded percentage", () => {
    expect(faceLabel({ fused_label: "happiness", confidence: 0.814 })).toBe("радость 81%");
    expect(faceLabel({ fused_label: "neutral", confidence: 1 })).toBe("нейтрально 100%");
  });
  it("has a Russian name for every emotion key", () => {
    for (const e of EMOTIONS) expect(EMOTION_RU[e]).toMatch(/^[а-яё]+$/);
  });
});
