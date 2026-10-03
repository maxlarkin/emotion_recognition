import { describe, expect, it } from "vitest";
import { encodeFrame, encodeFrameHeader, HEADER_BYTES, parseServerMessage } from "./protocol";

describe("encodeFrameHeader", () => {
  it("lays out 0x01 | f64 LE capture_ts | u32 LE frame_id", () => {
    // Same bytes as backend tests/test_ws_live.py::test_header_matches_client_layout
    expect(Array.from(encodeFrameHeader(1.5, 258))).toEqual([0x01, 0, 0, 0, 0, 0, 0, 0xf8, 0x3f, 0x02, 0x01, 0, 0]);
  });

  it("round-trips a realistic timestamp and the full u32 range", () => {
    const h = encodeFrameHeader(1727780000123.4, 4_000_000_000);
    const v = new DataView(h.buffer);
    expect(h.byteLength).toBe(HEADER_BYTES);
    expect(v.getFloat64(1, true)).toBe(1727780000123.4);
    expect(v.getUint32(9, true)).toBe(4_000_000_000);
  });

  it("appends the JPEG bytes", () => {
    const msg = encodeFrame(2, 7, new Uint8Array([0xff, 0xd8, 0xff]).buffer);
    expect(msg.byteLength).toBe(HEADER_BYTES + 3);
    expect(Array.from(msg.slice(HEADER_BYTES))).toEqual([0xff, 0xd8, 0xff]);
  });
});

describe("parseServerMessage", () => {
  const face = {
    track_id: 1,
    box: [10, 20, 30, 40],
    face: {},
    face_label: "happiness",
    fused: {},
    fused_label: "happiness",
    confidence: 0.81,
  };

  it("types frame_result, session and error", () => {
    const fr = parseServerMessage(
      JSON.stringify({ type: "frame_result", frame_id: 3, capture_ts: 1.5, proc_ms: 4, faces: [face], audio: null }),
    );
    expect(fr?.type).toBe("frame_result");
    expect(parseServerMessage('{"type":"session","session_id":"ab"}')).toEqual({ type: "session", session_id: "ab" });
    expect(parseServerMessage('{"type":"error","message":"Сессия не начата"}')).toEqual({
      type: "error",
      message: "Сессия не начата",
    });
  });

  it("rejects malformed messages", () => {
    expect(parseServerMessage("{oops")).toBeNull();
    expect(parseServerMessage('{"type":"other"}')).toBeNull();
    expect(parseServerMessage('{"type":"frame_result","frame_id":1,"capture_ts":1,"faces":[{"box":[1,2]}]}')).toBeNull();
  });
});
