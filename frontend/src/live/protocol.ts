// /ws/live wire format (interfaces.md, Live session). All binary fields little-endian.

export const EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"] as const;
export type Emotion = (typeof EMOTIONS)[number];
export type Probs = Record<Emotion, number>;

export const VIDEO_FRAME = 0x01;
export const HEADER_BYTES = 13;

/** 0x01 | f64 capture_ts (ms) | u32 frame_id */
export function encodeFrameHeader(captureTs: number, frameId: number): Uint8Array<ArrayBuffer> {
  const buf = new ArrayBuffer(HEADER_BYTES);
  const view = new DataView(buf);
  view.setUint8(0, VIDEO_FRAME);
  view.setFloat64(1, captureTs, true);
  view.setUint32(9, frameId >>> 0, true);
  return new Uint8Array(buf);
}

/** Header + JPEG bytes as one binary message. */
export function encodeFrame(captureTs: number, frameId: number, jpeg: ArrayBuffer | Uint8Array): Uint8Array<ArrayBuffer> {
  const body = jpeg instanceof Uint8Array ? jpeg : new Uint8Array(jpeg);
  const out = new Uint8Array(HEADER_BYTES + body.byteLength);
  out.set(encodeFrameHeader(captureTs, frameId), 0);
  out.set(body, HEADER_BYTES);
  return out;
}

export interface FaceResult {
  track_id: number;
  box: [number, number, number, number]; // x, y, w, h in pixels of the sent JPEG
  face: Probs;
  face_label: Emotion;
  fused: Probs;
  fused_label: Emotion;
  confidence: number;
}

export interface FrameResult {
  type: "frame_result";
  frame_id: number;
  capture_ts: number;
  proc_ms: number;
  faces: FaceResult[];
  audio: unknown | null;
}

export type ServerMessage =
  | FrameResult
  | { type: "session"; session_id: string }
  | { type: "error"; message: string };

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isEmotion(v: unknown): v is Emotion {
  return typeof v === "string" && (EMOTIONS as readonly string[]).includes(v);
}

function isFace(v: unknown): v is FaceResult {
  if (!isObj(v)) return false;
  const box = v.box;
  return (
    Array.isArray(box) &&
    box.length === 4 &&
    box.every((n) => typeof n === "number" && Number.isFinite(n)) &&
    isEmotion(v.fused_label) &&
    typeof v.confidence === "number"
  );
}

/** Parse one text message from the server; null if it is not a known, well-formed message. */
export function parseServerMessage(text: string): ServerMessage | null {
  let obj: unknown;
  try {
    obj = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isObj(obj)) return null;
  switch (obj.type) {
    case "session":
      return typeof obj.session_id === "string" ? { type: "session", session_id: obj.session_id } : null;
    case "error":
      return { type: "error", message: typeof obj.message === "string" ? obj.message : "Неизвестная ошибка" };
    case "frame_result":
      if (
        typeof obj.frame_id !== "number" ||
        typeof obj.capture_ts !== "number" ||
        !Array.isArray(obj.faces) ||
        !obj.faces.every(isFace)
      ) {
        return null;
      }
      return obj as unknown as FrameResult;
    default:
      return null;
  }
}
