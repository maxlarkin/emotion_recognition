// Analysed frame with face boxes positioned in percent of the frame (ui.md Face decision boxes).
import { useState } from "react";
import type { Probs, StreamFace } from "../api/types";
import { prob } from "../ui/format";

const NEG_KEYS = ["anger", "disgust", "fear", "sadness"] as const;

export const negativeProb = (p: Probs): number => NEG_KEYS.reduce((s, k) => s + p[k], 0);

export function boxLabel(f: Pick<StreamFace, "track_id" | "decision" | "fused">): string {
  return `#${f.track_id} · ${f.decision === "abstain" ? "?" : prob(negativeProb(f.fused))}`;
}

/** JPEG size for a source resolution: longest side 640 (interfaces.md analysed frame). */
export function tileSize(res: [number, number] | null): [number, number] {
  const [w, h] = res ?? [1280, 720];
  const k = 640 / Math.max(w, h);
  return [Math.round(w * k), Math.round(h * k)];
}

export default function FrameView({
  frameUrl,
  faces,
  sourceResolution,
  compact = false,
  children,
}: {
  frameUrl: string | null;
  faces: readonly StreamFace[];
  sourceResolution: [number, number] | null;
  compact?: boolean;
  children?: React.ReactNode;
}) {
  const [natural, setNatural] = useState<[number, number] | null>(null);
  const [fw, fh] = natural ?? tileSize(sourceResolution);
  return (
    <div className={compact ? "frame compact" : "frame"}>
      {frameUrl ? (
        <img src={frameUrl} alt="" onLoad={(e) => setNatural([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])} />
      ) : (
        <div className="frame-placeholder" aria-hidden="true">
          {compact ? "" : "[ кадр потока ]"}
        </div>
      )}
      {faces.map((f) => (
        <div
          key={f.track_id}
          className={`face-box face-${f.decision}`}
          style={{
            left: `${(f.box[0] / fw) * 100}%`,
            top: `${(f.box[1] / fh) * 100}%`,
            width: `${(f.box[2] / fw) * 100}%`,
            height: `${(f.box[3] / fh) * 100}%`,
          }}
        >
          {!compact && <span className="face-tag">{boxLabel(f)}</span>}
        </div>
      ))}
      {children}
    </div>
  );
}
