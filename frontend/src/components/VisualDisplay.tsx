import type { InspectResponse } from "../types";

interface Props {
  result: InspectResponse;
}

interface Frame {
  label: string;
  src: string;
  caption?: string;
}

function FrameStrip({ frames }: { frames: Frame[] }) {
  return (
    <div className="frame-strip">
      {frames.map((f) => (
        <div className="frame-stage" key={f.label}>
          <div className="frame-stage-label">{f.label}</div>
          <div className="frame">
            <img src={f.src} alt={f.label} />
          </div>
          {f.caption && <div className="frame-stage-caption">{f.caption}</div>}
        </div>
      ))}
    </div>
  );
}

export function VisualDisplay({ result }: Props) {
  const masterFrames: Frame[] = [
    { label: "Reference", src: result.master_png },
    {
      label: "Edges · full structure",
      src: result.master_contoured_png,
      caption: "Canny edges — used for profile check",
    },
  ];

  const liveFrames: Frame[] = [
    { label: "1 · Preprocessed", src: result.live_preprocessed_png },
    { label: "2 · Segmented", src: result.live_segmented_png },
    { label: "3 · Rotated", src: result.live_rotated_png },
    {
      label: "4 · Registered",
      src: result.live_registered_png,
      caption: "with full edge structure",
    },
  ];

  return (
    <div className="visual-display">
      <div className="display-cell">
        <div className="display-cell-header">
          <span className="display-cell-title">Master</span>
          <span className="display-cell-sub">reference · contours</span>
        </div>
        <FrameStrip frames={masterFrames} />
      </div>

      <div className="display-cell">
        <div className="display-cell-header">
          <span className="display-cell-title">Live</span>
          <span className="display-cell-sub">stage-by-stage progression</span>
        </div>
        <FrameStrip frames={liveFrames} />
      </div>
    </div>
  );
}
