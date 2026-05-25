import type { InfoResponse } from "../types";
import { InstrumentPanel } from "./InstrumentPanel";

interface Props {
  info: InfoResponse | null;
}

export function Hero({ info }: Props) {
  return (
    <div className="hero-grid">
      <div className="hero">
        <div className="hero-eyebrow">A Reference-Based Inspection Pipeline</div>
        <h1 className="hero-title">
          The verification<br />
          of <span className="accent">form</span>
          <span className="dot">.</span>
        </h1>
        <p className="hero-sub">
          A master jewel is held as ground truth. Each new specimen is aligned,
          rotated into the same frame, and judged against it across profile,
          decoration, and surface — offline, on-device, in milliseconds.
        </p>
      </div>
      <div>
        <div style={{ height: "1.5rem" }} />
        <InstrumentPanel info={info} />
      </div>
    </div>
  );
}
