import type { InspectResponse } from "../types";

interface Props {
  result: InspectResponse;
}

const WORD: Record<InspectResponse["decision"], string> = {
  ACCEPT: "Accept.",
  REVIEW: "Review.",
  REJECT: "Reject.",
};

const KLASS: Record<InspectResponse["decision"], string> = {
  ACCEPT: "accept",
  REVIEW: "review",
  REJECT: "reject",
};

export function Verdict({ result }: Props) {
  const word = WORD[result.decision];
  const klass = KLASS[result.decision];

  return (
    <div className="verdict">
      <div className="verdict-eyebrow">— Result of Verification</div>
      <div className="verdict-grid">
        <h2 className={`verdict-word ${klass}`}>{word}</h2>
        <div className="verdict-aside">
          <span className="num">{result.total_ms.toFixed(1)}ms</span>
          <span className="lbl">total latency</span>
          <br />
          <span className="num">{result.rotation_deg.toFixed(1)}°</span>
          <span className="lbl">rotation correction</span>
          <br />
          <span className="num">
            {result.registration.reliable ? "✓" : "✗"} {result.registration.ncc.toFixed(2)}
          </span>
          <span className="lbl">registration NCC</span>
        </div>
      </div>
      <div className="verdict-reasons">
        {result.reasons.map((r, i) => (
          <span key={i}>· {r}</span>
        ))}
      </div>
    </div>
  );
}
