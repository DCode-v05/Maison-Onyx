import type { InspectResponse } from "../types";

interface Props {
  result: InspectResponse;
}

export function ComparisonView({ result }: Props) {
  return (
    <div className="comparison-grid">
      <div className="comparison-cell">
        <div className="label">Master · reference</div>
        <div className="frame">
          <img src={result.master_png} alt="master" />
        </div>
      </div>
      <div className="comparison-cell">
        <div className="label">Live · aligned</div>
        <div className="frame">
          <img src={result.live_aligned_png} alt="live aligned" />
        </div>
      </div>
      <div className="comparison-cell">
        <div className="label">Difference · defects boxed</div>
        <div className="frame">
          <img src={result.difference_overlay_png} alt="difference overlay" />
        </div>
      </div>
    </div>
  );
}
