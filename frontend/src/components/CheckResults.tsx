import type { CheckResult, InspectResponse } from "../types";

const TITLE: Record<CheckResult["name"], string> = {
  profile: "Profile",
  decoration: "Decoration",
  surface: "Surface",
};

const VERDICT_GLYPH: Record<CheckResult["verdict"], string> = {
  PASS: "PASS",
  BORDERLINE: "REVIEW",
  FAIL: "FAIL",
};

const VERDICT_CLASS: Record<CheckResult["verdict"], string> = {
  PASS: "pass",
  BORDERLINE: "borderline",
  FAIL: "fail",
};

function formatNumber(n: number): string {
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 1) return n.toFixed(2);
  return n.toFixed(4);
}

function CheckCard({ check }: { check: CheckResult }) {
  return (
    <div className="check-card">
      <div className="title">
        <span className="name">{TITLE[check.name]}</span>
        <span className={`verdict ${VERDICT_CLASS[check.verdict]}`}>
          {VERDICT_GLYPH[check.verdict]}
        </span>
      </div>

      {check.heatmap_png && (
        <img className="heatmap" src={check.heatmap_png} alt={`${check.name} heatmap`} />
      )}

      <div className="metric-list">
        {Object.entries(check.metrics).map(([k, v]) => (
          <div className="row" key={k}>
            <span>{k.replace(/_/g, " ")}</span>
            <span>{typeof v === "number" ? formatNumber(v) : String(v)}</span>
          </div>
        ))}
      </div>

      {check.boxes.length > 0 && (
        <div className="boxes-summary">
          ◈ {check.boxes.length} region{check.boxes.length === 1 ? "" : "s"} flagged
        </div>
      )}
    </div>
  );
}

export function CheckResults({ result }: { result: InspectResponse }) {
  return (
    <div className="checks-grid">
      <CheckCard check={result.profile} />
      <CheckCard check={result.decoration} />
      <CheckCard check={result.surface} />
    </div>
  );
}
