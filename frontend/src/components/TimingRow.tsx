import type { StageTiming } from "../types";

interface Props {
  timings: StageTiming[];
}

const LABELS: Record<string, string> = {
  preprocess: "Preprocess",
  segmentation: "Segmentation",
  geometric_alignment: "Alignment",
  fine_registration: "Registration",
  profile_check: "Profile",
  decoration_check: "Decoration",
  surface_check: "Surface",
  decision: "Decision",
  total: "Total",
};

export function TimingRow({ timings }: Props) {
  // Show: preprocess+segmentation, alignment (similarity+fine SIFT), profile, decoration, surface, total
  const grouped: { label: string; value: number; gold?: boolean }[] = [];

  const get = (n: string) => timings.find((t) => t.name === n)?.ms ?? 0;

  grouped.push({ label: "Prep+Seg", value: get("preprocess") + get("segmentation") });
  grouped.push({
    label: "Alignment",
    value: get("geometric_alignment") + get("fine_registration"),
  });
  grouped.push({ label: "Profile", value: get("profile_check") });
  grouped.push({ label: "Decoration", value: get("decoration_check") });
  grouped.push({ label: "Surface", value: get("surface_check") });
  // Total card spans an extra row on small screens via CSS grid
  grouped.push({ label: "Total", value: get("total"), gold: true });

  return (
    <div className="timing-row">
      {grouped.slice(0, 5).map((g) => (
        <Card key={g.label} {...g} />
      ))}
      <Card {...grouped[5]} />
    </div>
  );
}

function Card({ label, value, gold }: { label: string; value: number; gold?: boolean }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value${gold ? " gold" : ""}`}>
        {value.toFixed(1)}
        <span className="unit">ms</span>
      </div>
    </div>
  );
}
