export interface BoundingBoxModel {
  x: number;
  y: number;
  w: number;
  h: number;
  color: string;
  label: string;
  score: number | null;
}

export interface CheckResult {
  name: "profile" | "decoration" | "surface";
  verdict: "PASS" | "BORDERLINE" | "FAIL";
  metrics: Record<string, number>;
  boxes: BoundingBoxModel[];
  heatmap_png: string | null;
}

export interface StageTiming {
  name: string;
  ms: number;
}

export interface InspectResponse {
  decision: "ACCEPT" | "REVIEW" | "REJECT";
  reasons: string[];
  rotation_deg: number;
  registration: {
    reliable: boolean;
    ncc: number;
    num_inliers: number;
    inlier_ratio: number;
  };
  profile: CheckResult;
  decoration: CheckResult;
  surface: CheckResult;
  timings: StageTiming[];
  total_ms: number;
  master_png: string;
  live_aligned_png: string;
  difference_overlay_png: string;
}

export interface InfoResponse {
  architecture: string;
  model_name: string;
  device: string;
  working_resolution: number;
  pipeline_stages: string[];
  bbox_colors: Record<string, string>;
}
