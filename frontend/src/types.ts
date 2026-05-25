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

export interface StageImages {
  master_input: string;
  live_input: string;
  master_with_bbox: string;
  live_with_bbox: string;
  master_with_mask: string;
  live_with_mask: string;
  master_cropped: string;
  live_cropped: string;
  live_after_alignment: string;
  live_after_registration: string;
}

export interface InspectResponse {
  decision: "ACCEPT" | "REVIEW" | "REJECT";
  reasons: string[];
  rotation_deg: number;
  scale_factor: number;
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
  stage_images: StageImages;
}

export interface InfoResponse {
  architecture: string;
  model_name: string;
  device: string;
  working_resolution: number;
  pipeline_stages: string[];
  bbox_colors: Record<string, string>;
}
