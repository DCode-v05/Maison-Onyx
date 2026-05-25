import type { InspectResponse } from "../types";

interface Props {
  result: InspectResponse;
}

interface StageStep {
  num: string;
  title: string;
  subtitle: string;
  image: string;
}

export function PipelineStages({ result }: Props) {
  const s = result.stage_images;

  // The live image's journey through the pipeline, plus the master pair so
  // operators can see how segmentation found the piece on both sides.
  const masterSteps: StageStep[] = [
    {
      num: "1",
      title: "Master · Input",
      subtitle: "after preprocess + pad",
      image: s.master_input,
    },
    {
      num: "2",
      title: "Master · Segmented",
      subtitle: "refined mask (informational)",
      image: s.master_with_mask,
    },
    {
      num: "3",
      title: "Master · Bounding Box",
      subtitle: "min/max x,y of piece",
      image: s.master_with_bbox,
    },
    {
      num: "4",
      title: "Master · Cropped",
      subtitle: "input to next stage",
      image: s.master_cropped,
    },
  ];

  const liveSteps: StageStep[] = [
    {
      num: "1",
      title: "Live · Input",
      subtitle: "after preprocess + pad",
      image: s.live_input,
    },
    {
      num: "2",
      title: "Live · Segmented",
      subtitle: "refined mask (informational)",
      image: s.live_with_mask,
    },
    {
      num: "3",
      title: "Live · Bounding Box",
      subtitle: "min/max x,y of piece",
      image: s.live_with_bbox,
    },
    {
      num: "4",
      title: "Live · Cropped",
      subtitle: "input to next stage",
      image: s.live_cropped,
    },
    {
      num: "5",
      title: "Live · After Alignment",
      subtitle: `similarity: ${result.rotation_deg.toFixed(1)}° · ${result.scale_factor.toFixed(2)}×`,
      image: s.live_after_alignment,
    },
    {
      num: "6",
      title: "Live · After Registration",
      subtitle: `SIFT · NCC ${result.registration.ncc.toFixed(2)}`,
      image: s.live_after_registration,
    },
  ];

  return (
    <div className="pipeline-stages">
      <div className="stage-rail master">
        <div className="rail-label">master</div>
        <div className="stage-track" data-count={masterSteps.length}>
          {masterSteps.map((step, i) => (
            <StageCell key={step.title} step={step} arrow={i < masterSteps.length - 1} />
          ))}
        </div>
      </div>

      <div className="stage-rail live">
        <div className="rail-label">live</div>
        <div className="stage-track" data-count={liveSteps.length}>
          {liveSteps.map((step, i) => (
            <StageCell key={step.title} step={step} arrow={i < liveSteps.length - 1} />
          ))}
        </div>
      </div>
    </div>
  );
}

function StageCell({ step, arrow }: { step: StageStep; arrow: boolean }) {
  return (
    <div className="stage-cell">
      <div className="stage-cell-head">
        <span className="stage-num">{step.num}</span>
        <div className="stage-titles">
          <div className="stage-title">{step.title}</div>
          <div className="stage-subtitle">{step.subtitle}</div>
        </div>
      </div>
      <div className="stage-frame">
        <img src={step.image} alt={step.title} />
      </div>
      {arrow && <div className="stage-arrow">→</div>}
    </div>
  );
}
