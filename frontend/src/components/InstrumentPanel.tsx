import type { InfoResponse } from "../types";

interface Props {
  info: InfoResponse | null;
}

export function InstrumentPanel({ info }: Props) {
  return (
    <div className="info-panel">
      <div className="info-panel-title">Instrument</div>
      <div className="info-row">
        <span className="info-key">Architecture</span>
        <span className="info-val">{info?.architecture ?? "—"}</span>
      </div>
      <div className="info-row">
        <span className="info-key">Model</span>
        <span className="info-val">{info?.model_name ?? "—"}</span>
      </div>
      <div className="info-row">
        <span className="info-key">Device</span>
        <span className="info-val gold">{info?.device?.toUpperCase() ?? "—"}</span>
      </div>
      <div className="info-row">
        <span className="info-key">Working Res.</span>
        <span className="info-val">
          {info ? `${info.working_resolution}px` : "—"}
        </span>
      </div>
      <div className="info-row">
        <span className="info-key">Stages</span>
        <span className="info-val">{info?.pipeline_stages.length ?? "—"}</span>
      </div>
    </div>
  );
}
