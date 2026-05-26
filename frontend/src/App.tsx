import { useCallback, useEffect, useState } from "react";

import { fetchInfo, inspect } from "./api";
import { rotateImageFile } from "./utils/rotate";
import type { InfoResponse, InspectResponse } from "./types";

import { CheckResults } from "./components/CheckResults";
import { Divider } from "./components/Divider";
import { Footer } from "./components/Footer";
import { Hero } from "./components/Hero";
import { Masthead } from "./components/Masthead";
import { TimingRow } from "./components/TimingRow";
import { UploadRow } from "./components/UploadRow";
import { Verdict } from "./components/Verdict";
import { VisualDisplay } from "./components/VisualDisplay";

export default function App() {
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [master, setMaster] = useState<File | null>(null);
  const [live, setLive] = useState<File | null>(null);
  const [rotation, setRotation] = useState<number>(0);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<InspectResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchInfo().then(setInfo).catch(() => setInfo(null));
  }, []);

  const handleRun = useCallback(async () => {
    if (!master || !live) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const liveToSend =
        rotation % 360 === 0 ? live : await rotateImageFile(live, rotation);
      const res = await inspect(master, liveToSend);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, [master, live, rotation]);

  const canRun = master !== null && live !== null && !running;

  return (
    <div className="app-shell">
      <Masthead />

      <Hero info={info} />

      <Divider />

      <UploadRow
        master={master}
        live={live}
        setMaster={setMaster}
        setLive={setLive}
        rotation={rotation}
        setRotation={setRotation}
      />

      {!master || !live ? (
        <div className="stage">awaiting both specimens</div>
      ) : (
        <>
          <div style={{ height: "2rem" }} />
          <button
            className="btn-primary"
            disabled={!canRun}
            onClick={handleRun}
          >
            {running ? "computing inspection..." : "Run Verification"}
          </button>
        </>
      )}

      {running && (
        <div className="spinner-wrap">
          <span className="spinner" /> ◈ running 8-stage pipeline...
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <>
          <Verdict result={result} />

          <div className="section-label">
            <span className="num">iii.</span>
            Visual Display
            <span className="line" />
          </div>
          <VisualDisplay result={result} />

          <div style={{ height: "3rem" }} />

          <div className="section-label">
            <span className="num">iv.</span>
            Per-Check Results
            <span className="line" />
          </div>
          <CheckResults result={result} />

          <div style={{ height: "3rem" }} />

          <div className="section-label">
            <span className="num">v.</span>
            Per-Stage Timing
            <span className="line" />
          </div>
          <TimingRow timings={result.timings} />

          <p className="annotation">
            Each new specimen passes through eight stages — preprocessing,
            segmentation, rotation estimation, SIFT registration, profile
            check, decoration check (DINOv2 patch features), surface check
            (SSIM + Laplacian), and a decision engine that fuses the three
            verdicts. Bounding boxes are color-coded by check: red for
            profile deviations, orange for decoration mismatches, magenta
            for surface defects. Green and yellow outlines on the difference
            overlay show the master and live silhouettes respectively.
            Per-check thresholds are calibrated from a labeled validation
            set, not adjusted at runtime — see the validation plan for the
            calibration procedure.
          </p>
        </>
      )}

      <Footer />
    </div>
  );
}
