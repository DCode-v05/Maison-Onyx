import { Dropzone } from "./Dropzone";
import { RotationControl } from "./RotationControl";

interface Props {
  master: File | null;
  live: File | null;
  setMaster: (f: File | null) => void;
  setLive: (f: File | null) => void;
  rotation: number;
  setRotation: (deg: number) => void;
}

export function UploadRow({
  master,
  live,
  setMaster,
  setLive,
  rotation,
  setRotation,
}: Props) {
  return (
    <div className="upload-row">
      <Dropzone
        label="Master Template"
        number="i."
        onFile={setMaster}
        file={master}
      />
      <div>
        <Dropzone
          label="Specimen Input"
          number="ii."
          onFile={setLive}
          file={live}
          previewRotation={rotation}
        />
        <RotationControl
          value={rotation}
          onChange={setRotation}
          disabled={!live}
        />
      </div>
    </div>
  );
}
