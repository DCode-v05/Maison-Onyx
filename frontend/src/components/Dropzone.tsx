import { useCallback, useRef, useState } from "react";

interface Props {
  label: string;
  number: string;
  onFile: (file: File | null) => void;
  file: File | null;
  previewRotation?: number;
}

export function Dropzone({ label, number, onFile, file, previewRotation }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    (f: File | null) => {
      onFile(f);
      if (f) {
        const url = URL.createObjectURL(f);
        setPreview(url);
      } else {
        setPreview(null);
      }
    },
    [onFile]
  );

  return (
    <div>
      <div className="section-label">
        <span className="num">{number}</span>
        {label}
        <span className="line" />
      </div>
      <div
        className={`dropzone ${file ? "has-file" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files?.[0] ?? null;
          handleFile(f);
        }}
        style={dragOver ? { borderColor: "var(--gold)" } : undefined}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
        />
        {!file && (
          <>
            <div className="dropzone-instruction">
              Drag and drop file here
            </div>
            <div className="dropzone-hint">
              Limit 200MB per file · PNG, JPG, JPEG
            </div>
            <button
              className="browse-btn"
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                inputRef.current?.click();
              }}
            >
              Browse files
            </button>
          </>
        )}
        {file && (
          <div className="file-chip">
            {file.name} · {(file.size / 1024).toFixed(1)} KB
          </div>
        )}
      </div>
      {preview && (
        <img
          className="preview-img"
          src={preview}
          alt="preview"
          style={
            previewRotation
              ? { transform: `rotate(${previewRotation}deg)` }
              : undefined
          }
        />
      )}
    </div>
  );
}
