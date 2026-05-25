interface Props {
  value: number;
  onChange: (deg: number) => void;
  disabled?: boolean;
}

const PRESETS = [0, 45, 90, 135, 180, 225, 270, 315];

export function RotationControl({ value, onChange, disabled }: Props) {
  return (
    <div className={`rotation-control${disabled ? " is-disabled" : ""}`}>
      <label>Specimen Rotation</label>
      <div className="row">
        <input
          type="range"
          min={-180}
          max={180}
          step={1}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(parseInt(e.target.value, 10))}
        />
        <span className="value">{value}°</span>
      </div>
      <div className="presets">
        {PRESETS.map((p) => (
          <button
            key={p}
            type="button"
            disabled={disabled}
            className={`preset-btn${value === p ? " active" : ""}`}
            onClick={() => onChange(p)}
          >
            {p}°
          </button>
        ))}
        <button
          type="button"
          disabled={disabled || value === 0}
          className="preset-btn reset"
          onClick={() => onChange(0)}
        >
          reset
        </button>
      </div>
    </div>
  );
}
