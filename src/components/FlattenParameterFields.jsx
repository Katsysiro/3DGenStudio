import { FLATTEN_SHADERS } from '../utils/meshFlatten'

const FLATTEN_RESOLUTION_OPTIONS = [512, 1024, 2048, 4096]
const FLATTEN_SAMPLE_OPTIONS = [16, 32, 64, 128, 256]

// The "Flatten to one lit albedo" option set, rendered as params-card fields —
// the same choices the Export dialog offers (shader, resolution, samples,
// exposure), for surfaces that run the flatten outside it (the graph's Flatten
// to Albedo node).
//
// `options` is the current value bag, `onChange(key, value)` writes one field
// back; `controlClassName` is appended to every control ("nodrag" in React Flow).
export default function FlattenParameterFields({
  options,
  onChange,
  disabled = false,
  controlClassName = '',
  selectClassName = 'params-card__select'
}) {
  const withControlClass = (base) => `${base}${controlClassName ? ` ${controlClassName}` : ''}`
  const shader = FLATTEN_SHADERS.find(entry => entry.value === options?.shader) || FLATTEN_SHADERS[0]

  return (
    <>
      <div className="params-card__field">
        <label className="params-card__label font-label">Target shader</label>
        <select
          className={withControlClass(selectClassName)}
          value={shader.value}
          disabled={disabled}
          onChange={event => onChange('shader', event.target.value)}
        >
          {FLATTEN_SHADERS.map(entry => (
            <option key={entry.value} value={entry.value}>{entry.label}</option>
          ))}
        </select>
        <span className="image-card__param-hint">{shader.hint}</span>
      </div>

      <div className="params-card__field">
        <label className="params-card__label font-label">Resolution</label>
        <select
          className={withControlClass(selectClassName)}
          value={String(options?.resolution ?? 2048)}
          disabled={disabled}
          onChange={event => onChange('resolution', Number(event.target.value))}
        >
          {FLATTEN_RESOLUTION_OPTIONS.map(size => (
            <option key={size} value={String(size)}>{size} × {size}</option>
          ))}
        </select>
        <span className="image-card__param-hint">Albedo size in pixels. Cost scales with the square of this.</span>
      </div>

      <div className="params-card__field">
        <label className="params-card__label font-label">Samples</label>
        <select
          className={withControlClass(selectClassName)}
          value={String(options?.samples ?? 64)}
          disabled={disabled}
          onChange={event => onChange('samples', Number(event.target.value))}
        >
          {FLATTEN_SAMPLE_OPTIONS.map(count => (
            <option key={count} value={String(count)}>{count}{count === 16 ? ' (preview)' : ''}</option>
          ))}
        </select>
        <span className="image-card__param-hint">Raise it if cavities look grainy.</span>
      </div>

      <div className="params-card__field">
        <label className="params-card__label font-label">Exposure (stops)</label>
        <input
          type="number"
          min={-3}
          max={3}
          step={0.25}
          className={withControlClass('params-card__input')}
          value={options?.exposure ?? 0}
          disabled={disabled}
          onChange={event => onChange('exposure', event.target.value === '' ? '' : Math.min(3, Math.max(-3, Number(event.target.value) || 0)))}
        />
        <span className="image-card__param-hint">Applied before the highlight roll-off. Lower it if the result looks washed out.</span>
      </div>
    </>
  )
}
