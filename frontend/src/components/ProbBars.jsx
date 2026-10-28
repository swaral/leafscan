import { pct } from '../api';
import { useTooltip } from './Tooltip';

/** Horizontal bar list for a single series of 0..1 values. */
export default function ProbBars({ items, max = 1, digits = 1, format = (v) => pct(v, digits) }) {
  const [tip, show, hide] = useTooltip();
  return (
    <div className="hbars">
      {items.map((it, i) => (
        <div
          key={it.id ?? i}
          className={`hbar ${i === 0 && it.highlight !== false ? 'top' : ''}`}
          onMouseMove={(e) => show(e, <><b>{it.label}</b>{it.sub ? <div className="muted">{it.sub}</div> : null}<div>{format(it.value)}</div>{it.extra}</>)}
          onMouseLeave={hide}
        >
          <div className="name" title={it.label}>
            {it.label} {it.sub ? <small>· {it.sub}</small> : null}
          </div>
          <div className="track">
            <div className="fill" style={{ width: `${Math.max(0, Math.min(1, it.value / max)) * 100}%` }} />
          </div>
          <div className="val">{format(it.value)}</div>
        </div>
      ))}
      {tip}
    </div>
  );
}
