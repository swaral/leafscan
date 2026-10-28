import { useRef, useState } from 'react';

/**
 * Small multi-series line chart over a shared x (epochs).
 * series: [{ key, label, color, values: number[], dashed?, endLabel? }]
 */
export default function LineChart({ x, series, yFormat = (v) => v.toFixed(2), height = 240, yDomain, yMax }) {
  const ref = useRef(null);
  const [hover, setHover] = useState(null);
  const W = 640, H = height, m = { t: 12, r: 70, b: 28, l: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;

  const all = series.flatMap((s) => s.values);
  let [lo, hi] = yDomain || [Math.min(...all), Math.max(...all)];
  if (!yDomain) { const pad = (hi - lo) * 0.08 || 0.01; lo -= pad; hi += pad; }
  if (yMax !== undefined) hi = Math.min(hi, yMax);
  const xs = (i) => m.l + (x.length === 1 ? iw / 2 : (i / (x.length - 1)) * iw);
  const ys = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);

  const onMove = (e) => {
    const box = ref.current.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.round(((px - m.l) / iw) * (x.length - 1));
    setHover(Math.max(0, Math.min(x.length - 1, i)));
  };

  return (
    <div style={{ position: 'relative' }}>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} width="100%" onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img"
        aria-label={`Line chart of ${series.map((s) => s.label).join(' and ')} by epoch`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={ys(t)} y2={ys(t)} stroke="var(--line)" strokeWidth="1" />
            <text x={m.l - 8} y={ys(t) + 4} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        <line x1={m.l} x2={W - m.r} y1={m.t + ih} y2={m.t + ih} stroke="var(--axis)" />
        {x.map((v, i) => (x.length <= 16 || i % 2 === 0) && (
          <text key={v} x={xs(i)} y={H - 8} textAnchor="middle">{v}</text>
        ))}
        {hover !== null && <line x1={xs(hover)} x2={xs(hover)} y1={m.t} y2={m.t + ih} stroke="var(--axis)" />}
        {series.map((s) => (
          <g key={s.key}>
            <polyline fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
              strokeDasharray={s.dashed ? '5 4' : undefined}
              points={s.values.map((v, i) => `${xs(i)},${ys(v)}`).join(' ')} />
            <circle cx={xs(s.values.length - 1)} cy={ys(s.values.at(-1))} r="4" fill={s.color} stroke="var(--surface)" strokeWidth="2" />
            {s.endLabel && (
              <text x={xs(s.values.length - 1) + 8} y={ys(s.values.at(-1)) + 4} style={{ fill: 'var(--ink-2)' }}>{yFormat(s.values.at(-1))}</text>
            )}
            {hover !== null && (
              <circle cx={xs(hover)} cy={ys(s.values[hover])} r="4.5" fill={s.color} stroke="var(--surface)" strokeWidth="2" />
            )}
          </g>
        ))}
        <rect x={m.l} y={m.t} width={iw} height={ih} fill="transparent" />
      </svg>
      {hover !== null && (
        <div className="tip" style={{
          position: 'absolute', left: `${(xs(hover) / W) * 100}%`, top: 0,
          transform: xs(hover) > W * 0.6 ? 'translateX(calc(-100% - 12px))' : 'translateX(12px)',
        }}>
          <b>Epoch {x[hover]}</b>
          {series.map((s) => <div key={s.key}>{s.label}: {yFormat(s.values[hover])}</div>)}
        </div>
      )}
    </div>
  );
}
