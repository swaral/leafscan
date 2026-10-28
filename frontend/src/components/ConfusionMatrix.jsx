import { useTooltip } from './Tooltip';

// Row-normalised (recall) bins on a single blue ramp; light = near zero.
const BINS = [
  { max: 0, fill: 'var(--seq-0)', label: '0' },
  { max: 0.02, fill: 'var(--seq-1)', label: '≤2%' },
  { max: 0.1, fill: 'var(--seq-2)', label: '≤10%' },
  { max: 0.5, fill: 'var(--seq-3)', label: '≤50%' },
  { max: 1, fill: 'var(--seq-4)', label: '>50%' },
];
const binFor = (v) => BINS.find((b) => v <= b.max) || BINS[BINS.length - 1];

export default function ConfusionMatrix({ matrix, labels }) {
  const [tip, show, hide] = useTooltip();
  const n = matrix.length;
  const cell = 14, gap = 1, left = 190, top = 8;
  const size = n * (cell + gap);
  const rowSums = matrix.map((r) => r.reduce((a, b) => a + b, 0) || 1);

  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="cm-wrap">
        <svg viewBox={`0 0 ${left + size + 8} ${top + size + 40}`} width="100%" style={{ minWidth: 560, maxWidth: 820 }}
          role="img" aria-label="Confusion matrix, rows are actual classes, columns predicted classes">
          {labels.map((l, i) => (
            <text key={l} x={left - 6} y={top + i * (cell + gap) + cell - 3} textAnchor="end" style={{ fontSize: 9 }}>{l}</text>
          ))}
          {labels.map((l, j) => (j % 2 === 0) && (
            <text key={l} x={left + j * (cell + gap) + cell / 2} y={top + size + 12} textAnchor="middle" style={{ fontSize: 8 }}>{j + 1}</text>
          ))}
          <text x={left + size / 2} y={top + size + 32} textAnchor="middle">Predicted class (index)</text>
          {matrix.map((row, i) => row.map((v, j) => {
            const r = v / rowSums[i];
            return (
              <rect key={`${i}-${j}`} className="cm-cell"
                x={left + j * (cell + gap)} y={top + i * (cell + gap)} width={cell} height={cell} rx="2"
                fill={binFor(r).fill}
                onMouseMove={(e) => show(e, <>
                  <div><span className="muted">Actual</span> <b>{labels[i]}</b></div>
                  <div><span className="muted">Predicted</span> <b>{labels[j]}</b></div>
                  <div>{v} image{v === 1 ? '' : 's'} · {(r * 100).toFixed(1)}% of actual</div>
                </>)}
                onMouseLeave={hide}
              />
            );
          }))}
        </svg>
      </div>
      <div className="legend" aria-label="Share of actual class">
        <span style={{ color: 'var(--muted)' }}>Share of actual class:</span>
        {BINS.map((b) => (
          <span key={b.label}><i style={{ width: 12, height: 12, borderRadius: 3, background: b.fill, border: '1px solid var(--ring)' }} />{b.label}</span>
        ))}
      </div>
      {tip}
    </div>
  );
}
