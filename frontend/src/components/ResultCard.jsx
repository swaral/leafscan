import { AlertTriangle, Bug, Info, Pill, ShieldCheck, Stethoscope } from 'lucide-react';
import { pct } from '../api';
import ProbBars from './ProbBars';
import StatusBadge from './StatusBadge';

export function DiseaseDetails({ d }) {
  const healthy = d.type === 'Healthy' || d.type === 'None';
  return (
    <div className="stack">
      <div className="chips">
        <span className="chip">{d.crop}</span>
        <span className="chip">{d.type}</span>
        {d.pathogen && <span className="chip"><i>{d.pathogen}</i></span>}
        {!healthy && <span className={`chip sev-${d.severity}`}>Severity: {d.severity}</span>}
      </div>
      <p style={{ color: 'var(--ink-2)' }}>{d.summary}</p>
      <div className="info-cols">
        {d.symptoms?.length > 0 && (
          <div><h4><Stethoscope size={14} /> Symptoms</h4><ul>{d.symptoms.map((s) => <li key={s}>{s}</li>)}</ul></div>
        )}
        {d.treatment?.length > 0 && (
          <div><h4><Pill size={14} /> Treatment</h4><ul>{d.treatment.map((s) => <li key={s}>{s}</li>)}</ul></div>
        )}
        {d.prevention?.length > 0 && (
          <div><h4><ShieldCheck size={14} /> {healthy ? 'Keep it healthy' : 'Prevention'}</h4><ul>{d.prevention.map((s) => <li key={s}>{s}</li>)}</ul></div>
        )}
      </div>
    </div>
  );
}

export default function ResultCard({ r }) {
  const d = r.disease;
  const title = d.type === 'Healthy' ? `Healthy ${d.crop.toLowerCase()} leaf` : d.name;
  return (
    <div className="stack fade-in">
      <div className="card">
        <div className="result-hero">
          <div>
            <div className="label">Diagnosis</div>
            <h2>{title}</h2>
            <div className="row" style={{ gap: 8 }}>
              <StatusBadge status={r.status} />
              {d.crop !== 'None' && <span className="crop">{d.crop}</span>}
            </div>
          </div>
          <div className="conf">
            <div className="v">{pct(r.confidence)}</div>
            <div className="k">calibrated confidence</div>
          </div>
        </div>

        {(r.status === 'uncertain' || r.quality.warnings.length > 0) && (
          <div className="stack" style={{ marginTop: 14, gap: 8 }}>
            {r.status === 'uncertain' && (
              <div className="notice warn"><AlertTriangle size={16} />
                The model is not confident. Compare the top candidates below, or retake a closer, well-lit photo of a single leaf.
              </div>
            )}
            {r.quality.warnings.map((w) => (
              <div key={w} className="notice warn"><AlertTriangle size={16} />{w}</div>
            ))}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title"><Bug size={16} /><h3>Top predictions</h3><span className="sub">probability</span></div>
        <ProbBars items={r.top_k.map((k) => ({
          id: k.key,
          label: k.crop === 'None' ? k.name : `${k.name}`,
          sub: k.crop === 'None' ? null : k.crop,
          value: k.probability,
        }))} />
      </div>

      <div className="card">
        <div className="card-title"><Info size={16} /><h3>About this condition</h3></div>
        <DiseaseDetails d={d} />
      </div>

      <div className="row" style={{ color: 'var(--muted)', fontSize: '0.8rem', gap: 14 }}>
        <span>Inference {r.inference_ms} ms</span>
        <span>{r.tta ? 'Test-time augmentation on' : 'Single pass'}</span>
        <span>Uncertainty (entropy) {r.entropy.toFixed(3)}</span>
        {r.id && <span>History #{r.id}</span>}
      </div>
      <p style={{ color: 'var(--muted)', fontSize: '0.78rem' }}>
        AI-assisted guidance only. Confirm serious diagnoses with a local agricultural extension officer and follow product label instructions.
      </p>
    </div>
  );
}
