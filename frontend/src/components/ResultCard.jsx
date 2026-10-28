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

const NO_DIAGNOSIS = {
  unusable_photo: {
    title: 'Retake the photo',
    tips: ['Photograph one leaf so it fills most of the frame', 'Use even daylight, avoid deep shade and glare',
      'Hold steady and tap the leaf to focus'],
  },
  no_leaf: {
    title: 'No leaf found',
    tips: ['Point the camera at a single leaf, close up', 'Make sure the leaf, not the background, is in focus'],
  },
  crop_mismatch: {
    title: 'The selected crop does not match',
    tips: ['Check the crop selector, or set it to auto-detect', 'If the crop is right, retake a clearer photo'],
  },
  uncertain: {
    title: 'Not sure: possible conditions',
    tips: ['Retake a closer, well-lit photo of the most affected leaf', 'Select the crop above if you know it',
      'For a firm diagnosis, ask a local agricultural extension officer'],
  },
};

export default function ResultCard({ r }) {
  const d = r.disease;
  const noDx = NO_DIAGNOSIS[r.status];
  const title = noDx ? noDx.title : d.type === 'Healthy' ? `Healthy ${d.crop.toLowerCase()} leaf` : d.name;
  // Candidates are shown for diagnoses and 'uncertain'; for a crop mismatch they would be the
  // forced (wrong-crop) classes, so only the model's own reading (in the reason) is shown.
  const showCandidates = !noDx || r.status === 'uncertain';
  return (
    <div className="stack fade-in">
      <div className="card">
        <div className="result-hero">
          <div>
            <div className="label">{noDx ? 'No diagnosis' : 'Diagnosis'}</div>
            <h2>{title}</h2>
            <div className="row" style={{ gap: 8 }}>
              <StatusBadge status={r.status} />
              {!noDx && d.crop !== 'None' && <span className="crop">{d.crop}</span>}
            </div>
          </div>
          {!noDx && (
            <div className="conf">
              <div className="v">{pct(r.confidence)}</div>
              <div className="k">estimated confidence</div>
            </div>
          )}
        </div>

        {noDx && (
          <div className="stack" style={{ marginTop: 14, gap: 8 }}>
            <div className="notice warn"><AlertTriangle size={16} />{r.decision_reason}</div>
            <ul style={{ margin: 0, paddingLeft: 20, color: 'var(--ink-2)', fontSize: '0.9rem' }}>
              {noDx.tips.map((t) => <li key={t}>{t}</li>)}
            </ul>
          </div>
        )}
        {!noDx && r.quality.warnings.length > 0 && (
          <div className="stack" style={{ marginTop: 14, gap: 8 }}>
            {r.quality.warnings.map((w) => (
              <div key={w} className="notice warn"><AlertTriangle size={16} />{w}</div>
            ))}
          </div>
        )}
      </div>

      {showCandidates && (
        <div className="card">
          <div className="card-title">
            <Bug size={16} /><h3>{noDx ? 'Possible conditions (not a diagnosis)' : 'Top predictions'}</h3>
            <span className="sub">probability</span>
          </div>
          <ProbBars items={r.top_k.map((k) => ({
            id: k.key,
            label: k.crop === 'None' ? k.name : `${k.name}`,
            sub: k.crop === 'None' ? null : k.crop,
            value: k.probability,
          }))} />
        </div>
      )}

      {!noDx && (
        <div className="card">
          <div className="card-title"><Info size={16} /><h3>About this condition</h3></div>
          <DiseaseDetails d={d} />
        </div>
      )}

      <div className="row" style={{ color: 'var(--muted)', fontSize: '0.8rem', gap: 14 }}>
        <span>Inference {r.inference_ms} ms</span>
        <span>{r.tta ? 'Test-time augmentation on' : 'Single pass'}</span>
        {r.crop_filter && <span>Crop: {r.crop_filter}</span>}
        {r.id && <span>History #{r.id}</span>}
      </div>
      <p style={{ color: 'var(--muted)', fontSize: '0.78rem' }}>
        AI-assisted guidance only. In tests on field photos outside its training data the model was right
        about 9 times in 10, but it was only tested on a few crops and can be wrong on unfamiliar farms.
        Confirm serious diagnoses with a local agricultural extension officer and follow product label
        instructions.
      </p>
    </div>
  );
}
