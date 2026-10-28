import { useEffect, useMemo, useState } from 'react';
import { BookOpen, X } from 'lucide-react';
import { api } from '../api';
import { DiseaseDetails } from '../components/ResultCard';

export default function Library() {
  const [items, setItems] = useState([]);
  const [q, setQ] = useState('');
  const [crop, setCrop] = useState('');
  const [open, setOpen] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.diseases().then((d) => setItems(d.filter((x) => x.crop !== 'None'))).catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && setOpen(null);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const crops = useMemo(() => [...new Set(items.map((d) => d.crop))].sort(), [items]);
  const shown = items.filter((d) =>
    (!crop || d.crop === crop) &&
    (!q || `${d.crop} ${d.name} ${d.pathogen || ''} ${d.type}`.toLowerCase().includes(q.toLowerCase())));
  const diseases = shown.filter((d) => d.type !== 'Healthy');

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Disease library</h1>
          <p>Every condition the model can recognise, with symptoms, treatment and prevention guidance.</p>
        </div>
        <div className="row">
          <input type="search" placeholder="Search diseases, pathogens…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search" />
          <select value={crop} onChange={(e) => setCrop(e.target.value)} aria-label="Filter by crop">
            <option value="">All crops ({crops.length})</option>
            {crops.map((c) => <option key={c}>{c}</option>)}
          </select>
        </div>
      </div>
      {error && <div className="notice err">{error}</div>}

      <div className="lib-grid">
        {diseases.map((d) => (
          <button key={d.key} className="card lib-card" onClick={() => setOpen(d)}>
            <div className="row" style={{ gap: 6 }}>
              <span className="chip">{d.crop}</span>
              <span className="chip">{d.type}</span>
              <span className={`chip sev-${d.severity}`}>{d.severity}</span>
            </div>
            <h3 style={{ marginTop: 10 }}>{d.name}</h3>
            {d.pathogen && <div style={{ fontSize: '0.82rem', color: 'var(--muted)', fontStyle: 'italic' }}>{d.pathogen}</div>}
            <p>{d.summary}</p>
          </button>
        ))}
      </div>
      {!diseases.length && !error && <div className="card empty"><BookOpen size={40} /><p>No matching diseases.</p></div>}

      {open && (
        <div className="modal-back" onClick={() => setOpen(null)}>
          <div className="card modal fade-in" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label={open.name}>
            <div className="row" style={{ marginBottom: 12 }}>
              <h2>{open.crop} · {open.name}</h2>
              <div className="spacer" />
              <button className="btn ghost icon-btn" onClick={() => setOpen(null)} aria-label="Close"><X size={18} /></button>
            </div>
            <DiseaseDetails d={open} />
          </div>
        </div>
      )}
    </>
  );
}
