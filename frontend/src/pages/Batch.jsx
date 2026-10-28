import { useMemo, useState } from 'react';
import { Download, Layers, ScanSearch, Trash2, X } from 'lucide-react';
import { api, pct } from '../api';
import Dropzone from '../components/Dropzone';
import StatusBadge, { statusLabel } from '../components/StatusBadge';

const MAX = 20;

export default function Batch() {
  const [items, setItems] = useState([]); // {file, url, result?, error?}
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const add = (files) => {
    setItems((prev) => [...prev, ...files.map((f) => ({ file: f, url: URL.createObjectURL(f) }))].slice(0, MAX));
    setError('');
  };
  const remove = (i) => setItems((prev) => prev.filter((_, j) => j !== i));
  const clear = () => { items.forEach((it) => URL.revokeObjectURL(it.url)); setItems([]); };

  const run = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await api.predictBatch(items.map((it) => it.file));
      setItems((prev) => prev.map((it, i) => ({ ...it, result: res.items[i]?.result, error: res.items[i]?.error })));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const summary = useMemo(() => {
    const s = { healthy: 0, diseased: 0, uncertain: 0, no_leaf: 0 };
    items.forEach((it) => it.result && (s[it.result.status] += 1));
    return s;
  }, [items]);
  const done = items.some((it) => it.result);

  const exportCsv = () => {
    const rows = [['filename', 'status', 'crop', 'prediction', 'confidence', 'second_choice', 'second_probability']];
    items.forEach((it) => {
      const r = it.result;
      if (!r) return;
      const second = r.top_k[1];
      rows.push([it.file.name, r.status, r.disease.crop, r.disease.name, r.confidence.toFixed(4),
        second ? `${second.crop} ${second.name}` : '', second ? second.probability.toFixed(4) : '']);
    });
    const csv = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `leafscan-batch-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Batch scan</h1>
          <p>Scan up to {MAX} leaves at once, for example a quick field survey, and export the results as CSV.</p>
        </div>
        <div className="row">
          {done && <button className="btn" onClick={exportCsv}><Download size={16} /> Export CSV</button>}
          {items.length > 0 && <button className="btn ghost danger" onClick={clear}><Trash2 size={16} /> Clear</button>}
        </div>
      </div>

      <div className="stack">
        <Dropzone multiple onFiles={add} title="Drop several leaf photos" hint={`${items.length}/${MAX} selected · click or drag to add more`} />

        {items.length > 0 && (
          <div className="row">
            <button className="btn primary" onClick={run} disabled={loading}>
              {loading ? <><span className="spinner" /> Scanning {items.length}…</> : <><ScanSearch size={18} /> Scan {items.length} image{items.length > 1 ? 's' : ''}</>}
            </button>
            {done && (
              <div className="row" style={{ gap: 8 }}>
                {Object.entries(summary).filter(([, n]) => n).map(([k, n]) => (
                  <span key={k} className="chip">{statusLabel(k)}: <b>{n}</b></span>
                ))}
              </div>
            )}
          </div>
        )}
        {error && <div className="notice err">{error}</div>}

        {items.length > 0 ? (
          <div className="card table-wrap" style={{ padding: 8 }}>
            <table>
              <thead>
                <tr><th></th><th>File</th><th>Prediction</th><th>Status</th><th className="num">Confidence</th><th></th></tr>
              </thead>
              <tbody>
                {items.map((it, i) => (
                  <tr key={it.url}>
                    <td style={{ width: 60 }}><img className="thumb" src={it.url} alt="" /></td>
                    <td style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.file.name}</td>
                    <td>
                      {it.result ? <><b>{it.result.disease.name}</b><div style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>{it.result.disease.crop}</div></>
                        : it.error ? <span style={{ color: 'var(--critical-ink)' }}>{it.error}</span>
                        : <span style={{ color: 'var(--muted)' }}>Pending</span>}
                    </td>
                    <td>{it.result && <StatusBadge status={it.result.status} />}</td>
                    <td className="num">{it.result ? pct(it.result.confidence) : ''}</td>
                    <td className="num"><button className="btn ghost icon-btn" onClick={() => remove(i)} aria-label="Remove"><X size={16} /></button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="card empty"><Layers size={40} /><p>No images selected yet.</p></div>
        )}
      </div>
    </>
  );
}
