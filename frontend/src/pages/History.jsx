import { useCallback, useEffect, useState } from 'react';
import { History as HistoryIcon, RefreshCw, Trash2, X } from 'lucide-react';
import { api, pct } from '../api';
import ProbBars from '../components/ProbBars';
import StatusBadge from '../components/StatusBadge';

const FILTERS = [['', 'All'], ['diseased', 'Diseased'], ['healthy', 'Healthy'], ['uncertain', 'Uncertain'], ['no_leaf', 'No leaf'], ['unusable_photo', 'Retake'], ['crop_mismatch', 'Crop mismatch']];
const PAGE = 25;

export default function History() {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState(null);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');

  const load = useCallback(async (offset = 0) => {
    try {
      const [page, st] = await Promise.all([api.history(PAGE, offset, status), api.historyStats()]);
      setItems((prev) => (offset ? [...prev, ...page.items] : page.items));
      setTotal(page.total);
      setStats(st);
      setError('');
    } catch (e) { setError(e.message); }
  }, [status]);

  useEffect(() => { load(0); }, [load]);

  const del = async (id) => { await api.deleteHistory(id); load(0); };
  const clearAll = async () => {
    if (!window.confirm('Delete all saved predictions? This cannot be undone.')) return;
    await api.clearHistory();
    load(0);
  };

  const bs = stats?.by_status || {};
  return (
    <>
      <div className="page-head">
        <div>
          <h1>History</h1>
          <p>Every saved diagnosis, stored in the API's SQLite database.</p>
        </div>
        <div className="row">
          <button className="btn" onClick={() => load(0)}><RefreshCw size={16} /> Refresh</button>
          {total > 0 && <button className="btn ghost danger" onClick={clearAll}><Trash2 size={16} /> Clear all</button>}
        </div>
      </div>

      {stats && stats.total > 0 && (
        <div className="grid-2" style={{ marginBottom: 20, gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)' }}>
          <div className="stat-row" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
            <div className="stat"><div className="k">Total scans</div><div className="v">{stats.total}</div></div>
            <div className="stat"><div className="k">Disease detected</div><div className="v">{bs.diseased || 0}</div>
              <div className="s">{stats.total ? pct((bs.diseased || 0) / stats.total, 0) : '0%'} of scans</div></div>
            <div className="stat"><div className="k">Healthy</div><div className="v">{bs.healthy || 0}</div></div>
            <div className="stat"><div className="k">No diagnosis given</div><div className="v">{(bs.uncertain || 0) + (bs.no_leaf || 0) + (bs.unusable_photo || 0) + (bs.crop_mismatch || 0)}</div>
              <div className="s">uncertain, no leaf, retake, crop mismatch</div></div>
          </div>
          <div className="card">
            <div className="card-title"><h3>Most frequent findings</h3><span className="sub">scans</span></div>
            {stats.top_classes.length ? (
              <ProbBars
                max={Math.max(...stats.top_classes.map((c) => c.count))}
                format={(v) => String(v)}
                items={stats.top_classes.slice(0, 6).map((c) => ({ id: c.class_key, label: c.name, sub: c.crop, value: c.count, highlight: false }))}
              />
            ) : <p style={{ color: 'var(--muted)' }}>No findings yet.</p>}
          </div>
        </div>
      )}

      <div className="row" style={{ marginBottom: 12 }}>
        <div className="seg">
          {FILTERS.map(([k, label]) => (
            <button key={k} className={status === k ? 'on' : ''} onClick={() => setStatus(k)}>{label}</button>
          ))}
        </div>
        <span style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>{total} result{total === 1 ? '' : 's'}</span>
      </div>

      {error && <div className="notice err" style={{ marginBottom: 12 }}>{error}</div>}

      {items.length ? (
        <div className="card table-wrap" style={{ padding: 8 }}>
          <table>
            <thead><tr><th></th><th>Prediction</th><th>Status</th><th className="num">Confidence</th><th>Date</th><th></th></tr></thead>
            <tbody>
              {items.map((it) => (
                <tr key={it.id}>
                  <td style={{ width: 60 }}>{it.thumbnail ? <img className="thumb" src={it.thumbnail} alt="" /> : <div className="thumb" />}</td>
                  <td><b>{it.name}</b><div style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>{it.crop} · {it.filename}</div></td>
                  <td><StatusBadge status={it.status} /></td>
                  <td className="num">{pct(it.confidence)}</td>
                  <td style={{ color: 'var(--ink-2)', whiteSpace: 'nowrap' }}>{new Date(it.created_at).toLocaleString()}</td>
                  <td className="num"><button className="btn ghost icon-btn" onClick={() => del(it.id)} aria-label="Delete"><X size={16} /></button></td>
                </tr>
              ))}
            </tbody>
          </table>
          {items.length < total && (
            <div style={{ padding: 10, textAlign: 'center' }}><button className="btn sm" onClick={() => load(items.length)}>Load more</button></div>
          )}
        </div>
      ) : (
        <div className="card empty"><HistoryIcon size={40} /><p>No predictions saved yet. Diagnose a leaf to get started.</p></div>
      )}
    </>
  );
}
