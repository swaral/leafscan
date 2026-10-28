// Same-origin by default (Vite proxy in dev, FastAPI static serving in prod).
const BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return res.json();
}

export const api = {
  health: () => request('/api/health'),
  model: () => request('/api/model'),
  diseases: () => request('/api/diseases'),
  crops: () => request('/api/crops'),
  predict(file, { tta = false, save = true, crop = '' } = {}) {
    const fd = new FormData();
    fd.append('file', file);
    const c = crop ? `&crop=${encodeURIComponent(crop)}` : '';
    return request(`/api/predict?tta=${tta}&save=${save}${c}`, { method: 'POST', body: fd });
  },
  predictBatch(files, { tta = false } = {}) {
    const fd = new FormData();
    files.forEach((f) => fd.append('files', f));
    return request(`/api/predict/batch?tta=${tta}`, { method: 'POST', body: fd });
  },
  history: (limit = 50, offset = 0, status = '') =>
    request(`/api/history?limit=${limit}&offset=${offset}${status ? `&status=${status}` : ''}`),
  historyStats: () => request('/api/history/stats'),
  deleteHistory: (id) => request(`/api/history/${id}`, { method: 'DELETE' }),
  clearHistory: () => request('/api/history', { method: 'DELETE' }),
};

export const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
