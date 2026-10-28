import { useEffect, useState } from 'react';
import { Camera, Leaf, Printer, RotateCcw, ScanSearch, Sparkles } from 'lucide-react';
import { api } from '../api';
import CameraCapture from '../components/CameraCapture';
import Dropzone from '../components/Dropzone';
import HeatmapViewer from '../components/HeatmapViewer';
import ResultCard from '../components/ResultCard';

const SAMPLES = [
  'tomato_early_blight', 'potato_late_blight', 'apple_scab',
  'corn_common_rust', 'grape_black_rot', 'pepper_healthy',
];

export default function Diagnose() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [camera, setCamera] = useState(false);
  const [tta, setTta] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [coverage, setCoverage] = useState(null);
  const [crops, setCrops] = useState([]);
  const [crop, setCrop] = useState('');

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);
  useEffect(() => {
    api.diseases().then((d) => {
      const plants = d.filter((x) => x.crop !== 'None');
      setCoverage({ conditions: plants.length, crops: new Set(plants.map((x) => x.crop)).size });
    }).catch(() => {});
    api.crops().then(setCrops).catch(() => {});
  }, []);

  const choose = (f) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError('');
    setCamera(false);
  };

  const loadSample = async (name) => {
    const blob = await fetch(`/samples/${name}.jpg`).then((r) => r.blob());
    choose(new File([blob], `${name}.jpg`, { type: 'image/jpeg' }));
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    try {
      setResult(await api.predict(file, { tta, crop }));
    } catch (e) {
      setError(e.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setFile(null); setPreview(null); setResult(null); setError(''); };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Diagnose a leaf</h1>
          <p>Upload or snap a photo of a single leaf. The model identifies {coverage ? `${coverage.conditions} diseases and healthy states across ${coverage.crops} crops` : 'plant diseases'}, highlights what it looked at, and suggests treatment.</p>
        </div>
        {result && (
          <button className="btn no-print" onClick={() => window.print()}><Printer size={16} /> Print report</button>
        )}
      </div>

      <div className="grid-2">
        <div className="stack">
          <div className="card stack">
            {camera ? (
              <CameraCapture onCapture={choose} onClose={() => setCamera(false)} />
            ) : preview ? (
              <HeatmapViewer src={preview} heatmap={result?.heatmap} preprocessing={result?.preprocessing} />
            ) : (
              <Dropzone onFiles={([f]) => choose(f)} />
            )}

            <div className="row no-print">
              {!camera && navigator.mediaDevices && (
                <button className="btn" onClick={() => setCamera(true)}><Camera size={16} /> Camera</button>
              )}
              {preview && <button className="btn ghost" onClick={reset}><RotateCcw size={16} /> New photo</button>}
              <div className="spacer" />
              {crops.length > 0 && (
                <select value={crop} onChange={(e) => setCrop(e.target.value)} aria-label="Crop"
                  title="Knowing the crop makes the diagnosis much more accurate">
                  <option value="">Crop: auto-detect</option>
                  {crops.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              )}
              <label className="toggle" title="Averages predictions over flipped and rotated copies">
                <input type="checkbox" checked={tta} onChange={(e) => setTta(e.target.checked)} />
                <Sparkles size={14} /> TTA
              </label>
            </div>

            <button className="btn primary block no-print" onClick={analyze} disabled={!file || loading}>
              {loading ? <><span className="spinner" /> Analyzing…</> : <><ScanSearch size={18} /> Analyze leaf</>}
            </button>
            {error && <div className="notice err">{error}</div>}
          </div>

          <div className="card no-print">
            <div className="card-title"><Leaf size={16} /><h3>Try a sample</h3><span className="sub">held-out test images</span></div>
            <div className="samples">
              {SAMPLES.map((s) => (
                <button key={s} onClick={() => loadSample(s)} title={s.replace(/_/g, ' ')}>
                  <img src={`/samples/${s}.jpg`} alt={s.replace(/_/g, ' ')} loading="lazy" />
                </button>
              ))}
            </div>
          </div>
        </div>

        <div>
          {result ? (
            <ResultCard r={result} />
          ) : (
            <div className="card empty">
              <ScanSearch size={40} />
              <h3 style={{ color: 'var(--ink)' }}>Results will appear here</h3>
              <p style={{ marginTop: 6 }}>
                For best results, fill the frame with one leaf, use even daylight and keep the photo sharp.
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
