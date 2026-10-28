import { useEffect, useMemo, useState } from 'react';
import { Activity, BarChart3, Cpu, Database, Globe, Grid3x3, Shuffle } from 'lucide-react';
import { api, pct } from '../api';
import ConfusionMatrix from '../components/ConfusionMatrix';
import LineChart from '../components/LineChart';
import ProbBars from '../components/ProbBars';

const EXT_NAMES = {
  potato_india: 'Potato · India (PLDD-UP)', potato_ethiopia: 'Potato · Ethiopia', maize_tanzania: 'Maize · Tanzania',
  bean_uganda: 'Bean · Uganda (iBean)', coffee_ecuador: 'Coffee · Ecuador (RoCoLe)', cassava_uganda: 'Cassava · Uganda',
};

const SOURCE_NAMES = {
  plantvillage: 'PlantVillage', plantdoc: 'PlantDoc', plantwild: 'PlantWild',
  ccmt: 'CCMT (Ghana)', plantpathology: 'Plant Pathology 2021',
};

export default function ModelPage() {
  const [info, setInfo] = useState(null);
  const [names, setNames] = useState({});
  const [error, setError] = useState('');
  const [curve, setCurve] = useState('acc');
  const [f1View, setF1View] = useState('lowest');

  useEffect(() => {
    Promise.all([api.model(), api.diseases()])
      .then(([m, d]) => {
        setInfo(m);
        setNames(Object.fromEntries(d.map((x) => [x.key, x])));
      })
      .catch((e) => setError(e.message));
  }, []);

  const label = (k) => {
    const d = names[k];
    if (!d) return k;
    return d.crop === 'None' ? 'Background (no leaf)' : `${d.crop} · ${d.name}`;
  };

  const m = info?.metrics;
  const ext = info?.external;
  const t = info?.training;

  const confused = useMemo(() => {
    if (!m) return [];
    const out = [];
    m.confusion_matrix.forEach((row, i) => row.forEach((v, j) => {
      if (i !== j && v > 0) out.push({ i, j, v, share: v / row.reduce((a, b) => a + b, 0) });
    }));
    return out.sort((a, b) => b.v - a.v).slice(0, 8);
  }, [m]);

  if (error) return <div className="notice err">{error}</div>;
  if (!info) return <div className="empty"><span className="spinner" style={{ margin: '0 auto' }} /></div>;
  if (!m) return <div className="notice">No evaluation metrics found. Run <code>python ml/evaluate.py</code>.</div>;

  const perClass = [...m.per_class].sort((a, b) => a.f1 - b.f1);
  const shownF1 = f1View === 'lowest' ? perClass.slice(0, 10) : perClass.slice().reverse();
  const epochs = t?.history.map((h) => h.epoch) || [];
  const counts = info.dataset_stats
    ? Object.entries(info.dataset_stats).map(([k, v]) => ({ k, n: v.train + v.val + v.test })).sort((a, b) => b.n - a.n)
    : [];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Model performance</h1>
          <p>
            {info.name}, exported to ONNX and evaluated end to end on {m.n_images.toLocaleString()} held-out test
            images the model never saw during training or tuning.
          </p>
        </div>
      </div>

      <div className="stack" style={{ gap: 20 }}>
        <div className="stat-row">
          <div className="stat hero"><div className="k">Internal test accuracy</div><div className="v">{pct(m.accuracy, 2)}</div><div className="s">{m.classes.length} classes · same sources as training</div></div>
          <div className="stat"><div className="k">Macro F1</div><div className="v">{m.macro_f1.toFixed(3)}</div><div className="s">class-balanced</div></div>
          <div className="stat"><div className="k">Top-3 accuracy</div><div className="v">{pct(m.top3_accuracy, 2)}</div></div>
          <div className="stat"><div className="k">Calibration error</div><div className="v">{pct(m.ece, 2)}</div>
            <div className="s">ECE after temperature scaling{t ? ` (T=${t.temperature.toFixed(2)})` : ''}</div></div>
          <div className="stat"><div className="k">CPU latency</div><div className="v">{m.latency_ms_p50.toFixed(0)} ms</div>
            <div className="s">p50 · {m.model_size_mb.toFixed(1)} MB model</div></div>
        </div>

        {ext && (
          <div className="card">
            <div className="card-title"><Globe size={16} /><h3>Accuracy on field photos outside training</h3>
              <span className="sub">{ext.images.toLocaleString()} photos not used for training</span></div>
            <div className="stat-row" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', marginBottom: 14 }}>
              <div className="stat"><div className="k">Exact disease</div><div className="v">{pct(ext.overall.accuracy, 1)}</div>
                {ext.previous_overall_accuracy != null && <div className="s">previous model {pct(ext.previous_overall_accuracy, 1)}</div>}</div>
              <div className="stat"><div className="k">With crop selected</div><div className="v">{pct(ext.overall.accuracy_given_crop, 1)}</div></div>
              <div className="stat"><div className="k">Correct crop</div><div className="v">{pct(ext.overall.crop_accuracy, 1)}</div></div>
              <div className="stat"><div className="k">Avg. confidence</div><div className="v">{pct(ext.overall.mean_confidence, 0)}</div>
                <div className="s">{ext.overall.mean_confidence - ext.overall.accuracy > 0.05 ? 'overconfident on new data' : 'matches accuracy (calibrated)'}</div></div>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Dataset</th><th className="num">Photos</th><th className="num">Accuracy</th><th className="num">Crop selected</th><th className="num">Previous model</th></tr></thead>
                <tbody>
                  {Object.entries(ext.per_dataset).sort((a, b) => b[1].accuracy - a[1].accuracy).map(([k, v]) => (
                    <tr key={k}><td>{EXT_NAMES[k] || k}
                      {v.unseen_dataset != null && <div style={{ color: 'var(--muted)', fontSize: '0.78rem' }}>
                        {v.unseen_dataset ? 'never seen in training' : 'held-out 30% of a training dataset'}</div>}</td>
                      <td className="num">{v.n.toLocaleString()}</td>
                      <td className="num">{pct(v.accuracy, 1)}</td><td className="num">{pct(v.accuracy_given_crop, 1)}</td>
                      <td className="num">{v.previous_accuracy != null ? pct(v.previous_accuracy, 1) : '–'}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: 8 }}>{ext.scope}</p>
          </div>
        )}

        <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)' }}>
          {t && (
            <div className="card">
              <div className="card-title">
                <Activity size={16} /><h3>Training curves</h3>
                <div className="spacer" />
                <div className="seg">
                  <button className={curve === 'acc' ? 'on' : ''} onClick={() => setCurve('acc')}>Accuracy</button>
                  <button className={curve === 'loss' ? 'on' : ''} onClick={() => setCurve('loss')}>Loss</button>
                </div>
              </div>
              <div className="legend" style={{ marginBottom: 6 }}>
                <span><i style={{ background: 'var(--series-1)' }} />Validation</span>
                <span style={{ color: 'var(--series-2)' }}><i className="dash" style={{ background: 'var(--series-2)' }} /><span style={{ color: 'var(--ink-2)' }}>Training (augmented)</span></span>
              </div>
              <LineChart
                x={epochs}
                yFormat={curve === 'acc' ? (v) => pct(v, 1) : (v) => v.toFixed(2)}
                yMax={curve === 'acc' ? 1 : undefined}
                series={curve === 'acc' ? [
                  { key: 'val', label: 'Validation', color: 'var(--series-1)', endLabel: true, values: t.history.map((h) => h.val_acc) },
                  { key: 'train', label: 'Training', color: 'var(--series-2)', dashed: true, values: t.history.map((h) => h.train_acc) },
                ] : [
                  { key: 'val', label: 'Validation', color: 'var(--series-1)', endLabel: true, values: t.history.map((h) => h.val_loss) },
                  { key: 'train', label: 'Training', color: 'var(--series-2)', dashed: true, values: t.history.map((h) => h.train_loss) },
                ]}
              />
              <p style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: 6 }}>
                Training loss includes label smoothing and runs on augmented data, so it sits above validation loss.
              </p>
            </div>
          )}

          <div className="card">
            <div className="card-title">
              <BarChart3 size={16} /><h3>F1 score by class</h3>
              <div className="spacer" />
              <div className="seg">
                <button className={f1View === 'lowest' ? 'on' : ''} onClick={() => setF1View('lowest')}>Lowest 10</button>
                <button className={f1View === 'all' ? 'on' : ''} onClick={() => setF1View('all')}>All</button>
              </div>
            </div>
            <div style={{ maxHeight: 330, overflowY: 'auto', paddingRight: 4 }}>
              <ProbBars
                format={(v) => v.toFixed(3)}
                items={shownF1.map((c) => ({
                  id: c.class, label: label(c.class), value: c.f1, highlight: false,
                  extra: <div className="muted">P {c.precision.toFixed(3)} · R {c.recall.toFixed(3)} · n={c.support}</div>,
                }))}
              />
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-title"><Grid3x3 size={16} /><h3>Confusion matrix</h3><span className="sub">rows = actual, columns = predicted</span></div>
          <ConfusionMatrix matrix={m.confusion_matrix} labels={m.classes.map((k, i) => `${i + 1}. ${label(k)}`)} />
        </div>

        <div className="grid-2" style={{ gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)' }}>
          <div className="card">
            <div className="card-title"><Shuffle size={16} /><h3>Most confused pairs</h3></div>
            {confused.length ? (
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Actual</th><th>Predicted as</th><th className="num">Images</th></tr></thead>
                  <tbody>
                    {confused.map((c) => (
                      <tr key={`${c.i}-${c.j}`}>
                        <td>{label(m.classes[c.i])}</td>
                        <td>{label(m.classes[c.j])}</td>
                        <td className="num">{c.v} <span style={{ color: 'var(--muted)' }}>({pct(c.share, 1)})</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <p style={{ color: 'var(--muted)' }}>No confusions on the test set.</p>}
          </div>

          <div className="card">
            <div className="card-title"><Database size={16} /><h3>Dataset: images per class</h3><span className="sub">{m.per_source ? `${Object.keys(m.per_source).length} sources` : 'PlantVillage'}</span></div>
            <div style={{ maxHeight: 330, overflowY: 'auto', paddingRight: 4 }}>
              <ProbBars
                max={counts[0]?.n || 1}
                format={(v) => v.toLocaleString()}
                items={counts.map((c) => ({ id: c.k, label: label(c.k), value: c.n, highlight: false }))}
              />
            </div>
          </div>
        </div>

        {m.per_source && (
          <div className="card">
            <div className="card-title"><Database size={16} /><h3>Accuracy by data source</h3><span className="sub">held-out test images</span></div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Source</th><th>Photos</th><th className="num">Test images</th><th className="num">Accuracy</th><th className="num">Macro F1</th><th className="num">Top-5</th></tr></thead>
                <tbody>
                  {Object.entries(m.per_source).map(([src, v]) => (
                    <tr key={src}>
                      <td><b>{SOURCE_NAMES[src] || src}</b></td>
                      <td style={{ color: 'var(--ink-2)' }}>{src === 'plantvillage' ? 'Lab' : 'Field / web'}</td>
                      <td className="num">{v.n_images.toLocaleString()}</td>
                      <td className="num">{pct(v.accuracy, 1)}</td>
                      <td className="num">{v.macro_f1.toFixed(3)}</td>
                      <td className="num">{pct(v.top5_accuracy, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {t && (
          <div className="card">
            <div className="card-title"><Cpu size={16} /><h3>Training setup</h3></div>
            <div className="chips">
              <span className="chip">Backbone: {t.config.backbone}</span>
              {t.config.params_millions && <span className="chip">{t.config.params_millions}M parameters</span>}
              {t.config.trainable_from_block !== undefined && <span className="chip">Fine-tuned from block {t.config.trainable_from_block}</span>}
              <span className="chip">{t.history.length} epochs</span>
              <span className="chip">AdamW + {t.config.warmup !== undefined ? 'warmup/cosine' : 'one-cycle'} LR</span>
              <span className="chip">Label smoothing {t.config.label_smoothing}</span>
              {t.config.mixup !== undefined
                ? <span className="chip">Mixup/CutMix, EMA weights, class-balanced sampling</span>
                : <span className="chip">Class-weighted loss</span>}
              <span className="chip">Augmentation: crop, flip, rotate, color{t.config.mixup !== undefined ? ', random erasing' : ', blur'}</span>
              <span className="chip">Temperature scaling: ECE {pct(t.val_ece_before, 2)} → {pct(t.val_ece_after, 2)} (val)</span>
              <span className="chip">Split 80/10/10 stratified{m.per_source ? ', duplicate-group aware' : ''}</span>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
