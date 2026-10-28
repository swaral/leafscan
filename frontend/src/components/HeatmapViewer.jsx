import { useState } from 'react';

const GATE_TEXT = {
  ok: 'Leaf isolated: background suppressed, colour normalised.',
  too_small: 'No clear single leaf found, so the full photo was kept (colour normalised).',
  fills_frame: 'The leaf fills the frame, so no background to remove (colour normalised).',
  fragmented: 'Several separate objects detected, so the full photo was kept (colour normalised).',
  cuts_leaf: 'Removing the background would have cut into leaf tissue, so the full photo was kept (colour normalised).',
};

/**
 * Shows the uploaded image with the model's class-activation heatmap overlaid.
 * When the model uses leaf preprocessing, the heatmap is drawn on the image the model
 * actually saw, and a "Model input" view shows that image on its own.
 */
export default function HeatmapViewer({ src, heatmap, preprocessing }) {
  const modelInput = preprocessing?.image;
  const [mode, setMode] = useState('overlay');
  const [opacity, setOpacity] = useState(0.6);
  const modes = ['original', ...(modelInput ? ['model input'] : []), 'overlay', 'heatmap'];
  const base = mode === 'original' || !modelInput ? src : modelInput;

  return (
    <div className="stack">
      <div className="preview-box">
        <div className="stacked-img">
          <img src={base} alt={mode === 'original' ? 'Uploaded leaf' : 'Image as seen by the model'} />
          {heatmap && (mode === 'overlay' || mode === 'heatmap') && (
            <img className="heat" src={heatmap} alt="Model attention heatmap" style={{ opacity: mode === 'heatmap' ? 1 : opacity }} />
          )}
        </div>
      </div>
      {heatmap && (
        <div className="row no-print">
          <div className="seg" role="tablist" aria-label="Image view">
            {modes.map((m) => (
              <button key={m} className={mode === m ? 'on' : ''} onClick={() => setMode(m)}>
                {m[0].toUpperCase() + m.slice(1)}
              </button>
            ))}
          </div>
          {mode === 'overlay' && (
            <label className="row" style={{ gap: 8, fontSize: '0.85rem', color: 'var(--ink-2)' }}>
              Opacity
              <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
            </label>
          )}
        </div>
      )}
      {preprocessing && (
        <p style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
          {GATE_TEXT[preprocessing.gate] || 'Colour normalised.'}
          {preprocessing.preprocess_ms ? ` (${(preprocessing.preprocess_ms / 1000).toFixed(1)} s)` : ''}
        </p>
      )}
      {heatmap && (
        <p style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
          Bright areas show the regions that most influenced the prediction (class activation map).
        </p>
      )}
    </div>
  );
}
