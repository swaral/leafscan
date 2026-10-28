import { useEffect, useRef, useState } from 'react';
import { Aperture, RefreshCcw, X } from 'lucide-react';

export default function CameraCapture({ onCapture, onClose }) {
  const video = useRef(null);
  const [stream, setStream] = useState(null);
  const [facing, setFacing] = useState('environment');
  const [error, setError] = useState('');

  useEffect(() => {
    let active = true;
    let s;
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 } }, audio: false })
      .then((st) => {
        if (!active) { st.getTracks().forEach((t) => t.stop()); return; }
        s = st;
        setStream(st);
        if (video.current) video.current.srcObject = st;
      })
      .catch(() => setError('Camera unavailable: permission denied or no camera found.'));
    return () => { active = false; s?.getTracks().forEach((t) => t.stop()); };
  }, [facing]);

  const capture = () => {
    const v = video.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement('canvas');
    c.width = v.videoWidth;   // keep full resolution and aspect ratio
    c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    c.toBlob((b) => b && onCapture(new File([b], `camera-${Date.now()}.jpg`, { type: 'image/jpeg' })), 'image/jpeg', 0.92);
  };

  return (
    <div className="camera stack">
      {error ? <div className="notice err">{error}</div> : <video ref={video} autoPlay playsInline muted />}
      <div className="row">
        <button className="btn primary" onClick={capture} disabled={!stream}><Aperture size={16} /> Capture</button>
        <button className="btn" onClick={() => setFacing((f) => (f === 'environment' ? 'user' : 'environment'))}>
          <RefreshCcw size={16} /> Flip
        </button>
        <div className="spacer" />
        <button className="btn ghost" onClick={onClose}><X size={16} /> Close</button>
      </div>
    </div>
  );
}
