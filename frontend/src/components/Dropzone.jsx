import { useRef, useState } from 'react';
import { ImagePlus } from 'lucide-react';

export default function Dropzone({ onFiles, multiple = false, title, hint, children }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);

  const pick = (list) => {
    const files = Array.from(list || []).filter((f) => f.type.startsWith('image/'));
    if (files.length) onFiles(multiple ? files : [files[0]]);
  };

  return (
    <div
      className={`dropzone ${over ? 'over' : ''}`}
      role="button"
      tabIndex={0}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files); }}
    >
      <input
        ref={input}
        type="file"
        accept="image/*"
        multiple={multiple}
        hidden
        onChange={(e) => { pick(e.target.files); e.target.value = ''; }}
      />
      {children || (
        <div>
          <div className="big-icon"><ImagePlus size={26} /></div>
          <strong>{title || (over ? 'Drop it here' : 'Drag & drop a leaf photo')}</strong>
          <small>{hint || 'or click to browse · JPG, PNG, HEIC up to 10 MB'}</small>
        </div>
      )}
    </div>
  );
}
