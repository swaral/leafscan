import { useCallback, useState } from 'react';

/** Cursor-following tooltip. Returns [element, show(event, content), hide]. */
export function useTooltip() {
  const [tip, setTip] = useState(null);
  const show = useCallback((e, content) => {
    const pad = 14;
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    if (x > window.innerWidth - 300) x = e.clientX - 290;
    if (y > window.innerHeight - 90) y = e.clientY - 70;
    setTip({ x, y, content });
  }, []);
  const hide = useCallback(() => setTip(null), []);
  const el = tip ? (
    <div className="tip" style={{ left: tip.x, top: tip.y }} role="tooltip">
      {tip.content}
    </div>
  ) : null;
  return [el, show, hide];
}
