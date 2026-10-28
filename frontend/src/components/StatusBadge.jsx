import { CheckCircle2, HelpCircle, ImageOff, ShieldAlert } from 'lucide-react';

const META = {
  healthy: { label: 'Healthy', Icon: CheckCircle2 },
  diseased: { label: 'Disease detected', Icon: ShieldAlert },
  uncertain: { label: 'Uncertain', Icon: HelpCircle },
  no_leaf: { label: 'No leaf', Icon: ImageOff },
};

export default function StatusBadge({ status, short = false }) {
  const m = META[status] || META.uncertain;
  return (
    <span className={`badge ${status}`}>
      <m.Icon aria-hidden />
      {short ? m.label.split(' ')[0] : m.label}
    </span>
  );
}

export const statusLabel = (s) => (META[s] || META.uncertain).label;
