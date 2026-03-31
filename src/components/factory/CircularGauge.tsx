interface CircularGaugeProps {
  value: number;
  label: string;
  unit?: string;
  size?: 'sm' | 'md' | 'lg';
}

const CircularGauge = ({ value, label, unit = '%', size = 'md' }: CircularGaugeProps) => {
  const sizes = { sm: 64, md: 80, lg: 100 };
  const s = sizes[size];
  const strokeWidth = size === 'sm' ? 4 : 5;
  const r = (s - strokeWidth * 2) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference - (value / 100) * circumference;

  const color = value >= 90 ? 'hsl(348, 100%, 50%)' : value >= 75 ? 'hsl(18, 100%, 57%)' : 'hsl(155, 100%, 50%)';
  const textSize = size === 'sm' ? 'text-sm' : size === 'md' ? 'text-lg' : 'text-xl';

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={s} height={s} className="-rotate-90">
        <circle cx={s / 2} cy={s / 2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={strokeWidth} />
        <circle
          cx={s / 2} cy={s / 2} r={r} fill="none"
          stroke={color} strokeWidth={strokeWidth}
          strokeDasharray={circumference} strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 800ms cubic-bezier(0.4,0,0.2,1), stroke 300ms' }}
        />
      </svg>
      <span className={`metric-value ${textSize} -mt-${s === 64 ? 10 : s === 80 ? 12 : 14}`} style={{ marginTop: -(s / 2 + (size === 'sm' ? 6 : 8)), position: 'relative' }}>
        {value}{unit}
      </span>
      <span className="section-label mt-1">{label}</span>
    </div>
  );
};

export default CircularGauge;
