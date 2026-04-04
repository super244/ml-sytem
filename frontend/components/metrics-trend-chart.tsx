import { type MetricPoint } from '@/lib/api';

type MetricSeries = {
  name: string;
  values: number[];
};

const PREFERRED_SERIES = [
  'loss',
  'train_loss',
  'eval_loss',
  'accuracy',
  'parse_rate',
  'verifier_agreement_rate',
  'avg_latency_s',
];

function toSeries(points: MetricPoint[]): MetricSeries[] {
  const grouped = new Map<string, number[]>();
  for (const point of points) {
    if (typeof point.value !== 'number' || !Number.isFinite(point.value)) {
      continue;
    }
    const bucket = grouped.get(point.name) ?? [];
    bucket.push(point.value);
    grouped.set(point.name, bucket);
  }

  const orderedNames = [
    ...PREFERRED_SERIES.filter((name) => grouped.has(name)),
    ...[...grouped.keys()].filter((name) => !PREFERRED_SERIES.includes(name)),
  ];

  return orderedNames
    .map((name) => ({ name, values: (grouped.get(name) ?? []).slice(-48) }))
    .filter((series) => series.values.length >= 2)
    .slice(0, 3);
}

function formatValue(value: number): string {
  if (Math.abs(value) >= 100) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1);
  }
  return value.toFixed(3);
}

function buildPolyline(values: number[]): string {
  if (values.length < 2) {
    return '';
  }
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  return values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 100;
      const y = 100 - ((value - min) / range) * 100;
      return `${x},${y}`;
    })
    .join(' ');
}

export function MetricsTrendChart({ points }: { points: MetricPoint[] }) {
  const series = toSeries(points);
  if (!series.length) {
    return (
      <div className="metrics-empty-state">
        <strong>No numeric metric stream yet.</strong>
        <p>
          The control center will render learning curves as soon as the instance writes metric
          points.
        </p>
      </div>
    );
  }

  return (
    <div className="metrics-series-grid">
      {series.map((item) => {
        const latest = item.values[item.values.length - 1];
        return (
          <article key={item.name} className="metrics-series-card">
            <div className="message-meta">
              <span>{item.name}</span>
              <span className="status-pill">{item.values.length} points</span>
            </div>
            <div className="metrics-series-value">{formatValue(latest)}</div>
            <svg
              className="metrics-chart"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              role="img"
              aria-label={`${item.name} trend`}
            >
              <polyline
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
                points={buildPolyline(item.values)}
              />
            </svg>
          </article>
        );
      })}
    </div>
  );
}
