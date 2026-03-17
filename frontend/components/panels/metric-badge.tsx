type MetricBadgeProps = {
  label: string;
  value: string;
  tone?: "default" | "accent" | "secondary";
};

export function MetricBadge({ label, value, tone = "default" }: MetricBadgeProps) {
  return (
    <div className={`metric-badge ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
