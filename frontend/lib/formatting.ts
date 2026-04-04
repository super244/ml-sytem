export function formatCount(value?: number | null): string {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'n/a';
  }
  return value.toLocaleString();
}

export function formatBytes(value?: number | null): string {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'n/a';
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 ** 2) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  if (value < 1024 ** 3) {
    return `${(value / 1024 ** 2).toFixed(1)} MB`;
  }
  return `${(value / 1024 ** 3).toFixed(1)} GB`;
}

export function formatPercent(value?: number | null, digits = 0): string {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'n/a';
  }
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatLatency(value?: number | null): string {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'n/a';
  }
  return `${value.toFixed(2)}s`;
}

export function formatFixed(value?: number | null, digits = 3): string {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 'n/a';
  }
  return value.toFixed(digits);
}
