export interface PerformanceMetrics {
  ttfb?: number;
  fcp?: number;
  lcp?: number;
  cls?: number;
  inp?: number;
}

function toMetricName(entryName: string): keyof PerformanceMetrics | undefined {
  if (entryName === 'first-contentful-paint') return 'fcp';
  if (entryName === 'largest-contentful-paint') return 'lcp';
  return undefined;
}

export class PerformanceMonitor {
  private readonly metrics: PerformanceMetrics = {};

  start(): void {
    if (typeof window === 'undefined' || typeof PerformanceObserver === 'undefined') {
      return;
    }

    const nav = performance.getEntriesByType('navigation')[0] as
      | PerformanceNavigationTiming
      | undefined;
    if (nav) {
      this.metrics.ttfb = nav.responseStart;
    }

    const paintObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const key = toMetricName(entry.name);
        if (key) {
          this.metrics[key] = entry.startTime;
        }
      }
    });
    paintObserver.observe({ type: 'paint', buffered: true });

    const lcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const last = entries[entries.length - 1];
      if (last) {
        this.metrics.lcp = last.startTime;
      }
    });
    lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true });

    const clsObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries() as PerformanceEntry[]) {
        const shifted = entry as PerformanceEntry & { hadRecentInput?: boolean; value?: number };
        if (!shifted.hadRecentInput && typeof shifted.value === 'number') {
          this.metrics.cls = (this.metrics.cls ?? 0) + shifted.value;
        }
      }
    });
    clsObserver.observe({ type: 'layout-shift', buffered: true });
  }

  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }
}

export const performanceMonitor = new PerformanceMonitor();
