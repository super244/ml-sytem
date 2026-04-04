'use client';

import Link from 'next/link';

import { formatCount } from '@/lib/formatting';
import { useLabMetadata } from '@/hooks/use-lab-metadata';
import { ROUTES } from '@/lib/routes';

import { AppShell } from '@/components/layout/app-shell';
import { PageHeader } from '@/components/ui/page-header';
import { StatePanel } from '@/components/ui/state-panel';

export function BenchmarksView() {
  const metadata = useLabMetadata();

  return (
    <AppShell>
      <section className="route-stack">
        <PageHeader
          eyebrow="Benchmark Library"
          title="Held-out slices and verification suites"
          description="AI-Factory serves benchmark metadata directly from the same registry used by the offline evaluation pipeline."
          metrics={[
            { label: 'Benchmarks', value: formatCount(metadata.benchmarks.length) },
            { label: 'Models', value: formatCount(metadata.models.length), tone: 'secondary' },
            { label: 'Runs', value: formatCount(metadata.runs.length), tone: 'accent' },
          ]}
          actions={
            <>
              <Link className="ghost-button small" href={ROUTES.datasets}>
                Back to datasets
              </Link>
              <Link className="primary-button small" href={ROUTES.runs}>
                Inspect runs
              </Link>
            </>
          }
        />

        {metadata.loading && !metadata.benchmarks.length ? (
          <StatePanel
            eyebrow="Loading"
            title="Benchmark metadata is loading."
            description="AI-Factory is discovering held-out benchmark packs and verification suites."
            tone="loading"
          />
        ) : null}

        {metadata.error && !metadata.benchmarks.length ? (
          <StatePanel
            eyebrow="Unavailable"
            title="Benchmark metadata could not be loaded."
            description={metadata.error}
            tone="error"
          />
        ) : null}

        <div className="card-grid compact">
          {metadata.benchmarks.map((benchmark) => (
            <article key={benchmark.id} className="panel catalog-panel">
              <div className="message-meta">
                <span>{benchmark.id}</span>
                <div className="pill-row">
                  {benchmark.tags.map((tag) => (
                    <span key={`${benchmark.id}-${tag}`} className="status-pill">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
              <h2>{benchmark.title}</h2>
              <p>{benchmark.description}</p>
              <div className="preview-block subtle">
                <strong>Benchmark path</strong>
                <p>{benchmark.path}</p>
              </div>
            </article>
          ))}
        </div>
      </section>
    </AppShell>
  );
}
