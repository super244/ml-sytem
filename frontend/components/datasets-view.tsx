"use client";

import Link from "next/link";

import { formatBytes, formatCount } from "@/lib/formatting";
import { useLabMetadata } from "@/hooks/use-lab-metadata";

import { AppShell } from "@/components/layout/app-shell";
import { PageHeader } from "@/components/ui/page-header";
import { StatePanel } from "@/components/ui/state-panel";

export function DatasetsView() {
  const metadata = useLabMetadata();
  const dashboard = metadata.datasets;

  return (
    <AppShell>
      <section className="route-stack">
        <PageHeader
          eyebrow="Dataset Explorer"
          title="Packs, adapters, and preview slices"
          description="Browse the synthetic families, public adapters, and derived training and benchmark packs that feed Atlas Math Lab."
          metrics={[
            { label: "Datasets", value: formatCount(dashboard?.summary.num_datasets) },
            { label: "Total rows", value: formatCount(dashboard?.summary.total_rows), tone: "secondary" },
            { label: "Pack files", value: formatCount(dashboard?.packs?.length), tone: "accent" },
          ]}
          actions={
            <>
              <Link className="ghost-button small" href="/benchmarks">
                View benchmarks
              </Link>
              <Link className="primary-button small" href="/">
                Open solve workspace
              </Link>
            </>
          }
        />

        {metadata.loading && !dashboard ? (
          <StatePanel
            eyebrow="Loading"
            title="Dataset metadata is loading."
            description="Atlas is fetching the catalog, pack manifests, and preview examples."
            tone="loading"
          />
        ) : null}

        {metadata.error && !dashboard ? (
          <StatePanel
            eyebrow="Unavailable"
            title="Dataset metadata could not be loaded."
            description={metadata.error}
            tone="error"
          />
        ) : null}

        {dashboard ? (
          <>
            <div className="section-head">
              <div>
                <div className="section-title">Dataset Registry</div>
                <p className="hero-copy">
                  Each dataset card shows family, reasoning style, weighting intent, and preview
                  slices from the shared catalog.
                </p>
              </div>
            </div>

            <div className="card-grid">
              {dashboard.datasets.map((dataset) => (
                <article key={dataset.id} className="panel catalog-panel">
                  <div className="message-meta">
                    <span>{dataset.kind}</span>
                    <div className="pill-row">
                      <span className="status-pill">{dataset.family}</span>
                      {dataset.reasoning_style ? (
                        <span className="status-pill">{dataset.reasoning_style}</span>
                      ) : null}
                    </div>
                  </div>
                  <h2>{dataset.title}</h2>
                  <p>{dataset.description}</p>
                  <div className="badge-row">
                    <span className="status-pill">{dataset.topic}</span>
                    <span className="status-pill">{formatCount(dataset.num_rows)} rows</span>
                    <span className="status-pill">{formatBytes(dataset.size_bytes)}</span>
                  </div>
                  <div className="badge-row">
                    {dataset.usage ? <span className="status-pill">{dataset.usage}</span> : null}
                    {typeof dataset.default_weight === "number" ? (
                      <span className="status-pill">weight {dataset.default_weight.toFixed(2)}</span>
                    ) : null}
                  </div>
                  {dataset.preview_examples.slice(0, 2).map((preview) => (
                    <div key={preview.id} className="preview-block">
                      <strong>{preview.difficulty}</strong>
                      <p>{preview.question}</p>
                    </div>
                  ))}
                </article>
              ))}
            </div>

            <div className="section-head">
              <div>
                <div className="section-title">Derived Packs</div>
                <p className="hero-copy">
                  Derived packs are the ready-to-train or ready-to-benchmark slices created from
                  the processed corpus.
                </p>
              </div>
            </div>

            <div className="card-grid compact">
              {dashboard.packs?.map((pack) => (
                <article key={pack.id} className="panel catalog-panel">
                  <h2>{pack.id}</h2>
                  <p>{pack.description}</p>
                  <div className="badge-row">
                    <span className="status-pill">{formatCount(pack.num_rows)} rows</span>
                    <span className="status-pill">{formatBytes(pack.size_bytes)}</span>
                  </div>
                  <div className="preview-block subtle">
                    <strong>Artifact</strong>
                    <p>{pack.path}</p>
                  </div>
                </article>
              ))}
            </div>
          </>
        ) : null}
      </section>
    </AppShell>
  );
}
