'use client';

import Link from 'next/link';

import { formatBytes, formatCount } from '@/lib/formatting';
import { useLabMetadata } from '@/hooks/use-lab-metadata';
import { ROUTES } from '@/lib/routes';

import { AppShell } from '@/components/layout/app-shell';
import { PageHeader } from '@/components/ui/page-header';
import { StatePanel } from '@/components/ui/state-panel';

export function DatasetsView() {
  const metadata = useLabMetadata();
  const dashboard = metadata.datasets;
  const provenance = dashboard?.provenance;
  const processedManifest = provenance?.processed_manifest ?? null;
  const lineageSummary = provenance?.lineage_summary ?? null;
  const packManifests = provenance?.pack_manifests ?? [];

  return (
    <AppShell>
      <section className="route-stack">
        <PageHeader
          eyebrow="Dataset Explorer"
          title="Packs, adapters, and preview slices"
          description="Browse the synthetic families, public adapters, and derived training and benchmark packs that feed AI-Factory."
          metrics={[
            { label: 'Datasets', value: formatCount(dashboard?.summary.num_datasets) },
            {
              label: 'Total rows',
              value: formatCount(dashboard?.summary.total_rows),
              tone: 'secondary',
            },
            { label: 'Pack files', value: formatCount(dashboard?.packs?.length), tone: 'accent' },
          ]}
          actions={
            <>
              <Link className="ghost-button small" href={ROUTES.benchmarks}>
                View benchmarks
              </Link>
              <Link className="primary-button small" href={ROUTES.solve}>
                Open solve workspace
              </Link>
            </>
          }
        />

        {metadata.loading && !dashboard ? (
          <StatePanel
            eyebrow="Loading"
            title="Dataset metadata is loading."
            description="AI-Factory is fetching the catalog, pack manifests, and preview examples."
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
            {provenance ? (
              <>
                <div className="section-head">
                  <div>
                    <div className="section-title">Provenance Ledger</div>
                    <p className="hero-copy">
                      The processed corpus now carries build metadata, lineage aggregation, and pack
                      manifest details so you can inspect how the current dataset snapshot was
                      assembled.
                    </p>
                  </div>
                </div>

                <div className="card-grid compact">
                  <article className="panel catalog-panel">
                    <div className="message-meta">
                      <span>Processed manifest</span>
                      {processedManifest?.build?.build_id ? (
                        <span className="status-pill">{processedManifest.build.build_id}</span>
                      ) : null}
                    </div>
                    <h2>Corpus build provenance</h2>
                    <div className="preview-block">
                      <strong>Git / config</strong>
                      <p>
                        {processedManifest?.build?.git_sha ?? 'unknown git sha'}
                        {processedManifest?.build?.config_path ? (
                          <>
                            <br />
                            {processedManifest.build.config_path}
                          </>
                        ) : null}
                      </p>
                    </div>
                    <div className="badge-row">
                      <span className="status-pill">
                        {formatCount(processedManifest?.outputs?.length)} outputs
                      </span>
                      <span className="status-pill">
                        {formatCount(processedManifest?.source_lineage?.length)} lineage entries
                      </span>
                    </div>
                    {processedManifest?.metadata?.lineage_summary_path ? (
                      <div className="preview-block subtle">
                        <strong>Lineage summary</strong>
                        <p>{String(processedManifest.metadata.lineage_summary_path)}</p>
                      </div>
                    ) : null}
                  </article>

                  <article className="panel catalog-panel">
                    <div className="message-meta">
                      <span>Lineage summary</span>
                      {lineageSummary ? (
                        <span className="status-pill">
                          {formatCount(lineageSummary.total_records)} rows
                        </span>
                      ) : null}
                    </div>
                    <h2>Source aggregation</h2>
                    <div className="badge-row">
                      <span className="status-pill">
                        {formatCount(lineageSummary?.contamination?.contaminated_records)}{' '}
                        contaminated
                      </span>
                      <span className="status-pill">
                        {formatCount(lineageSummary?.contamination?.failure_cases)} failure cases
                      </span>
                    </div>
                    <div className="preview-block">
                      <strong>By loader</strong>
                      <p>
                        {lineageSummary
                          ? Object.entries(lineageSummary.by_loader)
                              .map(([loader, count]) => `${loader}: ${count}`)
                              .join(' · ')
                          : 'No lineage summary available'}
                      </p>
                    </div>
                    <div className="preview-block subtle">
                      <strong>By split</strong>
                      <p>
                        {lineageSummary
                          ? Object.entries(lineageSummary.by_split)
                              .map(([split, count]) => `${split}: ${count}`)
                              .join(' · ')
                          : 'No split summary available'}
                      </p>
                    </div>
                  </article>

                  <article className="panel catalog-panel">
                    <div className="message-meta">
                      <span>Pack manifests</span>
                      <span className="status-pill">
                        {formatCount(packManifests.length)} manifests
                      </span>
                    </div>
                    <h2>Derived pack provenance</h2>
                    {packManifests.slice(0, 3).map((packManifest) => (
                      <div
                        key={`${packManifest.pack_id ?? packManifest.description ?? 'pack'}`}
                        className="preview-block"
                      >
                        <strong>{packManifest.pack_id ?? 'pack'}</strong>
                        <p>
                          {packManifest.description ?? 'Derived pack'}
                          <br />
                          {packManifest.build?.build_id ?? 'unknown build'}
                        </p>
                      </div>
                    ))}
                    {packManifests.length === 0 ? (
                      <div className="preview-block subtle">
                        <p>No pack manifests were found for this snapshot.</p>
                      </div>
                    ) : null}
                  </article>
                </div>
              </>
            ) : null}

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
                    {typeof dataset.default_weight === 'number' ? (
                      <span className="status-pill">
                        weight {dataset.default_weight.toFixed(2)}
                      </span>
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
                  Derived packs are the ready-to-train or ready-to-benchmark slices created from the
                  processed corpus.
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
                  {pack.manifest_path ? (
                    <div className="preview-block subtle">
                      <strong>Manifest</strong>
                      <p>{pack.manifest_path}</p>
                    </div>
                  ) : null}
                  {pack.card_path ? (
                    <div className="preview-block subtle">
                      <strong>Card</strong>
                      <p>{pack.card_path}</p>
                    </div>
                  ) : null}
                  {pack.build_id ? (
                    <div className="badge-row">
                      <span className="status-pill">build {pack.build_id}</span>
                      {pack.stats?.num_records !== undefined ? (
                        <span className="status-pill">
                          {formatCount(Number(pack.stats.num_records))} manifest rows
                        </span>
                      ) : null}
                    </div>
                  ) : null}
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
