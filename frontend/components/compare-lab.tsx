'use client';

import Link from 'next/link';
import { useState, useTransition } from 'react';

import {
  compareModels,
  type CompareResponse,
  type Difficulty,
  type OutputFormat,
  type SolverMode,
} from '@/lib/api';
import { FALLBACK_EXAMPLES, FALLBACK_MODELS, FALLBACK_PROMPTS } from '@/lib/demo-content';
import { formatCount, formatLatency, formatPercent } from '@/lib/formatting';
import { DIFFICULTY_OPTIONS, OUTPUT_FORMAT_OPTIONS, SOLVER_MODE_OPTIONS } from '@/lib/options';
import {
  isDemoMode,
  pickPrimaryModel,
  pickPromptPreset,
  pickSecondaryModel,
} from '@/lib/runtime-mode';
import { ROUTES } from '@/lib/routes';
import { useLabMetadata } from '@/hooks/use-lab-metadata';

import { MathBlock } from '@/components/math-block';
import { AppShell } from '@/components/layout/app-shell';
import { MetricBadge } from '@/components/panels/metric-badge';
import { ModelChip } from '@/components/panels/model-chip';
import { PageHeader } from '@/components/ui/page-header';
import { StatePanel } from '@/components/ui/state-panel';

export function CompareLab() {
  const metadata = useLabMetadata();
  const demoMode = isDemoMode();
  const promptLibrary = metadata.promptLibrary;
  const models = metadata.models.length ? metadata.models : demoMode ? FALLBACK_MODELS : [];
  const examples =
    promptLibrary && promptLibrary.examples.length > 0
      ? promptLibrary.examples.slice(0, 4)
      : demoMode
        ? FALLBACK_EXAMPLES
        : [];
  const promptPresets =
    promptLibrary && promptLibrary.presets.length > 0
      ? promptLibrary.presets
      : demoMode
        ? FALLBACK_PROMPTS
        : [];
  const metadataDegraded =
    !demoMode && (metadata.status?.status === 'degraded' || Boolean(metadata.error));

  const [question, setQuestion] = useState('Evaluate \\int_0^1 x e^{x^2} dx.');
  const [primaryModel, setPrimaryModel] = useState('');
  const [secondaryModel, setSecondaryModel] = useState('');
  const [difficultyTarget, setDifficultyTarget] = useState<Difficulty>('hard');
  const [solverMode, setSolverMode] = useState<SolverMode>('rigorous');
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('text');
  const [promptPreset, setPromptPreset] = useState('');
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const compareDisabled =
    isPending ||
    !question.trim() ||
    !primaryModel ||
    !secondaryModel ||
    !promptPreset ||
    primaryModel === secondaryModel ||
    metadataDegraded;

  const modelsStr = JSON.stringify(models);
  const [prevModelsStr, setPrevModelsStr] = useState(modelsStr);
  if (modelsStr !== prevModelsStr) {
    setPrevModelsStr(modelsStr);
    setPrimaryModel(pickPrimaryModel(models, primaryModel));
  }

  const [prevCompareVariant, setPrevCompareVariant] = useState(primaryModel);
  if (modelsStr !== prevModelsStr || primaryModel !== prevCompareVariant) {
    setPrevCompareVariant(primaryModel);
    setSecondaryModel(pickSecondaryModel(models, primaryModel, secondaryModel));
  }

  const promptPresetsStr = JSON.stringify(promptPresets);
  const [prevPresetsStr, setPrevPresetsStr] = useState(promptPresetsStr);
  if (promptPresetsStr !== prevPresetsStr) {
    setPrevPresetsStr(promptPresetsStr);
    setPromptPreset(pickPromptPreset(promptPresets, ['atlas_rigorous'], promptPreset));
  }

  return (
    <AppShell>
      <section className="route-stack">
        <PageHeader
          eyebrow="Compare Workspace"
          title="Base vs Specialist"
          description="Run the same prompt through two model variants and inspect how answer quality, verification behavior, and latency diverge under identical solver settings."
          metrics={[
            { label: 'Models', value: formatCount(models.length) },
            {
              label: 'Prompt presets',
              value: formatCount(promptPresets.length),
              tone: 'secondary',
            },
            { label: 'Benchmarks', value: formatCount(metadata.benchmarks.length), tone: 'accent' },
          ]}
          actions={
            <>
              <Link className="ghost-button small" href={ROUTES.solve}>
                Back to solve
              </Link>
              <Link className="primary-button small" href={ROUTES.runs}>
                Inspect runs
              </Link>
            </>
          }
        />

        {metadata.error ? (
          <StatePanel
            eyebrow={demoMode ? 'Demo Metadata' : 'Metadata Degraded'}
            title={
              demoMode
                ? 'The compare lab is using demo metadata.'
                : 'The compare lab cannot trust live metadata.'
            }
            description={metadata.error}
            tone="error"
          />
        ) : null}

        {requestError ? (
          <StatePanel
            eyebrow="Request Error"
            title="The comparison request failed."
            description={requestError}
            tone="error"
          />
        ) : null}

        <div className="panel form-panel">
          <div className="thread-toolbar">
            <div>
              <div className="section-title">Comparison setup</div>
              <p>
                Use the same prompt envelope for both models, then inspect the answer cards side by
                side with latency and agreement metadata.
              </p>
            </div>
          </div>

          <div className="meta-grid model-chip-grid">
            {models.slice(0, 3).map((model) => (
              <ModelChip key={model.name} model={model} />
            ))}
          </div>

          <div className="control-row compare-controls">
            <div className="control-group">
              <label className="control-label">Primary model</label>
              <select
                value={primaryModel}
                onChange={(event) => setPrimaryModel(event.target.value)}
              >
                {models.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.label ?? model.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label className="control-label">Secondary model</label>
              <select
                value={secondaryModel}
                onChange={(event) => setSecondaryModel(event.target.value)}
              >
                {models.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.label ?? model.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label className="control-label">Prompt preset</label>
              <select
                value={promptPreset}
                onChange={(event) => setPromptPreset(event.target.value)}
              >
                {promptPresets.map((preset) => (
                  <option key={preset.id} value={preset.id}>
                    {preset.title}
                  </option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label className="control-label">Difficulty</label>
              <select
                value={difficultyTarget}
                onChange={(event) => setDifficultyTarget(event.target.value as Difficulty)}
              >
                {DIFFICULTY_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label className="control-label">Mode</label>
              <select
                value={solverMode}
                onChange={(event) => setSolverMode(event.target.value as SolverMode)}
              >
                {SOLVER_MODE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="control-group">
              <label className="control-label">Output</label>
              <select
                value={outputFormat}
                onChange={(event) => setOutputFormat(event.target.value as OutputFormat)}
              >
                {OUTPUT_FORMAT_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <textarea
            value={question}
            rows={5}
            onChange={(event) => setQuestion(event.target.value)}
          />

          <div className="starter-grid compact">
            {examples.map((example) => (
              <button
                key={`${example.dataset_id}-${example.question}`}
                className="starter-card"
                type="button"
                onClick={() => setQuestion(example.question)}
              >
                <div className="starter-meta">
                  <span>{example.dataset_title}</span>
                  <span>{example.difficulty}</span>
                </div>
                <strong>{example.question}</strong>
                <span className="starter-topic">{example.topic}</span>
              </button>
            ))}
          </div>

          <div className="composer-actions">
            <div className="hint-text">
              {primaryModel === secondaryModel
                ? 'Choose two different model variants to run a meaningful side-by-side comparison.'
                : 'Structured output is useful for inspecting verifier metadata in the comparison cards.'}
            </div>
            <div className="action-row">
              <button className="ghost-button small" type="button" onClick={() => setQuestion('')}>
                Clear prompt
              </button>
              <button
                className="primary-button"
                type="button"
                disabled={compareDisabled}
                onClick={() =>
                  startTransition(async () => {
                    try {
                      setRequestError(null);
                      const response = await compareModels({
                        question,
                        primary_model: primaryModel,
                        secondary_model: secondaryModel,
                        prompt_preset: promptPreset,
                        temperature: 0.2,
                        top_p: 0.95,
                        max_new_tokens: 768,
                        show_reasoning: true,
                        difficulty_target: difficultyTarget,
                        num_samples: 3,
                        use_calculator: true,
                        solver_mode: solverMode,
                        output_format: outputFormat,
                        use_cache: true,
                      });
                      setResult(response);
                    } catch (error) {
                      setRequestError(
                        error instanceof Error ? error.message : 'Unknown comparison failure.',
                      );
                    }
                  })
                }
              >
                {isPending ? 'Running...' : 'Compare models'}
              </button>
            </div>
          </div>
        </div>

        {result ? (
          <div className="comparison-grid wide">
            {[result.primary, result.secondary].map((entry) => (
              <article key={entry.model_variant} className="comparison-card tall panel">
                <div className="message-meta">
                  <span>{entry.model_variant}</span>
                  <div className="pill-row">
                    {entry.prompt_preset ? (
                      <span className="status-pill">{entry.prompt_preset}</span>
                    ) : null}
                    {entry.final_answer ? (
                      <span className="status-pill success">{entry.final_answer}</span>
                    ) : null}
                  </div>
                </div>
                <div className="badge-row">
                  <MetricBadge label="Final" value={entry.final_answer ?? 'n/a'} />
                  <MetricBadge
                    label="Agreement"
                    value={formatPercent(entry.candidate_agreement ?? 0, 0)}
                    tone="secondary"
                  />
                  <MetricBadge
                    label="Latency"
                    value={formatLatency(entry.latency_s)}
                    tone="accent"
                  />
                </div>
                <MathBlock content={entry.answer} />
              </article>
            ))}
          </div>
        ) : null}
      </section>
    </AppShell>
  );
}
