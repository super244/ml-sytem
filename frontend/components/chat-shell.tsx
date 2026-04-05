'use client';

import clsx from 'clsx';
import Link from 'next/link';
import { useEffect, useRef, useState, useTransition } from 'react';

import {
  generateAnswer,
  type Difficulty,
  type GenerateResponse,
  type ModelVariant,
  type OutputFormat,
  type SolverMode,
} from '@/lib/api';
import {
  FALLBACK_EXAMPLES,
  FALLBACK_MODELS,
  FALLBACK_PROMPTS,
  RESEARCH_RESOURCES,
} from '@/lib/demo-content';
import { formatCount, formatLatency, formatPercent } from '@/lib/formatting';
import {
  DIFFICULTY_OPTIONS,
  OUTPUT_FORMAT_OPTIONS,
  SAMPLE_OPTIONS,
  SOLVER_MODE_OPTIONS,
} from '@/lib/options';
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
import { CandidateInspector } from '@/components/panels/candidate-inspector';
import { MetricBadge } from '@/components/panels/metric-badge';
import { ModelChip } from '@/components/panels/model-chip';
import { PageHeader } from '@/components/ui/page-header';
import { StatePanel } from '@/components/ui/state-panel';

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  result?: GenerateResponse;
};

type WorkspaceMode = 'focus' | 'research' | 'verification';
type WorkspaceDensity = 'compact' | 'balanced' | 'expanded';

const INITIAL_MESSAGES: ChatMessage[] = [
  {
    id: 'intro',
    role: 'system',
    content:
      'AI-Factory specializes in verifier-aware reasoning, controlled sampling, and model comparison. Use the control rail to switch presets, compare models, and inspect reranked candidates.',
  },
];

export function ChatShell() {
  const metadata = useLabMetadata();
  const demoMode = isDemoMode();
  const promptLibrary = metadata.promptLibrary;
  const availableModels = metadata.models.length
    ? metadata.models
    : demoMode
      ? FALLBACK_MODELS
      : [];
  const promptPresets =
    promptLibrary && promptLibrary.presets.length > 0
      ? promptLibrary.presets
      : demoMode
        ? FALLBACK_PROMPTS
        : [];
  const promptExamples =
    promptLibrary && promptLibrary.examples.length > 0
      ? promptLibrary.examples.slice(0, 6)
      : demoMode
        ? FALLBACK_EXAMPLES
        : [];
  const metadataDegraded =
    !demoMode && (metadata.status?.status === 'degraded' || Boolean(metadata.error));

  const [messages, setMessages] = useState<ChatMessage[]>(INITIAL_MESSAGES);
  const [question, setQuestion] = useState('');
  const [modelVariant, setModelVariant] = useState<ModelVariant>('');
  const [compareToModel, setCompareToModel] = useState<string>('');
  const [showReasoning, setShowReasoning] = useState(true);
  const [difficultyTarget, setDifficultyTarget] = useState<Difficulty>('olympiad');
  const [useCalculator, setUseCalculator] = useState(true);
  const [solverMode, setSolverMode] = useState<SolverMode>('rigorous');
  const [temperature, setTemperature] = useState(0.2);
  const [numSamples, setNumSamples] = useState(3);
  const [promptPreset, setPromptPreset] = useState('');
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('text');
  const [selectedCandidateIndex, setSelectedCandidateIndex] = useState(0);
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>('research');
  const [density, setDensity] = useState<WorkspaceDensity>('balanced');
  const [isPending, startTransition] = useTransition();
  const messageSequenceRef = useRef(0);
  const activeModelVariant = pickPrimaryModel(availableModels, modelVariant);
  const activeCompareToModel = pickSecondaryModel(availableModels, activeModelVariant, compareToModel);
  const activePromptPreset = pickPromptPreset(promptPresets, ['atlas_rigorous'], promptPreset);
  const canSubmit =
    question.trim().length > 0 &&
    Boolean(activeModelVariant) &&
    Boolean(activePromptPreset) &&
    !metadataDegraded;

  useEffect(() => {
    try {
      const storedMode = window.localStorage.getItem('ai-factory-workspace-mode');
      const storedDensity = window.localStorage.getItem('ai-factory-workspace-density');
      if (storedMode === 'focus' || storedMode === 'research' || storedMode === 'verification') {
        setTimeout(() => setWorkspaceMode(storedMode), 0);
      }
      if (
        storedDensity === 'compact' ||
        storedDensity === 'balanced' ||
        storedDensity === 'expanded'
      ) {
        setTimeout(() => setDensity(storedDensity), 0);
      }
    } catch {
      // LocalStorage can be disabled in privacy-focused environments.
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem('ai-factory-workspace-mode', workspaceMode);
      window.localStorage.setItem('ai-factory-workspace-density', density);
    } catch {
      // Ignore persistence failures and keep the session interactive.
    }
  }, [density, workspaceMode]);

  const latestResult = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant' && message.result)?.result;

  function resetConversation() {
    setMessages(INITIAL_MESSAGES);
    setQuestion('');
    setSelectedCandidateIndex(0);
  }

  function nextMessageId(prefix: 'user' | 'assistant') {
    messageSequenceRef.current += 1;
    return `${prefix}-${messageSequenceRef.current}`;
  }

  function applyWorkspacePreset(mode: WorkspaceMode) {
    setWorkspaceMode(mode);
    if (mode === 'focus') {
      setPromptPreset(pickPromptPreset(promptPresets, ['atlas_exam', 'atlas_rigorous']));
      setDifficultyTarget('medium');
      setSolverMode('exam');
      setShowReasoning(false);
      setUseCalculator(true);
      setNumSamples(1);
      setTemperature(0.0);
      setOutputFormat('text');
      return;
    }
    if (mode === 'verification') {
      setPromptPreset(pickPromptPreset(promptPresets, ['atlas_verifier', 'atlas_rigorous']));
      setDifficultyTarget('hard');
      setSolverMode('verification');
      setShowReasoning(true);
      setUseCalculator(true);
      setNumSamples(4);
      setTemperature(0.1);
      setOutputFormat('json');
      return;
    }
    setPromptPreset(pickPromptPreset(promptPresets, ['atlas_rigorous']));
    setDifficultyTarget('olympiad');
    setSolverMode('rigorous');
    setShowReasoning(true);
    setUseCalculator(true);
    setNumSamples(3);
    setTemperature(0.2);
    setOutputFormat('text');
  }

  async function submitQuestion(submittedQuestion: string) {
    const trimmed = submittedQuestion.trim();
    if (!trimmed || !activeModelVariant || !activePromptPreset || metadataDegraded) {
      return;
    }
    const userMessage: ChatMessage = {
      id: nextMessageId('user'),
      role: 'user',
      content: trimmed,
    };
    setMessages((current) => [...current, userMessage]);
    setQuestion('');
    try {
      const response = await generateAnswer({
        question: trimmed,
        model_variant: activeModelVariant,
        compare_to_base: false,
        compare_to_model: activeCompareToModel || null,
        prompt_preset: activePromptPreset,
        temperature,
        top_p: 0.95,
        max_new_tokens: 768,
        show_reasoning: showReasoning,
        difficulty_target: difficultyTarget,
        num_samples: numSamples,
        use_calculator: useCalculator,
        solver_mode: solverMode,
        output_format: outputFormat,
        use_cache: true,
      });
      setSelectedCandidateIndex(0);
      setMessages((current) => [
        ...current,
        {
          id: nextMessageId('assistant'),
          role: 'assistant',
          content: response.answer,
          result: response,
        },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown request failure.';
      setMessages((current) => [
        ...current,
        {
          id: nextMessageId('assistant'),
          role: 'assistant',
          content: `The request failed.\n\n\`\`\`\n${message}\n\`\`\``,
        },
      ]);
    }
  }

  return (
    <AppShell density={density} surfaceMode={workspaceMode}>
      <section className="route-stack">
        <PageHeader
          eyebrow="Assistant Workspace"
          title="AI-Factory Assistant Console"
          description="A polished interface for the same verifier-aware inference system used in offline evaluation. Tune model routing, sampling, and reasoning visibility while inspecting candidates, workspace metadata, and benchmark signals in one place."
          metrics={[
            {
              label: 'Models',
              value: formatCount(metadata.models.length || availableModels.length),
            },
            {
              label: 'Benchmarks',
              value: formatCount(metadata.benchmarks.length),
              tone: 'secondary',
            },
            {
              label: 'Runs',
              value: formatCount(metadata.runs.length),
              tone: 'accent',
            },
            {
              label: 'Cache entries',
              value: formatCount(metadata.status?.cache.entries),
            },
          ]}
          actions={
            <>
              <Link className="ghost-button small" href={ROUTES.datasets}>
                Explore datasets
              </Link>
              <Link className="primary-button small" href={ROUTES.compare}>
                Open compare lab
              </Link>
            </>
          }
        />

        <section className="panel control-bay">
          <div className="control-bay-copy">
            <div className="eyebrow">Command surface</div>
            <h2 className="workspace-title">Tune the workspace without leaving the page.</h2>
            <p className="hero-copy">
              Switch the session between focused answering, rigorous research, and verification
              mode. The same controls stay available across the CLI, TUI, web, and desktop shells.
            </p>
          </div>

          <div className="control-bay-actions">
            <div className="control-chip-group">
              {(['focus', 'research', 'verification'] as WorkspaceMode[]).map((mode) => (
                <button
                  key={mode}
                  className={clsx('ghost-button small', { active: workspaceMode === mode })}
                  type="button"
                  aria-pressed={workspaceMode === mode}
                  onClick={() => applyWorkspacePreset(mode)}
                >
                  {mode}
                </button>
              ))}
            </div>
            <div className="control-chip-group">
              {(['compact', 'balanced', 'expanded'] as WorkspaceDensity[]).map((nextDensity) => (
                <button
                  key={nextDensity}
                  className={clsx('secondary-button small', { active: density === nextDensity })}
                  type="button"
                  aria-pressed={density === nextDensity}
                  onClick={() => setDensity(nextDensity)}
                >
                  {nextDensity}
                </button>
              ))}
            </div>
            <div className="badge-row">
              <MetricBadge label="Mode" value={workspaceMode} tone="accent" />
              <MetricBadge label="Density" value={density} tone="secondary" />
              <MetricBadge label="Signals" value={showReasoning ? 'full' : 'minimal'} />
            </div>
          </div>
        </section>

        {metadata.error ? (
          <StatePanel
            eyebrow={demoMode ? 'Demo Metadata' : 'Metadata Degraded'}
            title={
              demoMode
                ? 'The workspace is running with demo metadata.'
                : 'The workspace cannot trust live metadata.'
            }
            description={metadata.error}
            tone="error"
            action={
              <button className="secondary-button small" type="button" onClick={resetConversation}>
                Start clean session
              </button>
            }
          />
        ) : null}

        {!demoMode && !availableModels.length ? (
          <StatePanel
            eyebrow="Metadata Required"
            title="No live model inventory is available."
            description="Restore `/v1/status` and `/v1/models` metadata before sending prompts from the assistant workspace."
            tone="error"
          />
        ) : null}

        <section className="workspace-grid">
          <aside className="panel control-rail">
            <div className="aside-section">
              <div className="section-title">Live Session</div>
              <p className="hero-copy">
                Route queries through the specialist stack, enable verification hooks, and keep the
                thread ready for direct model-vs-model inspection.
              </p>
              <div className="badge-row">
                <MetricBadge label="Primary" value={activeModelVariant} />
                <MetricBadge label="Samples" value={`${numSamples}`} tone="secondary" />
                <MetricBadge
                  label="Reasoning"
                  value={showReasoning ? 'visible' : 'hidden'}
                  tone="accent"
                />
              </div>
            </div>

            <div className="aside-section">
              <div className="section-title">Model Stack</div>
              <div className="meta-grid">
                {availableModels.slice(0, 4).map((model) => (
                  <ModelChip key={model.name} model={model} />
                ))}
              </div>
            </div>

            <div className="aside-section">
              <div className="section-title">Routing</div>
              <div className="control-group">
                <label className="control-label" htmlFor="modelVariant">
                  Primary model
                </label>
                <select
                  id="modelVariant"
                  value={activeModelVariant}
                  onChange={(event) => {
                    const nextModel = event.target.value as ModelVariant;
                    setModelVariant(nextModel);
                    if (activeCompareToModel === nextModel) {
                      setCompareToModel('');
                    }
                  }}
                >
                  {availableModels.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.label ?? model.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="control-group">
                <label className="control-label" htmlFor="compareModel">
                  Compare against
                </label>
                <select
                  id="compareModel"
                  value={activeCompareToModel}
                  onChange={(event) => setCompareToModel(event.target.value)}
                >
                  <option value="">No comparison</option>
                  {availableModels
                    .filter((model) => model.name !== activeModelVariant)
                    .map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.label ?? model.name}
                      </option>
                    ))}
                </select>
              </div>

              <div className="control-row">
                <div className="control-group">
                  <label className="control-label" htmlFor="preset">
                    Prompt preset
                  </label>
                  <select
                    id="preset"
                    value={activePromptPreset}
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
                  <label className="control-label" htmlFor="solverMode">
                    Solver mode
                  </label>
                  <select
                    id="solverMode"
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
              </div>
            </div>

            <div className="aside-section">
              <div className="section-title">Decoding</div>
              <div className="control-row">
                <div className="control-group">
                  <label className="control-label" htmlFor="difficultyTarget">
                    Difficulty
                  </label>
                  <select
                    id="difficultyTarget"
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
                  <label className="control-label" htmlFor="outputFormat">
                    Output
                  </label>
                  <select
                    id="outputFormat"
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

              <div className="control-row">
                <div className="control-group">
                  <label className="control-label" htmlFor="samples">
                    Samples
                  </label>
                  <select
                    id="samples"
                    value={numSamples}
                    onChange={(event) => setNumSamples(Number(event.target.value))}
                  >
                    {SAMPLE_OPTIONS.map((count) => (
                      <option key={count} value={count}>
                        {count}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="control-group">
                  <label className="control-label" htmlFor="temperature">
                    Temperature {temperature.toFixed(1)}
                  </label>
                  <input
                    id="temperature"
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={temperature}
                    onChange={(event) => setTemperature(Number(event.target.value))}
                  />
                </div>
              </div>

              <div className="toggle-grid">
                <label className="toggle-card" htmlFor="showReasoning">
                  <span>
                    <strong>Show reasoning</strong>
                    <span className="control-label">
                      Render the complete derivation in the workspace.
                    </span>
                  </span>
                  <input
                    id="showReasoning"
                    type="checkbox"
                    checked={showReasoning}
                    onChange={(event) => setShowReasoning(event.target.checked)}
                  />
                </label>

                <label className="toggle-card" htmlFor="useCalculator">
                  <span>
                    <strong>Enable calculator checks</strong>
                    <span className="control-label">
                      Allow safe `[[calc: ...]]` verification hooks.
                    </span>
                  </span>
                  <input
                    id="useCalculator"
                    type="checkbox"
                    checked={useCalculator}
                    onChange={(event) => setUseCalculator(event.target.checked)}
                  />
                </label>
              </div>
            </div>
          </aside>

          <section className="panel conversation-panel">
            <div className="thread-toolbar">
              <div>
                <div className="section-title">Conversation</div>
                <p>
                  Ask calculus, olympiad, or verification-heavy questions. You can also seed the
                  composer from the dataset-backed examples below.
                </p>
              </div>
              <div className="action-row">
                <button
                  className="secondary-button small"
                  type="button"
                  onClick={resetConversation}
                >
                  New thread
                </button>
                <Link className="ghost-button small" href={ROUTES.compare}>
                  Compare response
                </Link>
              </div>
            </div>

            <div className="badge-row workspace-badges">
              <MetricBadge label="Preset" value={activePromptPreset} />
              <MetricBadge label="Mode" value={solverMode} tone="secondary" />
              <MetricBadge label="Target" value={difficultyTarget} tone="accent" />
              <MetricBadge
                label="Cache"
                value={
                  metadata.status?.cache.enabled === true
                    ? 'enabled'
                    : metadata.status?.cache.enabled === false
                      ? 'disabled'
                      : 'unknown'
                }
              />
            </div>

            <div className="starter-grid">
              {promptExamples.map((example) => (
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

            <div className="message-list">
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={clsx('message-card', {
                    user: message.role === 'user',
                    assistant: message.role === 'assistant',
                    system: message.role === 'system',
                  })}
                >
                  <div className="message-meta">
                    <span>
                      {message.role === 'user'
                        ? 'You'
                        : message.role === 'assistant'
                          ? 'AI-Factory'
                          : 'Session guide'}
                    </span>
                    <div className="pill-row">
                      {message.result?.prompt_preset ? (
                        <span className="status-pill">{message.result.prompt_preset}</span>
                      ) : null}
                      {message.result?.final_answer ? (
                        <span className="status-pill success">{message.result.final_answer}</span>
                      ) : null}
                    </div>
                  </div>

                  <MathBlock content={message.content} />

                  {message.result ? (
                    <>
                      <div className="badge-row">
                        <MetricBadge label="Final" value={message.result.final_answer ?? 'n/a'} />
                        <MetricBadge
                          label="Rerank"
                          value={message.result.selected_score.toFixed(2)}
                          tone="accent"
                        />
                        <MetricBadge
                          label="Latency"
                          value={formatLatency(message.result.latency_s)}
                          tone="secondary"
                        />
                      </div>
                      {message.result.comparison ? (
                        <div className="comparison-grid">
                          <div className="comparison-card">
                            <h3>{message.result.comparison.model_variant}</h3>
                            <MathBlock content={message.result.comparison.answer} />
                          </div>
                          <div className="comparison-card">
                            <h3>{message.result.model_variant}</h3>
                            <MathBlock content={message.result.answer} />
                          </div>
                        </div>
                      ) : null}
                    </>
                  ) : null}
                </article>
              ))}
            </div>

            <form
              className="composer-form"
              onSubmit={(event) => {
                event.preventDefault();
                startTransition(() => {
                  void submitQuestion(question);
                });
              }}
            >
              <div className="composer-shell">
                <textarea
                  value={question}
                  rows={5}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Ask a calculus, olympiad, or advanced reasoning problem..."
                />
                <div className="composer-toolbar">
                  <div className="composer-hint">
                    <span className="hint-text">
                      {metadata.status?.cache.entries
                        ? `${metadata.status.cache.entries} cached generations ready for replay`
                        : 'Inference cache will warm as you explore prompts.'}
                    </span>
                    <div className="pill-row">
                      <span className="status-pill">
                        {showReasoning ? 'reasoning visible' : 'reasoning hidden'}
                      </span>
                      <span className="status-pill">
                        {outputFormat === 'json' ? 'structured' : 'text'}
                      </span>
                      <span className="status-pill">
                        {useCalculator ? 'calculator on' : 'calculator off'}
                      </span>
                    </div>
                  </div>
                  <div className="action-row">
                    <button
                      className="ghost-button small"
                      type="button"
                      onClick={() => setQuestion('')}
                    >
                      Clear draft
                    </button>
                    <button
                      className="primary-button"
                      type="submit"
                      disabled={isPending || !canSubmit}
                    >
                      {isPending ? 'Solving...' : 'Solve problem'}
                    </button>
                  </div>
                </div>
              </div>
            </form>
          </section>

          <div className="inspector-stack">
            <CandidateInspector
              candidates={latestResult?.candidates ?? []}
              selectedIndex={selectedCandidateIndex}
              onSelect={setSelectedCandidateIndex}
            />

            <section className="panel detail-panel">
              <div className="section-title">Reasoning and Verification</div>
              {latestResult ? (
                <>
                  <div className="badge-row">
                    <MetricBadge
                      label="Agreement"
                      value={formatPercent(latestResult.candidate_agreement ?? 0, 0)}
                    />
                    <MetricBadge
                      label="Verifier"
                      value={
                        latestResult.verification?.equivalent
                          ? 'match'
                          : (latestResult.verification?.error_type ?? 'unknown')
                      }
                      tone="secondary"
                    />
                    <MetricBadge
                      label="Latency"
                      value={formatLatency(latestResult.latency_s)}
                      tone="accent"
                    />
                  </div>
                  <div className="preview-block subtle">
                    <strong>Prompt envelope</strong>
                    <p>{latestResult.prompt}</p>
                  </div>
                  <div className="reasoning-scroll">
                    <MathBlock
                      content={
                        latestResult.structured?.reasoning ||
                        latestResult.reasoning_steps.join('\n') ||
                        'No reasoning available.'
                      }
                    />
                  </div>
                </>
              ) : (
                <p className="hero-copy">
                  Latest reasoning traces, verifier signals, and candidate metadata will appear here
                  after the first generation.
                </p>
              )}
            </section>

            <section className="panel detail-panel">
              <div className="section-title">Research Resources</div>
              <div className="resource-list">
                {RESEARCH_RESOURCES.map((resource) => (
                  <div key={resource.path} className="resource-card">
                    <strong>{resource.label}</strong>
                    <span>{resource.path}</span>
                    <p>{resource.detail}</p>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </section>
      </section>
    </AppShell>
  );
}
