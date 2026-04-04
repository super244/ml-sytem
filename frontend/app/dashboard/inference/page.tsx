'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect, useMemo, useRef, useState } from 'react';

import {
  compareModels,
  flagTelemetry,
  generateAnswer,
  getInstances,
  getModels,
  getPromptLibrary,
  getStatus,
  startManagedInference,
  type InstanceSummary,
  type ModelInfo,
  type PromptPreset,
  type StatusInfo,
} from '@/lib/api';
import { FALLBACK_MODELS, FALLBACK_PROMPTS } from '@/lib/demo-content';
import {
  isDemoMode,
  pickPrimaryModel,
  pickPromptPreset,
  pickSecondaryModel,
} from '@/lib/runtime-mode';

type AssistantVariant = {
  model: string;
  label: string;
  content: string;
  latency?: number;
  flagged?: boolean;
};

type ChatMessage =
  | {
      role: 'user';
      content: string;
    }
  | {
      role: 'assistant';
      variants: AssistantVariant[];
    };

type FlagTarget = {
  messageIndex: number;
  variantIndex: number;
} | null;

const QUICK_PROMPTS = [
  {
    label: 'Derivatives',
    prompt: 'Differentiate f(x) = (3x^2 - 5x + 2)e^(2x). Show the steps and end with Final Answer.',
  },
  {
    label: 'Integrals',
    prompt: 'Evaluate integral from 0 to 1 of (4x^3 - 2x + 1) dx. Explain every step.',
  },
  {
    label: 'Proof',
    prompt: 'Prove that the derivative of ln(x) is 1/x for x > 0.',
  },
  {
    label: 'Olympiad',
    prompt: 'Solve the system x + y = 7 and x^2 + y^2 = 29. Show reasoning carefully.',
  },
];

function averageLatency(messages: ChatMessage[]): number | null {
  const latencies = messages
    .flatMap((message) => (message.role === 'assistant' ? message.variants : []))
    .map((variant) => variant.latency)
    .filter((value): value is number => typeof value === 'number');
  if (!latencies.length) {
    return null;
  }
  return latencies.reduce((sum, value) => sum + value, 0) / latencies.length;
}

export default function InferencePage() {
  const router = useRouter();
  const [instances, setInstances] = useState<InstanceSummary[]>([]);
  const [launchSources, setLaunchSources] = useState<InstanceSummary[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [promptPresets, setPromptPresets] = useState<PromptPreset[]>([]);
  const [status, setStatus] = useState<StatusInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [secondaryModel, setSecondaryModel] = useState('');
  const [promptPreset, setPromptPreset] = useState('');
  const [compareMode, setCompareMode] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [generating, setGenerating] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [launchingId, setLaunchingId] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);
  const [launchNotice, setLaunchNotice] = useState<string | null>(null);
  const [flagTarget, setFlagTarget] = useState<FlagTarget>(null);
  const [flagReason, setFlagReason] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const demoMode = isDemoMode();
  const availableModels = models.length ? models : demoMode ? FALLBACK_MODELS : [];
  const availablePromptPresets = promptPresets.length
    ? promptPresets
    : demoMode
      ? FALLBACK_PROMPTS
      : [];
  const metadataDegraded = !demoMode && (status?.status === 'degraded' || Boolean(loadError));

  useEffect(() => {
    let active = true;
    setLoadError(null);
    Promise.allSettled([getInstances(), getModels(), getPromptLibrary(), getStatus()]).then(
      ([instancesResult, modelsResult, promptResult, statusResult]) => {
        if (!active) {
          return;
        }

        const errors: string[] = [];

        if (instancesResult.status === 'fulfilled') {
          const list = instancesResult.value;
          setInstances(
            list.filter(
              (instance) =>
                (instance.type === 'inference' && instance.status === 'running') ||
                (instance.type === 'deploy' && instance.status === 'completed'),
            ),
          );
          setLaunchSources(
            list.filter(
              (instance) =>
                instance.status === 'completed' &&
                (instance.type === 'train' ||
                  instance.type === 'finetune' ||
                  instance.type === 'deploy'),
            ),
          );
        } else {
          errors.push(
            instancesResult.reason instanceof Error
              ? instancesResult.reason.message
              : 'Instances could not be loaded.',
          );
        }

        if (modelsResult.status === 'fulfilled') {
          setModels(modelsResult.value);
        } else {
          errors.push(
            modelsResult.reason instanceof Error
              ? modelsResult.reason.message
              : 'Model registry could not be loaded.',
          );
        }

        if (promptResult.status === 'fulfilled') {
          setPromptPresets(promptResult.value.presets);
          if (promptResult.value.status === 'degraded' && promptResult.value.errors?.length) {
            errors.push(...promptResult.value.errors);
          }
        } else {
          errors.push(
            promptResult.reason instanceof Error
              ? promptResult.reason.message
              : 'Prompt metadata could not be loaded.',
          );
        }

        if (statusResult.status === 'fulfilled') {
          setStatus(statusResult.value);
          if (statusResult.value.status === 'degraded' && statusResult.value.errors?.length) {
            errors.push(...statusResult.value.errors);
          }
        } else {
          errors.push(
            statusResult.reason instanceof Error
              ? statusResult.reason.message
              : 'Status metadata could not be loaded.',
          );
        }

        setLoadError(errors.length ? errors.join(' | ') : null);
        setLoading(false);
      },
    );

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    setSelectedModel((current) => pickPrimaryModel(availableModels, current));
  }, [availableModels]);

  useEffect(() => {
    setSecondaryModel((current) => pickSecondaryModel(availableModels, selectedModel, current));
  }, [availableModels, selectedModel]);

  useEffect(() => {
    setPromptPreset((current) =>
      pickPromptPreset(availablePromptPresets, ['atlas_rigorous'], current),
    );
  }, [availablePromptPresets]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, generating]);

  useEffect(() => {
    if (selectedModel && selectedModel === secondaryModel) {
      const fallback = availableModels.find((model) => model.name !== selectedModel)?.name;
      if (fallback) {
        setSecondaryModel(fallback);
      }
    }
  }, [availableModels, secondaryModel, selectedModel]);

  const modelLabelLookup = useMemo(
    () => new Map(availableModels.map((model) => [model.name, model.label ?? model.name])),
    [availableModels],
  );
  const flaggedCount = useMemo(
    () =>
      messages.reduce((count, message) => {
        if (message.role !== 'assistant') {
          return count;
        }
        return count + message.variants.filter((variant) => variant.flagged).length;
      }, 0),
    [messages],
  );
  const sessionAverageLatency = useMemo(() => averageLatency(messages), [messages]);

  function findPromptForMessage(messageIndex: number): string {
    for (let index = messageIndex - 1; index >= 0; index -= 1) {
      const candidate = messages[index];
      if (candidate?.role === 'user') {
        return candidate.content;
      }
    }
    return '';
  }

  async function send() {
    if (!input.trim() || generating || metadataDegraded || !selectedModel || !promptPreset) {
      return;
    }

    const question = input.trim();
    setMessages((current) => [...current, { role: 'user', content: question }]);
    setInput('');
    setGenerating(true);
    const startedAt = Date.now();

    try {
      if (compareMode && secondaryModel && secondaryModel !== selectedModel) {
        const result = await compareModels({
          question,
          primary_model: selectedModel,
          secondary_model: secondaryModel,
          prompt_preset: promptPreset,
          temperature,
          top_p: 0.95,
          max_new_tokens: maxTokens,
          show_reasoning: false,
          difficulty_target: 'medium',
          num_samples: 1,
          use_calculator: false,
          solver_mode: 'concise',
          output_format: 'text',
          use_cache: false,
        });
        setMessages((current) => [
          ...current,
          {
            role: 'assistant',
            variants: [
              {
                model: selectedModel,
                label: modelLabelLookup.get(selectedModel) ?? selectedModel,
                content: result.primary.answer || result.primary.raw_text,
                latency: result.primary.latency_s ?? (Date.now() - startedAt) / 1000,
              },
              {
                model: secondaryModel,
                label: modelLabelLookup.get(secondaryModel) ?? secondaryModel,
                content: result.secondary.answer || result.secondary.raw_text,
                latency: result.secondary.latency_s ?? (Date.now() - startedAt) / 1000,
              },
            ],
          },
        ]);
      } else {
        const result = await generateAnswer({
          question,
          model_variant: selectedModel,
          compare_to_base: false,
          prompt_preset: promptPreset,
          temperature,
          top_p: 0.95,
          max_new_tokens: maxTokens,
          show_reasoning: false,
          difficulty_target: 'medium',
          num_samples: 1,
          use_calculator: false,
          solver_mode: 'concise',
          output_format: 'text',
          use_cache: false,
        });
        setMessages((current) => [
          ...current,
          {
            role: 'assistant',
            variants: [
              {
                model: selectedModel,
                label: modelLabelLookup.get(selectedModel) ?? selectedModel,
                content: result.answer || result.raw_text,
                latency: result.latency_s ?? (Date.now() - startedAt) / 1000,
              },
            ],
          },
        ]);
      }
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          variants: [
            {
              model: selectedModel,
              label: 'System',
              content: `Error: ${error instanceof Error ? error.message : 'Inference failed'}`,
            },
          ],
        },
      ]);
    } finally {
      setGenerating(false);
    }
  }

  async function submitFlag() {
    if (!flagTarget) {
      return;
    }
    const targetMessage = messages[flagTarget.messageIndex];
    if (targetMessage?.role !== 'assistant') {
      return;
    }
    const targetVariant = targetMessage.variants[flagTarget.variantIndex];
    if (!targetVariant) {
      return;
    }

    await flagTelemetry({
      prompt: findPromptForMessage(flagTarget.messageIndex),
      assistant_output: targetVariant.content,
      expected_output: flagReason,
      model_variant: targetVariant.model,
      latency_s: targetVariant.latency,
    });

    setMessages((current) =>
      current.map((message, messageIndex) => {
        if (message.role !== 'assistant' || messageIndex !== flagTarget.messageIndex) {
          return message;
        }
        return {
          ...message,
          variants: message.variants.map((variant, variantIndex) =>
            variantIndex === flagTarget.variantIndex ? { ...variant, flagged: true } : variant,
          ),
        };
      }),
    );
    setFlagReason('');
    setFlagTarget(null);
  }

  async function launchInference(instanceId: string) {
    if (metadataDegraded) {
      return;
    }
    setLaunching(true);
    setLaunchingId(instanceId);
    setLaunchNotice(null);
    try {
      const instance = await startManagedInference(instanceId, { start: true });
      setLaunchNotice(`Started inference branch ${instance.name}.`);
      router.push(`/runs/${instance.id}`);
    } finally {
      setLaunchingId(null);
      setLaunching(false);
    }
  }

  return (
    <div className="dashboard-content">
      <div className="dash-page-header panel">
        <div>
          <span className="eyebrow">Lifecycle → Inference</span>
          <h1 className="dash-page-title">Inference Studio</h1>
          <p className="dash-page-desc">
            Ollama-style chat workspace for live prompting, model comparison, launch control, and
            telemetry capture from the same AI-Factory dashboard.
          </p>
        </div>
      </div>

      {loadError && (
        <div className="dash-error-banner panel">
          <span>⚠</span> {loadError}
        </div>
      )}

      {launchNotice && (
        <div className="dash-note-banner panel">
          <span>◎</span> {launchNotice}
        </div>
      )}

      <div className="inference-layout">
        <div className="inference-sidebar">
          <div className="panel inference-settings-panel">
            <h2 className="section-title">Session Cockpit</h2>
            <div className="input-group">
              <label className="control-label" htmlFor="response-mode">
                Response mode
              </label>
              <select
                id="response-mode"
                value={compareMode ? 'compare' : 'single'}
                onChange={(event) => setCompareMode(event.target.value === 'compare')}
              >
                <option value="single">Single model</option>
                <option value="compare">Compare two models</option>
              </select>
            </div>
            <div className="input-group">
              <label className="control-label" htmlFor="primary-model">
                Primary model
              </label>
              <select
                id="primary-model"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                disabled={!availableModels.length || metadataDegraded}
              >
                {availableModels.length > 0 ? (
                  availableModels.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.label ?? model.name}
                    </option>
                  ))
                ) : (
                  <option value="">No live models</option>
                )}
              </select>
            </div>
            {compareMode && (
              <div className="input-group">
                <label className="control-label" htmlFor="secondary-model">
                  Comparison model
                </label>
                <select
                  id="secondary-model"
                  value={secondaryModel}
                  onChange={(event) => setSecondaryModel(event.target.value)}
                  disabled={!availableModels.length || metadataDegraded}
                >
                  {availableModels
                    .filter((model) => model.name !== selectedModel)
                    .map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.label ?? model.name}
                      </option>
                    ))}
                </select>
              </div>
            )}

            <div className="inference-stat-grid">
              <div className="inference-stat-card">
                <span className="inference-stat-label">Turns</span>
                <strong>{messages.filter((message) => message.role === 'assistant').length}</strong>
              </div>
              <div className="inference-stat-card">
                <span className="inference-stat-label">Avg latency</span>
                <strong>
                  {sessionAverageLatency ? `${sessionAverageLatency.toFixed(2)}s` : '—'}
                </strong>
              </div>
              <div className="inference-stat-card">
                <span className="inference-stat-label">Flagged</span>
                <strong>{flaggedCount}</strong>
              </div>
            </div>
          </div>

          <div className="panel inference-settings-panel">
            <h2 className="section-title">Quick Prompts</h2>
            <div className="prompt-chip-list">
              {QUICK_PROMPTS.map((prompt) => (
                <button
                  key={prompt.label}
                  type="button"
                  className="prompt-chip"
                  onClick={() => setInput(prompt.prompt)}
                >
                  {prompt.label}
                </button>
              ))}
            </div>
            <p className="control-label">
              Tap a prompt to preload the composer with a structured task.
            </p>
          </div>

          <div className="panel inference-settings-panel">
            <h2 className="section-title">Launch Sandbox</h2>
            <p className="control-label">
              Start a managed inference branch from a completed train, finetune, or deploy run.
            </p>
            {launchSources.length > 0 ? (
              <>
                <div className="source-list">
                  {launchSources.slice(0, 5).map((source) => (
                    <button
                      key={source.id}
                      type="button"
                      className={`source-item ${launchingId === source.id ? 'active' : ''}`}
                      disabled={launching || metadataDegraded}
                      onClick={() =>
                        setLaunchingId((current) => (current === source.id ? null : source.id))
                      }
                    >
                      <span className="source-name">{source.name}</span>
                      <span className="source-type">
                        {source.type} · {source.lifecycle.learning_mode ?? '—'}
                      </span>
                    </button>
                  ))}
                </div>
                <button
                  type="button"
                  className="primary-button small"
                  disabled={!launchingId || launching || metadataDegraded}
                  onClick={() => {
                    if (launchingId) {
                      void launchInference(launchingId);
                    }
                  }}
                >
                  {launching
                    ? '⟳ Launching…'
                    : launchingId
                      ? '◎ Launch inference branch'
                      : 'Select a source'}
                </button>
              </>
            ) : (
              <p className="control-label">No completed sources are available yet.</p>
            )}
          </div>

          <div className="panel inference-settings-panel">
            <h2 className="section-title">Runtime Settings</h2>
            <div className="input-group">
              <label className="control-label" htmlFor="temperature">
                Temperature: {temperature.toFixed(1)}
              </label>
              <input
                id="temperature"
                type="range"
                min={0}
                max={2}
                step={0.1}
                value={temperature}
                onChange={(event) => setTemperature(Number(event.target.value))}
              />
            </div>
            <div className="input-group">
              <label className="control-label" htmlFor="max-tokens">
                Max tokens: {maxTokens}
              </label>
              <input
                id="max-tokens"
                type="range"
                min={64}
                max={2048}
                step={64}
                value={maxTokens}
                onChange={(event) => setMaxTokens(Number(event.target.value))}
              />
            </div>
            {instances.length > 0 && (
              <>
                <p className="control-label">Live inference instances</p>
                {instances.map((instance) => (
                  <div key={instance.id} className="inference-instance-chip">
                    <span className={`monitor-status-dot status-${instance.status}`} />
                    <span className="inference-chip-name">{instance.name}</span>
                  </div>
                ))}
              </>
            )}
          </div>

          <div className="panel inference-settings-panel">
            <h2 className="section-title">Model Catalog</h2>
            <div className="model-catalog">
              {availableModels.slice(0, 6).map((model) => (
                <button
                  key={model.name}
                  type="button"
                  className={`model-catalog-item ${selectedModel === model.name ? 'active' : ''}`}
                  disabled={metadataDegraded}
                  onClick={() => setSelectedModel(model.name)}
                >
                  <strong>{model.label ?? model.name}</strong>
                  <span>{model.description ?? 'Local registry model'}</span>
                </button>
              ))}
            </div>
            {!loading && availableModels.length === 0 && (
              <Link href="/dashboard/deploy" className="secondary-button small">
                ⬆ Deploy a model
              </Link>
            )}
          </div>

          {messages.length > 0 && (
            <button type="button" className="ghost-button" onClick={() => setMessages([])}>
              Clear session
            </button>
          )}
        </div>

        <div className="panel inference-chat-panel">
          <div className="inference-chat-header">
            <div>
              <span className="eyebrow">Prompt Console</span>
              <h2 className="section-title">Live session</h2>
            </div>
            <div className="inference-chat-status">
              <span>{modelLabelLookup.get(selectedModel) ?? selectedModel}</span>
              {compareMode && secondaryModel && (
                <span>vs {modelLabelLookup.get(secondaryModel) ?? secondaryModel}</span>
              )}
            </div>
          </div>

          <div className="inference-messages">
            {messages.length === 0 && (
              <div className="inference-empty-state">
                <span className="inference-empty-icon">◎</span>
                <p>Start a conversation with your model stack.</p>
                <p className="inference-empty-hint">
                  Mode: <strong>{compareMode ? 'Compare' : 'Single'}</strong> · Primary model:{' '}
                  <strong>{modelLabelLookup.get(selectedModel) ?? selectedModel}</strong>
                </p>
              </div>
            )}

            {messages.map((message, messageIndex) =>
              message.role === 'user' ? (
                <div key={messageIndex} className="inference-message user">
                  <div className="inference-message-role">You</div>
                  <div className="inference-message-content">{message.content}</div>
                </div>
              ) : (
                <div key={messageIndex} className="inference-message assistant">
                  <div className="inference-message-role">Model</div>
                  {message.variants.length > 1 ? (
                    <div className="inference-compare-grid">
                      {message.variants.map((variant, variantIndex) => (
                        <div
                          key={`${variant.model}-${variantIndex}`}
                          className="inference-compare-card"
                        >
                          <div className="inference-compare-header">
                            <strong>{variant.label}</strong>
                            <button
                              type="button"
                              className="ghost-button small"
                              disabled={variant.flagged}
                              onClick={() => setFlagTarget({ messageIndex, variantIndex })}
                            >
                              {variant.flagged ? '✓ Flagged' : 'Flag'}
                            </button>
                          </div>
                          <div className="inference-message-content">{variant.content}</div>
                          <div className="inference-message-meta">
                            {variant.latency != null
                              ? `${variant.latency.toFixed(2)}s`
                              : 'No latency'}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    message.variants.map((variant, variantIndex) => (
                      <div
                        key={`${variant.model}-${variantIndex}`}
                        className="inference-single-output"
                      >
                        <div className="inference-output-actions">
                          <strong>{variant.label}</strong>
                          <button
                            type="button"
                            className="ghost-button small"
                            disabled={variant.flagged}
                            onClick={() => setFlagTarget({ messageIndex, variantIndex })}
                          >
                            {variant.flagged ? '✓ Flagged for dataset' : 'Flag as failure'}
                          </button>
                        </div>
                        <div className="inference-message-content">{variant.content}</div>
                        {variant.latency != null && (
                          <div className="inference-message-meta">
                            {variant.latency.toFixed(2)}s
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              ),
            )}

            {generating && (
              <div className="inference-message assistant">
                <div className="inference-message-role">Model</div>
                <div className="inference-generating">
                  <span className="inference-thinking-dot" />
                  <span className="inference-thinking-dot" />
                  <span className="inference-thinking-dot" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {flagTarget && (
            <div className="inference-flag-sheet">
              <div className="inference-flag-copy">
                Capture this response as a failure case for future dataset synthesis or replay
                training.
              </div>
              <textarea
                value={flagReason}
                onChange={(event) => setFlagReason(event.target.value)}
                placeholder="Expected answer or what failed..."
                className="inference-input"
                rows={3}
              />
              <div className="inference-flag-actions">
                <button
                  type="button"
                  className="primary-button small"
                  disabled={!flagReason.trim()}
                  onClick={() => void submitFlag()}
                >
                  Submit to telemetry
                </button>
                <button
                  type="button"
                  className="ghost-button small"
                  onClick={() => setFlagTarget(null)}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          <div className="inference-composer">
            <textarea
              className="inference-input"
              placeholder="Ask your model anything…"
              value={input}
              rows={3}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  void send();
                }
              }}
            />
            <button
              type="button"
              className="primary-button inference-send-btn"
              disabled={
                generating || !input.trim() || metadataDegraded || !selectedModel || !promptPreset
              }
              onClick={() => void send()}
            >
              {generating ? '⟳' : compareMode ? 'Compare →' : 'Send →'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
