"use client";

import { useMemo, useState } from "react";

import {
  createManagedInstance,
  type CreateManagedInstanceRequest,
  type DeploymentTarget,
  type InstanceDetail,
  type LifecycleStage,
  type WorkspaceOrchestrationTemplate,
} from "@/lib/api";

const DEPLOYMENT_TARGETS: DeploymentTarget[] = [
  "ollama",
  "huggingface",
  "lmstudio",
  "openai_compatible_api",
];

function parsePortForwards(raw: string) {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const parts = line.split(":").map((part) => part.trim()).filter(Boolean);
      if (parts.length < 2) {
        return null;
      }
      return {
        local_port: Number(parts[0]),
        remote_port: Number(parts[1]),
        bind_host: parts[2] || "127.0.0.1",
      };
    })
    .filter((value): value is { local_port: number; remote_port: number; bind_host: string } => Boolean(value));
}

function stageFromTemplate(
  template: WorkspaceOrchestrationTemplate | undefined,
): Extract<LifecycleStage, "train" | "finetune"> {
  if (!template) {
    return "train";
  }
  if (template.instance_type === "finetune") {
    return "finetune";
  }
  return "train";
}

export function NewInstancePanel({
  templates,
  onCreated,
}: {
  templates: WorkspaceOrchestrationTemplate[];
  onCreated: (detail: InstanceDetail) => void;
}) {
  const trainingTemplates = useMemo(
    () => templates.filter((template) => ["train", "finetune"].includes(template.instance_type)),
    [templates],
  );
  const defaultTemplate = trainingTemplates[0];

  const [configPath, setConfigPath] = useState(defaultTemplate?.path ?? "configs/train.yaml");
  const [name, setName] = useState("");
  const [userLevel, setUserLevel] = useState<"beginner" | "hobbyist" | "dev">("hobbyist");
  const [environmentKind, setEnvironmentKind] = useState<"local" | "cloud">("local");
  const [cloudHost, setCloudHost] = useState("");
  const [cloudUser, setCloudUser] = useState("");
  const [cloudPort, setCloudPort] = useState("22");
  const [cloudKeyPath, setCloudKeyPath] = useState("");
  const [remoteRepoRoot, setRemoteRepoRoot] = useState("/tmp/ai-factory");
  const [portForwards, setPortForwards] = useState("6006:6006");
  const [origin, setOrigin] = useState<"existing_model" | "from_scratch">("existing_model");
  const [learningMode, setLearningMode] = useState<
    "supervised" | "unsupervised" | "rlhf" | "dpo" | "orpo" | "ppo" | "lora" | "qlora" | "full_finetune"
  >("qlora");
  const [sourceModel, setSourceModel] = useState("Qwen/Qwen2.5-Math-1.5B-Instruct");
  const [architectureFamily, setArchitectureFamily] = useState("transformer");
  const [hiddenSize, setHiddenSize] = useState("");
  const [layers, setLayers] = useState("");
  const [heads, setHeads] = useState("");
  const [contextWindow, setContextWindow] = useState("");
  const [parameterSizeB, setParameterSizeB] = useState("");
  const [quantization, setQuantization] = useState<"4bit" | "8bit" | "16bit" | "none">("none");
  const [evaluationSuite, setEvaluationSuite] = useState("evaluation/configs/base_vs_finetuned.yaml");
  const [deploymentTargets, setDeploymentTargets] = useState<DeploymentTarget[]>(["ollama"]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedTemplate = trainingTemplates.find((template) => template.path === configPath) ?? defaultTemplate;

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const payload: CreateManagedInstanceRequest = {
        config_path: configPath,
        start: true,
        name: name || undefined,
        user_level: userLevel,
        environment:
          environmentKind === "cloud"
            ? {
                kind: "cloud",
                host: cloudHost || undefined,
                user: cloudUser || undefined,
                port: Number(cloudPort) || 22,
                key_path: cloudKeyPath || undefined,
                remote_repo_root: remoteRepoRoot || undefined,
                port_forwards: parsePortForwards(portForwards),
              }
            : { kind: "local" },
        lifecycle: {
          stage: stageFromTemplate(selectedTemplate),
          origin,
          learning_mode: learningMode,
          source_model: origin === "existing_model" ? sourceModel : undefined,
          architecture:
            origin === "from_scratch"
              ? {
                  base_model: architectureFamily || "transformer",
                  context_window: contextWindow ? Number(contextWindow) : undefined,
                  parameter_size_b: parameterSizeB ? Number(parameterSizeB) : undefined,
                  quantization: quantization,
                }
              : undefined,
          evaluation_suite: evaluationSuite
            ? {
                id: evaluationSuite.split("/").pop()?.replace(".yaml", ""),
                label: evaluationSuite.split("/").pop()?.replace(".yaml", "").replace(/[_-]/g, " "),
                benchmark_config: evaluationSuite,
                compare_to_models: ["base"],
              }
            : undefined,
          deployment_targets: deploymentTargets,
        },
        metadata: {
          source: "control_center_form",
        },
      };
      const detail = await createManagedInstance(payload);
      onCreated(detail);
      setName("");
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Failed to create the managed instance.");
    } finally {
      setSubmitting(false);
    }
  }

  function toggleTarget(target: DeploymentTarget) {
    setDeploymentTargets((current) =>
      current.includes(target) ? current.filter((item) => item !== target) : [...current, target],
    );
  }

  return (
    <section className="panel workspace-section">
      <div className="section-heading">
        <div>
          <div className="eyebrow">Launch Flow</div>
          <h2 className="workspace-title">Start a managed training branch</h2>
        </div>
      </div>

      <form className="control-form" onSubmit={handleSubmit}>
        <div className="control-form-grid">
          <label className="field-stack">
            <span>Template</span>
            <select value={configPath} onChange={(event) => setConfigPath(event.target.value)}>
              {trainingTemplates.map((template) => (
                <option key={template.path} value={template.path}>
                  {template.title} ({template.instance_type})
                </option>
              ))}
            </select>
          </label>

          <label className="field-stack">
            <span>Instance name</span>
            <input
              type="text"
              value={name}
              placeholder="atlas-iteration-01"
              onChange={(event) => setName(event.target.value)}
            />
          </label>

          <label className="field-stack">
            <span>Experience level</span>
            <select value={userLevel} onChange={(event) => setUserLevel(event.target.value as typeof userLevel)}>
              <option value="beginner">Beginner</option>
              <option value="hobbyist">Hobbyist</option>
              <option value="dev">Dev</option>
            </select>
          </label>

          <label className="field-stack">
            <span>Environment</span>
            <select
              value={environmentKind}
              onChange={(event) => setEnvironmentKind(event.target.value as typeof environmentKind)}
            >
              <option value="local">Local</option>
              <option value="cloud">Cloud / SSH</option>
            </select>
          </label>

          <label className="field-stack">
            <span>Training origin</span>
            <select value={origin} onChange={(event) => setOrigin(event.target.value as typeof origin)}>
              <option value="existing_model">Existing model</option>
              <option value="from_scratch">From scratch</option>
            </select>
          </label>

          <label className="field-stack">
            <span>Learning mode</span>
            <select
              value={learningMode}
              onChange={(event) => setLearningMode(event.target.value as typeof learningMode)}
            >
              <option value="supervised">Supervised</option>
              <option value="unsupervised">Unsupervised</option>
              <option value="rlhf">RLHF</option>
              <option value="dpo">DPO</option>
              <option value="orpo">ORPO</option>
              <option value="ppo">PPO</option>
              <option value="lora">LoRA</option>
              <option value="qlora">QLoRA</option>
              <option value="full_finetune">Full finetune</option>
            </select>
          </label>

          {origin === "existing_model" ? (
            <label className="field-stack control-form-span-2">
              <span>Source model</span>
              <input
                type="text"
                value={sourceModel}
                onChange={(event) => setSourceModel(event.target.value)}
              />
            </label>
          ) : (
            <>
              <label className="field-stack">
                <span>Architecture family</span>
                <input
                  type="text"
                  value={architectureFamily}
                  onChange={(event) => setArchitectureFamily(event.target.value)}
                />
              </label>
              <label className="field-stack">
                <span>Hidden size</span>
                <input type="number" value={hiddenSize} onChange={(event) => setHiddenSize(event.target.value)} />
              </label>
              <label className="field-stack">
                <span>Layers</span>
                <input type="number" value={layers} onChange={(event) => setLayers(event.target.value)} />
              </label>
              <label className="field-stack">
                <span>Attention heads</span>
                <input type="number" value={heads} onChange={(event) => setHeads(event.target.value)} />
              </label>
              <label className="field-stack">
                <span>Context window</span>
                <input
                  type="number"
                  value={contextWindow}
                  onChange={(event) => setContextWindow(event.target.value)}
                />
              </label>
            </>
          )}

          <label className="field-stack control-form-span-2">
            <span>Evaluation suite</span>
            <input
              type="text"
              value={evaluationSuite}
              onChange={(event) => setEvaluationSuite(event.target.value)}
            />
          </label>
        </div>

        {environmentKind === "cloud" ? (
          <div className="control-form-grid">
            <label className="field-stack">
              <span>SSH host</span>
              <input type="text" value={cloudHost} onChange={(event) => setCloudHost(event.target.value)} />
            </label>
            <label className="field-stack">
              <span>SSH user</span>
              <input type="text" value={cloudUser} onChange={(event) => setCloudUser(event.target.value)} />
            </label>
            <label className="field-stack">
              <span>SSH port</span>
              <input type="number" value={cloudPort} onChange={(event) => setCloudPort(event.target.value)} />
            </label>
            <label className="field-stack">
              <span>SSH key path</span>
              <input type="text" value={cloudKeyPath} onChange={(event) => setCloudKeyPath(event.target.value)} />
            </label>
            <label className="field-stack control-form-span-2">
              <span>Remote repo root</span>
              <input
                type="text"
                value={remoteRepoRoot}
                onChange={(event) => setRemoteRepoRoot(event.target.value)}
              />
            </label>
            <label className="field-stack control-form-span-2">
              <span>Port forwards</span>
              <textarea
                rows={3}
                value={portForwards}
                onChange={(event) => setPortForwards(event.target.value)}
              />
            </label>
          </div>
        ) : null}

        <div className="field-stack">
          <span>Deployment targets</span>
          <div className="selection-grid">
            {DEPLOYMENT_TARGETS.map((target) => (
              <button
                key={target}
                className={`selection-chip${deploymentTargets.includes(target) ? " active" : ""}`}
                type="button"
                onClick={() => toggleTarget(target)}
              >
                {target.replace(/_/g, " ")}
              </button>
            ))}
          </div>
        </div>

        {error ? <p className="control-form-error">{error}</p> : null}

        <div className="action-row">
          <button className="primary-button" type="submit" disabled={submitting}>
            {submitting ? "Launching..." : "Launch managed instance"}
          </button>
        </div>
      </form>
    </section>
  );
}
