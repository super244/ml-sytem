import type { ModelInfo, PromptExample, PromptPreset } from "@/lib/api";

export const FALLBACK_MODELS: ModelInfo[] = [
  {
    name: "finetuned",
    label: "Atlas Specialist",
    description: "Default local specialist adapter.",
    base_model: "Qwen2.5-Math-1.5B-Instruct",
    available: true,
    tags: ["specialist", "verification"],
  },
  {
    name: "base",
    label: "Base Model",
    description: "Reference model for side-by-side comparison.",
    base_model: "Qwen2.5-Math-1.5B-Instruct",
    available: true,
    tags: ["baseline"],
  },
];

export const FALLBACK_PROMPTS: PromptPreset[] = [
  {
    id: "atlas_rigorous",
    title: "Rigorous",
    description: "Detailed derivations with strong final-answer formatting.",
    style_instructions: "Show rigorous step-by-step reasoning and end with Final Answer: ...",
  },
  {
    id: "atlas_exam",
    title: "Exam",
    description: "Fast, test-friendly solutions with compact derivations.",
    style_instructions: "Optimize for concise but legible exam-style solving.",
  },
  {
    id: "atlas_verifier",
    title: "Verifier",
    description: "Verification-oriented reasoning with explicit checkpoints.",
    style_instructions: "Expose intermediate checks and double-check calculations.",
  },
];

export const FALLBACK_EXAMPLES: PromptExample[] = [
  {
    dataset_id: "custom_integral_arena",
    dataset_title: "Integral Arena",
    question: "Evaluate \\int_0^1 x e^{x^2} dx.",
    difficulty: "hard",
    topic: "calculus",
  },
  {
    dataset_id: "custom_limits_series_lab",
    dataset_title: "Limits and Series Laboratory",
    question: "Determine whether \\sum_{n=2}^{\\infty} \\frac{1}{n (\\log n)^2} converges.",
    difficulty: "hard",
    topic: "calculus",
  },
  {
    dataset_id: "custom_olympiad_reasoning_studio",
    dataset_title: "Olympiad Reasoning Studio",
    question: "Find all positive integers n such that n^2 + n + 1 is divisible by n + 3.",
    difficulty: "olympiad",
    topic: "number theory",
  },
  {
    dataset_id: "custom_multivariable_studio",
    dataset_title: "Multivariable Studio",
    question: "Find the tangent plane to z = x^2 + xy - y^2 at (1, 2).",
    difficulty: "medium",
    topic: "multivariable calculus",
  },
];

export const RESEARCH_RESOURCES = [
  {
    label: "Architecture",
    path: "docs/architecture.md",
    detail: "Shared-core architecture, layers, and artifact conventions.",
  },
  {
    label: "Quickstart",
    path: "quickstart.md",
    detail: "Fastest path to a working local demo and dry-run validation.",
  },
  {
    label: "Experiment Playbook",
    path: "docs/experiment-playbook.md",
    detail: "Recommended ablation ladder and failure-driven iteration loop.",
  },
  {
    label: "Notebook Lab",
    path: "notebooks/README.md",
    detail: "Reproducible notebook workbench for exploration and analysis.",
  },
];
