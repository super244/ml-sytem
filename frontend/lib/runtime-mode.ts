"use client";

import type { ModelInfo, PromptPreset } from "@/lib/api";

export function isDemoMode(): boolean {
  return process.env.NEXT_PUBLIC_AI_FACTORY_DEMO_MODE === "1";
}

export function pickPrimaryModel(models: ModelInfo[], current?: string | null): string {
  if (current && models.some((model) => model.name === current)) {
    return current;
  }
  return models[0]?.name ?? "";
}

export function pickSecondaryModel(
  models: ModelInfo[],
  primaryModel: string,
  current?: string | null,
): string {
  if (current && current !== primaryModel && models.some((model) => model.name === current)) {
    return current;
  }
  return models.find((model) => model.name !== primaryModel)?.name ?? "";
}

export function pickPromptPreset(
  presets: PromptPreset[],
  preferredIds: string[] = [],
  current?: string | null,
): string {
  if (current && presets.some((preset) => preset.id === current)) {
    return current;
  }
  for (const preferredId of preferredIds) {
    const match = presets.find((preset) => preset.id === preferredId);
    if (match) {
      return match.id;
    }
  }
  return presets[0]?.id ?? "";
}
