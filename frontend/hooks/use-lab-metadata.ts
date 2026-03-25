"use client";

import { useEffect, useState } from "react";

import {
  getBenchmarks,
  getDatasetDashboard,
  getModels,
  getPromptLibrary,
  getRuns,
  getStatus,
  type BenchmarkInfo,
  type DatasetDashboard,
  type ModelInfo,
  type PromptLibrary,
  type RunInfo,
  type StatusInfo,
} from "@/lib/api";

type LabMetadataState = {
  datasets: DatasetDashboard | null;
  promptLibrary: PromptLibrary | null;
  models: ModelInfo[];
  benchmarks: BenchmarkInfo[];
  runs: RunInfo[];
  status: StatusInfo | null;
  loading: boolean;
  error: string | null;
};

const INITIAL_STATE: LabMetadataState = {
  datasets: null,
  promptLibrary: null,
  models: [],
  benchmarks: [],
  runs: [],
  status: null,
  loading: true,
  error: null,
};

export function useLabMetadata() {
  const [state, setState] = useState<LabMetadataState>(INITIAL_STATE);

  useEffect(() => {
    let active = true;
    async function load() {
      setState((current) => ({ ...current, loading: true, error: null }));
      try {
        const [
          datasetsResult,
          promptLibraryResult,
          modelsResult,
          benchmarksResult,
          runsResult,
          statusResult,
        ] = await Promise.allSettled([
          getDatasetDashboard(),
          getPromptLibrary(),
          getModels(),
          getBenchmarks(),
          getRuns(),
          getStatus(),
        ]);
        if (!active) {
          return;
        }
        const errors: string[] = [];
        const resolveResult = <T,>(
          result: PromiseSettledResult<T>,
          fallback: T,
          label: string,
        ): T => {
          if (result.status === "fulfilled") {
            return result.value;
          }
          errors.push(
            `${label}: ${
              result.reason instanceof Error ? result.reason.message : "request failed"
            }`,
          );
          return fallback;
        };
        setState({
          datasets: resolveResult(datasetsResult, null, "datasets"),
          promptLibrary: resolveResult(promptLibraryResult, null, "prompts"),
          models: resolveResult(modelsResult, [], "models"),
          benchmarks: resolveResult(benchmarksResult, [], "benchmarks"),
          runs: resolveResult(runsResult, [], "runs"),
          status: resolveResult(statusResult, null, "status"),
          loading: false,
          error: errors.length ? errors.join(" | ") : null,
        });
      } catch (error) {
        if (!active) {
          return;
        }
        setState((current) => ({
          ...current,
          loading: false,
          error: error instanceof Error ? error.message : "Failed to load metadata.",
        }));
      }
    }
    void load();
    return () => {
      active = false;
    };
  }, []);

  return state;
}
