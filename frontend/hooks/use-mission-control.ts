"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { getMissionControl, type MissionControlSnapshot } from "@/lib/api";

type MissionControlState = {
  mission: MissionControlSnapshot | null;
  loading: boolean;
  error: string | null;
};

export function useMissionControl(intervalMs: number) {
  const [state, setState] = useState<MissionControlState>({
    mission: null,
    loading: true,
    error: null,
  });
  const mountedRef = useRef(true);
  const inFlightRef = useRef(false);
  const missionRef = useRef<MissionControlSnapshot | null>(null);

  const refresh = useCallback(async () => {
    if (inFlightRef.current) {
      return missionRef.current;
    }
    inFlightRef.current = true;
    try {
      const mission = await getMissionControl();
      missionRef.current = mission;
      if (mountedRef.current) {
        setState({ mission, loading: false, error: null });
      }
      return mission;
    } catch (error) {
      missionRef.current = null;
      if (mountedRef.current) {
        setState({
          mission: null,
          loading: false,
          error: error instanceof Error ? error.message : "Mission control is unavailable.",
        });
      }
      return null;
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    function pollIfVisible() {
      if (typeof document !== "undefined" && document.visibilityState === "hidden") {
        return;
      }
      void refresh();
    }

    pollIfVisible();
    const intervalId = window.setInterval(pollIfVisible, intervalMs);
    const onVisibility = () => pollIfVisible();
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      mountedRef.current = false;
      window.clearInterval(intervalId);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [intervalMs, refresh]);

  return {
    ...state,
    refresh,
    replaceMission(mission: MissionControlSnapshot) {
      missionRef.current = mission;
      setState({ mission, loading: false, error: null });
    },
  };
}
