import { useCallback, useEffect, useState } from "react";
import type { AnalyzeResponse } from "./api";

export interface HistoryEntry {
  id: string;
  filename: string;
  timestamp: number;
  verdict: "genuine" | "fake" | null;
  confidence: number | null;
  rt60_seconds: number;
  drr_db: number;
  c50_db: number;
}

const STORAGE_KEY = "acousticspace.history.v1";
const MAX_ENTRIES = 25;

function readHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    // corrupted/unavailable storage shouldn't crash the app
    return [];
  }
}

function writeHistory(entries: HistoryEntry[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // storage full or unavailable (e.g. private browsing) — fail silently,
    // history just won't persist this session
  }
}

export function toHistoryEntry(result: AnalyzeResponse): HistoryEntry {
  return {
    id: `${result.filename}-${Date.now()}`,
    filename: result.filename,
    timestamp: Date.now(),
    verdict: result.baseline_prediction?.label ?? null,
    confidence: result.baseline_prediction?.confidence ?? null,
    rt60_seconds: result.rir.rt60_seconds,
    drr_db: result.rir.drr_db,
    c50_db: result.rir.c50_db,
  };
}

/** Persists analysis history to localStorage and keeps components in sync with it. */
export function useAnalysisHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>(() => readHistory());

  useEffect(() => {
    writeHistory(entries);
  }, [entries]);

  const addEntry = useCallback((result: AnalyzeResponse) => {
    setEntries((prev) => [toHistoryEntry(result), ...prev].slice(0, MAX_ENTRIES));
  }, []);

  const clear = useCallback(() => setEntries([]), []);

  return { entries, addEntry, clear };
}
