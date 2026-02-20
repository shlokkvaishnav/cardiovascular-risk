"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { AssessmentData, RiskCategory } from "../lib/risk";

export const HISTORY_KEY = "cardiorisk_history";
const HISTORY_LIMIT = 24;

export type AssessmentHistoryEntry = {
  id: string;
  timestamp: string;
  score: number;
  category: RiskCategory;
  age: number | null;
  systolicBp: number | null;
  totalCholesterol: number | null;
  hdlCholesterol: number | null;
  smoking: boolean | null;
  activityLevel: AssessmentData["activityLevel"];
};

function parseHistory(raw: string | null): AssessmentHistoryEntry[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as AssessmentHistoryEntry[];
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item) => item?.id && item?.timestamp).slice(0, HISTORY_LIMIT);
  } catch {
    return [];
  }
}

export function appendAssessmentHistory(entry: AssessmentHistoryEntry) {
  const raw = window.localStorage.getItem(HISTORY_KEY);
  const current = parseHistory(raw);
  const next = [entry, ...current].slice(0, HISTORY_LIMIT);
  window.localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
}

export function useAssessmentHistory() {
  const [history, setHistory] = useState<AssessmentHistoryEntry[]>([]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const raw = window.localStorage.getItem(HISTORY_KEY);
      setHistory(parseHistory(raw));
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  const persist = useCallback((next: AssessmentHistoryEntry[]) => {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(next.slice(0, HISTORY_LIMIT)));
    setHistory(next.slice(0, HISTORY_LIMIT));
  }, []);

  const addEntry = useCallback(
    (entry: AssessmentHistoryEntry) => {
      const raw = window.localStorage.getItem(HISTORY_KEY);
      const current = parseHistory(raw);
      const next = [entry, ...current].slice(0, HISTORY_LIMIT);
      persist(next);
    },
    [persist],
  );

  const clearHistory = useCallback(() => {
    window.localStorage.removeItem(HISTORY_KEY);
    setHistory([]);
  }, []);

  const sortedHistory = useMemo(
    () =>
      [...history].sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
      ),
    [history],
  );

  return {
    history: sortedHistory,
    historyCount: sortedHistory.length,
    addEntry,
    clearHistory,
  };
}
