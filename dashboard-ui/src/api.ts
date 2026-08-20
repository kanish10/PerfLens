import type { EntryKernel, Finding, HistoryPoint, Run } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function getJson<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(path, API_BASE);
  for (const [key, value] of Object.entries(params ?? {})) {
    url.searchParams.set(key, String(value));
  }
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(res.status, body || res.statusText);
  }
  return (await res.json()) as T;
}

export const api = {
  listRuns: (branch?: string): Promise<Run[]> =>
    getJson<Run[]>("/api/runs", branch ? { branch } : undefined),

  getFindings: (runId: number): Promise<Finding[]> =>
    getJson<Finding[]>(`/api/runs/${runId}/findings`),

  listEntries: (runId?: number): Promise<EntryKernel[]> =>
    getJson<EntryKernel[]>("/api/entries", runId !== undefined ? { run_id: runId } : undefined),

  getMetricHistory: (
    entry: string,
    kernel: string,
    metric: string,
    n = 20,
  ): Promise<HistoryPoint[]> => getJson<HistoryPoint[]>("/api/metrics/history", { entry, kernel, metric, n }),
};

export { ApiError };
