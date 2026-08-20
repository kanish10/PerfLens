import { useEffect, useState } from "react";
import { api, ApiError } from "./api";
import { FindingsTable } from "./components/FindingsTable";
import { MetricChart } from "./components/MetricChart";
import { RunsTable } from "./components/RunsTable";
import type { Finding, HistoryPoint, Run } from "./types";

export function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [selectedFinding, setSelectedFinding] = useState<Finding | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .listRuns()
      .then((r) => {
        setRuns(r);
        if (r.length > 0) setSelectedRunId(r[0].id);
      })
      .catch((e: unknown) => setError(describeError(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedRunId === null) return;
    setSelectedFinding(null);
    setHistory([]);
    // Guards against out-of-order responses: if the user reselects a run
    // before this request resolves, a stale response must not overwrite
    // state that a newer, still-in-flight request will also try to set.
    let stale = false;
    api
      .getFindings(selectedRunId)
      .then((f) => {
        if (!stale) setFindings(f);
      })
      .catch((e: unknown) => {
        if (!stale) setError(describeError(e));
      });
    return () => {
      stale = true;
    };
  }, [selectedRunId]);

  useEffect(() => {
    if (selectedFinding === null) return;
    let stale = false;
    api
      .getMetricHistory(selectedFinding.entry, selectedFinding.kernel, selectedFinding.metric)
      .then((h) => {
        if (!stale) setHistory(h);
      })
      .catch((e: unknown) => {
        if (!stale) setError(describeError(e));
      });
    return () => {
      stale = true;
    };
  }, [selectedFinding]);

  return (
    <div className="app">
      <header>
        <h1>PerfLens</h1>
        <p className="subtitle">CUDA kernel performance-regression dashboard</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {loading ? (
        <p className="empty">Loading...</p>
      ) : (
        <>
          <section>
            <h2>Runs</h2>
            <RunsTable runs={runs} selectedRunId={selectedRunId} onSelect={setSelectedRunId} />
          </section>

          {selectedRunId !== null && (
            <section>
              <h2>Findings for run #{selectedRunId}</h2>
              <FindingsTable
                findings={findings}
                selectedFinding={selectedFinding}
                onSelect={setSelectedFinding}
              />
            </section>
          )}

          {selectedFinding && (
            <section>
              <MetricChart
                title={`${selectedFinding.entry} / ${selectedFinding.kernel} -- ${selectedFinding.metric}`}
                points={history}
              />
            </section>
          )}
        </>
      )}
    </div>
  );
}

function describeError(e: unknown): string {
  if (e instanceof ApiError) return `API error (${e.status}): ${e.message}`;
  if (e instanceof Error) return e.message;
  return "Unknown error. Is the PerfLens dashboard API running (perflens dashboard)?";
}
