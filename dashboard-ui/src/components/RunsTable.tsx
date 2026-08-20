import type { Run } from "../types";

interface Props {
  runs: Run[];
  selectedRunId: number | null;
  onSelect: (runId: number) => void;
}

export function RunsTable({ runs, selectedRunId, onSelect }: Props) {
  if (runs.length === 0) {
    return <p className="empty">No runs yet. Run `perflens run` to collect one.</p>;
  }

  return (
    <table className="runs-table">
      <thead>
        <tr>
          <th>id</th>
          <th>timestamp</th>
          <th>branch</th>
          <th>git sha</th>
          <th>gpu</th>
          <th>driver</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => (
          <tr
            key={run.id}
            className={run.id === selectedRunId ? "selected" : ""}
            onClick={() => onSelect(run.id)}
          >
            <td>{run.id}</td>
            <td>{new Date(run.ts).toLocaleString()}</td>
            <td>{run.branch}</td>
            <td>
              <code>{run.git_sha.slice(0, 10)}</code>
            </td>
            <td>{run.gpu}</td>
            <td>{run.driver}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
