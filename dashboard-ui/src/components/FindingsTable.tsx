import type { Finding } from "../types";

interface Props {
  findings: Finding[];
  selectedFinding: Finding | null;
  onSelect: (finding: Finding) => void;
}

function badgeLabel(finding: Finding): string {
  return finding.kind === "improvement" ? "improvement" : (finding.severity ?? "?");
}

function badgeClass(finding: Finding): string {
  if (finding.kind === "improvement") return "badge badge-improvement";
  return finding.severity === "major" ? "badge badge-major" : "badge badge-minor";
}

export function FindingsTable({ findings, selectedFinding, onSelect }: Props) {
  if (findings.length === 0) {
    return <p className="empty">No regressions or improvements against baseline for this run.</p>;
  }

  return (
    <table className="findings-table">
      <thead>
        <tr>
          <th></th>
          <th>entry</th>
          <th>kernel</th>
          <th>metric</th>
          <th>baseline</th>
          <th>new</th>
          <th>delta</th>
        </tr>
      </thead>
      <tbody>
        {findings.map((f) => (
          <tr
            key={`${f.entry}/${f.kernel}/${f.metric}`}
            className={f === selectedFinding ? "selected" : ""}
            onClick={() => onSelect(f)}
          >
            <td>
              <span className={badgeClass(f)}>{badgeLabel(f)}</span>
            </td>
            <td>{f.entry}</td>
            <td>
              <code>{f.kernel}</code>
            </td>
            <td>{f.metric}</td>
            <td>{f.base_median.toLocaleString()}</td>
            <td>{f.new_median.toLocaleString()}</td>
            <td className={f.delta_pct > 0 ? "delta-up" : "delta-down"}>
              {f.delta_pct > 0 ? "+" : ""}
              {f.delta_pct.toFixed(2)}%
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
