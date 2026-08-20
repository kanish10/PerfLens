export interface Run {
  id: number;
  ts: string;
  git_sha: string;
  branch: string;
  gpu: string;
  driver: string;
  cuda_version: string;
  suite_hash: string;
}

export type FindingKind = "regression" | "improvement";
export type Severity = "minor" | "major";

export interface Finding {
  entry: string;
  kernel: string;
  metric: string;
  base_median: number;
  new_median: number;
  delta_pct: number;
  evidence: Record<string, number>;
  severity: Severity | null;
  kind: FindingKind;
  suggested_command: string | null;
}

export interface EntryKernel {
  entry: string;
  kernel: string;
}

export interface HistoryPoint {
  ts: string;
  value: number;
}
