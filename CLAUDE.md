# CLAUDE.md — P2: PerfLens — CUDA Performance-Regression Triage Agent
# Rename to CLAUDE.md in the PerfLens repo root before running Claude Code.
# PREREQUISITE: P1 shipped (its kernels are half the suite). Ray tracer repo available.

## Mission

A nightly CI system that catches and EXPLAINS CUDA kernel performance regressions:
GitHub Actions (self-hosted runner on the RTX 3060 box) runs Nsight Compute across a
kernel suite, logs hardware metrics to SQLite, a noise-aware detector flags
regressions against baselines, and a Claude Agent SDK agent correlates metric deltas
with kernel source diffs to produce root-cause reports that gate PRs.

Design lineage (say this in interviews): the statistical discipline is the Rakuten
benchmark orchestrator (repeated trials against nondeterminism, hermetic baselines);
the nightly-pipeline plumbing is the options screener (Actions -> analyze -> report).
PerfLens fuses both and points them at GPU kernels.

## Hard constraints
1. HONESTY: the README case study must be a real caught regression (deliberately
   introduced in a branch), with the actual agent report shown. No mockups.
2. MVP is CLI + Markdown reports + PR comments. React dashboard is stretch — do not
   start it before M6.
3. The agent EXPLAINS with evidence; it never auto-reverts or auto-merges. Human gate.
4. Self-hosted runner security: repo must be private OR workflows restricted to
   same-repo branches only (`pull_request` from forks disabled). Runner runs as a
   dedicated non-admin user. Only secret: ANTHROPIC_API_KEY.

## Repo layout
```
perflens/
  CLAUDE.md
  pyproject.toml            (python 3.11+, click CLI, pydantic, anthropic/claude-agent-sdk, pytest)
  suite.yaml                (kernel suite manifest)
  perflens/
    cli.py                  (perflens run|baseline|detect|triage|report|gate)
    collect.py              (ncu invocation, CSV parse -> records)
    store.py                (SQLite schema + queries)
    detect.py               (median/MAD regression logic)
    triage.py               (Claude Agent SDK agent + tools)
    report.py               (Markdown renderer; gh PR comment poster)
    gitctx.py               (diffs and commit logs scoped to kernel source paths)
  tests/
    test_detect.py          (synthetic distributions: known regressions/noise)
    test_store.py
    test_collect_parse.py   (fixture ncu CSV files)
  .github/workflows/
    nightly.yml             (cron 15:00 UTC = midnight JST; workflow_dispatch)
    pr-gate.yml             (same-repo PRs only; label-triggered to save GPU time)
  reports/                  (generated Markdown, committed nightly)
```

## Suite manifest (suite.yaml)
```yaml
gpu: "RTX 3060"
reps: 5
entries:
  - name: inference.paged_attention
    cwd: ../gpu-engine
    cmd: ./build/bench_decode --batch 8 --steps 64 --once
    kernel_regex: "paged_attention.*"
    source_paths: ["src/kernels/attention.cu"]
  - name: inference.fused_gelu
    cwd: ../gpu-engine
    cmd: ./build/bench_decode --batch 8 --steps 64 --once
    kernel_regex: "bias_gelu.*"
    source_paths: ["src/kernels/gelu.cu"]
  - name: raytracer.trace
    cwd: ../cuda-ray-tracer
    cmd: ./build/rt --scene bench.scene --frames 3 --headless
    kernel_regex: "trace_rays.*"
    source_paths: ["src/trace.cu"]
  # + layernorm, kv scatter, argmax entries — same pattern
```

## Collection (collect.py)
- Invocation per entry:
  `ncu --csv --kernel-name-base demangled --kernel-name regex:<kernel_regex>
   --launch-skip 2 --launch-count <reps> --metrics <METRIC_SET> -- <cmd>`
- METRIC_SET (curated, keep exactly these to start):
  gpu__time_duration.sum,
  sm__warps_active.avg.pct_of_peak_sustained_active,      # achieved occupancy
  dram__throughput.avg.pct_of_peak_sustained_elapsed,
  lts__t_sector_hit_rate.pct,                              # L2 hit rate
  launch__registers_per_thread,
  launch__block_size, launch__grid_size,
  smsp__inst_executed.avg.per_cycle_active                 # rough IPC
- --launch-skip 2 discards warmup launches. Parse CSV -> one record per (kernel, rep).
- KNOWN CAVEAT (encode in code comments and README): ncu serializes and replays
  kernels; durations are kernel-level, not app wall-clock. That is exactly what we
  want for kernel regression detection. App-level throughput regressions belong to
  P1's bench_decode CSVs — PerfLens may ingest those as a second channel (stretch M7).
- If ncu exits nonzero (counter permissions): fail loudly with the fix hint
  (`nvidia persistence + NVreg_RestrictProfilingToAdminUsers=0` or run via sudo-less
  profiling group).

## Storage (store.py — SQLite, WAL mode)
```sql
CREATE TABLE runs(
  id INTEGER PRIMARY KEY, ts TEXT, git_sha TEXT, branch TEXT,
  gpu TEXT, driver TEXT, cuda_version TEXT, suite_hash TEXT);
CREATE TABLE kernel_results(
  run_id INT, entry TEXT, kernel TEXT, rep INT,
  duration_ns REAL, occupancy_pct REAL, dram_pct REAL, l2_hit_pct REAL,
  regs_per_thread INT, block_size INT, grid_size INT, ipc REAL, raw_json TEXT);
CREATE TABLE baselines(
  entry TEXT, kernel TEXT, metric TEXT,
  median REAL, mad REAL, run_id INT, set_ts TEXT,
  PRIMARY KEY(entry, kernel, metric));
```
- `perflens baseline set --run <id>` writes medians+MADs for every metric.
- Baselines are per-GPU (key includes runs.gpu implicitly via discipline: refuse to
  compare across different gpu/driver strings — print a warning and skip instead).

## Detection (detect.py)
For each (entry, kernel):
- Primary metric: duration_ns. regression iff BOTH:
  median_new > median_base * (1 + THRESHOLD)   # THRESHOLD default 0.05
  AND median_new > median_base + 3 * MAD_base  # noise floor guard
- Secondary metrics (occupancy_pct down, dram_pct up, regs_per_thread up, l2_hit down)
  are DIAGNOSTIC ONLY — attached as evidence, never gate alone.
- Improvements (mirror condition) are flagged as `improvement` with a suggested
  `baseline set` command.
- Output: list[Finding]{entry, kernel, metric, base_median, new_median, delta_pct,
  evidence: dict of secondary deltas, severity: minor(5-15%)/major(>15%)}.
- tests/test_detect.py: synthetic cases — clean 20% regression caught; 3% noise not
  flagged; high-MAD kernel needs bigger delta; improvement path.

## Triage agent (triage.py — Claude Agent SDK)
- Agent tools (all read-only):
  get_findings() -> the detector output
  get_metric_history(entry, kernel, metric, n=10) -> recent medians
  get_kernel_diff(entry, base_sha, head_sha) -> unified diff limited to source_paths
  get_commit_log(entry, base_sha, head_sha) -> messages+authors touching source_paths
- System prompt core: "You are a senior GPU performance engineer. For each finding,
  produce: (1) severity, (2) evidence summary quoting exact metric deltas,
  (3) root-cause hypothesis tied to specific diff hunks or commits,
  (4) confidence, (5) next diagnostic step (which ncu section/flag to inspect,
  e.g., __launch_bounds__, smem tiling, --set full on that kernel).
  If the diff cannot explain the delta, say so and hypothesize environment causes
  (clocks, driver) — never invent code changes."
- Output schema (pydantic-validated JSON) -> report.py renders Markdown.
- Classic patterns to encode as few-shot examples in the prompt:
  regs_per_thread up + occupancy down -> register pressure / spills;
  l2_hit down + dram up -> access-pattern/coalescing change;
  duration up with flat counters -> launch config or clock state.

## Reporting & gating (report.py)
- reports/YYYY-MM-DD.md: run header (sha, driver, gpu), findings table, agent
  narrative per finding, improvement notes.
- PR mode: post as PR comment via `gh pr comment`; set commit status
  perflens/regression = failure on any `major` finding (this is the gate).
- Nightly mode: commit the report to reports/ and open an issue on major findings.

## CI (.github/workflows)
- nightly.yml: runs-on [self-hosted, gpu]; cron "0 15 * * *"; steps: checkout suite
  repos (pinned paths on the runner box), build if HEAD moved, `perflens run`,
  `perflens detect`, `perflens triage`, `perflens report --nightly`, commit report.
- pr-gate.yml: pull_request (same repo only) + label `perf`; run/detect/triage,
  `perflens report --pr`, commit status.
- Runner setup doc in README: dedicated user, `./config.sh --labels gpu`,
  systemd service, ANTHROPIC_API_KEY as repo secret.

## Milestones & acceptance
- M1: collect one suite entry end-to-end into SQLite (fixture-tested CSV parse).
- M2: full suite + `baseline set` + reps handling; runs table populated with env info.
- M3: detector green on synthetic tests; `perflens detect` prints findings table.
- M4: triage agent produces a schema-valid report on a HAND-INJECTED finding
  (fixture), before touching live data.
- M5: nightly workflow runs on the self-hosted runner two nights in a row without
  babysitting.
- M6 (THE README CASE STUDY): branch `demo-regression` deliberately regresses the
  paged-attention kernel (e.g., add per-thread locals to force register spills, or
  drop __restrict__). PR triggers the gate; agent report correctly identifies the
  mechanism. Screenshot + full report land in README. This is the artifact you show
  NVIDIA.
- M7 (stretch): ingest bench_decode wall-clock CSVs as a second channel; React
  dashboard reading SQLite via a tiny FastAPI shim.

## Claims ledger (resume bullet <- measurement)
| Claim | Source |
|---|---|
| nightly Nsight Compute across a kernel suite, 5 reps, SQLite baselines | suite.yaml + runs table + nightly.yml history |
| median + 3xMAD noise-aware gating on PRs | detect.py + test_detect.py + a real failed status on demo PR |
| agent correlates metric deltas with source diffs into root-cause reports | reports/ + M6 case study |
| e.g. register-pressure occupancy drop identified | the M6 report itself |

## Timebox
Week 1 evenings: M1–M3. Weekend: M4–M5. Following evenings: M6.
Total ~1–2 weeks part-time after P1.