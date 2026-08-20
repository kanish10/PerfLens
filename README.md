# PerfLens

A nightly CI system that catches and **explains** CUDA kernel performance
regressions. GitHub Actions (on a self-hosted RTX 3060 runner) profiles a
kernel suite with Nsight Compute, logs hardware metrics to SQLite, a
noise-aware statistical detector flags regressions against stored baselines,
and a Claude-powered triage agent correlates the metric deltas with the
kernel source diff to produce a root-cause report that gates PRs.

The agent explains. It never auto-reverts or auto-merges anything -- a human
reads the report and decides. See [`CLAUDE.md`](CLAUDE.md) for the full
design spec this repo implements.

## Status

This repo currently ships the **full pipeline** (M1-M5 from `CLAUDE.md`) with
committed, passing tests -- but honestly, not everything in `CLAUDE.md` can be
*proven* from a laptop with no NVIDIA GPU, no Nsight Compute, and no
self-hosted runner attached, and `CLAUDE.md`'s hard constraint #1 is explicit
that the README case study must be a real caught regression, never a mockup.
So here's exactly where things stand:

| Milestone | Status | Evidence |
|---|---|---|
| M1: collect one suite entry -> SQLite | **Done** | `perflens/collect.py`, `tests/test_collect_parse.py` (fixture ncu CSV) |
| M2: full suite + `baseline set` + reps | **Done** | `perflens/store.py`, `perflens/cli.py`, `tests/test_store.py`, `tests/test_cli.py` |
| M3: detector green on synthetic tests | **Done** | `perflens/detect.py`, `tests/test_detect.py` (clean regression / noise / high-MAD / improvement cases from the spec) |
| M4: triage agent, schema-valid report on a hand-injected fixture | **Done** | `perflens/triage.py`, `tests/test_triage.py` (stubbed client, runs in CI); `scripts/demo_fixture_triage.py` (same scenario against the *real* Claude API -- requires `ANTHROPIC_API_KEY`, not run in this environment, output intentionally not committed) |
| M5: nightly workflow, two unattended nights on the self-hosted runner | **Pending hardware** | `.github/workflows/nightly.yml` is written and YAML-valid, but has no self-hosted runner to execute on yet |
| M6: the README case study (real regression, real agent report) | **Pending hardware** | Cannot be produced without the RTX 3060 box + Nsight Compute + the paired `gpu-engine`/`cuda-ray-tracer` checkouts. **Deliberately not faked.** Once the runner is online, `git checkout -b demo-regression`, introduce a real regression (e.g. drop `__restrict__` or add per-thread locals to force spills), let `pr-gate.yml` catch it, and paste the real report below. |
| M7 (stretch): wall-clock ingestion + dashboard | **Dashboard done, wall-clock ingestion not started** | `perflens/dashboard/` (FastAPI) + `dashboard-ui/` (React), both read real SQLite data; bench_decode CSV ingestion is future work |

**Why this matters:** every synthetic/fixture test in this repo runs in CI
against fake-but-realistic data and proves the *logic* is correct. None of
them are a substitute for M6. When the case study lands, it replaces this
whole section with the real branch, the real PR, and the real agent output.

## Architecture

```mermaid
flowchart LR
    subgraph runner ["self-hosted RTX 3060 runner"]
        ncu["Nsight Compute\n(ncu --csv)"] --> collect["collect.py\nparse CSV"]
        collect --> db[("SQLite\nWAL mode")]
        db --> detect["detect.py\nmedian + 3*MAD"]
        detect --> triage["triage.py\nClaude agent\n(read-only tools)"]
        gitctx["gitctx.py\ndiff + log,\nscoped to source_paths"] --> triage
        triage --> report["report.py\nMarkdown + gh"]
    end
    report --> pr["PR comment +\ncommit status"]
    report --> issue["nightly issue\non major finding"]
    db --> dash["dashboard/ (FastAPI)\ndashboard-ui/ (React)"]
```

Design lineage: the statistical discipline (repeated trials against
nondeterminism, hermetic baselines) comes from a benchmark orchestrator built
for a prior project; the nightly-pipeline plumbing (Actions -> analyze ->
report) comes from an options-screener pipeline. PerfLens fuses both and
points them at GPU kernels.

## Repo layout

```
perflens/
  cli.py       perflens run|baseline|detect|triage|report|gate
  collect.py   ncu invocation + CSV parsing -> KernelResultRecord
  store.py     SQLite (WAL) schema + queries: runs, kernel_results, baselines
  detect.py    median + 3*MAD noise-aware regression/improvement detector
  triage.py    Claude agent: read-only tools + strict-schema final report
  report.py    Markdown rendering + gh PR comment / commit status
  gitctx.py    diffs and commit logs scoped to a suite entry's source_paths
  models.py    shared pydantic models
  dashboard/   FastAPI read-only API over the SQLite store (M7)
tests/         54 tests: synthetic detector cases, fixture CSV parsing,
                a stubbed-client triage run, a real temp git repo for
                gitctx, and full CLI wiring
dashboard-ui/  React + TypeScript dashboard (M7)
scripts/       demo_fixture_triage.py -- run the real agent on fixture data
.github/workflows/
  ci.yml        lint + type-check + test, every push/PR (GitHub-hosted)
  nightly.yml   the M5/M6 nightly pipeline (self-hosted GPU runner)
  pr-gate.yml   the PR gate (self-hosted GPU runner, same-repo + label only)
reports/        generated Markdown lands here (nightly + PR runs)
suite.yaml      kernel suite manifest
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"

pytest                    # 54 tests, no GPU required
ruff check perflens tests
mypy perflens
```

Everything above runs on any machine. The commands below need a real NVIDIA
GPU with Nsight Compute (`ncu`) on `PATH`, and the sibling `gpu-engine` /
`cuda-ray-tracer` checkouts that `suite.yaml` points at:

```bash
perflens run --branch main                       # collect the whole suite
perflens baseline set --run 1                     # seed a baseline from run 1
perflens run --branch main                        # a second run to compare
perflens detect --run 2                           # print findings vs. baseline
perflens triage --run 2                           # needs ANTHROPIC_API_KEY
perflens report --run 2 --nightly                 # render reports/YYYY-MM-DD.md
perflens gate --run 2 --pr 42                      # detect+triage+report+gate, exit 1 on major
```

`perflens triage`/`gate` don't take `--base-sha`/`--head-sha` in the normal
path -- each suite entry's own source commits are resolved automatically per
entry (see "How the triage agent works" below); those flags exist only to
force one explicit pair, which is only correct when every entry shares one
repo. `perflens --help` and `perflens <command> --help` document every flag.

## How detection works

For each `(entry, kernel)` in a run, compare the median `duration_ns` across
`suite.yaml`'s `reps` (5 by default) against a stored baseline median + MAD
(median absolute deviation). It's a **regression** iff *both*:

```
median_new > median_base * 1.05        # 5% relative-delta threshold
median_new > median_base + 3 * MAD_base  # noise-floor guard
```

Both gates matter: the first alone would flag a kernel with high natural
variance on every noisy run; the second alone would let a low-variance
kernel's real 3% regression slide by unflagged forever. Secondary metrics
(`occupancy_pct`, `dram_pct`, `l2_hit_pct`, `regs_per_thread`, block/grid
size, IPC) are attached as **diagnostic evidence only** -- they never gate a
finding by themselves, they just give the triage agent something to reason
about. The mirror condition on the other side flags an **improvement**, with
a suggested `perflens baseline set` command so the new normal gets adopted.
Baselines never compare across a different `gpu`/`driver` string -- that
comparison is silently skipped rather than producing a misleading finding.

See `tests/test_detect.py` for the exact synthetic cases this logic is held
to: a clean 20% regression is caught, 3% noise is not flagged, a kernel with
naturally high variance needs a proportionally bigger delta to trip the gate,
and the improvement path is exercised too.

## How the triage agent works

`triage.py` gives Claude four **read-only** tools -- `get_findings`,
`get_metric_history`, `get_kernel_diff`, `get_commit_log` -- all scoped to the
specific suite entry's `source_paths`, so the agent can never see a diff
outside the kernel that actually regressed. It never gets write access to the
repo, the database, or CI. Its final answer is forced through a
`submit_triage_report` tool with a strict JSON schema, so the output is
always a validated Pydantic `TriageReport`, never freeform prose that a
downstream renderer has to parse hopefully. Only `regression` findings go to
the agent -- an `improvement` has no severity to report and no root cause to
explain, and forcing one through the same schema would make the agent invent
a severity that isn't there.

**Resolving the right source commits across sibling repos.** A suite entry's
kernel source doesn't live in this repo -- `suite.yaml`'s `cwd` points at a
sibling checkout (`../gpu-engine`, `../cuda-ray-tracer`), each with its own
independent commit history that shares no SHA namespace with this repo or
with each other. So `get_kernel_diff`/`get_commit_log` don't take `base_sha`/
`head_sha` as agent-supplied arguments -- the agent can't know the right
values for a repo it's never seen, and asking it to guess (or reusing this
repo's own `git rev-parse HEAD~1`, which an earlier version of this pipeline
did) surfaces the wrong repo's commits or a "bad revision" tool error on
every real finding. Instead, `collect.py` records each entry's own repo HEAD
(`source_sha`) alongside every measurement at collection time, and
`TriageContext.resolve_shas` looks up, per entry: `head` = this run's
recorded `source_sha` for that entry, `base` = the `source_sha` recorded for
that entry in whichever run its baseline was set from. The agent just names
the `entry`; the correct commit pair for that entry's own repo comes back
automatically. `tests/test_triage.py::test_resolve_shas_uses_recorded_source_sha_not_this_repos_history`
is the regression test for this.

The system prompt asks for, per finding: severity (as set by the detector),
an evidence summary quoting exact metric deltas, a root-cause hypothesis tied
to a *specific* diff hunk or commit the agent actually saw via the tools,
a confidence level, and a next diagnostic step. If the diff can't explain the
delta, the agent is instructed to say so and hypothesize environment causes
(clock state, driver version, thermal throttling) rather than invent a code
change that isn't there.

## Self-hosted runner setup

`nightly.yml` and `pr-gate.yml` both require `runs-on: [self-hosted, gpu]`.
Per `CLAUDE.md` hard constraint #4:

1. **The repo must be private, or PR-triggered self-hosted jobs must be
   restricted to same-repo branches.** `pr-gate.yml` checks
   `github.event.pull_request.head.repo.full_name == github.repository`
   before running anything, and is additionally label-gated (`perf`) so a
   maintainer has to opt a PR in -- this repo is currently public, so both
   layers matter, not just one.
2. **Run the runner as a dedicated non-admin user.** Don't reuse an existing
   login; create one scoped to this job (`sudo adduser --disabled-password
   perflens-runner`).
3. **Register the runner with the `gpu` label:**
   ```bash
   ./config.sh --url https://github.com/kanish10/PerfLens \
               --token <runner-registration-token> \
               --labels gpu --name rtx3060-runner
   ./svc.sh install perflens-runner
   ./svc.sh start
   ```
4. **The only secret the runner needs is `ANTHROPIC_API_KEY`**, set as a repo
   secret (Settings -> Secrets and variables -> Actions). `GITHUB_TOKEN` is
   provided automatically by Actions for the `gh` calls in `report.py`.
5. Install Nsight Compute and confirm profiling counters are accessible to
   `perflens-runner` (not just root) -- `perflens run` fails loudly with the
   exact fix (`NVreg_RestrictProfilingToAdminUsers=0` or a profiling group)
   if `ncu` exits nonzero for permission reasons.
6. Clone `gpu-engine` and `cuda-ray-tracer` as siblings of this repo's
   checkout on the runner box, matching the `cwd` paths in `suite.yaml`.

**Known gap, stated plainly rather than papered over:** `pr-gate.yml`
triggers on PRs to *this* repo, but a real kernel regression (M6's
`demo-regression` scenario) is a commit in a *sibling* repo (`gpu-engine` or
`cuda-ray-tracer`) -- `CLAUDE.md` doesn't specify how one repo's PR is
supposed to make the other repo's sibling checkout advance to the matching
commit before `perflens run` executes. `nightly.yml` sidesteps this (it just
pulls whatever is latest on each sibling repo's default branch on its own
schedule), but `pr-gate.yml` as written assumes the sibling checkouts are
already sitting at the commit under test. Until the actual M6 case study
pins this down, the working assumption is: a demo regression PR would go
against `gpu-engine`, and either `gpu-engine`'s own CI cross-triggers this
workflow (`repository_dispatch` / `workflow_dispatch` with the target SHA),
or a maintainer updates the sibling checkout by hand before applying the
`perf` label. `perflens triage` itself is correct regardless of how the
sibling repo got to that commit -- see "How the triage agent works" above --
it's specifically the cross-repo *trigger* plumbing that's unbuilt.

## Dashboard (M7 stretch)

`perflens/dashboard/api.py` is a small read-only FastAPI service over the
same SQLite file `perflens run` writes to -- no separate data pipeline.
`dashboard-ui/` is a React + TypeScript frontend for it. See
[`dashboard-ui/README.md`](dashboard-ui/README.md) for how to run both
together locally.

## Claims ledger

The table `CLAUDE.md` asks for, kept honest about what's actually verified
right now vs. what's still pending the self-hosted runner:

| Claim | Source | Verified? |
|---|---|---|
| nightly Nsight Compute across a kernel suite, 5 reps, SQLite baselines | `suite.yaml` + `runs` table + `nightly.yml` | Code path tested end-to-end with fixture data; not yet run against real hardware |
| median + 3xMAD noise-aware gating on PRs | `detect.py` + `tests/test_detect.py` + `pr-gate.yml` | **Yes** -- synthetic tests pass in CI on every push |
| agent correlates metric deltas with source diffs into root-cause reports | `triage.py` + `tests/test_triage.py` | Loop + schema validation tested against a stubbed client; not yet run against the live API in this environment |
| e.g. register-pressure occupancy drop identified | the M6 report itself | **Not yet -- pending real hardware, see Status above** |

## License

MIT, see [`LICENSE`](LICENSE).
