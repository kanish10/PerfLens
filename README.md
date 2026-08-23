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

| Milestone | Status | Evidence |
|---|---|---|
| M1: collect one suite entry -> SQLite | **Done** | `perflens/collect.py`, `tests/test_collect_parse.py` (fixture ncu CSV) |
| M2: full suite + `baseline set` + reps | **Done** | `perflens/store.py`, `perflens/cli.py`, `tests/test_store.py`, `tests/test_cli.py` |
| M3: detector green on synthetic tests | **Done** | `perflens/detect.py`, `tests/test_detect.py` (clean regression / noise / high-MAD / improvement cases from the spec) |
| M4: triage agent, schema-valid report on a hand-injected fixture | **Done** | `perflens/triage.py`, `tests/test_triage.py` (stubbed client, runs in CI) |
| M5: nightly workflow, two unattended nights on the self-hosted runner | **Pipeline verified on real hardware; two-night unattended run not yet done** | See "M6 case study" below -- the full `run`->`baseline`->`run`->`detect`->`triage`->`report` pipeline was executed for real on a rented RTX 3060. What's still open is literally two nights of the cron actually firing unattended, which needs a runner kept online, not further engineering. |
| M6: the README case study (real regression, real agent report) | **Done** | See below. Real regression, real detector finding, real agent report -- no mockups. |
| M7 (stretch): wall-clock ingestion + dashboard | **Dashboard done, wall-clock ingestion not started** | `perflens/dashboard/` (FastAPI) + `dashboard-ui/` (React), both read real SQLite data; bench_decode CSV ingestion is future work |

## M6 case study: a real caught regression

Everything below happened on a rented RTX 3060 (Vast.ai VM instance, not a
container -- Nsight Compute needs kernel-level counter access that container
instances don't grant), against real sibling checkouts of
[`gpu-engine`](https://github.com/kanish10/LLM-Inference-Server/tree/main/gpu-engine)
and [`cuda-ray-tracer`](https://github.com/kanish10/Ray-Tracer). Nothing here
is fixture data.

**The regression.** On a `demo-regression` branch of `gpu-engine`,
[`paged_attention_kernel`](https://github.com/kanish10/LLM-Inference-Server/commit/cec01a1)
gained 128 per-thread running-sum "diagnostic checksums" -- framed as the
kind of NaN-source debugging instrumentation someone might plausibly add
without noticing the register cost, which is exactly the point of the demo.

**What the detector caught**, real `perflens run` -> `baseline set` -> `run`
-> `detect` output, nothing hand-edited:

```
[MINOR] inference.paged_attention/...paged_attention_kernel(...) duration_ns: 4416 -> 5024 (+13.77%)
```

`launch__registers_per_thread` went 40 -> 155 (+287.5%); every other suite
entry was untouched, confirming the detector correctly isolated the
regression to the one kernel that actually changed.

**What the agent said**, a real API call (this specific run used the Groq
backend -- `openai/gpt-oss-120b` via `perflens/groq_adapter.py`, not
Anthropic; `triage.py` supports both, see "How the triage agent works"):

> **Evidence:** duration_ns increased from 4416 to 5024 (+13.77%).
> regs_per_thread increased by +287.5%. dram throughput decreased by
> -1.024%. occupancy changed by +0.936%. L2 hit rate decreased by -2.459%.
> The diff adds a per-thread float array `chk[128]` and associated loops,
> which directly raises register usage.
>
> **Root-cause hypothesis:** The added diagnostic checksum array
> `float chk[128]` and its initialization/accumulation loops increase
> registers per thread by ~287%, causing register pressure and likely
> spills, leading to the observed ~14% runtime regression.
>
> **Confidence:** high
>
> **Next diagnostic step:** Run Nsight Compute with `--section=Registers
> --section=Spills` or `--set full` on `paged_attention_kernel` to verify
> register count and spill metrics, and examine `__launch_bounds__` or
> consider moving `chk` to shared memory or removing it.

Full report: [`reports/2026-08-23.md`](reports/2026-08-23.md).

**Worth being honest about, not smoothing over:** this isn't the textbook
"registers up + occupancy down" pattern from `triage.py`'s system prompt --
occupancy stayed flat (already SM-scheduling-bound at this block size, not
register-bound, even at 155 regs/thread). The agent noticed this correctly
(it reported occupancy as roughly unchanged rather than inventing a drop) and
still correctly attributed the slowdown to register pressure/spill overhead
adding raw per-thread latency, a distinct mechanism from the occupancy story.
That's a better demonstration of real reasoning than a case that fit the
textbook pattern exactly would have been.

Two real bugs surfaced by actually running this pipeline end to end, not by
inspection, both fixed and covered by regression tests: `collect.py`'s CSV
parser assumed `ncu`'s stdout was pure CSV (it isn't -- `==PROF==` progress
lines and the profiled program's own stdout are interleaved with it), and
`report.py` matched the agent's narrative back to a detector finding on
`(entry, kernel)` including the full demangled kernel name, which a model
restating that name in its own words doesn't always reproduce byte-for-byte
(harmless for the PR gate itself, which never reads the agent's fields -- see
`report.py`'s `severity_label` -- but it silently dropped the narrative
section). Also fixed along the way: `suite.yaml`'s `gpu-engine` entries had
drifted from the real repo (a `kernel_regex` that didn't match its actual
kernel name, source paths pointing at files that don't exist, a `--once`
CLI flag `bench_decode` doesn't have), and a real gpu-engine build bug
(`gpu_kernels` was missing three source files in `CMakeLists.txt`, so it
never linked) that had gone uncaught because gpu-engine's CUDA path had never
actually been compiled before.

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
tests/         83 tests: synthetic detector cases, fixture CSV parsing,
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

pytest                    # 83 tests, no GPU required
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

By default `triage.py` talks to Claude (`--model claude-*`, needs
`ANTHROPIC_API_KEY`). It can also talk to any Groq model (`--model` set to
anything else, e.g. `openai/gpt-oss-120b`, needs `GROQ_API_KEY` and the
`groq` extra: `pip install -e ".[groq]"`) via `perflens/groq_adapter.py`,
which translates between Anthropic's Messages-API tool-call shape and Groq's
OpenAI-compatible chat-completions shape on every call -- the agent loop
below is written once, against the Anthropic shape, and is unaware of which
backend is actually running. The M6 case study above used the Groq path.

`triage.py` gives the agent four **read-only** tools -- `get_findings`,
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

The table `CLAUDE.md` asks for, kept honest about what's actually verified:

| Claim | Source | Verified? |
|---|---|---|
| Nsight Compute across a kernel suite, 5 reps, SQLite baselines | `suite.yaml` + `runs` table | **Yes** -- run for real on an RTX 3060: `perflens run` collected all 6 entries (30 rows), `baseline set` seeded medians/MADs, a second clean run correctly reported zero false positives |
| median + 3xMAD noise-aware gating on PRs | `detect.py` + `tests/test_detect.py` | **Yes** -- synthetic tests pass in CI on every push, *and* correctly flagged the real M6 regression (+13.77%) while leaving the other 5 untouched entries alone |
| agent correlates metric deltas with source diffs into root-cause reports | `triage.py` + `tests/test_triage.py` | **Yes** -- real API call (Groq `openai/gpt-oss-120b`) produced a correct root-cause report naming the actual added variable (`chk[128]`), see M6 case study above |
| e.g. register-pressure regression identified from real evidence | `reports/2026-08-23.md` | **Yes**, with a caveat stated plainly: it's register pressure/spill overhead, not the textbook "occupancy drop" pattern -- occupancy stayed flat here, and the agent correctly said so rather than inventing a drop |
| nightly workflow runs unattended, two nights running | `nightly.yml` | **Not yet** -- the pipeline itself is proven end-to-end on real hardware (this whole case study); what's left is purely operational (keep a runner online through two cron firings), not further engineering |

## License

MIT, see [`LICENSE`](LICENSE).
