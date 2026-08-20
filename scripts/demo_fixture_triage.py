#!/usr/bin/env python3
"""Run the REAL triage agent (a live Claude API call) against a hand-injected
fixture finding, in a scratch git repo with a synthetic register-pressure
regression. This exercises the exact same code path as `perflens triage`,
without needing real hardware, `ncu`, or a live SQLite run history.

This is M4 from CLAUDE.md ("triage agent produces a schema-valid report on a
HAND-INJECTED finding (fixture), before touching live data") run for real
instead of against a stubbed client (see tests/test_triage.py for the
stubbed, CI-safe version of this same scenario).

This is explicitly NOT the M6 case study: M6 requires a REAL regression
caught by the REAL detector against REAL Nsight Compute measurements on the
self-hosted RTX 3060 runner (see CLAUDE.md hard constraint #1: no mockups in
the README case study). This script's output is illustrative only and is
never checked into reports/.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/demo_fixture_triage.py [--out reports/demo-fixture-triage.md]
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perflens import report, store, triage  # noqa: E402
from perflens.models import (  # noqa: E402
    Finding,
    KernelResultRecord,
    RunInfo,
    SuiteEntry,
)

FINDING = Finding(
    entry="inference.paged_attention",
    kernel="paged_attention_kernel",
    metric="duration_ns",
    base_median=12450.0,
    new_median=15600.0,
    delta_pct=25.3,
    evidence={"regs_per_thread_delta_pct": 50.0, "occupancy_pct_delta_pct": -18.0},
    severity="major",
    kind="regression",
)

BASE_SOURCE = """\
__global__ void paged_attention_kernel(const float* q, const float* k, const float* v, float* out) {
    // straightforward accumulation, low register pressure
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) {
        acc += q[i] * k[i];
    }
    out[threadIdx.x] = acc;
}
"""

REGRESSED_SOURCE = """\
__global__ void paged_attention_kernel(const float* q, const float* k, const float* v, float* out) {
    // fused epilogue: unrolled accumulation into per-thread locals to avoid a
    // second kernel launch. Trades a kernel launch for register pressure.
    float acc[8] = {0};
    float q_local[8], k_local[8], v_local[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            int idx = j * 8 + i;
            q_local[i] = q[idx];
            k_local[i] = k[idx];
            v_local[i] = v[idx];
            acc[j] += q_local[i] * k_local[i] * v_local[i];
        }
    }
    float total = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; j++) total += acc[j];
    out[threadIdx.x] = total;
}
"""


def _build_scratch_repo(root: Path) -> tuple[str, str, str]:
    repo = root / "gpu-engine"
    (repo / "src" / "kernels").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "demo@perflens.local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "PerfLens Demo"], cwd=repo, check=True)

    src = repo / "src" / "kernels" / "attention.cu"
    src.write_text(BASE_SOURCE)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "attention kernel: baseline"], cwd=repo, check=True)
    base_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    src.write_text(REGRESSED_SOURCE)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "attention: fuse epilogue to skip a kernel launch"],
        cwd=repo,
        check=True,
    )
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()

    return base_sha, head_sha, str(repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/demo-fixture-triage.md")
    parser.add_argument("--model", default=triage.DEFAULT_MODEL)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        base_sha, head_sha, repo_path = _build_scratch_repo(Path(tmp))

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        store.init_db(conn)
        run_id = store.insert_run(
            conn,
            RunInfo(
                ts="2026-01-02T00:00:00+00:00",
                git_sha=head_sha,
                branch="demo-regression",
                gpu="RTX 3060",
                driver="550.54.14",
                cuda_version="12.4",
                suite_hash="demo",
            ),
        )
        store.insert_kernel_results(
            conn,
            run_id,
            [
                KernelResultRecord(
                    entry=FINDING.entry, kernel=FINDING.kernel, rep=i, duration_ns=15600.0 + i
                )
                for i in range(5)
            ],
        )

        entries = {
            FINDING.entry: SuiteEntry(
                name=FINDING.entry,
                cwd=repo_path,
                cmd="./build/bench_decode --batch 8 --steps 64 --once",
                kernel_regex="paged_attention.*",
                source_paths=["src/kernels/attention.cu"],
            )
        }
        ctx = triage.TriageContext(
            conn=conn, entries=entries, findings=[FINDING], base_sha=base_sha, head_sha=head_sha
        )

        print(f"Calling {args.model} to triage the hand-injected fixture finding...", file=sys.stderr)
        result = triage.run_triage(ctx, model=args.model)

        run = store.get_run(conn, run_id)
        assert run is not None
        md = report.render_markdown(dict(run), [FINDING], result)
        md = (
            "> **This is an illustrative fixture demo (M4), not the M6 case study.**\n"
            "> The finding, source diff, and \"regression\" above are synthetic, hand-injected\n"
            "> data in a scratch git repo -- not a real measurement from the self-hosted RTX\n"
            "> 3060 runner. See CLAUDE.md hard constraint #1 and README.md's Status section.\n\n"
        ) + md
        out_path = report.write_report(args.out, md)
        print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    if not shutil.which("git"):
        sys.exit("git is required")
    main()
