"""M4: the triage agent produces a schema-valid report on a hand-injected
finding (fixture), before ever touching live data or a real API key.

The Anthropic client is fully stubbed here -- these tests exercise the tool
dispatch loop and the pydantic validation of the agent's final answer, not
the model itself.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from perflens import triage
from perflens.models import Finding, SuiteEntry

FIXTURE = Path(__file__).parent / "fixtures" / "hand_injected_finding.json"


@dataclass
class FakeBlock:
    type: str
    name: str | None = None
    input: dict[str, Any] | None = None
    id: str | None = None
    text: str | None = None


@dataclass
class FakeResponse:
    content: list[FakeBlock]


class FakeMessages:
    def __init__(self, scripted: list[FakeResponse]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if not self._scripted:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted: list[FakeResponse]) -> None:
        self.messages = FakeMessages(scripted)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "gpu-engine"
    r.mkdir()
    for args in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "T"],
    ):
        subprocess.run(args, cwd=r, check=True)
    (r / "src").mkdir()
    (r / "src" / "attention.cu").write_text("// base\n__global__ void k() {}\n")
    subprocess.run(["git", "add", "."], cwd=r, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=r, check=True)
    (r / "src" / "attention.cu").write_text(
        "// regress: fused epilogue adds per-thread locals\n__global__ void k() { int x[64]; }\n"
    )
    subprocess.run(["git", "add", "."], cwd=r, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fuse epilogue into attention kernel"], cwd=r, check=True)
    return r


def _shas(repo: Path) -> tuple[str, str]:
    out = subprocess.run(
        ["git", "log", "--format=%H"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.split()
    return out[1], out[0]  # base (older), head (newer)


@pytest.fixture
def finding() -> Finding:
    return Finding(**json.loads(FIXTURE.read_text()))


@pytest.fixture
def ctx(conn: sqlite3.Connection, repo: Path, finding: Finding) -> triage.TriageContext:
    base_sha, head_sha = _shas(repo)
    entries = {
        "inference.paged_attention": SuiteEntry(
            name="inference.paged_attention",
            cwd=str(repo),
            cmd="./build/bench_decode --batch 8 --steps 64 --once",
            kernel_regex="paged_attention.*",
            source_paths=["src/attention.cu"],
        )
    }
    return triage.TriageContext(
        conn=conn, entries=entries, findings=[finding], base_sha=base_sha, head_sha=head_sha
    )


def _submit_block(**overrides: Any) -> FakeBlock:
    payload = {
        "entry": "inference.paged_attention",
        "kernel": "paged_attention_kernel",
        "severity": "major",
        "evidence_summary": "duration +25.3%, regs_per_thread +50%, occupancy -18%",
        "root_cause_hypothesis": "Fused epilogue added per-thread locals, increasing register "
        "pressure and forcing spills, which dropped occupancy.",
        "confidence": "high",
        "next_diagnostic_step": "Re-profile with --set full and inspect launch__registers_per_thread",
    }
    payload.update(overrides)
    return FakeBlock(
        type="tool_use", name="submit_triage_report", id="call_submit", input={"findings": [payload]}
    )


def test_triage_produces_schema_valid_report_after_gathering_evidence(
    ctx: triage.TriageContext,
) -> None:
    tool_call_input = {
        "entry": "inference.paged_attention",
        "base_sha": ctx.base_sha,
        "head_sha": ctx.head_sha,
    }
    scripted = [
        FakeResponse(
            content=[
                FakeBlock(type="text", text="Let me look at the diff and history."),
                FakeBlock(type="tool_use", name="get_kernel_diff", input=tool_call_input, id="c1"),
                FakeBlock(type="tool_use", name="get_commit_log", input=tool_call_input, id="c2"),
            ]
        ),
        FakeResponse(content=[_submit_block()]),
    ]
    client = FakeClient(scripted)

    result = triage.run_triage(ctx, client=client, model="claude-opus-5")

    assert len(result.findings) == 1
    t = result.findings[0]
    assert t.entry == "inference.paged_attention"
    assert t.kernel == "paged_attention_kernel"
    assert t.severity == "major"
    assert t.confidence == "high"
    hypothesis = t.root_cause_hypothesis.lower()
    assert "register pressure" in hypothesis or "spills" in hypothesis
    assert result.model == "claude-opus-5"
    assert len(client.messages.calls) == 2

    # tool results were fed back correctly: the diff for the second call must
    # contain the actual injected source change.
    second_call_messages = client.messages.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]
    diff_result = next(
        r["content"] for r in tool_result_msg["content"] if r["tool_use_id"] == "c1"
    )
    assert "int x[64]" in diff_result


def test_triage_dispatches_get_findings_and_metric_history(ctx: triage.TriageContext) -> None:
    from perflens import store

    # seed some history so get_metric_history has something to return
    history_points = [("2026-01-01T00:00:00+00:00", 12000.0), ("2026-01-02T00:00:00+00:00", 12450.0)]
    for ts, dur in history_points:
        from perflens.models import KernelResultRecord, RunInfo

        rid = store.insert_run(
            ctx.conn,
            RunInfo(
                ts=ts, git_sha="x", branch="main", gpu="RTX 3060",
                driver="550", cuda_version="12.4", suite_hash="h",
            ),
        )
        store.insert_kernel_results(
            ctx.conn,
            rid,
            [
                KernelResultRecord(
                    entry="inference.paged_attention",
                    kernel="paged_attention_kernel",
                    rep=i,
                    duration_ns=dur,
                )
                for i in range(3)
            ],
        )

    scripted = [
        FakeResponse(
            content=[
                FakeBlock(type="tool_use", name="get_findings", input={}, id="c1"),
                FakeBlock(
                    type="tool_use",
                    name="get_metric_history",
                    input={
                        "entry": "inference.paged_attention",
                        "kernel": "paged_attention_kernel",
                        "metric": "duration_ns",
                        "n": 5,
                    },
                    id="c2",
                ),
            ]
        ),
        FakeResponse(content=[_submit_block()]),
    ]
    client = FakeClient(scripted)

    result = triage.run_triage(ctx, client=client)
    assert len(result.findings) == 1

    tool_result_msg = client.messages.calls[1]["messages"][-1]
    findings_json = next(r["content"] for r in tool_result_msg["content"] if r["tool_use_id"] == "c1")
    assert json.loads(findings_json)[0]["entry"] == "inference.paged_attention"
    history_json = next(r["content"] for r in tool_result_msg["content"] if r["tool_use_id"] == "c2")
    assert len(json.loads(history_json)) == 2


def test_triage_empty_findings_short_circuits_without_calling_model(
    conn: sqlite3.Connection,
) -> None:
    ctx = triage.TriageContext(conn=conn, entries={}, findings=[], base_sha="a", head_sha="b")
    client = FakeClient([])  # would raise AssertionError if called

    result = triage.run_triage(ctx, client=client)

    assert result.findings == []


def test_triage_raises_when_agent_never_submits(ctx: triage.TriageContext) -> None:
    scripted = [FakeResponse(content=[FakeBlock(type="text", text="I don't know.")])]
    client = FakeClient(scripted)

    with pytest.raises(triage.TriageError):
        triage.run_triage(ctx, client=client)


def test_triage_raises_on_invalid_submit_payload(ctx: triage.TriageContext) -> None:
    bad_block = FakeBlock(
        type="tool_use",
        name="submit_triage_report",
        id="c1",
        input={"findings": [{"entry": "x"}]},  # missing required fields
    )
    client = FakeClient([FakeResponse(content=[bad_block])])

    with pytest.raises(triage.TriageError):
        triage.run_triage(ctx, client=client)


def test_triage_tool_error_surfaces_as_is_error(ctx: triage.TriageContext) -> None:
    scripted = [
        FakeResponse(
            content=[
                FakeBlock(type="tool_use", name="get_kernel_diff", input={"entry": "nonexistent"}, id="c1"),
            ]
        ),
        FakeResponse(content=[_submit_block()]),
    ]
    client = FakeClient(scripted)

    result = triage.run_triage(ctx, client=client)
    assert len(result.findings) == 1  # agent recovers and submits anyway

    tool_result_msg = client.messages.calls[1]["messages"][-1]
    result_block = next(r for r in tool_result_msg["content"] if r["tool_use_id"] == "c1")
    assert result_block["is_error"] is True
