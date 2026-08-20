"""Claude-powered triage agent: correlates detector findings with kernel source
diffs and commit history to produce root-cause reports.

Tools are all read-only. The agent never sees write access to the repo, the
database, or CI -- it explains, a human decides. The final answer is forced
through a `submit_triage_report` tool with a strict JSON schema so the output
is always a validated `TriageReport`, never freeform prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from sqlite3 import Connection
from typing import Any, Protocol, cast

from perflens import gitctx, store
from perflens.models import Finding, FindingTriage, SuiteEntry, TriageReport

DEFAULT_MODEL = "claude-opus-5"
MAX_ITERATIONS = 12

SYSTEM_PROMPT = """You are a senior GPU performance engineer triaging CUDA kernel \
performance regressions flagged by an automated statistical detector (median + \
3xMAD gate on kernel duration, from Nsight Compute measurements).

For each finding you are given, use the read-only tools to gather evidence, then \
produce:
  1. severity (as given by the detector -- do not change it)
  2. an evidence summary quoting exact metric deltas (numbers, not vibes)
  3. a root-cause hypothesis tied to SPECIFIC diff hunks or commits you actually \
     observed via the tools -- never invent a code change you did not see
  4. a confidence level (low/medium/high)
  5. a next diagnostic step: which ncu section or flag to inspect next \
     (e.g. --set full on that kernel, __launch_bounds__, shared-memory tiling, \
     the launch config)

If the source diff cannot explain the delta, say so explicitly in the evidence \
summary and hypothesize environment causes instead (GPU clocks, driver version, \
thermal throttling, a noisy neighbor process) -- never invent code changes that \
are not present in the diff you were shown.

Classic patterns worth recognizing:
  - registers_per_thread UP + occupancy DOWN -> register pressure / spills, \
    often from added locals, unrolling, or a fused epilogue
  - L2 hit rate DOWN + DRAM throughput UP -> an access-pattern or coalescing \
    regression (e.g. a stride or layout change breaking coalesced loads)
  - duration UP with all other counters flat -> launch configuration change, \
    GPU clock/power state, or contention, not a source-level regression

Investigate every finding with at least one tool call before you submit, unless \
the entry's source_paths genuinely have no changes in range (an empty diff is \
itself evidence -- report it as such rather than skipping the finding). When you \
have gathered enough evidence for every finding, call `submit_triage_report` \
exactly once with one report entry per finding, in the same order they were given."""


class TriageError(RuntimeError):
    pass


class _MessagesLike(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class AnthropicLike(Protocol):
    """Structural type for the Anthropic client -- lets tests inject a stub
    without depending on the real SDK's response classes.
    """

    @property
    def messages(self) -> _MessagesLike: ...


@dataclass
class TriageContext:
    conn: Connection
    entries: dict[str, SuiteEntry]
    findings: list[Finding]
    base_sha: str
    head_sha: str

    def entry(self, name: str) -> SuiteEntry:
        if name not in self.entries:
            raise KeyError(f"unknown suite entry: {name}")
        return self.entries[name]


def _tool_defs() -> list[dict[str, Any]]:
    return [
        {
            "name": "get_findings",
            "description": (
                "Return the detector's findings for this run: entry, kernel, metric, "
                "base/new medians, delta_pct, and secondary-metric evidence."
            ),
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_metric_history",
            "description": "Recent per-run medians for one metric of one (entry, kernel), most recent first.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "entry": {"type": "string"},
                    "kernel": {"type": "string"},
                    "metric": {"type": "string"},
                    "n": {"type": "integer"},
                },
                "required": ["entry", "kernel", "metric"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_kernel_diff",
            "description": (
                "Unified source diff between base_sha and head_sha, "
                "limited to the entry's source_paths."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "entry": {"type": "string"},
                    "base_sha": {"type": "string"},
                    "head_sha": {"type": "string"},
                },
                "required": ["entry", "base_sha", "head_sha"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_commit_log",
            "description": "Commits touching the entry's source_paths between base_sha and head_sha.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "entry": {"type": "string"},
                    "base_sha": {"type": "string"},
                    "head_sha": {"type": "string"},
                },
                "required": ["entry", "base_sha", "head_sha"],
                "additionalProperties": False,
            },
        },
        {
            "name": "submit_triage_report",
            "description": "Submit the final triage report: one entry per finding, in the given order.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entry": {"type": "string"},
                                "kernel": {"type": "string"},
                                "severity": {"type": "string", "enum": ["minor", "major"]},
                                "evidence_summary": {"type": "string"},
                                "root_cause_hypothesis": {"type": "string"},
                                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                                "next_diagnostic_step": {"type": "string"},
                            },
                            "required": [
                                "entry",
                                "kernel",
                                "severity",
                                "evidence_summary",
                                "root_cause_hypothesis",
                                "confidence",
                                "next_diagnostic_step",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["findings"],
                "additionalProperties": False,
            },
        },
    ]


def _dispatch(ctx: TriageContext, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    try:
        if name == "get_findings":
            return json.dumps([f.model_dump() for f in ctx.findings]), False
        if name == "get_metric_history":
            history = store.get_metric_history(
                ctx.conn,
                tool_input["entry"],
                tool_input["kernel"],
                tool_input["metric"],
                tool_input.get("n", 10),
            )
            return json.dumps(history), False
        if name == "get_kernel_diff":
            entry = ctx.entry(tool_input["entry"])
            diff = gitctx.get_kernel_diff(
                entry.cwd, tool_input["base_sha"], tool_input["head_sha"], entry.source_paths
            )
            return diff, False
        if name == "get_commit_log":
            entry = ctx.entry(tool_input["entry"])
            log = gitctx.get_commit_log(
                entry.cwd, tool_input["base_sha"], tool_input["head_sha"], entry.source_paths
            )
            return json.dumps(log), False
        return f"unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
        return f"error: {exc}", True


def _initial_prompt(ctx: TriageContext) -> str:
    lines = [
        f"base_sha={ctx.base_sha} head_sha={ctx.head_sha}",
        "Findings to triage (also available via get_findings):",
    ]
    for f in ctx.findings:
        lines.append(
            f"- [{f.kind}] {f.entry}/{f.kernel}: {f.metric} {f.base_median:.0f} -> "
            f"{f.new_median:.0f} ({f.delta_pct:+.2f}%), evidence={f.evidence}"
        )
    lines.append("\nInvestigate each finding, then call submit_triage_report.")
    return "\n".join(lines)


def _build_report(raw_input: dict[str, Any], model: str) -> TriageReport:
    findings = [FindingTriage(**f) for f in raw_input.get("findings", [])]
    return TriageReport(
        findings=findings, model=model, generated_at=datetime.now(UTC).isoformat()
    )


def run_triage(
    ctx: TriageContext,
    client: AnthropicLike | None = None,
    model: str = DEFAULT_MODEL,
    max_iterations: int = MAX_ITERATIONS,
) -> TriageReport:
    if not ctx.findings:
        return TriageReport(findings=[], model=model, generated_at=datetime.now(UTC).isoformat())

    active_client: AnthropicLike
    if client is None:
        import anthropic

        active_client = cast("AnthropicLike", anthropic.Anthropic())
    else:
        active_client = client

    tools = _tool_defs()
    messages: list[dict[str, Any]] = [{"role": "user", "content": _initial_prompt(ctx)}]

    for _ in range(max_iterations):
        response = active_client.messages.create(
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            tools=tools,
            thinking={"type": "adaptive"},
            messages=messages,
        )

        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        submit_call = next((b for b in tool_uses if b.name == "submit_triage_report"), None)
        if submit_call is not None:
            try:
                return _build_report(submit_call.input, model)
            except Exception as exc:  # noqa: BLE001
                raise TriageError(f"agent submitted an invalid report: {exc}") from exc

        if not tool_uses:
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for call in tool_uses:
            content, is_error = _dispatch(ctx, call.name, call.input)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": content,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    raise TriageError("agent did not submit a triage report within max_iterations")
