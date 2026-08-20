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
    run_id: int
    # Explicit overrides. Leave unset (the normal path): each entry's source
    # commits are resolved per-entry from recorded source_sha history, since
    # different entries can live in different sibling repos (see suite.yaml)
    # with independent commit histories -- there is no single (base_sha,
    # head_sha) pair that's valid across all of them.
    base_sha: str | None = None
    head_sha: str | None = None

    def entry(self, name: str) -> SuiteEntry:
        if name not in self.entries:
            raise KeyError(f"unknown suite entry: {name}")
        return self.entries[name]

    def resolve_shas(self, entry_name: str) -> tuple[str | None, str | None]:
        """(base_sha, head_sha) for one entry's own source repo.

        Explicit overrides win when both are set. Otherwise: head is the
        source_sha recorded for this entry in this run; base is the
        source_sha recorded for this entry in whichever run the entry's
        (entry, kernel) baseline was set from. Never falls back to this
        repo's (PerfLens's) own git history -- that repo is not where the
        kernel source lives.
        """
        if self.base_sha is not None and self.head_sha is not None:
            return self.base_sha, self.head_sha
        head = store.get_entry_source_sha(self.conn, self.run_id, entry_name)
        base = None
        finding = next((f for f in self.findings if f.entry == entry_name), None)
        if finding is not None:
            baseline = store.get_baseline(self.conn, entry_name, finding.kernel, "duration_ns")
            if baseline is not None:
                base = store.get_entry_source_sha(self.conn, baseline["run_id"], entry_name)
        return base, head


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
                "Unified source diff for this entry, limited to its source_paths, "
                "between its baseline run's source commit and this run's source "
                "commit. The correct commits for this entry's own repo are "
                "resolved automatically -- you only supply the entry name."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"entry": {"type": "string"}},
                "required": ["entry"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_commit_log",
            "description": (
                "Commits touching this entry's source_paths between its baseline "
                "run's source commit and this run's source commit. The correct "
                "commits for this entry's own repo are resolved automatically."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"entry": {"type": "string"}},
                "required": ["entry"],
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
            base_sha, head_sha = ctx.resolve_shas(tool_input["entry"])
            if not base_sha or not head_sha:
                return (
                    "no recorded source commit for this entry (baseline or current "
                    "run predates source_sha tracking, or the entry's cwd isn't a "
                    "git checkout) -- cannot diff; treat this as inconclusive, not "
                    "as evidence of no change",
                    True,
                )
            diff = gitctx.get_kernel_diff(entry.cwd, base_sha, head_sha, entry.source_paths)
            return diff, False
        if name == "get_commit_log":
            entry = ctx.entry(tool_input["entry"])
            base_sha, head_sha = ctx.resolve_shas(tool_input["entry"])
            if not base_sha or not head_sha:
                return "no recorded source commit for this entry -- cannot list commits", True
            log = gitctx.get_commit_log(entry.cwd, base_sha, head_sha, entry.source_paths)
            return json.dumps(log), False
        return f"unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
        return f"error: {exc}", True


def _initial_prompt(ctx: TriageContext, findings: list[Finding]) -> str:
    lines = ["Findings to triage (also available via get_findings):"]
    for f in findings:
        base_sha, head_sha = ctx.resolve_shas(f.entry)
        lines.append(
            f"- [{f.kind}] {f.entry}/{f.kernel}: {f.metric} {f.base_median:.0f} -> "
            f"{f.new_median:.0f} ({f.delta_pct:+.2f}%), evidence={f.evidence}, "
            f"source commits: base={base_sha or 'unknown'} head={head_sha or 'unknown'}"
        )
    lines.append(
        "\nFor get_kernel_diff/get_commit_log, pass just the entry name -- the "
        "correct source commits for that entry's own repo are resolved for you. "
        "Investigate each finding, then call submit_triage_report."
    )
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
    # Improvements have no severity to report and no root cause to explain --
    # only regressions go to the agent. submit_triage_report's schema requires
    # severity in {minor, major}, which an improvement (severity=None) can't
    # satisfy without the agent fabricating a value.
    regression_findings = [f for f in ctx.findings if f.kind == "regression"]
    if not regression_findings:
        return TriageReport(findings=[], model=model, generated_at=datetime.now(UTC).isoformat())

    active_client: AnthropicLike
    if client is None:
        try:
            import anthropic

            active_client = cast("AnthropicLike", anthropic.Anthropic())
        except Exception as exc:  # noqa: BLE001 - normalize any SDK/env error
            raise TriageError(f"could not construct Anthropic client: {exc}") from exc
    else:
        active_client = client

    tools = _tool_defs()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _initial_prompt(ctx, regression_findings)}
    ]

    for _ in range(max_iterations):
        try:
            response = active_client.messages.create(
                model=model,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                tools=tools,
                thinking={"type": "adaptive"},
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - normalize any SDK error (auth, rate limit, ...)
            raise TriageError(f"Claude API call failed: {exc}") from exc

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
