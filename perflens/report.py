"""Markdown report rendering and PR/commit-status posting via the `gh` CLI."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from perflens.models import Finding, RunInfo, TriageReport

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def severity_label(finding: Finding) -> str:
    """The one place that decides how a Finding's kind/severity renders as
    text -- shared by the Markdown table and the CLI's plain-text table so
    the two can't silently drift on what counts as "improvement" vs "?".
    """
    if finding.kind == "improvement":
        return "IMPROVEMENT"
    return finding.severity.upper() if finding.severity else "?"


def _findings_table(findings: list[Finding]) -> str:
    if not findings:
        return "_No regressions or improvements detected against baseline._\n"
    header = "| Severity | Entry | Kernel | Metric | Baseline | New | Delta |\n"
    header += "|---|---|---|---|---|---|---|\n"
    rows = []
    for f in findings:
        rows.append(
            f"| {severity_label(f)} | {f.entry} | `{f.kernel}` | {f.metric} | "
            f"{f.base_median:,.0f} | {f.new_median:,.0f} | {f.delta_pct:+.2f}% |"
        )
    return header + "\n".join(rows) + "\n"


def _triage_narrative(findings: list[Finding], triage: TriageReport | None) -> str:
    if triage is None or not triage.findings:
        return ""
    # Keyed on entry alone, not (entry, kernel): kernel names are long
    # compiler-demangled C++ signatures (e.g. ncu drops leading namespace
    # segments like "kernels::<" when only one kernel type is in the capture
    # window), and an agent restating one in its own words -- especially a
    # smaller/free model -- won't always reproduce it byte-for-byte. entry is
    # a short internal identifier the agent has no reason to paraphrase, and
    # detect.py only ever produces one regression Finding per entry (primary
    # metric is duration_ns; every suite.yaml entry targets one kernel), so
    # it's already an unambiguous key. Caught by a real run: a Groq
    # gpt-oss-120b report cleaned up "unnamed>::paged_attention_kernel(...)"
    # to "paged_attention_kernel(...)", silently dropping this section.
    by_key = {t.entry: t for t in triage.findings}
    sections = ["\n## Agent analysis\n"]
    for f in findings:
        t = by_key.get(f.entry)
        if t is None:
            continue
        # Severity always comes from the detector (severity_label(f)), never
        # from the agent's own restated t.severity -- the system prompt tells
        # the agent not to change it, but rendering shouldn't trust that a
        # model (especially a smaller/free one) always complies. Caught by a
        # real run: a Groq gpt-oss-120b report said "major" here for a
        # finding the detector had classified "minor".
        sections.append(f"### {f.entry} / `{f.kernel}` -- {severity_label(f)}\n")
        sections.append(f"**Evidence:** {t.evidence_summary}\n")
        sections.append(f"**Root-cause hypothesis:** {t.root_cause_hypothesis}\n")
        sections.append(f"**Confidence:** {t.confidence}\n")
        sections.append(f"**Next diagnostic step:** {t.next_diagnostic_step}\n")
    sections.append(f"\n_Analysis by {triage.model} at {triage.generated_at}._\n")
    return "\n".join(sections)


def _improvement_notes(findings: list[Finding]) -> str:
    improvements = [f for f in findings if f.kind == "improvement"]
    if not improvements:
        return ""
    lines = ["\n## Improvements\n"]
    for f in improvements:
        lines.append(
            f"- `{f.entry}/{f.kernel}`: {f.metric} improved {f.delta_pct:+.2f}% "
            f"({f.base_median:,.0f} -> {f.new_median:,.0f}). "
            f"Consider: `{f.suggested_command}`"
        )
    return "\n".join(lines) + "\n"


def render_markdown(
    run: RunInfo | dict[str, Any],
    findings: list[Finding],
    triage: TriageReport | None = None,
) -> str:
    if not isinstance(run, RunInfo):
        run = RunInfo(**dict(run))

    major = [f for f in findings if f.kind == "regression" and f.severity == "major"]
    minor = [f for f in findings if f.kind == "regression" and f.severity == "minor"]

    lines = [
        f"# PerfLens report -- {run.ts}",
        "",
        f"- **git sha:** `{run.git_sha}` (`{run.branch}`)",
        f"- **gpu:** {run.gpu}",
        f"- **driver:** {run.driver}",
        f"- **cuda:** {run.cuda_version}",
        f"- **suite hash:** `{run.suite_hash}`",
        f"- **gate:** {'FAIL' if major else 'PASS'} "
        f"({len(major)} major, {len(minor)} minor regression(s))",
        "",
        "## Findings",
        "",
        _findings_table(findings),
    ]
    narrative = _triage_narrative(findings, triage)
    if narrative:
        lines.append(narrative)
    notes = _improvement_notes(findings)
    if notes:
        lines.append(notes)
    return "\n".join(lines)


def write_report(path: str | Path, content: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _run_gh(runner: CommandRunner, cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return runner(cmd)
    except FileNotFoundError as exc:
        # subprocess.run raises this (not a nonzero return code) when the
        # binary isn't on PATH -- normalize it so every caller can catch a
        # single RuntimeError regardless of which way `gh` is unavailable.
        raise RuntimeError(f"'gh' not found on PATH: {exc}") from exc


def post_pr_comment(
    pr_number: int, body: str, runner: CommandRunner = _default_runner
) -> subprocess.CompletedProcess[str]:
    """Post `body` as a comment on the given PR via `gh pr comment`."""
    proc = _run_gh(runner, ["gh", "pr", "comment", str(pr_number), "--body", body])
    if proc.returncode != 0:
        raise RuntimeError(f"gh pr comment failed: {proc.stderr}")
    return proc


def set_commit_status(
    sha: str,
    state: str,
    description: str,
    context: str = "perflens/regression",
    runner: CommandRunner = _default_runner,
) -> subprocess.CompletedProcess[str]:
    """Set a commit status via the GitHub API (`gh api`). `state` is one of
    error|failure|pending|success.
    """
    if state not in {"error", "failure", "pending", "success"}:
        raise ValueError(f"invalid commit status state: {state}")
    proc = _run_gh(
        runner,
        [
            "gh",
            "api",
            f"repos/{{owner}}/{{repo}}/statuses/{sha}",
            "-f",
            f"state={state}",
            "-f",
            f"description={description}",
            "-f",
            f"context={context}",
        ],
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh api statuses failed: {proc.stderr}")
    return proc
