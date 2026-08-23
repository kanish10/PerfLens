from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from perflens import report
from perflens.models import Finding, FindingTriage, RunInfo, TriageReport

RUN = RunInfo(
    id=1,
    ts="2026-01-02T00:00:00+00:00",
    git_sha="deadbeef1234",
    branch="demo-regression",
    gpu="RTX 3060",
    driver="550.54.14",
    cuda_version="12.4",
    suite_hash="abc123",
)

MAJOR_FINDING = Finding(
    entry="inference.paged_attention",
    kernel="paged_attention_kernel",
    base_median=12450.0,
    new_median=15600.0,
    delta_pct=25.3,
    evidence={"regs_per_thread_delta_pct": 50.0, "occupancy_pct_delta_pct": -18.0},
    severity="major",
    kind="regression",
)

IMPROVEMENT_FINDING = Finding(
    entry="raytracer.trace",
    kernel="trace_rays_kernel",
    base_median=9000.0,
    new_median=7000.0,
    delta_pct=-22.2,
    evidence={},
    severity=None,
    kind="improvement",
    suggested_command="perflens baseline set --run 42",
)


def test_render_markdown_includes_run_header() -> None:
    md = report.render_markdown(RUN, [MAJOR_FINDING])
    assert "deadbeef1234" in md
    assert "RTX 3060" in md
    assert "demo-regression" in md


def test_render_markdown_no_findings_says_so() -> None:
    md = report.render_markdown(RUN, [])
    assert "No regressions or improvements" in md
    assert "PASS" in md


def test_render_markdown_major_finding_fails_gate() -> None:
    md = report.render_markdown(RUN, [MAJOR_FINDING])
    assert "FAIL" in md
    assert "MAJOR" in md
    assert "paged_attention_kernel" in md


def test_render_markdown_accepts_dict_run() -> None:
    md = report.render_markdown(dict(RUN.model_dump()), [MAJOR_FINDING])
    assert "RTX 3060" in md


def test_render_markdown_improvement_section() -> None:
    md = report.render_markdown(RUN, [IMPROVEMENT_FINDING])
    assert "PASS" in md  # improvements never fail the gate
    assert "## Improvements" in md
    assert "baseline set --run 42" in md


def test_render_markdown_includes_agent_narrative() -> None:
    triage_report = TriageReport(
        findings=[
            FindingTriage(
                entry="inference.paged_attention",
                kernel="paged_attention_kernel",
                severity="major",
                evidence_summary="duration +25.3%, regs_per_thread +50%, occupancy -18%",
                root_cause_hypothesis="Added per-thread locals increased register pressure, "
                "forcing spills and dropping occupancy.",
                confidence="high",
                next_diagnostic_step="Re-profile with --set full and inspect launch__registers_per_thread",
            )
        ],
        model="claude-opus-5",
        generated_at="2026-01-02T00:05:00+00:00",
    )
    md = report.render_markdown(RUN, [MAJOR_FINDING], triage_report)
    assert "## Agent analysis" in md
    assert "register pressure" in md
    assert "claude-opus-5" in md


def test_render_markdown_narrative_severity_always_comes_from_detector() -> None:
    # Real bug, caught on a live run: a Groq gpt-oss-120b triage said "major"
    # for a finding the detector had classified "minor" -- the system prompt
    # tells the agent not to change severity, but rendering must not trust
    # that every model complies. The narrative heading must show the
    # detector's severity, not the agent's restated (and here, wrong) one.
    minor_finding = MAJOR_FINDING.model_copy(update={"severity": "minor", "delta_pct": 13.77})
    triage_report = TriageReport(
        findings=[
            FindingTriage(
                entry="inference.paged_attention",
                kernel="paged_attention_kernel",
                severity="major",  # agent got this wrong; must be ignored
                evidence_summary="duration +13.77%, regs_per_thread +287.5%",
                root_cause_hypothesis="Added per-thread checksum array increased register pressure.",
                confidence="high",
                next_diagnostic_step="Re-profile with --set full",
            )
        ],
        model="openai/gpt-oss-120b",
        generated_at="2026-01-02T00:05:00+00:00",
    )
    md = report.render_markdown(RUN, [minor_finding], triage_report)
    assert "-- MINOR" in md
    assert "-- MAJOR" not in md


def test_render_markdown_narrative_matches_on_entry_despite_kernel_name_mismatch() -> None:
    # Real bug, caught on a live run: a Groq gpt-oss-120b report restated the
    # kernel name as "paged_attention_kernel(...)" when the detector's Finding
    # (and the DB) had it as "unnamed>::paged_attention_kernel(...)" -- ncu
    # drops leading namespace segments in single-kernel capture windows, and
    # the agent silently "cleaned up" the odd-looking prefix when repeating
    # it. An exact (entry, kernel) match on that string dropped the whole
    # narrative section for a real finding.
    finding = MAJOR_FINDING.model_copy(
        update={"kernel": "unnamed>::paged_attention_kernel(const __half *, int)"}
    )
    triage_report = TriageReport(
        findings=[
            FindingTriage(
                entry="inference.paged_attention",
                kernel="paged_attention_kernel(const __half *, int)",  # agent paraphrased this
                severity="major",
                evidence_summary="duration +25.3%",
                root_cause_hypothesis="Register pressure from added locals.",
                confidence="high",
                next_diagnostic_step="Re-profile with --set full",
            )
        ],
        model="openai/gpt-oss-120b",
        generated_at="2026-01-02T00:05:00+00:00",
    )
    md = report.render_markdown(RUN, [finding], triage_report)
    assert "Register pressure from added locals." in md


def test_write_report_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "report.md"
    result = report.write_report(out, "# hello")
    assert result == out
    assert out.read_text() == "# hello"


def test_post_pr_comment_invokes_gh_with_body() -> None:
    calls = []

    def fake_runner(cmd: list[str]) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    report.post_pr_comment(42, "hello world", runner=fake_runner)

    assert calls[0][:3] == ["gh", "pr", "comment"]
    assert "42" in calls[0]
    assert "hello world" in calls[0]


def test_post_pr_comment_raises_on_failure() -> None:
    def fake_runner(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")

    with pytest.raises(RuntimeError):
        report.post_pr_comment(42, "hello", runner=fake_runner)


def test_set_commit_status_validates_state() -> None:
    with pytest.raises(ValueError):
        report.set_commit_status("sha", "bogus_state", "desc")


def test_set_commit_status_invokes_gh_api() -> None:
    calls = []

    def fake_runner(cmd: list[str]) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    report.set_commit_status("deadbeef", "failure", "1 major regression", runner=fake_runner)

    cmd = calls[0]
    assert cmd[:2] == ["gh", "api"]
    assert any("state=failure" in c for c in cmd)
    assert any("context=perflens/regression" in c for c in cmd)
