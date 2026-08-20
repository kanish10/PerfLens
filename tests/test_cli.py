"""End-to-end CLI tests using fixture data -- no real `ncu`, GPU, or network access.

Exercises: run (patched collect.run_entry) -> baseline set -> a second run ->
detect -> report, verifying the full pipeline wires together correctly and
that reports/*.md actually get written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from perflens import collect
from perflens.cli import main
from perflens.models import KernelResultRecord

SUITE_YAML = {
    "gpu": "RTX 3060",
    "reps": 5,
    "entries": [
        {
            "name": "inference.paged_attention",
            "cwd": ".",
            "cmd": "./build/bench_decode --once",
            "kernel_regex": "paged_attention.*",
            "source_paths": ["src/attention.cu"],
        }
    ],
}


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(SUITE_YAML))
    return tmp_path


def _fake_run_entry(durations: list[float]):
    def _run(entry, reps, metrics=collect.METRIC_SET, ncu_path="ncu"):
        return [
            KernelResultRecord(
                entry=entry.name,
                kernel="paged_attention_kernel",
                rep=i,
                duration_ns=d,
                occupancy_pct=78.0,
                regs_per_thread=32,
            )
            for i, d in enumerate(durations)
        ]

    return _run


def test_run_command_stores_kernel_results(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect, "run_entry", _fake_run_entry([100.0, 101.0, 99.0, 100.0, 100.0]))
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--git-sha", "abc123", "--branch", "main"])
    assert result.exit_code == 0, result.output
    assert "run 1: stored 5 kernel_results rows" in result.output


def test_baseline_set_then_detect_flags_regression(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collect, "run_entry", _fake_run_entry([100.0, 101.0, 99.0, 100.0, 100.0]))
    runner = CliRunner()
    r1 = runner.invoke(main, ["run", "--git-sha", "abc123"])
    assert r1.exit_code == 0, r1.output

    r2 = runner.invoke(main, ["baseline", "set", "--run", "1"])
    assert r2.exit_code == 0, r2.output
    assert "baseline set" in r2.output

    monkeypatch.setattr(collect, "run_entry", _fake_run_entry([130.0, 131.0, 129.0, 130.0, 130.0]))
    r3 = runner.invoke(main, ["run", "--git-sha", "def456"])
    assert r3.exit_code == 0, r3.output

    r4 = runner.invoke(main, ["detect", "--run", "2", "--json-out", "findings.json"])
    assert r4.exit_code == 0, r4.output
    assert "MAJOR" in r4.output

    findings = json.loads((project / "findings.json").read_text())
    assert len(findings) == 1
    assert findings[0]["kind"] == "regression"
    assert findings[0]["severity"] == "major"


def test_report_command_writes_markdown(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect, "run_entry", _fake_run_entry([100.0] * 5))
    runner = CliRunner()
    runner.invoke(main, ["run", "--git-sha", "abc123"])
    runner.invoke(main, ["baseline", "set", "--run", "1"])

    result = runner.invoke(main, ["report", "--run", "1", "--nightly"])
    assert result.exit_code == 0, result.output

    reports = list((project / "reports").glob("*.md"))
    assert len(reports) == 1
    content = reports[0].read_text()
    assert "PASS" in content
    assert "abc123" in content


def test_gate_exits_nonzero_on_major_regression(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collect, "run_entry", _fake_run_entry([100.0] * 5))
    runner = CliRunner()
    runner.invoke(main, ["run", "--git-sha", "abc123"])
    runner.invoke(main, ["baseline", "set", "--run", "1"])

    monkeypatch.setattr(collect, "run_entry", _fake_run_entry([200.0] * 5))
    runner.invoke(main, ["run", "--git-sha", "def456"])

    result = runner.invoke(main, ["gate", "--run", "2", "--skip-triage"])
    assert result.exit_code == 1
    assert "MAJOR" in result.output


def test_gate_exits_zero_when_clean(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect, "run_entry", _fake_run_entry([100.0] * 5))
    runner = CliRunner()
    runner.invoke(main, ["run", "--git-sha", "abc123"])
    runner.invoke(main, ["baseline", "set", "--run", "1"])

    monkeypatch.setattr(collect, "run_entry", _fake_run_entry([100.5] * 5))
    runner.invoke(main, ["run", "--git-sha", "def456"])

    result = runner.invoke(main, ["gate", "--run", "2", "--skip-triage"])
    assert result.exit_code == 0


def test_detect_unknown_run_raises(project: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--run", "999"])
    assert result.exit_code != 0


def test_run_print_run_id_outputs_only_the_id(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(collect, "run_entry", _fake_run_entry([100.0] * 5))
    runner = CliRunner()
    result = runner.invoke(main, ["run", "--git-sha", "abc123", "--print-run-id"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "1"


def test_baseline_set_entry_filter_scopes_to_one_entry(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    two_entry_suite = {
        "gpu": "RTX 3060",
        "reps": 5,
        "entries": [
            SUITE_YAML["entries"][0],
            {
                "name": "inference.fused_gelu",
                "cwd": ".",
                "cmd": "./build/bench_decode --once",
                "kernel_regex": "bias_gelu.*",
                "source_paths": ["src/gelu.cu"],
            },
        ],
    }
    (project / "suite.yaml").write_text(yaml.safe_dump(two_entry_suite))

    def fake_run_entry_two(entry, reps, metrics=collect.METRIC_SET, ncu_path="ncu"):
        kernel = "paged_attention_kernel" if "paged" in entry.name else "bias_gelu_kernel"
        return [
            KernelResultRecord(entry=entry.name, kernel=kernel, rep=i, duration_ns=100.0)
            for i in range(5)
        ]

    monkeypatch.setattr(collect, "run_entry", fake_run_entry_two)
    runner = CliRunner()
    r1 = runner.invoke(main, ["run", "--git-sha", "abc123"])
    assert r1.exit_code == 0, r1.output

    r2 = runner.invoke(
        main, ["baseline", "set", "--run", "1", "--entry", "inference.paged_attention"]
    )
    assert r2.exit_code == 0, r2.output
    assert "baseline set: 1" in r2.output  # only duration_ns for the one filtered entry

    from perflens import store

    conn = store.get_connection(project / "perflens.db")
    baselined = store.get_baseline(
        conn, "inference.paged_attention", "paged_attention_kernel", "duration_ns"
    )
    not_baselined = store.get_baseline(conn, "inference.fused_gelu", "bias_gelu_kernel", "duration_ns")
    assert baselined is not None
    assert not_baselined is None
