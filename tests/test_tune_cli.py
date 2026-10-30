"""`perflens tune ...` wiring, end to end on the synthetic runner (no GPU, no API key)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from perflens import cli
from perflens.triage import TriageError
from perflens.tune import agent as tune_agent
from tests.test_tune_agent import ScriptedClient, gelu, propose


def _rows(db: Path, sql: str) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return list(conn.execute(sql))


def _run(*args: str) -> Any:
    return CliRunner().invoke(cli.main, list(args), catch_exceptions=False)


def test_spaces_lists_every_kernel() -> None:
    result = _run("tune", "spaces")
    assert result.exit_code == 0
    kernels = [json.loads(line)["kernel"] for line in result.output.splitlines()]
    assert kernels == ["bias_gelu", "add_layernorm", "argmax"]


def test_run_and_report_on_fake_runner(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    common = ["--runner", "fake", "--db", str(db), "--batch", "8"]
    r1 = _run("tune", "run", "--kernel", "argmax", "--strategy", "exhaustive", *common)
    assert r1.exit_code == 0, r1.output
    assert "finished (space_exhausted)" in r1.output
    r2 = _run("tune", "run", "--kernel", "argmax", "--strategy", "random", "--budget", "5", *common)
    assert "finished (budget)" in r2.output

    runs = _rows(db, "SELECT strategy, budget, stop_reason, gpu FROM tune_runs ORDER BY id")
    assert [tuple(r) for r in runs] == [
        ("exhaustive", 12, "space_exhausted", "fake"),
        ("random", 5, "budget", "fake"),
    ]
    assert _rows(db, "SELECT COUNT(*) AS n FROM tune_evals")[0]["n"] == 17

    out = tmp_path / "results"
    r3 = _run("tune", "report", "--db", str(db), "--out", str(out))
    assert r3.exit_code == 0
    assert (out / "summary.csv").exists()
    assert "SYNTHETIC DATA" in (out / "REPORT.md").read_text()


def test_agent_strategy_records_rationales(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = ScriptedClient([propose(1, (gelu(256, bpr=4), "8 blocks only"), (gelu(128, bpr=8), "more"))])
    real = tune_agent.run_agent
    monkeypatch.setattr(cli.tune_agent, "run_agent", lambda ev, model: real(ev, client=client, model=model))
    db = tmp_path / "t.db"
    result = _run(
        "tune", "run", "--kernel", "bias_gelu", "--strategy", "agent", "--budget", "3",
        "--runner", "fake", "--db", str(db), "--model", "claude-test",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    evals = _rows(db, "SELECT idx, rationale FROM tune_evals ORDER BY idx")
    assert [e["rationale"] for e in evals] == ["default (shipped) config", "8 blocks only", "more"]
    assert _rows(db, "SELECT model FROM tune_runs")[0]["model"] == "claude-test"


def test_agent_failure_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(ev: Any, model: str) -> Any:
        ev.evaluate(ev.space.default)
        raise TriageError("no API key")

    monkeypatch.setattr(cli.tune_agent, "run_agent", boom)
    db = tmp_path / "t.db"
    result = CliRunner().invoke(
        cli.main,
        ["tune", "run", "--kernel", "argmax", "--strategy", "agent", "--runner", "fake", "--db", str(db)],
    )
    assert result.exit_code == 1
    assert "agent failed after 1 evaluations: no API key" in result.output
    assert _rows(db, "SELECT stop_reason FROM tune_runs")[0]["stop_reason"] == "agent_error"


def test_local_runner_needs_the_harness(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli.main,
        ["tune", "run", "--kernel", "argmax", "--strategy", "random", "--engine-dir", str(tmp_path),
         "--db", str(tmp_path / "t.db")],
    )  # fmt: skip
    assert result.exit_code == 1
    assert "build_candidate.sh not found" in result.output


def test_verify_requires_ground_truth(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli.main, ["tune", "verify", "--kernel", "argmax", "--db", str(tmp_path / "t.db")]
    )
    assert result.exit_code == 1
    assert "no exhaustive run for argmax batch 8" in result.output
