"""Report generation: ground truth, canonical scoring, the claims ledger, and the fake-data banner."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from perflens.tune import db, report
from perflens.tune.runner import FakeRunner
from perflens.tune.space import get_space
from perflens.tune.strategies import (
    Evaluator,
    Outcome,
    run_evolutionary,
    run_exhaustive,
    run_random,
)


def _record(conn: sqlite3.Connection, strategy: str, seed: int, budget: int, gpu: str = "fake") -> int:
    space = get_space("bias_gelu")
    run_id = db.insert_run(
        conn,
        kernel="bias_gelu",
        batch=8,
        strategy=strategy,
        seed=seed,
        budget=budget,
        gpu=gpu,
        driver="test",
        engine_sha="abc123def456",
        model="claude-test" if strategy == "agent" else None,
        space=space.describe(),
    )
    ev = Evaluator(space, FakeRunner(seed=seed), 8, budget, on_eval=lambda r: db.insert_eval(conn, run_id, r))
    outcome: Outcome
    if strategy == "exhaustive":
        outcome = run_exhaustive(ev)
    elif strategy == "random":
        outcome = run_random(ev, seed)
    elif strategy == "evo":
        outcome = run_evolutionary(ev, seed)
    else:  # stand-in for an agent run: fixed picks with rationales
        ev.evaluate(space.default, "default (shipped) config")
        for cfg, why in [
            ({"threads": 256, "blocks_per_row": 4, "vec": 1, "fast_math": False}, "only 8 blocks (#0)"),
            ({"threads": 256, "blocks_per_row": 4, "vec": 2, "fast_math": False}, "halve loads"),
            ({"threads": 256, "blocks_per_row": 4, "vec": 2, "fast_math": True}, "tanhf is hot"),
        ]:
            ev.evaluate(space.validate(cfg), why)
        outcome = Outcome(ev.records, "budget", invalid_proposals=1)
    db.finish_run(conn, run_id, outcome.stop_reason, outcome.invalid_proposals)
    return run_id


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_tune_db(conn)
    return conn


def test_summaries_score_against_exhaustive_ground_truth() -> None:
    conn = _conn()
    exh = _record(conn, "exhaustive", 0, 120)
    _record(conn, "random", 1, 20)
    _record(conn, "evo", 1, 20)
    _record(conn, "agent", 1, 4)
    summaries, truth = report.summarize(conn)

    gt = truth[("bias_gelu", 8)]
    assert gt.run_id == exh and gt.n_configs == 120
    assert gt.best_us == min(gt.canonical_us.values())

    by_strategy = {s.strategy: s for s in summaries}
    assert by_strategy["exhaustive"].regret_pct == 0.0
    assert by_strategy["exhaustive"].evals_to_5pct is None
    for name in ("random", "evo", "agent"):
        s = by_strategy[name]
        assert s.regret_pct is not None and s.regret_pct >= 0.0
        assert s.best_us_canonical is not None
        if s.evals_to_5pct is not None:
            assert 1 <= s.evals_to_5pct <= s.evals
    assert by_strategy["agent"].invalid_proposals == 1
    assert by_strategy["agent"].compile_s_total > 0


def test_report_files_banner_ledger_and_rationales(tmp_path: Path) -> None:
    conn = _conn()
    _record(conn, "exhaustive", 0, 120)
    _record(conn, "evo", 1, 20)
    _record(conn, "agent", 1, 4)
    csv_path, md_path = report.write_report(conn, tmp_path / "out", "test.db")

    rows = list(csv.DictReader(csv_path.open()))
    assert [r["strategy"] for r in rows] == ["exhaustive", "evo", "agent"]
    md = md_path.read_text()
    assert "SYNTHETIC DATA" in md
    assert "## bias_gelu, batch 8" in md
    assert "Exhaustive search (run #1, 120 configs)" in md
    assert "## Claims ledger" in md
    assert "bias_gelu b8: best config vs. default" in md
    assert "#1 `t256/bpr4`: only 8 blocks (#0)" in md


def test_real_gpu_runs_get_no_banner_and_missing_ground_truth_is_explicit(tmp_path: Path) -> None:
    conn = _conn()
    _record(conn, "random", 1, 10, gpu="NVIDIA GeForce RTX 3060")
    summaries, truth = report.summarize(conn)
    assert truth == {}
    assert summaries[0].regret_pct is None and summaries[0].evals_to_5pct is None
    md = report.render_markdown(conn, summaries, truth, "x.db")
    assert "SYNTHETIC" not in md
    assert "No exhaustive run for this kernel and batch" in md


def test_empty_database() -> None:
    conn = _conn()
    summaries, truth = report.summarize(conn)
    assert "No finished tuning runs" in report.render_markdown(conn, summaries, truth, "x.db")


def test_unfinished_runs_are_ignored() -> None:
    conn = _conn()
    db.insert_run(
        conn,
        kernel="argmax",
        batch=8,
        strategy="random",
        seed=1,
        budget=5,
        gpu="fake",
        driver="t",
        engine_sha=None,
        model=None,
        space={},
    )
    assert report.summarize(conn) == ([], {})


def test_exhaustive_with_nothing_correct_says_so() -> None:
    conn = _conn()
    space = get_space("argmax")
    run_id = db.insert_run(
        conn,
        kernel="argmax",
        batch=8,
        strategy="exhaustive",
        seed=0,
        budget=space.size,
        gpu="STUB",
        driver="t",
        engine_sha=None,
        model=None,
        space=space.describe(),
    )
    ev = Evaluator(
        space,
        FakeRunner(incorrect_fast_math=True),
        8,
        space.size,
        on_eval=lambda r: db.insert_eval(
            conn, run_id, type(r)(idx=r.idx, result=r.result.model_copy(update={"correct": False}))
        ),
    )
    run_exhaustive(ev)
    db.finish_run(conn, run_id, "space_exhausted", 0)
    summaries, truth = report.summarize(conn)
    md = report.render_markdown(conn, summaries, truth, "x.db")
    assert "found no configuration that built and passed the correctness check (12 of 12 rejected)" in md
    assert "| strategy |" not in md  # no strategy runs, no empty table
