"""Search strategies on the synthetic landscape: budget, de-duplication, determinism, correctness gate."""

from __future__ import annotations

import pytest

from perflens.tune.runner import FakeRunner
from perflens.tune.space import get_space
from perflens.tune.strategies import (
    EvalRecord,
    Evaluator,
    run_evolutionary,
    run_exhaustive,
    run_random,
)


def _ev(kernel: str = "bias_gelu", budget: int = 20, runner: FakeRunner | None = None) -> Evaluator:
    return Evaluator(get_space(kernel), runner or FakeRunner(seed=0), batch=8, budget=budget)


@pytest.mark.parametrize("strategy", ["random", "evo"])
def test_budgeted_strategies_start_at_default_and_never_repeat(strategy: str) -> None:
    ev = _ev(budget=20)
    outcome = run_random(ev, seed=3) if strategy == "random" else run_evolutionary(ev, seed=3)
    keys = [r.result.config.key() for r in outcome.records]
    assert outcome.records[0].result.config == ev.space.default
    assert len(keys) == len(set(keys)) == 20
    assert outcome.stop_reason == "budget"
    assert [r.idx for r in outcome.records] == list(range(20))


@pytest.mark.parametrize("strategy", ["random", "evo"])
def test_same_seed_same_sequence(strategy: str) -> None:
    def keys(seed: int) -> list[str]:
        ev = _ev(budget=15)
        out = run_random(ev, seed) if strategy == "random" else run_evolutionary(ev, seed)
        return [r.result.config.key() for r in out.records]

    assert keys(7) == keys(7)
    assert keys(7) != keys(8)


def test_exhaustive_covers_the_space_once_with_default_first() -> None:
    space = get_space("bias_gelu")
    ev = _ev(budget=space.size)
    outcome = run_exhaustive(ev)
    assert outcome.stop_reason == "space_exhausted"
    assert len(outcome.records) == space.size
    assert outcome.records[0].result.config == space.default
    assert {r.result.config.key() for r in outcome.records} == {c.key() for c in space.all()}


@pytest.mark.parametrize("strategy", ["random", "evo"])
def test_budget_larger_than_space_stops_when_space_is_exhausted(strategy: str) -> None:
    ev = _ev("argmax", budget=40)
    outcome = run_random(ev, 1) if strategy == "random" else run_evolutionary(ev, 1)
    assert len(outcome.records) == 12
    assert outcome.stop_reason == "space_exhausted"


def test_incorrect_configs_are_never_best() -> None:
    ev = _ev(budget=120, runner=FakeRunner(seed=0, incorrect_fast_math=True))
    run_exhaustive(ev)
    best = ev.best()
    assert best is not None
    assert best.result.config.fast_math is False
    assert all(r.result.fitness == float("inf") for r in ev.records if r.result.config.fast_math)


def test_evaluator_skips_duplicates_and_reports_each_eval() -> None:
    seen: list[EvalRecord] = []
    ev = Evaluator(get_space("argmax"), FakeRunner(), batch=8, budget=3, on_eval=seen.append)
    assert ev.evaluate(ev.space.default, "first") is not None
    assert ev.evaluate(ev.space.default) is None  # duplicate: no budget spent
    assert ev.remaining == 2
    assert [r.rationale for r in seen] == ["first"]
    assert ev.index_of(ev.space.default) == 0


def test_evolution_improves_on_default() -> None:
    ev = _ev(budget=20)
    run_evolutionary(ev, seed=1)
    best = ev.best()
    assert best is not None
    assert best.result.fitness < ev.records[0].result.fitness
