"""Search strategies over a kernel's space, all at one shared evaluation budget.

Every strategy starts by evaluating the kernel's default (shipped) config, so
each run has the same baseline and "speedup vs. default" means the same thing
everywhere. After that:

- exhaustive: every config in the space (ground truth; ignores the budget)
- random: uniform sampling without replacement
- evo: (mu + lambda) evolution -- tournament-selected parents, one-dimension
  mutation, survivors chosen by measured fitness
- agent: see perflens.tune.agent
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from perflens.tune.runner import EvalResult, Runner
from perflens.tune.space import SearchSpace, TuneConfig

STRATEGIES = ("exhaustive", "random", "evo", "agent")


@dataclass
class EvalRecord:
    idx: int
    result: EvalResult
    rationale: str | None = None


@dataclass
class Evaluator:
    """Evaluates configs once each, enforces the budget, and reports every eval."""

    space: SearchSpace
    runner: Runner
    batch: int
    budget: int
    on_eval: Callable[[EvalRecord], None] | None = None
    records: list[EvalRecord] = field(default_factory=list)
    _seen: dict[str, int] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return self.budget - len(self.records)

    @property
    def seen(self) -> set[str]:
        return set(self._seen)

    def index_of(self, cfg: TuneConfig) -> int | None:
        return self._seen.get(cfg.key())

    def evaluate(self, cfg: TuneConfig, rationale: str | None = None) -> EvalRecord | None:
        """Evaluate `cfg` unless it was already evaluated or the budget is spent."""
        if self.remaining <= 0 or cfg.key() in self._seen:
            return None
        result = self.runner.evaluate(self.space, self.batch, cfg)
        record = EvalRecord(idx=len(self.records), result=result, rationale=rationale)
        self._seen[cfg.key()] = record.idx
        self.records.append(record)
        if self.on_eval is not None:
            self.on_eval(record)
        return record

    def best(self) -> EvalRecord | None:
        candidates = [r for r in self.records if math.isfinite(r.result.fitness)]
        return min(candidates, key=lambda r: r.result.fitness) if candidates else None


@dataclass
class Outcome:
    records: list[EvalRecord]
    stop_reason: str
    invalid_proposals: int = 0


def run_exhaustive(ev: Evaluator) -> Outcome:
    ev.evaluate(ev.space.default, "default (shipped) config")
    for cfg in ev.space.all():
        ev.evaluate(cfg)
    return Outcome(ev.records, "space_exhausted")


def run_random(ev: Evaluator, seed: int) -> Outcome:
    rng = random.Random(seed)
    ev.evaluate(ev.space.default, "default (shipped) config")
    order = [c for c in ev.space.all() if c.key() not in ev.seen]
    rng.shuffle(order)
    for cfg in order:
        if ev.remaining <= 0:
            return Outcome(ev.records, "budget")
        ev.evaluate(cfg)
    return Outcome(ev.records, "budget" if ev.remaining <= 0 else "space_exhausted")


def run_evolutionary(ev: Evaluator, seed: int, mu: int = 4, lam: int = 4, tournament: int = 2) -> Outcome:
    rng = random.Random(seed)
    first = ev.evaluate(ev.space.default, "default (shipped) config")
    population = [first] if first is not None else []
    while len(population) < mu and ev.remaining > 0:
        cfg = ev.space.sample_unseen(rng, ev.seen)
        if cfg is None:
            break
        rec = ev.evaluate(cfg, "initial population (random)")
        if rec is not None:
            population.append(rec)

    while ev.remaining > 0:
        children: list[EvalRecord] = []
        for _ in range(lam):
            if ev.remaining <= 0:
                break
            contenders = rng.sample(population, min(tournament, len(population)))
            parent = min(contenders, key=lambda r: r.result.fitness)
            child_cfg = None
            for _attempt in range(25):
                candidate = ev.space.mutate(parent.result.config, rng)
                if candidate.key() not in ev.seen:
                    child_cfg = candidate
                    break
            note = f"mutation of #{parent.idx}"
            if child_cfg is None:  # neighbourhood exhausted: fall back to a fresh random point
                child_cfg = ev.space.sample_unseen(rng, ev.seen)
                note = "random restart (mutation neighbourhood exhausted)"
            if child_cfg is None:
                return Outcome(ev.records, "space_exhausted")
            rec = ev.evaluate(child_cfg, note)
            if rec is not None:
                children.append(rec)
        population = sorted(population + children, key=lambda r: r.result.fitness)[:mu]
    return Outcome(ev.records, "budget")
