"""Agent-guided search with a scripted client: proposal validation, budget accounting,
read-only tools, stall handling. Exercises the loop, not the model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from perflens.triage import TriageError
from perflens.tune import agent
from perflens.tune.runner import FakeRunner
from perflens.tune.space import get_space
from perflens.tune.strategies import Evaluator


@dataclass
class Block:
    type: str
    name: str | None = None
    input: dict[str, Any] | None = None
    id: str | None = None
    text: str | None = None


@dataclass
class Response:
    content: list[Block]


class ScriptedMessages:
    def __init__(self, scripted: list[Response]) -> None:
        self.scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Response:
        # Snapshot the conversation so later appends don't rewrite history.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.scripted:
            raise AssertionError("scripted client ran out of responses")
        return self.scripted.pop(0)


class ScriptedClient:
    def __init__(self, scripted: list[Response]) -> None:
        self.messages = ScriptedMessages(scripted)


def gelu(threads: int, bpr: int = 1, vec: int = 1, fast_math: bool = False) -> dict[str, Any]:
    return {"threads": threads, "blocks_per_row": bpr, "vec": vec, "fast_math": fast_math}


def call(name: str, tool_input: dict[str, Any], i: int) -> Block:
    return Block(type="tool_use", name=name, input=tool_input, id=f"tu_{i}")


def propose(i: int, *items: tuple[dict[str, Any], str]) -> Response:
    return Response(
        [call("propose_candidates", {"candidates": [{"config": c, "rationale": r} for c, r in items]}, i)]
    )


def _ev(budget: int) -> Evaluator:
    return Evaluator(get_space("bias_gelu"), FakeRunner(seed=0), batch=8, budget=budget)


def test_agent_run_validates_proposals_and_spends_budget_only_on_new_configs() -> None:
    client = ScriptedClient(
        [
            Response([call("get_kernel_source", {}, 1), call("get_ptxas_log", {"index": 0}, 2)]),
            propose(
                3,
                (gelu(256, bpr=4), "only 8 blocks at batch 8 (#0); split rows across SMs"),
                (gelu(256, vec=2), "halve load instructions"),
                (gelu(96), "not a power of two"),
                (gelu(256), "same as default"),
            ),
            Response([Block(type="text", text="Thinking about what to try next.")]),
            propose(4, (gelu(512, bpr=4), "bigger blocks"), (gelu(128, bpr=8), "smaller blocks")),
        ]
    )
    ev = _ev(budget=4)
    outcome = agent.run_agent(ev, client=client, model="claude-test")

    assert outcome.stop_reason == "budget"
    assert outcome.invalid_proposals == 2  # out-of-space + duplicate; budget overflow is not "invalid"
    picked = [r.result.config.short() for r in outcome.records]
    assert picked == ["t256", "t256/bpr4", "t256/vec2", "t512/bpr4"]
    assert outcome.records[1].rationale == "only 8 blocks at batch 8 (#0); split rows across SMs"

    calls = client.messages.calls
    assert len(calls) == 4
    first = calls[0]
    assert first["model"] == "claude-test"
    assert "Result #0 (default config)" in first["messages"][0]["content"]
    tool_names = [t["name"] for t in first["tools"]]
    assert tool_names == ["get_kernel_source", "get_ptxas_log", "propose_candidates"]
    schema = first["tools"][2]["input_schema"]["properties"]["candidates"]["items"]["properties"]["config"]
    assert schema["properties"]["vec"] == {"enum": [1, 2]}

    # Read-only tool results came back on the next call.
    tool_results = calls[1]["messages"][-1]["content"]
    assert "synthetic source for bias_gelu" in tool_results[0]["content"]
    assert "Used 14 registers" in tool_results[1]["content"]

    # Rejections are explained to the agent.
    proposal_result = json.loads(calls[2]["messages"][-1]["content"][0]["content"])
    rejections = [r["rejected"] for r in proposal_result["results"] if "rejected" in r]
    assert any("not in search space" in r for r in rejections)
    assert any("already evaluated as #0" in r for r in rejections)
    assert proposal_result["remaining_budget"] == 1

    # A text-only turn gets a nudge, not a silent stop.
    assert calls[3]["messages"][-1]["content"] == "1 evaluations remain. Call propose_candidates."


def test_agent_stops_after_repeated_stalls() -> None:
    text = Response([Block(type="text", text="I am done.")])
    client = ScriptedClient([text, text, text])
    outcome = agent.run_agent(_ev(budget=10), client=client, model="claude-test")
    assert outcome.stop_reason == "agent_stalled"
    assert len(outcome.records) == 1  # only the default


def test_only_invalid_proposals_count_as_stalls() -> None:
    bad = propose(1, (gelu(96), "invalid"))
    client = ScriptedClient([bad, bad, bad])
    outcome = agent.run_agent(_ev(budget=10), client=client, model="claude-test")
    assert outcome.stop_reason == "agent_stalled"
    assert outcome.invalid_proposals == 3


def test_bad_ptxas_index_is_a_tool_error() -> None:
    client = ScriptedClient(
        [Response([call("get_ptxas_log", {"index": 7}, 1)]), propose(2, (gelu(128), "try"))]
    )
    agent.run_agent(_ev(budget=2), client=client, model="claude-test")
    result = client.messages.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "no evaluated candidate" in result["content"]


def test_api_failure_raises_triage_error() -> None:
    class Boom:
        class messages:  # noqa: N801 - mimics the SDK attribute
            @staticmethod
            def create(**_: Any) -> Any:
                raise RuntimeError("rate limited")

    with pytest.raises(TriageError, match="rate limited"):
        agent.run_agent(_ev(budget=3), client=Boom(), model="claude-test")  # type: ignore[arg-type]
