"""Agent-guided search: an LLM proposes the next candidates from compiler and runtime feedback.

Same client pattern as perflens.triage (Anthropic Messages API, or Groq via
groq_adapter for non-Claude model names). The agent's only action is
`propose_candidates`; its proposals are validated against the kernel's search
space before anything is built, and duplicates or out-of-space proposals are
rejected without spending budget (they are counted and reported). Its other
two tools are read-only: the kernel source and the raw ptxas log of any
candidate already evaluated. It has no shell and cannot write files.
"""

from __future__ import annotations

import json
from typing import Any, cast

from perflens.triage import DEFAULT_MODEL, AnthropicLike, TriageError
from perflens.tune.space import SpaceError
from perflens.tune.strategies import Evaluator, Outcome

MAX_PER_CALL = 4
MAX_STALLS = 3
PTXAS_LOG_CHARS = 4000

SYSTEM_PROMPT = """You are a GPU performance engineer tuning one CUDA kernel's compile \
flags and launch parameters. Every evaluation compiles the kernel with your chosen \
settings (a few seconds of nvcc), checks it against a CPU reference, and times it \
on the GPU. Evaluations are limited by a fixed budget, so choose them carefully.

Goal: find the lowest median runtime (median_us) among configurations that build \
and pass the correctness check, within the budget. A configuration that fails \
correctness is never acceptable, however fast it is.

How to work:
  - The default (shipped) configuration has already been evaluated; its result \
is in the first message.
  - Read the kernel source (get_kernel_source) before your first proposal, so \
you know what each dimension actually changes.
  - Use the compiler feedback, not only runtimes: registers per thread, spill \
bytes, and ptxas warnings (get_ptxas_log for the full log). If the feedback \
shows a dimension cannot matter, stop spending budget on it.
  - Treat differences within about 3x the reported MAD as noise.
  - Propose 1-4 new configurations per propose_candidates call. Each needs a \
one-sentence rationale citing specific evidence (an earlier result by its \
index, a register count, a warning, a line of the source). Never repeat a \
configuration that was already evaluated.
  - Keep calling propose_candidates until the remaining budget is 0."""


def _tool_defs(ev: Evaluator) -> list[dict[str, Any]]:
    return [
        {
            "name": "get_kernel_source",
            "description": "Read-only: the kernel's CUDA source and the header defining its tunable macros.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_ptxas_log",
            "description": "Read-only: the raw nvcc/ptxas -v output for an already-evaluated candidate.",
            "input_schema": {
                "type": "object",
                "properties": {"index": {"type": "integer", "minimum": 0}},
                "required": ["index"],
                "additionalProperties": False,
            },
        },
        {
            "name": "propose_candidates",
            "description": (
                "Evaluate 1-4 new configurations (each is compiled, correctness-checked, and timed). "
                "Returns their results and the remaining budget."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_PER_CALL,
                        "items": {
                            "type": "object",
                            "properties": {
                                "config": ev.space.config_schema(),
                                "rationale": {"type": "string"},
                            },
                            "required": ["config", "rationale"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["candidates"],
                "additionalProperties": False,
            },
        },
    ]


def _initial_prompt(ev: Evaluator) -> str:
    default = ev.records[0]
    return "\n".join(
        [
            f"Kernel: {ev.space.kernel}   batch: {ev.batch}",
            f"Search space: {json.dumps(ev.space.describe())}",
            f"Budget: {ev.budget} evaluations in total, {ev.remaining} remaining "
            "(the default config used the first one).",
            "",
            "Result #0 (default config):",
            json.dumps(default.result.summary()),
            "",
            "Start by reading the source, then propose candidates.",
        ]
    )


def _handle_propose(ev: Evaluator, tool_input: dict[str, Any]) -> tuple[str, int, int]:
    """Validate and evaluate proposals. Returns (tool result text, evaluated, invalid)."""
    evaluated, invalid = 0, 0
    lines: list[dict[str, Any]] = []
    candidates = tool_input.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return "error: `candidates` must be a non-empty list", 0, 1
    for raw in candidates[:MAX_PER_CALL]:
        raw_cfg = raw.get("config") if isinstance(raw, dict) else None
        rationale = str(raw.get("rationale", "")).strip() if isinstance(raw, dict) else ""
        try:
            cfg = ev.space.validate(raw_cfg)  # type: ignore[arg-type]
        except (SpaceError, TypeError, ValueError) as exc:
            invalid += 1
            lines.append({"proposal": raw_cfg, "rejected": f"not in search space: {exc}"})
            continue
        existing = ev.index_of(cfg)
        if existing is not None:
            invalid += 1
            lines.append({"proposal": raw_cfg, "rejected": f"already evaluated as #{existing}"})
            continue
        if ev.remaining <= 0:
            lines.append({"proposal": raw_cfg, "rejected": "budget exhausted"})
            continue
        record = ev.evaluate(cfg, rationale or None)
        if record is not None:
            evaluated += 1
            lines.append({"index": record.idx, **record.result.summary()})
    invalid += max(0, len(candidates) - MAX_PER_CALL)
    best = ev.best()
    payload = {
        "results": lines,
        "remaining_budget": ev.remaining,
        "best_so_far": None if best is None else {"index": best.idx, **best.result.summary()},
    }
    return json.dumps(payload), evaluated, invalid


def run_agent(
    ev: Evaluator,
    client: AnthropicLike | None = None,
    model: str = DEFAULT_MODEL,
    max_turns: int | None = None,
) -> Outcome:
    if client is None:
        try:
            if model.startswith("claude-"):
                import anthropic

                client = cast("AnthropicLike", anthropic.Anthropic())
            else:
                from perflens.groq_adapter import GroqAdapterClient

                client = cast("AnthropicLike", GroqAdapterClient())
        except Exception as exc:  # noqa: BLE001 - normalize any SDK/env error
            raise TriageError(f"could not construct model client: {exc}") from exc

    ev.evaluate(ev.space.default, "default (shipped) config")
    tools = _tool_defs(ev)
    messages: list[dict[str, Any]] = [{"role": "user", "content": _initial_prompt(ev)}]
    invalid_total = 0
    stalls = 0
    turns = max_turns if max_turns is not None else 2 * ev.budget + 10

    for _ in range(turns):
        if ev.remaining <= 0:
            return Outcome(ev.records, "budget", invalid_total)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                tools=tools,
                thinking={"type": "adaptive"},
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - normalize any SDK error (auth, rate limit, ...)
            raise TriageError(f"model API call failed: {exc}") from exc

        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            stalls += 1
            if stalls >= MAX_STALLS:
                return Outcome(ev.records, "agent_stalled", invalid_total)
            messages.append(
                {
                    "role": "user",
                    "content": f"{ev.remaining} evaluations remain. Call propose_candidates.",
                }
            )
            continue

        results = []
        proposed_any = False
        evaluated_this_turn = 0
        for call in tool_uses:
            is_error = False
            if call.name == "get_kernel_source":
                content = ev.runner.kernel_source(ev.space)
            elif call.name == "get_ptxas_log":
                idx = call.input.get("index") if isinstance(call.input, dict) else None
                if isinstance(idx, int) and 0 <= idx < len(ev.records):
                    log = ev.runner.ptxas_log(ev.space, ev.records[idx].result.config)
                    content = log[-PTXAS_LOG_CHARS:]
                else:
                    content, is_error = f"no evaluated candidate with index {idx!r}", True
            elif call.name == "propose_candidates":
                proposed_any = True
                content, n_eval, n_invalid = _handle_propose(ev, call.input)
                evaluated_this_turn += n_eval
                invalid_total += n_invalid
            else:
                content, is_error = f"unknown tool: {call.name}", True
            results.append(
                {"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": is_error}
            )
        messages.append({"role": "user", "content": results})

        if proposed_any and evaluated_this_turn == 0:
            stalls += 1
            if stalls >= MAX_STALLS:
                return Outcome(ev.records, "agent_stalled", invalid_total)
        elif evaluated_this_turn > 0:
            stalls = 0

    return Outcome(ev.records, "budget" if ev.remaining <= 0 else "max_turns", invalid_total)
