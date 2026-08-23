"""Groq (OpenAI-compatible) adapter for triage.py's `AnthropicLike` protocol.

triage.py is written directly against Anthropic's Messages API shape:
`system=` as a separate kwarg, response content as a list of blocks with a
`.type` attribute, tool_use blocks carrying `.name`/`.input`/`.id`, and tool
results appended back into the message history as Anthropic-shaped
`tool_result` blocks. Groq mirrors OpenAI's chat-completions shape instead:
system folded into `messages`, tool calls on `message.tool_calls` with
`.function.name`/`.function.arguments` (a JSON *string*, not a dict), and
tool results as separate `role: "tool"` messages keyed by `tool_call_id`.

This module translates one shape to the other on every call, so triage.py's
agent loop (and its message-history bookkeeping) stays backend-agnostic. It
never sees Groq/OpenAI types directly -- everything it touches is the same
lightweight `SimpleNamespace` shape whether the real backend is Claude or
Groq.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def _to_groq_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groq_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user" and isinstance(content, str):
            groq_messages.append({"role": "user", "content": content})
        elif role == "user" and isinstance(content, list):
            # Anthropic tool_result blocks -> one OpenAI "tool" message each.
            for block in content:
                prefix = "ERROR: " if block.get("is_error") else ""
                groq_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": prefix + str(block["content"]),
                    }
                )
        elif role == "assistant":
            # content is a list of the SimpleNamespace blocks *we* returned
            # from a previous create() call, carried back in by triage.py.
            text_parts = [b.text for b in content if getattr(b, "type", None) == "text" and b.text]
            tool_calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": json.dumps(b.input)},
                }
                for b in content
                if getattr(b, "type", None) == "tool_use"
            ]
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            groq_messages.append(assistant_msg)
        else:
            raise ValueError(f"unrecognized message shape for role={role!r}: {content!r}")
    return groq_messages


def _to_groq_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groq_tools = []
    for t in tools:
        fn: dict[str, Any] = {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        }
        if t.get("strict"):
            fn["strict"] = True
        groq_tools.append({"type": "function", "function": fn})
    return groq_tools


GROQ_FREE_TIER_TPM = 8000
# Groq's free-tier tokens-per-minute limit is charged against
# prompt_tokens + max_tokens (the requested completion *ceiling*, not actual
# usage) -- triage.py asks for max_tokens=8000 unconditionally (sized for
# Anthropic's much higher limits), which alone exhausts the entire per-minute
# budget before a single prompt token is counted. Capped well under the
# ceiling here, not in triage.py, since this is a Groq-specific constraint
# the shared agent loop shouldn't need to know about.
GROQ_MAX_TOKENS_CAP = 4000


class _GroqMessages:
    def __init__(self, client: Any) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        thinking: dict[str, Any] | None = None,  # Anthropic-only; not forwarded to Groq
    ) -> Any:
        completion = self._client.chat.completions.create(
            model=model,
            max_tokens=min(max_tokens, GROQ_MAX_TOKENS_CAP),
            messages=_to_groq_messages(system, messages),
            tools=_to_groq_tools(tools),
        )
        choice = completion.choices[0].message

        blocks: list[Any] = []
        if choice.content:
            blocks.append(SimpleNamespace(type="text", text=choice.content))
        for call in choice.tool_calls or []:
            try:
                parsed_input = json.loads(call.function.arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Groq returned non-JSON tool arguments for "
                    f"{call.function.name}: {call.function.arguments!r}"
                ) from exc
            blocks.append(
                SimpleNamespace(
                    type="tool_use", id=call.id, name=call.function.name, input=parsed_input
                )
            )
        return SimpleNamespace(content=blocks)


class GroqAdapterClient:
    """Implements triage.py's `AnthropicLike` protocol (a `.messages.create(...)`
    surface) on top of Groq's OpenAI-compatible chat-completions API.
    """

    def __init__(self, api_key: str | None = None) -> None:
        import groq  # lazy import, mirrors triage.py's lazy `import anthropic`

        self._client = groq.Groq(api_key=api_key)
        self.messages = _GroqMessages(self._client)
