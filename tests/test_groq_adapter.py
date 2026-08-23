"""Groq (OpenAI-compatible) adapter: message/tool translation to and from the
Anthropic-shaped interface triage.py's agent loop is written against.

These exercise _GroqMessages/_to_groq_messages/_to_groq_tools directly against
a fake OpenAI-shaped client -- no real `groq` package or network needed (only
GroqAdapterClient.__init__ imports the `groq` SDK, and nothing here touches
that constructor).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from perflens.groq_adapter import _GroqMessages, _to_groq_messages, _to_groq_tools


@dataclass
class FakeToolCall:
    id: str
    name: str
    arguments: str

    @property
    def function(self) -> SimpleNamespace:
        return SimpleNamespace(name=self.name, arguments=self.arguments)


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeCompletion:
    message: FakeMessage

    @property
    def choices(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(message=self.message)]


class FakeCompletions:
    def __init__(self, response: FakeCompletion) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeCompletion:
        self.calls.append(kwargs)
        return self._response


@dataclass
class FakeGroqClient:
    completions: FakeCompletions

    @property
    def chat(self) -> SimpleNamespace:
        return SimpleNamespace(completions=self.completions)


def _tool_def(name: str = "get_findings", strict: bool = False) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": name,
        "description": "does a thing",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    }
    if strict:
        d["strict"] = True
    return d


def test_to_groq_messages_prepends_system_and_passes_user_string() -> None:
    out = _to_groq_messages("be helpful", [{"role": "user", "content": "hi"}])
    assert out[0] == {"role": "system", "content": "be helpful"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_to_groq_messages_translates_assistant_tool_use_blocks() -> None:
    blocks = [
        SimpleNamespace(type="text", text="thinking..."),
        SimpleNamespace(type="tool_use", id="call_1", name="get_findings", input={"a": 1}),
    ]
    out = _to_groq_messages("sys", [{"role": "assistant", "content": blocks}])
    assistant_msg = out[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "thinking..."
    assert assistant_msg["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_findings", "arguments": json.dumps({"a": 1})},
        }
    ]


def test_to_groq_messages_translates_tool_result_blocks_to_tool_messages() -> None:
    tool_results = [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "ok", "is_error": False},
        {"type": "tool_result", "tool_use_id": "call_2", "content": "boom", "is_error": True},
    ]
    out = _to_groq_messages("sys", [{"role": "user", "content": tool_results}])
    assert out[1] == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}
    assert out[2] == {"role": "tool", "tool_call_id": "call_2", "content": "ERROR: boom"}


def test_to_groq_tools_wraps_in_function_type_and_maps_input_schema() -> None:
    out = _to_groq_tools([_tool_def("get_findings"), _tool_def("submit_triage_report", strict=True)])
    assert out[0] == {
        "type": "function",
        "function": {
            "name": "get_findings",
            "description": "does a thing",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
    assert out[1]["function"]["strict"] is True


def test_groq_messages_create_returns_text_block_when_no_tool_calls() -> None:
    completions = FakeCompletions(FakeCompletion(FakeMessage(content="hello")))
    gm = _GroqMessages(FakeGroqClient(completions))
    response = gm.create(
        model="llama-3.3-70b-versatile",
        max_tokens=100,
        system="sys",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
    )
    assert len(response.content) == 1
    assert response.content[0].type == "text"
    assert response.content[0].text == "hello"


def test_groq_messages_create_parses_tool_call_arguments_into_dict() -> None:
    call = FakeToolCall(id="call_1", name="get_findings", arguments=json.dumps({"x": 1}))
    completions = FakeCompletions(FakeCompletion(FakeMessage(content=None, tool_calls=[call])))
    gm = _GroqMessages(FakeGroqClient(completions))
    response = gm.create(
        model="llama-3.3-70b-versatile",
        max_tokens=100,
        system="sys",
        tools=[_tool_def()],
        messages=[{"role": "user", "content": "hi"}],
    )
    assert len(response.content) == 1
    block = response.content[0]
    assert block.type == "tool_use"
    assert block.id == "call_1"
    assert block.name == "get_findings"
    assert block.input == {"x": 1}


def test_groq_messages_create_raises_on_malformed_tool_arguments() -> None:
    call = FakeToolCall(id="call_1", name="get_findings", arguments="{not json")
    completions = FakeCompletions(FakeCompletion(FakeMessage(content=None, tool_calls=[call])))
    gm = _GroqMessages(FakeGroqClient(completions))
    with pytest.raises(ValueError, match="non-JSON tool arguments"):
        gm.create(
            model="llama-3.3-70b-versatile",
            max_tokens=100,
            system="sys",
            tools=[_tool_def()],
            messages=[{"role": "user", "content": "hi"}],
        )


def test_groq_messages_create_caps_max_tokens_for_free_tier_tpm_limit() -> None:
    # Groq's free-tier TPM limit counts prompt_tokens + max_tokens (the
    # requested ceiling), so an uncapped 8000 (sized for Anthropic) would
    # alone exceed an 8000 TPM budget before any prompt tokens are counted.
    completions = FakeCompletions(FakeCompletion(FakeMessage(content="ok")))
    gm = _GroqMessages(FakeGroqClient(completions))
    gm.create(
        model="openai/gpt-oss-120b",
        max_tokens=8000,
        system="sys",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
    )
    assert completions.calls[0]["max_tokens"] < 8000


def test_groq_messages_create_drops_thinking_kwarg_without_error() -> None:
    completions = FakeCompletions(FakeCompletion(FakeMessage(content="ok")))
    gm = _GroqMessages(FakeGroqClient(completions))
    gm.create(
        model="llama-3.3-70b-versatile",
        max_tokens=100,
        system="sys",
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
        thinking={"type": "adaptive"},
    )
    assert "thinking" not in completions.calls[0]
