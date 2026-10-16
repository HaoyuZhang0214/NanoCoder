"""LangGraph agent loop: routing nodes, tool round, and the round cap."""

from typing import ClassVar

from nanocoder.agent import Agent
from nanocoder.demo import ScriptedLLM
from nanocoder.llm import LLMResponse, ToolCall
from nanocoder.tools.base import Tool


class _Echo(Tool):
    name = "echo"
    description = "Echo a message."
    parameters: ClassVar[dict] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def execute(self, text: str) -> str:
        return text


def test_graph_runs_one_tool_round_then_returns_text():
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "ping"})]),
            LLMResponse(content="done"),
        ]),
        tools=[_Echo()],
    )

    assert "call_llm" in agent.graph.nodes
    assert "execute_tools" in agent.graph.nodes
    assert agent.chat("go") == "done"
    assert [m["role"] for m in agent.messages] == ["user", "assistant", "tool", "assistant"]
    assert agent.messages[2]["tool_call_id"] == "c1"
    assert agent.messages[2]["content"] == "ping"
    assert agent.messages[3]["content"] == "done"


def test_max_rounds_still_executes_the_last_tool_batch():
    """The cap is checked before the next model call, so the last round's tools land."""
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "once"})]),
            LLMResponse(content="should not be consumed"),
        ]),
        tools=[_Echo()],
        max_rounds=1,
    )

    assert agent.chat("go") == "(reached maximum tool-call rounds)"
    assert [m["role"] for m in agent.messages] == ["user", "assistant", "tool"]
    assert agent.messages[2]["content"] == "once"
