"""The safety layer as one system: permissions x plan mode x hooks.

Each mechanism has its own test file. This file pins the interactions, the
places where two policies meet and the order between them decides what the
model is told. These combinations are the part of Nanocoder that must never
be wrong, because it is where readers copy the pattern from.
"""

import sys
from typing import ClassVar

import pytest

from nanocoder import Agent
from nanocoder.demo import ScriptedLLM
from nanocoder.hooks import Hooks
from nanocoder.llm import LLMResponse, ToolCall
from nanocoder.permissions import Permission
from nanocoder.tools.base import Tool
from nanocoder.tools.grep import GrepTool
from nanocoder.tools.write import WriteFileTool


def _write_call(call_id, path):
    return ToolCall(id=call_id, name="write_file",
                    arguments={"file_path": str(path), "content": "x\n"})


def _blocking_hook(reason="hook says no"):
    # Hook commands run through the platform shell (cmd.exe on Windows), so
    # POSIX-only syntax (`>&2`, `;`, `&&`) breaks the veto there. python -c
    # runs identically under both shells, keeping the exit-2 contract real on
    # every lane.
    return {
        "matcher": "*",
        "command": f'{sys.executable} -c "import sys; sys.stderr.write({reason!r}); sys.exit(2)"',
    }


class _BoomTool(Tool):
    """A tool whose execute dies with KeyboardInterrupt, like Ctrl+C mid-run."""

    name = "boom"
    description = "raises KeyboardInterrupt"
    parameters: ClassVar[dict] = {"type": "object", "properties": {}}

    def execute(self):
        raise KeyboardInterrupt


def test_pre_hook_fires_before_plan_mode_and_its_reason_wins(tmp_path):
    """In plan mode a blocking hook still inspects first: the model gets the
    hook's reason, not the plan-mode refusal."""
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[_write_call("c1", tmp_path / "a.txt")]),
            LLMResponse(content="noted"),
        ]),
        tools=[WriteFileTool()],
        permission=Permission(allow_all=True),
        hooks=Hooks(pre=[_blocking_hook()], post=[]),
    )
    agent.plan_mode = True

    agent.chat("go")
    result = agent.messages[2]
    assert result["tool_call_id"] == "c1"
    assert result["content"] == "Blocked by hook: hook says no"
    assert not (tmp_path / "a.txt").exists()


def test_a_blocked_call_never_reaches_the_consent_prompt(tmp_path):
    """Hooks gate ahead of consent: when a hook vetoes, nobody gets asked."""
    asked = []
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[_write_call("c1", tmp_path / "a.txt")]),
            LLMResponse(content="noted"),
        ]),
        tools=[WriteFileTool()],
        permission=Permission(ask=lambda *a: asked.append(a) or "once"),
        hooks=Hooks(pre=[_blocking_hook()], post=[]),
    )

    agent.chat("go")
    assert asked == []
    assert agent.messages[2]["content"].startswith("Blocked by hook")


def test_a_hook_can_veto_a_read_only_tool(tmp_path):
    """Consent auto-passes read-only tools; a pre-hook still outranks that."""
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="c1", name="grep", arguments={"pattern": "x", "path": str(tmp_path)}),
            ]),
            LLMResponse(content="noted"),
        ]),
        tools=[GrepTool()],
        permission=Permission(allow_all=True),
        hooks=Hooks(pre=[_blocking_hook("no grep today")], post=[]),
    )

    agent.chat("go")
    assert agent.messages[2]["content"] == "Blocked by hook: no grep today"


def test_post_hook_observes_only_calls_that_actually_executed(tmp_path):
    """A blocked call must not show up in post-tool logs as if it ran."""
    log = tmp_path / "post.log"
    post = [
        {
            "matcher": "*",
            "command": (
                f'{sys.executable} -c "import sys; '
                f"open(sys.argv[1], 'a', encoding='utf-8').write(sys.stdin.read())"
                f'" "{log}"'
            ),
        }
    ]
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[
                _write_call("c1", tmp_path / "a.txt"),
                _write_call("c2", tmp_path / "b.txt"),
            ]),
            LLMResponse(content="noted"),
        ]),
        tools=[WriteFileTool()],
        permission=Permission(allow_all=True),
        hooks=Hooks(pre=[{"matcher": "write_file", "command": (
            f'{sys.executable} -c "import sys; '
            "sys.exit(2 if 'a.txt' in sys.stdin.read() else 0)\""
        )}], post=post),
    )

    agent.chat("go")
    tool_results = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("Blocked by hook" in m["content"] for m in tool_results)
    observed = log.read_text(encoding="utf-8")
    # c1 was blocked, so only c2's execution may be observed by the post hook
    assert observed.count("tool_response") == 1
    assert "b.txt" in observed


def test_plan_mode_parallel_batch_refuses_writes_but_runs_reads(tmp_path):
    """Plan mode inside a parallel batch: the read executes, the write is
    refused, and both answers land on their own call ids."""
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[
                _write_call("c1", tmp_path / "a.txt"),
                ToolCall(id="c2", name="grep",
                         arguments={"pattern": "hello", "path": str(tmp_path / "note.txt")}),
            ]),
            LLMResponse(content="plan ready"),
        ]),
        tools=[WriteFileTool(), GrepTool()],
        permission=Permission(allow_all=True),
    )
    agent.plan_mode = True

    agent.chat("go")
    by_id = {m["tool_call_id"]: m["content"] for m in agent.messages if m.get("role") == "tool"}
    assert "Plan mode is on" in by_id["c1"]
    assert "hello" in by_id["c2"]
    assert not (tmp_path / "a.txt").exists()


def test_ctrl_c_backfills_every_unanswered_tool_call(tmp_path):
    """Interrupting mid-execution leaves no dangling tool_calls: the history
    stays valid for the next request."""
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall(id="c1", name="boom", arguments={})]),
        ]),
        tools=[_BoomTool()],
    )

    with pytest.raises(KeyboardInterrupt):
        agent.chat("go")
    tool_results = [m for m in agent.messages if m.get("role") == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_results] == [("c1", "[interrupted]")]


def test_ctrl_c_in_a_parallel_batch_backfills_the_whole_batch(tmp_path):
    """One interrupted call must not strand its siblings either."""
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    agent = Agent(
        llm=ScriptedLLM([
            LLMResponse(tool_calls=[
                ToolCall(id="c1", name="grep",
                         arguments={"pattern": "hello", "path": str(tmp_path / "note.txt")}),
                ToolCall(id="c2", name="boom", arguments={}),
            ]),
        ]),
        tools=[GrepTool(), _BoomTool()],
    )

    with pytest.raises(KeyboardInterrupt):
        agent.chat("go")
    answered = {m.get("tool_call_id") for m in agent.messages if m.get("role") == "tool"}
    assert answered == {"c1", "c2"}
    assert any(
        m.get("tool_call_id") == "c2" and m.get("content") == "[interrupted]"
        for m in agent.messages
    )
