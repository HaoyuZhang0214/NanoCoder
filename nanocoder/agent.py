"""Core agent loop.

This is the heart of NanoCoder.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

The loop is a LangGraph StateGraph (call_llm, then execute_tools, then back)
so routing is explicit. Conversation history still lives on Agent.messages;
the graph state only carries the round counter and whether this turn is done.
"""

import concurrent.futures
import inspect
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .context import ContextManager
from .llm import LLM
from .permissions import Permission
from .prompt import PLAN_MODE_PROMPT, system_prompt
from .tools import ALL_TOOLS
from .tools.agent import AgentTool
from .tools.base import Tool
from .tools.todo import TodoWriteTool


class _LoopState(TypedDict):
    """Control state for one chat(). Messages stay on the Agent instance."""

    rounds: int
    has_tools: bool
    final_response: str


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        permission=None,
        hooks=None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self.permission = permission
        self.hooks = hooks
        self._tool_by_name = {t.name: t for t in self.tools}
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)
        self.plan_mode = False  # toggled by /plan; while on, mutating tools are refused

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

        self._todo = next((t for t in self.tools if isinstance(t, TodoWriteTool)), None)
        self._on_token = None
        self._on_tool = None
        self._on_reasoning = None
        self._pending_tool_calls = []
        self.graph = self._build_graph()

    def _build_graph(self):
        """call_llm -> (tools? execute_tools -> call_llm : END).

        The round limit is checked at the start of call_llm, matching the old
        for-loop: the last allowed round still runs its tools, then the next
        entry returns without another model call.
        """
        graph = StateGraph(_LoopState)
        graph.add_node("call_llm", self._node_call_llm)
        graph.add_node("execute_tools", self._node_execute_tools)
        graph.add_edge(START, "call_llm")
        graph.add_conditional_edges(
            "call_llm",
            self._route_after_llm,
            {"execute_tools": "execute_tools", "end": END},
        )
        graph.add_edge("execute_tools", "call_llm")
        return graph.compile()

    def _full_messages(self) -> list[dict]:
        system = self._system
        # re-injected every round, like the task list below, so a toggle made
        # between turns takes effect on the very next request
        if self.plan_mode:
            system += "\n\n" + PLAN_MODE_PROMPT
        # the task list is re-injected every round, so the model always sees the
        # current state rather than a stale copy buried in old tool results
        if self._todo is not None:
            rendered = self._todo.render()
            if rendered:
                system += "\n\n# Current task list\n" + rendered
        return [{"role": "system", "content": system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def chat(self, user_input: str, on_token=None, on_tool=None, on_reasoning=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self._on_token = on_token
        self._on_tool = on_tool
        self._on_reasoning = on_reasoning
        self.messages.append({"role": "user", "content": user_input})
        self.context.maybe_compress(self.messages, self.llm)

        # each LLM call and each tool batch is a step; the default limit of 25
        # would cut off a long tool loop before max_rounds
        final_state = self.graph.invoke(
            {"rounds": 0, "has_tools": False, "final_response": ""},
            {"recursion_limit": self.max_rounds * 2 + 5},
        )
        return final_state["final_response"]

    def _node_call_llm(self, state: _LoopState) -> dict:
        """One model turn. Stops before the call once max_rounds is spent."""
        rounds = state["rounds"]
        if rounds >= self.max_rounds:
            return {
                "rounds": rounds,
                "has_tools": False,
                "final_response": "(reached maximum tool-call rounds)",
            }

        resp = self.llm.chat(
            messages=self._full_messages(),
            tools=self._tool_schemas(),
            on_token=self._on_token,
            on_reasoning=self._on_reasoning,
        )
        rounds += 1

        if not resp.tool_calls:
            self.messages.append(resp.message)
            self._pending_tool_calls = []
            return {"rounds": rounds, "has_tools": False, "final_response": resp.content}

        self.messages.append(resp.message)
        self._pending_tool_calls = list(resp.tool_calls)
        return {"rounds": rounds, "has_tools": True, "final_response": ""}

    def _route_after_llm(self, state: _LoopState) -> str:
        return "execute_tools" if state["has_tools"] else "end"

    def _node_execute_tools(self, _state: _LoopState) -> dict:
        """Run the pending tool batch, then compress, then loop back to the model."""
        tool_calls = self._pending_tool_calls
        try:
            if len(tool_calls) == 1:
                tc = tool_calls[0]
                if self._on_tool:
                    self._on_tool(tc.name, tc.arguments)
                result = self._pre_hooks(tc) or self._permit(tc)
                if result is None:
                    result = self._exec_tool(tc)
                    self._post_hooks(tc, result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            else:
                # parallel execution for multiple tool calls
                results = self._exec_tools_parallel(tool_calls, self._on_tool)
                for tc, result in zip(tool_calls, results):
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
        except KeyboardInterrupt:
            # Ctrl+C mid-execution would leave the assistant tool_calls
            # message without replies, poisoning the next request; backfill
            self._answer_pending_tool_calls(tool_calls)
            raise

        self.context.maybe_compress(self.messages, self.llm)
        return {}

    def _pre_hooks(self, tc) -> str | None:
        """PreToolUse hooks, fired before consent. A string return blocks the
        call and becomes the tool result the model sees; None lets it through."""
        if self.hooks is None:
            return None
        return self.hooks.run_pre(tc.name, tc.arguments)

    def _post_hooks(self, tc, result: str):
        """PostToolUse hooks observe a finished call; they can never block."""
        if self.hooks is not None:
            self.hooks.run_post(tc.name, tc.arguments, result)

    def _permit(self, tc) -> str | None:
        """Consent check for one call. None means go ahead; a string is the
        refusal, returned as the tool result instead of executing."""
        # plan mode outranks consent, even --yes: while it's on nothing mutates
        if self.plan_mode and tc.name not in Permission.READ_ONLY:
            return (
                "Plan mode is on, so this call was refused: plan mode is "
                "read-only. Do not retry it. Keep investigating with the "
                "read-only tools, then present the plan and stop. The user can "
                'approve it by typing "approve", or exit plan mode with /plan.'
            )
        if self.permission is None:
            return None
        return self.permission.check(tc.name, tc.arguments)

    def _exec_tool(self, tc) -> str:
        """Execute a single tool call, returning the result string."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        # validate arguments first so a TypeError raised *inside* the tool isn't
        # mislabelled as a bad-arguments error from the caller
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        # a tool that blows up gets reported back as text, never kills the loop
        try:
            return tool.execute(**tc.arguments)
        except Exception as e:  # noqa: BLE001
            return f"Error executing {tc.name}: {e}"

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[str]:
        """Run multiple tool calls concurrently using threads.

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel.
        """
        from .tools.bash import get_tracked_cwd, set_tracked_cwd

        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        # hooks and consent are settled up front on this thread: prompting
        # from pool workers would interleave several prompts on one terminal
        results = [self._pre_hooks(tc) or self._permit(tc) for tc in tool_calls]
        # the tracked cwd is thread-local, so pool workers would otherwise
        # start from the launch directory: hand the session cwd in, and merge
        # any cd back in call order so the batch behaves like a sequence
        session_cwd = get_tracked_cwd()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            def run(i, tc, base_cwd):
                is_bash = tc.name == "bash"
                if is_bash and base_cwd is not None:
                    set_tracked_cwd(base_cwd)
                out = self._exec_tool(tc)
                if not is_bash:
                    return i, out, None
                cwd_after = get_tracked_cwd()
                # only a worker that actually moved the dir reports back; a
                # sibling that merely inherited the base must not revert a cd
                return i, out, cwd_after if cwd_after != base_cwd else None

            futures = {
                i: pool.submit(run, i, tc, session_cwd)
                for i, tc in enumerate(tool_calls)
                if results[i] is None
            }
            for i, future in futures.items():
                i, results[i], worker_cwd = future.result()
                if worker_cwd is not None:
                    session_cwd = worker_cwd
                    set_tracked_cwd(worker_cwd)
        for i in futures:
            self._post_hooks(tool_calls[i], results[i])
        return results

    def _answer_pending_tool_calls(self, tool_calls):
        """Backfill a tool reply for every call that didn't get one.

        OpenAI-compatible APIs reject a request where an assistant message has
        tool_calls without a matching tool reply for each id, so this keeps the
        history valid when execution is interrupted partway through.
        """
        answered = {m.get("tool_call_id") for m in self.messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })

    def reset(self):
        """Clear conversation history."""
        self.messages.clear()
        # a user-visible reset must not leak the dead conversation's checklist
        # into the next system prompt
        todo = self._tool_by_name.get("todo_write")
        if isinstance(todo, TodoWriteTool):
            todo._tasks = []
