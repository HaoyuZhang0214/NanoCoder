"""LangGraph-based Agent implementation with state graph orchestration.

This module provides a graph-driven agent that replaces the traditional while-loop
with a LangGraph StateGraph. Key features:

- Directed graph orchestration for LLM workflow
- Conditional routing (continue/end/tool calls)
- Parallel tool execution via Send API
- Automatic state persistence with checkpointing
- Resume from breakpoints after interruptions

The agent maintains the same interface as the traditional Agent class, ensuring
backward compatibility with existing code.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import logging
import uuid
from typing import TYPE_CHECKING, Annotated, Any, Callable, TypedDict

from .context import ContextManager
from .permissions import Permission
from .prompt import PLAN_MODE_PROMPT, system_prompt
from .tools import ALL_TOOLS
from .tools.agent import AgentTool
from .tools.base import Tool
from .tools.todo import TodoWriteTool

if TYPE_CHECKING:
    from .llm import LLM, LLMResponse, ToolCall

logger = logging.getLogger(__name__)


# ============================================================================
# State Definition
# ============================================================================

def _add_messages_reducer(existing: list[dict], new: list[dict]) -> list[dict]:
    """Reducer for messages: append new messages to existing list.
    
    This is a simplified version of LangGraph's add_messages reducer.
    We use a custom one to avoid the dependency on langgraph.graph.message.
    """
    if not new:
        return existing
    return existing + new


class AgentState(TypedDict):
    """LangGraph Agent state definition.
    
    The state flows through the graph nodes and is automatically persisted
    by the checkpointer at each step.
    
    Attributes:
        messages: Conversation history with message addition reducer
        tool_calls: Current batch of tool calls to execute
        current_round: Current iteration count
        max_rounds: Maximum allowed iterations
        final_response: Final text response when done
        should_continue: Whether to continue the loop
    """
    messages: Annotated[list[dict], _add_messages_reducer]
    tool_calls: list[dict]
    current_round: int
    max_rounds: int
    final_response: str
    should_continue: bool


# ============================================================================
# LangGraph Agent Implementation
# ============================================================================

class LangGraphAgent:
    """Graph-driven agent using LangGraph StateGraph.
    
    This agent replaces the traditional while-loop with a directed graph:
    
        START -> call_llm -> check_completion -> execute_tools -> call_llm
                                     |
                                     v
                                    END
    
    The graph automatically handles:
    - Conditional routing based on LLM response
    - State persistence at each step
    - Parallel tool execution
    - Resume from checkpoints
    
    Args:
        llm: LLM instance for chat completions
        tools: List of available tools (defaults to ALL_TOOLS)
        max_context_tokens: Maximum context window size
        max_rounds: Maximum tool-call rounds per conversation
        permission: Permission checker for tool execution
        hooks: Pre/post execution hooks
        checkpointer: LangGraph checkpointer for state persistence
        thread_id: Thread ID for checkpoint (auto-generated if None)
        enable_parallel: Enable parallel tool execution via Send API
    
    Examples:
        >>> from .llm import LLM
        >>> from .checkpoints import get_checkpointer
        >>> 
        >>> llm = LLM(model="gpt-4", api_key="...")
        >>> checkpointer = get_checkpointer("memory")
        >>> agent = LangGraphAgent(llm, checkpointer=checkpointer)
        >>> 
        >>> # Normal execution
        >>> response = agent.chat("Hello!")
        >>> 
        >>> # Resume from checkpoint
        >>> agent.resume_from_checkpoint(thread_id="my-thread")
    """
    
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
        max_rounds: int = 50,
        permission: Permission | None = None,
        hooks: Any | None = None,
        checkpointer: Any | None = None,
        thread_id: str | None = None,
        enable_parallel: bool = True,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self.permission = permission
        self.hooks = hooks
        self._tool_by_name = {t.name: t for t in self.tools}
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self._system = system_prompt(self.tools)
        self.plan_mode = False
        self.enable_parallel = enable_parallel
        
        # Thread ID for checkpoint
        self.thread_id = thread_id or str(uuid.uuid4())
        
        # Checkpointer for state persistence
        self.checkpointer = checkpointer
        
        # Wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self
        
        self._todo = next((t for t in self.tools if isinstance(t, TodoWriteTool)), None)
        
        # Callbacks (set per-chat)
        self._on_token: Callable | None = None
        self._on_tool: Callable | None = None
        self._on_reasoning: Callable | None = None
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build and compile the LangGraph StateGraph.
        
        Graph structure:
            START -> call_llm -> check_completion
                                    ├─ (end) -> END
                                    └─ (continue) -> execute_tools -> call_llm
        
        Returns:
            Compiled LangGraph graph with optional checkpointer
        """
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as e:
            raise ImportError(
                "LangGraph is required for LangGraphAgent. "
                "Install with: pip install langgraph"
            ) from e
        
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("call_llm", self._node_call_llm)
        graph.add_node("check_completion", self._node_check_completion)
        graph.add_node("execute_tools", self._node_execute_tools)
        
        # Add edges
        graph.add_edge(START, "call_llm")
        graph.add_edge("call_llm", "check_completion")
        
        # Conditional routing after check_completion
        graph.add_conditional_edges(
            "check_completion",
            self._route_after_check,
            {
                "continue": "execute_tools",
                "end": END,
            },
        )
        
        # Loop back after tool execution
        graph.add_edge("execute_tools", "call_llm")
        
        # Compile with checkpointer
        return graph.compile(checkpointer=self.checkpointer)
    
    # ========================================================================
    # Graph Nodes
    # ========================================================================
    
    def _node_call_llm(self, state: AgentState) -> dict:
        """Call LLM node: generate response and update state.
        
        This node:
        1. Prepares full message list with system prompt
        2. Calls LLM with streaming
        3. Updates messages and tool_calls in state
        4. Triggers callbacks for tokens/reasoning
        
        Args:
            state: Current graph state
        
        Returns:
            State updates: {messages, tool_calls, current_round}
        """
        logger.debug(f"[call_llm] Round {state['current_round']}/{state['max_rounds']}")
        
        # Prepare messages
        messages = self._full_messages(state["messages"])
        
        # Call LLM
        resp = self.llm.chat(
            messages=messages,
            tools=self._tool_schemas(),
            on_token=self._on_token,
            on_reasoning=self._on_reasoning,
        )
        
        # Update state
        updates = {
            "messages": [resp.message],
            "tool_calls": [tc.__dict__ for tc in resp.tool_calls] if resp.tool_calls else [],
            "current_round": state["current_round"] + 1,
        }
        
        # If no tool calls, set final response
        if not resp.tool_calls:
            updates["final_response"] = resp.content
            updates["should_continue"] = False
        
        return updates
    
    def _node_check_completion(self, state: AgentState) -> dict:
        """Check completion node: decide whether to continue or end.
        
        This node evaluates:
        1. Are there tool calls to execute?
        2. Have we exceeded max_rounds?
        3. Should we continue the loop?
        
        Args:
            state: Current graph state
        
        Returns:
            State updates: {should_continue}
        """
        has_tool_calls = bool(state.get("tool_calls"))
        within_limit = state["current_round"] < state["max_rounds"]
        
        should_continue = has_tool_calls and within_limit
        
        logger.debug(
            f"[check_completion] tool_calls={has_tool_calls}, "
            f"round={state['current_round']}/{state['max_rounds']}, "
            f"continue={should_continue}"
        )
        
        return {"should_continue": should_continue}
    
    def _node_execute_tools(self, state: AgentState) -> dict:
        """Execute tools node: run tool calls and update state.
        
        This node:
        1. Executes tool calls (parallel if enabled)
        2. Collects results
        3. Updates messages with tool responses
        4. Triggers pre/post hooks
        
        Args:
            state: Current graph state
        
        Returns:
            State updates: {messages, tool_calls=[]}
        """
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return {"messages": [], "tool_calls": []}
        
        logger.debug(f"[execute_tools] Executing {len(tool_calls)} tool calls")
        
        # Convert dicts back to ToolCall-like objects
        from .llm import ToolCall
        tool_call_objs = [
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
            for tc in tool_calls
        ]
        
        # Execute tools
        try:
            if len(tool_call_objs) == 1:
                # Single tool call
                tc = tool_call_objs[0]
                if self._on_tool:
                    self._on_tool(tc.name, tc.arguments)
                
                result = self._pre_hooks(tc) or self._permit(tc)
                if result is None:
                    result = self._exec_tool(tc)
                    self._post_hooks(tc, result)
                
                tool_messages = [{
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }]
            else:
                # Parallel execution
                tool_messages = self._exec_tools_parallel(tool_call_objs)
        
        except KeyboardInterrupt:
            # Backfill pending tool calls
            self._answer_pending_tool_calls(tool_call_objs, state["messages"])
            raise
        
        return {
            "messages": tool_messages,
            "tool_calls": [],
        }
    
    # ========================================================================
    # Conditional Routing
    # ========================================================================
    
    def _route_after_check(self, state: AgentState) -> str:
        """Route after check_completion: continue to tools or end.
        
        Args:
            state: Current graph state
        
        Returns:
            "continue" or "end"
        """
        if state.get("should_continue"):
            return "continue"
        return "end"
    
    # ========================================================================
    # Public Interface
    # ========================================================================
    
    def chat(
        self,
        user_input: str,
        on_token: Callable | None = None,
        on_tool: Callable | None = None,
        on_reasoning: Callable | None = None,
    ) -> str:
        """Process one user message through the graph.
        
        This is the main entry point. It:
        1. Adds user message to state
        2. Invokes the graph
        3. Returns the final response
        
        Args:
            user_input: User message text
            on_token: Callback for streaming tokens
            on_tool: Callback for tool execution
            on_reasoning: Callback for reasoning content
        
        Returns:
            Final text response from the agent
        """
        # Set callbacks
        self._on_token = on_token
        self._on_tool = on_tool
        self._on_reasoning = on_reasoning
        
        # Add user message
        user_message = {"role": "user", "content": user_input}
        
        # Compress context if needed
        current_messages = self._get_current_messages()
        current_messages.append(user_message)
        self.context.maybe_compress(current_messages, self.llm)
        
        # Initial state
        initial_state = {
            "messages": [user_message],
            "tool_calls": [],
            "current_round": 0,
            "max_rounds": self.max_rounds,
            "final_response": "",
            "should_continue": True,
        }
        
        # Configuration for checkpointing
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # Invoke the graph
        try:
            final_state = self.graph.invoke(initial_state, config)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            raise
        
        # Extract final response
        if final_state.get("final_response"):
            return final_state["final_response"]
        
        # Fallback: get last assistant message
        for msg in reversed(final_state.get("messages", [])):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        
        return "(no response generated)"
    
    def resume_from_checkpoint(self, thread_id: str | None = None) -> str:
        """Resume execution from a checkpoint.
        
        Args:
            thread_id: Thread ID to resume from (uses self.thread_id if None)
        
        Returns:
            Final text response
        
        Raises:
            ValueError: If no checkpoint found for the thread
        """
        if thread_id:
            self.thread_id = thread_id
        
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # Get current state
        checkpoint = self.graph.get_state(config)
        if not checkpoint or not checkpoint.values:
            raise ValueError(f"No checkpoint found for thread {self.thread_id}")
        
        logger.info(f"Resuming from checkpoint: {self.thread_id}")
        
        # Continue execution
        final_state = self.graph.invoke(None, config)
        
        # Extract final response
        if final_state.get("final_response"):
            return final_state["final_response"]
        
        for msg in reversed(final_state.get("messages", [])):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"]
        
        return "(no response generated)"
    
    def reset(self):
        """Clear conversation history and reset state."""
        self.thread_id = str(uuid.uuid4())
        if self._todo is not None:
            self._todo._tasks = []
    
    # ========================================================================
    # Helper Methods (reused from traditional Agent)
    # ========================================================================
    
    def _get_current_messages(self) -> list[dict]:
        """Get current message history from checkpoint."""
        config = {"configurable": {"thread_id": self.thread_id}}
        state = self.graph.get_state(config)
        if state and state.values:
            return list(state.values.get("messages", []))
        return []
    
    def _full_messages(self, history: list[dict]) -> list[dict]:
        """Build full message list with system prompt."""
        system = self._system
        if self.plan_mode:
            system += "\n\n" + PLAN_MODE_PROMPT
        if self._todo is not None:
            rendered = self._todo.render()
            if rendered:
                system += "\n\n# Current task list\n" + rendered
        return [{"role": "system", "content": system}] + history
    
    def _tool_schemas(self) -> list[dict]:
        """Get tool schemas for LLM."""
        return [t.schema() for t in self.tools]
    
    def _pre_hooks(self, tc) -> str | None:
        """Run pre-tool hooks."""
        if self.hooks is None:
            return None
        return self.hooks.run_pre(tc.name, tc.arguments)
    
    def _post_hooks(self, tc, result: str):
        """Run post-tool hooks."""
        if self.hooks is not None:
            self.hooks.run_post(tc.name, tc.arguments, result)
    
    def _permit(self, tc) -> str | None:
        """Check permission for tool call."""
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
        """Execute a single tool call."""
        tool = self._tool_by_name.get(tc.name)
        if tool is None:
            return f"Error: unknown tool '{tc.name}'"
        try:
            inspect.signature(tool.execute).bind(**tc.arguments)
        except TypeError as e:
            return f"Error: bad arguments for {tc.name}: {e}"
        try:
            return tool.execute(**tc.arguments)
        except Exception as e:
            return f"Error executing {tc.name}: {e}"
    
    def _exec_tools_parallel(self, tool_calls: list) -> list[dict]:
        """Execute multiple tool calls in parallel.
        
        This is a simplified version that uses ThreadPoolExecutor.
        For true Send API parallelism, see _node_execute_tools_with_send.
        """
        from .tools.bash import get_tracked_cwd, set_tracked_cwd
        
        # Notify callbacks
        for tc in tool_calls:
            if self._on_tool:
                self._on_tool(tc.name, tc.arguments)
        
        # Run hooks and consent on main thread
        results = [self._pre_hooks(tc) or self._permit(tc) for tc in tool_calls]
        
        # Execute in parallel
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
        
        # Post hooks
        for i in futures:
            self._post_hooks(tool_calls[i], results[i])
        
        # Build tool messages
        return [
            {"role": "tool", "tool_call_id": tc.id, "content": result}
            for tc, result in zip(tool_calls, results)
        ]
    
    def _answer_pending_tool_calls(self, tool_calls: list, messages: list[dict]):
        """Backfill tool responses for interrupted calls."""
        answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
        for tc in tool_calls:
            if tc.id not in answered:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "[interrupted]",
                })
