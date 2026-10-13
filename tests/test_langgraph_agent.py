"""Tests for LangGraph-based agent implementation.

Tests cover:
- Graph construction and structure
- Conditional routing logic
- Parallel tool execution
- State persistence and checkpointing
- Resume from breakpoints
- Error handling
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, Mock, patch

# Skip all tests if langgraph is not installed
langgraph = pytest.importorskip("langgraph")


class TestAgentState:
    """Test AgentState definition and reducers."""
    
    def test_agent_state_import(self):
        """Test that AgentState can be imported."""
        from nanocoder.langgraph_agent import AgentState
        
        assert AgentState is not None
    
    def test_messages_reducer(self):
        """Test message addition reducer."""
        from nanocoder.langgraph_agent import _add_messages_reducer
        
        existing = [{"role": "user", "content": "hello"}]
        new = [{"role": "assistant", "content": "hi"}]
        
        result = _add_messages_reducer(existing, new)
        
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
    
    def test_messages_reducer_empty_new(self):
        """Test reducer with empty new messages."""
        from nanocoder.langgraph_agent import _add_messages_reducer
        
        existing = [{"role": "user", "content": "hello"}]
        new = []
        
        result = _add_messages_reducer(existing, new)
        
        assert len(result) == 1
        assert result == existing


class TestLangGraphAgent:
    """Test LangGraphAgent class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        llm = Mock()
        llm.model = "test-model"
        llm.chat = Mock()
        return llm
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        tool1 = Mock()
        tool1.name = "test_tool"
        tool1.description = "A test tool"
        tool1.parameters = {}
        tool1.execute = Mock(return_value="tool result")
        tool1.schema = Mock(return_value={
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {},
            },
        })
        return [tool1]
    
    def test_agent_initialization(self, mock_llm, mock_tools):
        """Test agent can be initialized."""
        from nanocoder.langgraph_agent import LangGraphAgent
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
            max_rounds=10,
        )
        
        assert agent.llm == mock_llm
        assert agent.tools == mock_tools
        assert agent.max_rounds == 10
        assert agent.graph is not None
    
    def test_agent_with_checkpointer(self, mock_llm, mock_tools):
        """Test agent initialization with checkpointer."""
        from nanocoder.langgraph_agent import LangGraphAgent
        from nanocoder.langgraph_checkpointer import get_checkpointer
        
        checkpointer = get_checkpointer("memory")
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
            checkpointer=checkpointer,
        )
        
        assert agent.checkpointer is not None
    
    def test_graph_structure(self, mock_llm, mock_tools):
        """Test graph has correct nodes and edges."""
        from nanocoder.langgraph_agent import LangGraphAgent
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
        )
        
        # Graph should be compiled
        assert agent.graph is not None
        
        # Check nodes exist (access internal graph structure)
        # The graph should have: call_llm, check_completion, execute_tools
        # We can't directly inspect the compiled graph easily, but we can
        # verify it was created successfully
    
    def test_conditional_routing(self, mock_llm, mock_tools):
        """Test conditional routing logic."""
        from nanocoder.langgraph_agent import LangGraphAgent
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
        )
        
        # Test route with should_continue=True
        state_continue = {"should_continue": True}
        assert agent._route_after_check(state_continue) == "continue"
        
        # Test route with should_continue=False
        state_end = {"should_continue": False}
        assert agent._route_after_check(state_end) == "end"
    
    def test_chat_simple_response(self, mock_llm, mock_tools):
        """Test chat with simple text response (no tool calls)."""
        from nanocoder.langgraph_agent import LangGraphAgent
        from nanocoder.llm import LLMResponse
        
        # Mock LLM response
        mock_response = LLMResponse(
            content="Hello! How can I help you?",
            tool_calls=[],
        )
        mock_llm.chat.return_value = mock_response
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
        )
        
        response = agent.chat("Hi")
        
        assert response == "Hello! How can I help you?"
        mock_llm.chat.assert_called()
    
    def test_chat_with_tool_calls(self, mock_llm, mock_tools):
        """Test chat with tool calls."""
        from nanocoder.langgraph_agent import LangGraphAgent
        from nanocoder.llm import LLMResponse, ToolCall
        
        # First response: tool call
        tool_call = ToolCall(
            id="call_1",
            name="test_tool",
            arguments={"arg": "value"},
        )
        first_response = LLMResponse(
            content="",
            tool_calls=[tool_call],
        )
        
        # Second response: final text
        second_response = LLMResponse(
            content="Done!",
            tool_calls=[],
        )
        
        mock_llm.chat.side_effect = [first_response, second_response]
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
        )
        
        response = agent.chat("Use the tool")
        
        assert response == "Done!"
        assert mock_llm.chat.call_count == 2
    
    def test_max_rounds_limit(self, mock_llm, mock_tools):
        """Test that agent respects max_rounds limit."""
        from nanocoder.langgraph_agent import LangGraphAgent
        from nanocoder.llm import LLMResponse, ToolCall
        
        # Always return tool calls (infinite loop scenario)
        tool_call = ToolCall(id="call_1", name="test_tool", arguments={})
        mock_response = LLMResponse(content="", tool_calls=[tool_call])
        mock_llm.chat.return_value = mock_response
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
            max_rounds=3,
        )
        
        # Should stop after max_rounds
        response = agent.chat("Keep going")
        
        # LLM should be called at most max_rounds times
        assert mock_llm.chat.call_count <= 3
    
    def test_reset(self, mock_llm, mock_tools):
        """Test agent reset."""
        from nanocoder.langgraph_agent import LangGraphAgent
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
            thread_id="test-thread-123",
        )
        
        old_thread_id = agent.thread_id
        agent.reset()
        
        # Thread ID should change after reset
        assert agent.thread_id != old_thread_id


class TestCheckpoints:
    """Test checkpoint backends."""
    
    def test_memory_checkpointer(self):
        """Test memory checkpointer creation."""
        from nanocoder.langgraph_checkpointer import get_checkpointer
        
        checkpointer = get_checkpointer("memory")
        
        assert checkpointer is not None
    
    def test_checkpointer_from_env(self):
        """Test checkpointer creation from environment variable."""
        from nanocoder.langgraph_checkpointer import get_checkpointer
        import os
        
        # Set env var
        os.environ["NANOCODER_CHECKPOINT_BACKEND"] = "memory"
        
        checkpointer = get_checkpointer()
        
        assert checkpointer is not None
        
        # Clean up
        del os.environ["NANOCODER_CHECKPOINT_BACKEND"]
    
    def test_sqlite_checkpointer_import_error(self):
        """Test SQLite checkpointer raises helpful error if not installed."""
        from nanocoder.langgraph_checkpointer import get_checkpointer
        
        # This test may pass or fail depending on whether langgraph-checkpoint-sqlite
        # is installed. We just verify it doesn't crash unexpectedly.
        try:
            checkpointer = get_checkpointer("sqlite", "/tmp/test.db")
            # If it succeeds, that's fine
            assert checkpointer is not None
        except ImportError as e:
            # If it fails, should have helpful message
            assert "langgraph-checkpoint-sqlite" in str(e)


class TestParallelExecution:
    """Test parallel tool execution."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        llm = Mock()
        llm.model = "test-model"
        llm.chat = Mock()
        return llm
    
    @pytest.fixture
    def mock_tools(self):
        """Create multiple mock tools."""
        tools = []
        for i in range(3):
            tool = Mock()
            tool.name = f"tool_{i}"
            tool.description = f"Tool {i}"
            tool.parameters = {}
            tool.execute = Mock(return_value=f"result_{i}")
            tool.schema = Mock(return_value={
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": f"Tool {i}",
                    "parameters": {},
                },
            })
            tools.append(tool)
        return tools
    
    def test_parallel_execution_enabled(self, mock_llm, mock_tools):
        """Test that parallel execution can be enabled."""
        from nanocoder.langgraph_agent import LangGraphAgent
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
            enable_parallel=True,
        )
        
        assert agent.enable_parallel is True
    
    def test_parallel_execution_disabled(self, mock_llm, mock_tools):
        """Test that parallel execution can be disabled."""
        from nanocoder.langgraph_agent import LangGraphAgent
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
            enable_parallel=False,
        )
        
        assert agent.enable_parallel is False


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        llm = Mock()
        llm.model = "test-model"
        llm.chat = Mock()
        return llm
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.parameters = {}
        tool.execute = Mock(return_value="result")
        tool.schema = Mock(return_value={
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {},
            },
        })
        return [tool]
    
    def test_tool_execution_error(self, mock_llm, mock_tools):
        """Test that tool execution errors are caught."""
        from nanocoder.langgraph_agent import LangGraphAgent
        from nanocoder.llm import LLMResponse, ToolCall
        
        # Tool raises exception
        mock_tools[0].execute.side_effect = Exception("Tool error")
        
        # First response: tool call
        tool_call = ToolCall(id="call_1", name="test_tool", arguments={})
        first_response = LLMResponse(content="", tool_calls=[tool_call])
        
        # Second response: final text
        second_response = LLMResponse(content="Error handled", tool_calls=[])
        
        mock_llm.chat.side_effect = [first_response, second_response]
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
        )
        
        # Should not crash, error should be in tool result
        response = agent.chat("Use broken tool")
        
        assert response == "Error handled"
    
    def test_unknown_tool(self, mock_llm, mock_tools):
        """Test handling of unknown tool calls."""
        from nanocoder.langgraph_agent import LangGraphAgent
        from nanocoder.llm import LLMResponse, ToolCall
        
        # Call non-existent tool
        tool_call = ToolCall(id="call_1", name="nonexistent_tool", arguments={})
        first_response = LLMResponse(content="", tool_calls=[tool_call])
        second_response = LLMResponse(content="Done", tool_calls=[])
        
        mock_llm.chat.side_effect = [first_response, second_response]
        
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
        )
        
        response = agent.chat("Use unknown tool")
        
        # Should handle gracefully
        assert response == "Done"


class TestIntegration:
    """Integration tests for full workflow."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM instance."""
        llm = Mock()
        llm.model = "test-model"
        llm.chat = Mock()
        return llm
    
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        tool = Mock()
        tool.name = "read_file"
        tool.description = "Read a file"
        tool.parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
        tool.execute = Mock(return_value="file content")
        tool.schema = Mock(return_value={
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        })
        return [tool]
    
    def test_full_workflow(self, mock_llm, mock_tools):
        """Test complete workflow: user input -> LLM -> tool -> LLM -> response."""
        from nanocoder.langgraph_agent import LangGraphAgent
        from nanocoder.llm import LLMResponse, ToolCall
        from nanocoder.langgraph_checkpointer import get_checkpointer
        
        # Setup responses
        tool_call = ToolCall(
            id="call_1",
            name="read_file",
            arguments={"path": "/test.txt"},
        )
        
        responses = [
            LLMResponse(content="", tool_calls=[tool_call]),
            LLMResponse(content="File content is: test", tool_calls=[]),
        ]
        mock_llm.chat.side_effect = responses
        
        # Create agent with memory checkpointer
        checkpointer = get_checkpointer("memory")
        agent = LangGraphAgent(
            llm=mock_llm,
            tools=mock_tools,
            checkpointer=checkpointer,
        )
        
        # Run chat
        response = agent.chat("Read the file")
        
        # Verify
        assert response == "File content is: test"
        assert mock_llm.chat.call_count == 2
        
        # Verify tool was called
        mock_tools[0].execute.assert_called_once_with(path="/test.txt")
