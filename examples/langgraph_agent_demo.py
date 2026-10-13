"""LangGraph Agent usage examples.

This file demonstrates how to use the LangGraph-based agent with various
configurations including checkpointing, resume, and parallel execution.
"""

from nanocoder.llm import LLM
from nanocoder.langgraph_agent import LangGraphAgent
from nanocoder.langgraph_checkpointer import get_checkpointer
from nanocoder.tools import ALL_TOOLS


def example_basic_usage():
    """Basic usage without checkpointing."""
    print("=" * 60)
    print("Example 1: Basic LangGraph Agent Usage")
    print("=" * 60)
    
    # Initialize LLM
    llm = LLM(
        model="gpt-4",
        api_key="your-api-key-here",
    )
    
    # Create agent (no checkpointing)
    agent = LangGraphAgent(
        llm=llm,
        tools=ALL_TOOLS,
        max_rounds=10,
    )
    
    # Chat with the agent
    response = agent.chat("What is the capital of France?")
    print(f"\nResponse: {response}\n")


def example_with_memory_checkpoint():
    """Usage with in-memory checkpointing."""
    print("=" * 60)
    print("Example 2: In-Memory Checkpointing")
    print("=" * 60)
    
    llm = LLM(
        model="gpt-4",
        api_key="your-api-key-here",
    )
    
    # Create memory checkpointer
    checkpointer = get_checkpointer("memory")
    
    # Create agent with checkpointer
    agent = LangGraphAgent(
        llm=llm,
        tools=ALL_TOOLS,
        checkpointer=checkpointer,
        thread_id="my-conversation-1",
    )
    
    # First message
    response1 = agent.chat("My name is Alice")
    print(f"\nResponse 1: {response1}\n")
    
    # Second message (agent remembers context)
    response2 = agent.chat("What's my name?")
    print(f"\nResponse 2: {response2}\n")


def example_with_sqlite_checkpoint():
    """Usage with SQLite checkpointing for persistence."""
    print("=" * 60)
    print("Example 3: SQLite Checkpointing (Persistent)")
    print("=" * 60)
    
    llm = LLM(
        model="gpt-4",
        api_key="your-api-key-here",
    )
    
    # Create SQLite checkpointer (persists to disk)
    checkpointer = get_checkpointer(
        backend="sqlite",
        db_path="~/.nanocoder/checkpoints.db"
    )
    
    # Create agent with thread ID
    thread_id = "project-discussion-123"
    agent = LangGraphAgent(
        llm=llm,
        tools=ALL_TOOLS,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )
    
    # Have a conversation
    response = agent.chat("Help me design a REST API")
    print(f"\nResponse: {response}\n")
    
    print(f"Conversation saved with thread_id: {thread_id}")
    print("You can resume this conversation later using the same thread_id")


def example_resume_from_checkpoint():
    """Resume a conversation from checkpoint."""
    print("=" * 60)
    print("Example 4: Resume from Checkpoint")
    print("=" * 60)
    
    llm = LLM(
        model="gpt-4",
        api_key="your-api-key-here",
    )
    
    checkpointer = get_checkpointer("sqlite")
    
    # Resume from previous conversation
    thread_id = "project-discussion-123"
    agent = LangGraphAgent(
        llm=llm,
        tools=ALL_TOOLS,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )
    
    try:
        # Continue the conversation
        response = agent.chat("Now let's add authentication")
        print(f"\nResponse: {response}\n")
    except ValueError as e:
        print(f"Could not resume: {e}")
        print("Starting new conversation instead...")
        
        response = agent.chat("Help me design a REST API with authentication")
        print(f"\nResponse: {response}\n")


def example_parallel_tool_execution():
    """Demonstrate parallel tool execution."""
    print("=" * 60)
    print("Example 5: Parallel Tool Execution")
    print("=" * 60)
    
    llm = LLM(
        model="gpt-4",
        api_key="your-api-key-here",
    )
    
    agent = LangGraphAgent(
        llm=llm,
        tools=ALL_TOOLS,
        enable_parallel=True,  # Enable parallel execution
    )
    
    # This may trigger multiple tool calls that execute in parallel
    response = agent.chat(
        "Read the contents of file1.py, file2.py, and file3.py"
    )
    print(f"\nResponse: {response}\n")


def example_with_callbacks():
    """Usage with streaming callbacks."""
    print("=" * 60)
    print("Example 6: Streaming Callbacks")
    print("=" * 60)
    
    llm = LLM(
        model="gpt-4",
        api_key="your-api-key-here",
    )
    
    agent = LangGraphAgent(
        llm=llm,
        tools=ALL_TOOLS,
    )
    
    # Define callbacks
    def on_token(token):
        """Called for each streaming token."""
        print(token, end="", flush=True)
    
    def on_tool(tool_name, arguments):
        """Called before tool execution."""
        print(f"\n[Calling tool: {tool_name}({arguments})]\n")
    
    def on_reasoning(reasoning):
        """Called for reasoning tokens (thinking models)."""
        print(f"[Reasoning: {reasoning}]")
    
    # Chat with callbacks
    response = agent.chat(
        "Explain quantum computing",
        on_token=on_token,
        on_tool=on_tool,
        on_reasoning=on_reasoning,
    )
    print(f"\n\nFinal response: {response}\n")


def example_cli_usage():
    """Show CLI usage examples."""
    print("=" * 60)
    print("Example 7: CLI Usage")
    print("=" * 60)
    
    print("""
# Basic usage with LangGraph agent
nanocoder --use-langgraph

# With SQLite checkpointing
nanocoder --use-langgraph --checkpoint-backend sqlite

# With specific thread ID
nanocoder --use-langgraph --thread-id my-project-123

# Resume from checkpoint
nanocoder --use-langgraph --thread-id my-project-123

# One-shot mode
nanocoder --use-langgraph -p "Explain Python decorators"

# With auto-approval
nanocoder --use-langgraph --yes
    """)


def example_error_handlinging():
    """Demonstrate error handling."""
    print("=" * 60)
    print("Example 8: Error Handling")
    print("=" * 60)
    
    llm = LLM(
        model="gpt-4",
        api_key="your-api-key-here",
    )
    
    agent = LangGraphAgent(
        llm=llm,
        tools=ALL_TOOLS,
        max_rounds=5,  # Limit rounds to prevent infinite loops
    )
    
    try:
        response = agent.chat("Perform an impossible task")
        print(f"\nResponse: {response}\n")
    except Exception as e:
        print(f"Error occurred: {e}")
        # Agent handles errors gracefully and reports them to the LLM


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LangGraph Agent Examples")
    print("=" * 60 + "\n")
    
    # Note: These examples require a valid API key to run
    # Uncomment the example you want to test
    
    # example_basic_usage()
    # example_with_memory_checkpoint()
    # example_with_sqlite_checkpoint()
    # example_resume_from_checkpoint()
    # example_parallel_tool_execution()
    # example_with_callbacks()
    example_cli_usage()
    # example_error_handlinging()
    
    print("\nAll examples completed!")
