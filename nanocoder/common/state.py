from typing import TypedDict


class AgentState(TypedDict):
    """Control state for one chat(). Messages stay on the Agent instance."""

    rounds: int
    has_tools: bool
    final_response: str