"""Base class for all tools."""

import threading
from abc import ABC, abstractmethod

# File-mutating tools (write_file/edit_file) serialize on this, so a parallel
# batch can't interleave read-modify-write on the same file and lose an edit.
FILE_MUTATION_LOCK = threading.Lock()


class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Run the tool and return a text result."""
        ...

    def schema(self) -> dict:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
