from abc import ABC, abstractmethod
from contextlib import AsyncExitStack

from mcp import ClientSession


class MCPClient(ABC):
    """Abstract base class for MCP clients."""

    def __init__(self, name: str):
        # Initialize session and client objects
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.name = name

    @abstractmethod
    async def connect_to_server(self, server_script_path: str) -> None:
        """
        Connect to an MCP server
        This is left unimplemented because each MCP client could have its own way of connecting to the server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        # type: (...) -> None # sets self.session: ClientSession

    # In mcp_client.py
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if self.session:
                # Try to explicitly close the session first
                try:
                    raise ValueError(
                        "This should have never been called, self.session.shutdown is undefined"
                    )
                    await self.session.shutdown()
                except Exception as e:
                    print(f"Warning: Error shutting down session for {self.name}: {e}")

            # Then close the exit stack
            await self.exit_stack.aclose()
        except Exception as e:
            print(f"Warning: Error during cleanup of {self.name}: {e}")
