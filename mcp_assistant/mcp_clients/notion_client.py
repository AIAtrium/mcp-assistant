import os

from mcp.client.stdio import stdio_client
from mcp_clients.mcp_client import MCPClient

from mcp_assistant import ClientSession, StdioServerParameters


class NotionMCPClient(MCPClient):
    def __init__(self):
        super().__init__(name="Notion")

    async def connect_to_server(self, server_script_path: str) -> None:
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script .js, .ts, .mjs
        """
        command = "node"

        # Create a copy of the current environment and add our variables
        env = os.environ.copy()

        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=env,  # Pass the environment variables
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print(
            f"\nConnected to server {self.name} with tools: {[tool.name for tool in tools]}"
        )
