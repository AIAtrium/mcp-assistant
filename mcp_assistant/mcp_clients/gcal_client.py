from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_clients.mcp_client import MCPClient


class GCalMCPClient(MCPClient):
    def __init__(self):
        super().__init__(name="Google Calendar")

    async def connect_to_server(self, server_script_path: str):
        """
        Connect to an MCP server
        Usage: python gcal_client.py <path_to_server_script>
        Sample usage: python gcal_client.py /Users/myusername/Documents/mcp/google-calendar-mcp/dist/index.js

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session: ClientSession = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print(
            f"\nConnected to server {self.name} with tools: {[tool.name for tool in tools]}"
        )

        return self.session

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
