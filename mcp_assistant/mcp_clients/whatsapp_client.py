import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_clients.mcp_client import MCPClient


class WhatsappMCPClient(MCPClient):
    def __init__(self):
        super().__init__(name="Whatsapp")

    async def connect_to_server(self, server_script_path: str) -> None:
        """Connect to the whatsapp MCP server using a specific virtual environment.

        Args:
            server_script_path: Path to the server script
        """
        # Path to the Python interpreter in the WhatsApp server's virtual environment
        venv_path = os.getenv(
            "WHATSAPP_MCP_SERVER_VENV_PATH"
        )  # Replace with actual path
        if not venv_path:
            raise ValueError(
                "WHATSAPP_MCP_SERVER_VENV_PATH not found in env, perhaps you forgot to set it up"
            )
        venv_python = f"{venv_path}/bin/python"

        command = venv_python
        args = [server_script_path]

        # Create a copy of the current environment and add venv-specific variables
        env = os.environ.copy()

        # Set VIRTUAL_ENV environment variable (some packages check this)
        env["VIRTUAL_ENV"] = venv_path

        # Modify PATH to prioritize the venv's bin directory
        bin_dir = os.path.join(venv_path, "bin")
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

        # Ensure PYTHONHOME is unset as it can interfere with the venv
        if "PYTHONHOME" in env:
            del env["PYTHONHOME"]

        server_params = StdioServerParameters(command=command, args=args, env=env)

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
