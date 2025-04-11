import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextResourceContents, BlobResourceContents
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

# based on https://modelcontextprotocol.io/quickstart/client

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # List available resources
        response = await self.session.list_resources()
        resources = response.resources
        print("\nConnected to server with resources:", [resource.name for resource in resources])

        # List available resource templates
        response = await self.session.list_resource_templates()
        resource_templates = response.resourceTemplates
        print("\nConnected to server with resource templates:", [template.name for template in resource_templates])
    
    async def get_resources_info(self):
        resource_response = await self.session.list_resources()
        resources = resource_response.resources
        
        templates_response = await self.session.list_resource_templates()
        templates = templates_response.resourceTemplates

        formatted_info = "You may also access the following resources and resource templates to help you answer the user's query:\nAvailable resources:\n"
        for resource in resources:
            formatted_info += f"Name: {resource.name}, Description: {resource.description}, URI: {resource.uri._url}\n"

        formatted_info += "\nAvailable resource templates:\n"
        for template in templates:
            formatted_info += f"Name: {template.name}, Description: {template.description}, URI Template: {template.uriTemplate}\n"

        print(f"formatted_info: {formatted_info}")
        return formatted_info
    
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        resources_info = await self.get_resources_info()
        query += resources_info
        
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        server_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Add custom resource access tool
        resource_access_tool = {
            "name": "access_resource",
            "description": "Access a resource from the MCP server",
            "input_schema": {
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "The URI of the resource to access"
                    }
                },
                "required": ["uri"]
            }
        }
        
        available_tools = server_tools + [resource_access_tool]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        print(f"response: {response}")

        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input

                if tool_name == "access_resource":
                    # Handle resource access
                    uri = tool_args["uri"]
                    resource_result = await self.session.read_resource(uri)
                    final_text.append(f"[Accessing resource {uri}]")
                    
                    assistant_message_content.append(content)
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message_content
                    })

                    # Format the resource result as a tool result
                    resource_result_content = []
                    for resource_content in resource_result.contents:
                        if isinstance(resource_content, TextResourceContents):
                            resource_result_content.append(resource_content.text)
                        elif isinstance(resource_content, BlobResourceContents):
                            resource_result_content.append(str(resource_content.blob))

                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": "\n".join(resource_result_content)
                            }
                        ]
                    })

                else:
                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                    assistant_message_content.append(content)
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message_content
                    })
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }
                        ]
                    })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        response = await client.process_query("What is my app config?")
        print(response)
        # await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())