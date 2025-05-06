import asyncio
import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from anthropic import Anthropic
from langfuse.decorators import observe, langfuse_context
from mcp.types import TextResourceContents, BlobResourceContents, Tool
from mcp_clients.mcp_client import MCPClient
from mcp_clients.gcal_client import GCalMCPClient
from mcp_clients.gmail_client import GmailMCPClient
from mcp_clients.notion_client import NotionMCPClient
from mcp_clients.whatsapp_client import WhatsappMCPClient
from mcp_clients.exa_client import ExaMCPClient
from mcp_clients.outlook_client import OutlookMCPClient
from mcp_clients.slack_client import SlackMCPClient

load_dotenv()


class MCPHost:
    def __init__(self, default_system_prompt: str = None, user_context: str = None):
        self.anthropic = Anthropic()
        self.gcal_client = GCalMCPClient()
        self.gmail_client = GmailMCPClient()
        self.notion_client = NotionMCPClient()
        self.whatsapp_client = WhatsappMCPClient()
        self.exa_client = ExaMCPClient()
        self.outlook_client = OutlookMCPClient()
        self.slack_client = SlackMCPClient()
        
        # inject the user context into the system prompt if its provided
        if default_system_prompt and user_context:
            default_system_prompt = f"""
            {default_system_prompt}
            
            USER CONTEXT:
            {user_context}
            """
        
        # Store system prompt as instance variable with a default
        self.system_prompt = default_system_prompt or "You are a helpful assistant."
        self.user_context = user_context if user_context else ""

        # manually add each eligible MCP clients here
        self.mcp_clients = {
            self.gcal_client.name: self.gcal_client,
            self.gmail_client.name: self.gmail_client,
            self.notion_client.name: self.notion_client,
            self.whatsapp_client.name: self.whatsapp_client,
            self.exa_client.name: self.exa_client,
            self.outlook_client.name: self.outlook_client,
            self.slack_client.name: self.slack_client,
        }

        self.mcp_client_paths = {
            self.gcal_client.name: os.getenv("GCAL_MCP_SERVER_PATH"),
            self.gmail_client.name: os.getenv("GMAIL_MCP_SERVER_PATH"),
            self.notion_client.name: os.getenv("NOTION_MCP_SERVER_PATH"),
            self.whatsapp_client.name: os.getenv("WHATSAPP_MCP_SERVER_PATH"),
            self.exa_client.name: os.getenv("EXA_MCP_SERVER_PATH"),
            self.outlook_client.name: os.getenv("OUTLOOK_MCP_SERVER_PATH"),
            self.slack_client.name: os.getenv("SLACK_MCP_SERVER_PATH"),
        }

        # Map of tool names to client names
        self.tool_to_client_map: Dict[str, str] = {}

        # custom resource access tool
        self.resource_access_tool = {
            "name": "access_resource",
            "description": "Access a resource from the MCP server",
            "input_schema": {
                "type": "object",
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "The URI of the resource to access",
                    },
                    "client": {
                        "type": "string",
                        "description": "The name of the client to access the resource",
                    },
                },
                "required": ["uri", "client"],
            },
        }

        # Add a tool reference capability that allows the LLM to reference previous tool outputs
        self.reference_tool_output = {
            "name": "reference_tool_output",
            "description": "Reference the output of a previously called tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tool_id": {
                        "type": "string",
                        "description": "The ID of the previously called tool",
                    },
                    "extract_path": {
                        "type": "string",
                        "description": "Optional JSON path to extract specific data from the tool result",
                    },
                },
                "required": ["tool_id"],
            },
        }

    async def initialize_mcp_clients(self):
        for client_name, client_path in self.mcp_client_paths.items():
            print(f"Initializing {client_name} with path {client_path}")
            await self.mcp_clients[client_name].connect_to_server(client_path)

    async def get_all_resources_info(self):
        resources_start_message = "## You may also access the following resources and resource templates to help you answer the user's query:\n"
        resources_info = []
        for client in self.mcp_clients.values():
            resources_info.append(await self.get_resources_info(client))
        return resources_start_message + "\n".join(resources_info)

    async def get_resources_info(self, client: MCPClient):
        resources = []
        templates = []

        try:
            resource_response = await client.session.list_resources()
            resources = resource_response.resources
        except Exception as e:
            print(f"Error getting resources: {e}")

        try:
            templates_response = await client.session.list_resource_templates()
            templates = templates_response.resourceTemplates
        except Exception as e:
            print(f"Error getting resource templates: {e}")

        if not resources and not templates:
            return ""

        formatted_info = ""
        if resources:
            formatted_info += f"### Available resources for Client {client.name}:\n"
            for resource in resources:
                formatted_info += f"Name: {resource.name}, Description: {resource.description}, URI: {resource.uri._url}\n"

        if templates:
            formatted_info += (
                f"### Available resource templates for Client {client.name}:\n"
            )
            for template in templates:
                formatted_info += f"Name: {template.name}, Description: {template.description}, URI Template: {template.uriTemplate}\n"

        print(f"formatted_info: {formatted_info}")
        return formatted_info

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        tools, _ = await self.get_all_tools_from_servers()
        server_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ]

        # Add custom resource access tool
        available_tools = server_tools + [
            self.resource_access_tool,
            self.reference_tool_output,
        ]
        return available_tools

    async def get_all_tools_from_servers(self) -> Tuple[List[Tool], Dict[str, str]]:
        """Get all tools from all servers and map tool names to client names"""
        tools: List[Tool] = []
        tool_to_client_map: Dict[str, str] = {}

        for client_name, client in self.mcp_clients.items():
            response = await client.session.list_tools()
            if response.tools:
                for tool in response.tools:
                    tools.append(tool)
                    # Map this tool name to the client that provides it
                    tool_to_client_map[tool.name] = client_name

        # Store the map in the class for later use
        self.tool_to_client_map = tool_to_client_map
        return tools, tool_to_client_map
    @observe()
    async def process_input_with_agent_loop(self, query: str, system_prompt: str = None, langfuse_session_id: str = None, state: Dict = None):
        # Use provided system prompt or fall back to the instance variable
        current_system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        # Set the observation name to include the current step if available
        if state and "current_plan" in state and state["current_plan"]:
            current_step = state["current_plan"][0]
            langfuse_context.update_current_observation(name=f"{current_step}")

        # Prepare query with available resources information
        enriched_query = await self._prepare_query(query)

        # Initialize conversation context
        tool_results_context = {}
        messages = [{"role": "user", "content": enriched_query}]

        # Get available tools
        await self.get_all_tools_from_servers()
        available_tools = await self.get_all_tools()

        # Initial Claude API call
        response = await self._create_claude_message(
            messages, available_tools, current_system_prompt, langfuse_session_id
        )

        # Process response and handle tool calls
        final_text = []

        # Continue processing until we have a complete response
        while True:
            assistant_message_content = []
            has_tool_calls = False

            for content in response.content:
                if content.type == "text":
                    final_text.append(content.text)
                    assistant_message_content.append(content)
                elif content.type == "tool_use":
                    has_tool_calls = True
                    tool_name = content.name
                    tool_args = content.input
                    tool_id = content.id

                    # Process the specific tool call
                    updated_messages, result_content = await self._process_tool_call(
                        tool_name,
                        tool_args,
                        tool_id,
                        content,
                        assistant_message_content,
                        messages,
                        tool_results_context,
                        final_text,
                        langfuse_session_id,
                    )

                    # Update conversation context
                    messages = updated_messages
                    if result_content:
                        tool_results_context[tool_id] = result_content

                    # Get next response from Claude after a tool call
                    response = await self._create_claude_message(
                        messages,
                        available_tools,
                        current_system_prompt,
                        langfuse_session_id,
                    )

                    # Break the content loop to process the new response
                    break

            # If there are no more tool calls, add the final text and break the loop
            if not has_tool_calls:
                if len(response.content) > 0 and response.content[0].type == "text":
                    final_text.append(response.content[0].text)
                break

        # Add a line at the end, before returning the result
        if state is not None and "tool_results" in state:
            state["tool_results"].update(tool_results_context)
        
        return "\n".join(final_text)

    async def _prepare_query(self, query: str) -> str:
        """Enrich the user query with available resource information."""
        resources_info = await self.get_all_resources_info()
        return query + "\n\n" + resources_info

    @observe(as_type="generation")
    async def _create_claude_message(
        self, messages, available_tools, system_prompt=None, langfuse_session_id=None
    ):
        """Create a message using Claude API with the given messages and tools."""
        system = system_prompt if system_prompt is not None else self.system_prompt

        # Add langfuse input tracking
        langfuse_context.update_current_observation(
            input=messages,
            model="claude-3-5-sonnet-20241022",
            session_id=langfuse_session_id
        )

        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=available_tools,
        )

        # if no session id is provided, doesn't flush to langfuse
        if langfuse_session_id:
            langfuse_context.update_current_trace(session_id=langfuse_session_id)
            langfuse_context.flush()
        
            # Add cost tracking
            langfuse_context.update_current_observation(
                usage_details={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                    "cache_read_input_tokens": response.usage.cache_read_input_tokens
                }
            )

        return response

    @observe(as_type="tool")
    async def _process_tool_call(
        self,
        tool_name,
        tool_args,
        tool_id,
        content,
        assistant_message_content,
        messages,
        tool_results_context,
        final_text,
        langfuse_session_id,
    ):
        """Process a specific tool call and return updated messages and result content."""

        # Add langfuse tracking
        if langfuse_session_id:
            langfuse_context.update_current_observation(name=tool_name)
            langfuse_context.update_current_trace(session_id=langfuse_session_id)
            langfuse_context.flush()
  
        if tool_name == "reference_tool_output":
            return await self._handle_reference_tool(
                tool_id,
                tool_args,
                content,
                assistant_message_content,
                messages,
                tool_results_context,
            )
        elif tool_name == "access_resource":
            return await self._handle_resource_access(
                tool_id,
                tool_args,
                content,
                assistant_message_content,
                messages,
                final_text,
            )
        else:
            return await self._handle_standard_tool(
                tool_name,
                tool_args,
                tool_id,
                content,
                assistant_message_content,
                messages,
                final_text,
            )

    async def _handle_reference_tool(
        self,
        tool_id,
        tool_args,
        content,
        assistant_message_content,
        messages,
        tool_results_context,
    ):
        """Handle reference_tool_output tool."""
        referenced_tool_id = tool_args["tool_id"]
        extract_path = tool_args.get("extract_path", None)
        result_content = None

        if referenced_tool_id in tool_results_context:
            result_content = self._extract_reference_data(
                tool_results_context[referenced_tool_id], extract_path
            )
        else:
            result_content = (
                f"Error: No tool result found with ID '{referenced_tool_id}'"
            )

        # Add tool usage to message
        assistant_message_content.append(content)
        updated_messages = messages.copy()
        updated_messages.append(
            {"role": "assistant", "content": assistant_message_content}
        )

        # Add tool result to message
        updated_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_content,
                    }
                ],
            }
        )

        return updated_messages, None

    def _extract_reference_data(self, result_content, extract_path):
        """Extract data from a result using the given path."""
        if not extract_path or not result_content:
            return result_content

        import json

        try:
            data = json.loads(result_content)
            # Simple path extraction
            parts = extract_path.split(".")
            for part in parts:
                if part in data:
                    data = data[part]
                else:
                    data = None
                    break
            return json.dumps(data) if data else "Path not found in data"
        except json.JSONDecodeError:
            return "Cannot extract path: result is not valid JSON"

    async def _handle_resource_access(
        self,
        tool_id,
        tool_args,
        content,
        assistant_message_content,
        messages,
        final_text,
    ):
        """Handle access_resource tool."""
        uri = tool_args["uri"]
        client_name = tool_args["client"]

        # Get resource from MCP server
        resource_result = await self.mcp_clients[client_name].session.read_resource(uri)
        final_text.append(f"[Accessing resource {uri}]")

        # Format the resource result
        result_content = self._format_resource_content(resource_result)

        # Add tool usage to message
        assistant_message_content.append(content)
        updated_messages = messages.copy()
        updated_messages.append(
            {"role": "assistant", "content": assistant_message_content}
        )

        # Add tool result to message
        updated_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_content,
                    }
                ],
            }
        )

        return updated_messages, result_content

    def _format_resource_content(self, resource_result):
        """Format resource result into a string."""
        resource_result_content = []
        for resource_content in resource_result.contents:
            if isinstance(resource_content, TextResourceContents):
                resource_result_content.append(resource_content.text)
            elif isinstance(resource_content, BlobResourceContents):
                resource_result_content.append(str(resource_content.blob))

        return "\n".join(resource_result_content)

    async def _handle_standard_tool(
        self,
        tool_name,
        tool_args,
        tool_id,
        content,
        assistant_message_content,
        messages,
        final_text,
    ):
        """Handle standard tools that are provided by MCP clients."""
        result_content = None
        updated_messages = messages.copy()

        # Look up which client this tool belongs to
        if tool_name in self.tool_to_client_map:
            client_name = self.tool_to_client_map[tool_name]
            client = self.mcp_clients[client_name]

            # Call the tool through the appropriate client
            print(
                f"Calling tool {tool_name} with args {tool_args} via client {client_name}"
            )
            result = await client.session.call_tool(tool_name, tool_args)
            final_text.append(
                f"[Calling tool {tool_name} with args {tool_args} via client {client_name}]"
            )

            result_content = result.content
        else:
            error_message = f"Error: Tool '{tool_name}' not found in any client"
            print(error_message)
            final_text.append(error_message)
            result_content = f"Error: Tool '{tool_name}' is not available."

        # Add tool usage to message
        assistant_message_content.append(content)
        updated_messages.append(
            {"role": "assistant", "content": assistant_message_content}
        )

        # Add tool result to message
        updated_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_content,
                    }
                ],
            }
        )

        return updated_messages, result_content

    async def cleanup(self):
        cleanup_tasks = []

        # Create separate tasks for each client cleanup
        for client_name, client in self.mcp_clients.items():
            cleanup_tasks.append(
                asyncio.create_task(self._cleanup_client(client_name, client))
            )

        # Wait for all cleanup tasks to complete
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    async def _cleanup_client(self, client_name, client):
        """Helper method to clean up a single client"""
        try:
            await client.cleanup()
        except Exception as e:
            print(f"Warning: Error during cleanup of {client_name}: {e}")

    def _log_claude_response(self, response):
        """Log detailed analysis of Claude's response including text outputs and tool calls."""
        print("\n=== Initial Claude Response Analysis ===")
        text_outputs = [c for c in response.content if c.type == "text"]
        tool_calls = [c for c in response.content if c.type == "tool_use"]
        
        # Log text outputs
        if text_outputs:
            print(f"\n📝 Text Outputs ({len(text_outputs)}):")
            for i, text in enumerate(text_outputs, 1):
                print(f"  Output {i}: {text.text}")
        
        # Log tool calls
        if tool_calls:
            print(f"\n🔧 Tool Calls ({len(tool_calls)}):")
            for i, tool in enumerate(tool_calls, 1):
                print(f"\n  Tool {i}:")
                print(f"    Name: {tool.name}")
                print(f"    ID: {tool.id}")
                print("    Input Arguments:")
                for key, value in tool.input.items():
                    print(f"      {key}: {value}")
        
        print("\n" + "="*40 + "\n")


async def main():
    """
    This creates a sample daily briefing for today from my gmail and google calendar then writes it to a Notion database.
    Override these variables OR the `query` to customize the daily briefing to your liking.
    """
    # variables
    DATE = datetime.today().strftime("%Y-%m-%d")
    NOTION_PAGE_TITLE = "Daily Briefings"
    LANGFUSE_SESSION_ID = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    user_context = """
    I am David, the CTO / Co-Founder of a pre-seed startup based in San Francisco. 
    I handle all the coding and product development.
    We are a two person team, with my co-founder handling sales, marketing, and business development.
    
    When looking at my calendar, if you see anything titled 'b', that means it's a blocker.
    I often put blockers before or after calls that could go long.
    """

    base_system_prompt = f"""
    You are a helpful assistant. 
    """

    # Initialize host with default system prompt
    host = MCPHost(default_system_prompt=base_system_prompt, user_context=user_context)

    try:
        await host.initialize_mcp_clients()

        # can override the query to customize the daily briefing to your liking
        # NOTE: provide the model with step by step instructions for best results
        query = f"""
        Your goal is to create a daily briefing for today, {DATE}, from my gmail and google calendar.
        Do the following:
        1) check my gmail, look for unread emails and tell me if any are high priority
        2) check my google calendar, look for events from today and give me a summary of the events. 
           - If I have a meeting with anyone, search the internet for that person and write a quick summary of them.
        3) Go to my second email, which is my outlook account, and look for any unread emails. Write a summary of the unread emails.
        4) Write the output from the above steps into a new page in my Notion in the '{NOTION_PAGE_TITLE}' page. Title the entry '{DATE}', which is today's date. 
        """
        result = await host.process_input_with_agent_loop(query, LANGFUSE_SESSION_ID)
        print(result)
    finally:
        await host.cleanup()


if __name__ == "__main__":
    asyncio.run(main())