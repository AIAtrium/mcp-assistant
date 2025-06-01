import json
from typing import Any

from arcadepy import Arcade
from arcadepy.types import ExecuteToolResponse
from langfuse.decorators import langfuse_context, observe

from plan_exec_agent.arcade_utils import ModelProvider


class ToolProcessor:
    def __init__(self, arcade_client: Arcade):
        self.arcade_client = arcade_client

    @observe(as_type="tool")  # pyright:ignore[reportArgumentType]
    def process_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_id: str,
        content: Any,
        assistant_message_content: list[Any],
        messages: list[dict[str, Any]],
        tool_results_context: dict[str, Any],
        final_text: list[str],
        user_id: str,
        provider: ModelProvider,
        langfuse_session_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], Any]:
        """
        Process a specific tool call and return updated messages and result content.

        NOTE: this does not currently support MCP resources the way the MCP host does because
        Arcade's off the shelf tools don't support it and its not necessary.
        """

        # Add langfuse tracking
        if langfuse_session_id:
            langfuse_context.update_current_observation(name=tool_name)
            langfuse_context.update_current_trace(session_id=langfuse_session_id)
            langfuse_context.flush()

        if tool_name == "reference_tool_output":
            return self._handle_reference_tool(
                tool_id,
                tool_args,
                content,
                assistant_message_content,
                messages,
                tool_results_context,
            )
        else:
            return self._handle_standard_tool(
                tool_name,
                tool_args,
                tool_id,
                content,
                assistant_message_content,
                messages,
                final_text,
                user_id,
                provider,
            )

    def _handle_reference_tool(
        self,
        tool_id: str,
        tool_args: dict[str, Any],
        content: Any,
        assistant_message_content: list[Any],
        messages: list[dict[str, Any]],
        tool_results_context: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], Any]:
        """Handle reference_tool_output tool."""
        referenced_tool_id = tool_args["tool_id"]
        extract_path = tool_args.get("extract_path", ".")
        result_content = None

        if referenced_tool_id in tool_results_context:
            result_content = self._extract_reference_data(
                tool_results_context[referenced_tool_id], extract_path
            )
        else:
            result_content = (
                f"Error: No tool result found with ID '{referenced_tool_id}'"
            )

        return self._create_tool_response(
            tool_id,
            content,
            assistant_message_content,
            messages,
            result_content,
            ModelProvider.ANTHROPIC,
        )

    def _extract_reference_data(self, result_content: Any, extract_path: str) -> str:
        """Extract data from a result using the given path."""
        if not extract_path or not result_content:
            return result_content

        try:
            data = json.loads(result_content)
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

    def _handle_standard_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_id: str,
        content: Any,
        assistant_message_content: list[Any],
        messages: list[dict[str, Any]],
        final_text: list[str],
        user_id: str,
        provider: ModelProvider,
    ) -> tuple[list[dict[str, Any]], str]:
        """Handle standard tools using Arcade's tool execution flow."""
        try:
            # First authorize the tool
            auth_response = self.arcade_client.tools.authorize(
                tool_name=tool_name,
                user_id=user_id,
            )

            if auth_response.status != "completed":
                print(f"Authorization needed. URL: {auth_response.url}")
                # Wait for the authorization to complete
                self.arcade_client.auth.wait_for_completion(auth_response)

            # Convert tool_args to dict if it's a string
            tool_input = tool_args if isinstance(tool_args, dict) else eval(tool_args)

            # Execute the tool
            # NOTE: arcade has rate limits on the free plan
            response: ExecuteToolResponse = self.arcade_client.tools.execute(
                tool_name=tool_name,
                input=tool_input,
                user_id=user_id,
            )

            final_text.append(f"[Executing tool {tool_name} with args {tool_input}]")

            # Handle the response
            if response.success and response.output:
                output = response.output
                if output.error:
                    result_content = f"Error: {output.error.message}"
                else:
                    result_content = (
                        output.value
                        if isinstance(output.value, str)
                        else json.dumps(output.value)
                    )
            else:
                result_content = f"Tool execution failed with status: {response.status}"

        except Exception as e:
            error_message = f"Error executing tool '{tool_name}': {str(e)}"
            print(error_message)
            final_text.append(error_message)
            result_content = error_message

        return self._create_tool_response(
            tool_id,
            content,
            assistant_message_content,
            messages,
            result_content,
            provider,
        )

    def _create_tool_response(
        self,
        tool_id: str,
        content: Any,
        assistant_message_content: list[Any],
        messages: list[dict[str, Any]],
        result_content: str,
        provider: ModelProvider,
    ) -> tuple[list[dict[str, Any]], str]:
        """Create a standardized tool response format."""
        updated_messages = messages.copy()

        if provider == ModelProvider.ANTHROPIC:
            # Anthropic format
            assistant_message_content.append(content)
            updated_messages.append({
                "role": "assistant",
                "content": assistant_message_content,
            })
            updated_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_content,
                    }
                ],
            })
        else:
            # OpenAI format
            tool_name = content["name"] if isinstance(content, dict) else content.name
            tool_args = content["input"] if isinstance(content, dict) else content.input

            # Add the assistant's message with the tool call
            updated_messages.append({
                "role": "assistant",
                "content": None,  # Content is null when there's a tool call
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            })

            # Add the tool result
            updated_messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result_content,
            })

        return updated_messages, result_content
