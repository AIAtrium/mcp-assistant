import json
import os
from typing import Any, Dict, List, Tuple

from arcadepy import Arcade
from arcadepy.types import ExecuteToolResponse
from langfuse.decorators import langfuse_context, observe

from plan_exec_agent.arcade_utils import ModelProvider


class ToolProcessor:
    def __init__(self, arcade_client: Arcade):
        self.arcade_client = arcade_client

    @observe(as_type="tool")  # pyright: ignore
    def process_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_id: str,
        content: Any,
        assistant_message_content: list[Any],
        messages: list[Dict[str, Any]],
        state: dict[str, Any],
        final_text: list[str],
        user_id: str,
        provider: ModelProvider,
        langfuse_data: dict[str, Any] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Process a specific tool call and return updated messages and result content.

        NOTE: this does not currently support MCP resources the way the MCP host does because
        Arcade's off the shelf tools don't support it and its not necessary.
        """

        # Add langfuse tracking
        if (
            langfuse_data
            and "session_id" in langfuse_data
            and "user_id" in langfuse_data
        ):
            langfuse_context.update_current_observation(name=tool_name)
            langfuse_context.update_current_trace(
                session_id=langfuse_data["session_id"], user_id=langfuse_data["user_id"]
            )
            langfuse_context.flush()

        if tool_name == "reference_tool_output":
            return self._handle_reference_tool(
                tool_id,
                tool_args,
                content,
                assistant_message_content,
                messages,
                state,
                provider,
            )
        elif tool_name == "get_previous_step_result":
            return self._handle_previous_step_tool(
                tool_id,
                tool_args,
                content,
                assistant_message_content,
                messages,
                state,
                provider,
            )
        elif tool_name == "signal_insufficient_context":
            return self._handle_insufficient_context_tool(
                tool_id,
                tool_args,
                content,
                assistant_message_content,
                messages,
                provider,
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
        tool_args: Dict[str, Any],
        content: Any,
        assistant_message_content: List[Any],
        messages: List[Dict[str, Any]],
        state: Dict[str, Any],
        provider: ModelProvider,
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """Handle reference_tool_output tool."""
        referenced_tool_id = tool_args["tool_id"]
        result_content = None

        if (
            state
            and "tool_results" in state
            and referenced_tool_id in state["tool_results"]
        ):
            tool_name, result_content = state["tool_results"][referenced_tool_id]
            print(
                f"Successfully retrieved tool result for {tool_name} with ID {referenced_tool_id} with LLM tool call {tool_id}"
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
            provider,
        )

    def _handle_previous_step_tool(
        self,
        tool_id: str,
        tool_args: Dict[str, Any],
        content: Any,
        assistant_message_content: List[Any],
        messages: List[Dict[str, Any]],
        state: Dict[str, Any],
        provider: ModelProvider,
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """
        This enables the execution agent to reference the output of a previous step.
        The result of that previous step is added to the `updated_messages` array
        in the `_create_tool_response` function.
        These messages are fed back into the LLM during the next iteration of the execution loop,
        allowing it to take action on them.

        NOTE: if these results are large and this tool is called many times in one execution loop,
        it could cause the LLM to run out of context window.
        """
        step_number = tool_args.get("step_number")

        if not step_number or step_number < 1:
            result_content = (
                "Error: Invalid step number. Please provide a step number >= 1."
            )
            return self._create_tool_response(
                tool_id,
                content,
                assistant_message_content,
                messages,
                result_content,
                provider,
            )

        # Convert to 0-based index
        step_index = step_number - 1

        try:
            if (
                state
                and "past_results" in state
                and step_index < len(state["past_results"])
            ):
                step_name, raw_result = state["past_results"][step_index]
                # Convert list to string if needed
                if isinstance(raw_result, list):
                    raw_result = "\n".join(str(item) for item in raw_result)
                result_content = f"Step {step_number} ({step_name}):\n{raw_result}"
            else:
                result_content = f"Error: No raw result found for step {step_number}. Available steps: 1-{len(state.get('past_results', []))}"

        except Exception as e:
            result_content = f"Error retrieving step {step_number} result: {str(e)}"

        return self._create_tool_response(
            tool_id,
            content,
            assistant_message_content,
            messages,
            result_content,
            provider,
        )

    def _handle_insufficient_context_tool(
        self,
        tool_id: str,
        tool_args: Dict[str, Any],
        content: Any,
        assistant_message_content: List[Any],
        messages: List[Dict[str, Any]],
        provider: ModelProvider,
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """Handle the "signal_insufficient_context" tool."""
        reason = tool_args.get("reason")
        if not reason:
            result_content = "Error: No reason provided for insufficient context"
            return self._create_tool_response(
                tool_id,
                content,
                assistant_message_content,
                messages,
                result_content,
                provider,
            )

        result_content = f"STEP_FAILED_INSUFFICIENT_CONTEXT: {reason}"
        return self._create_tool_response(
            tool_id,
            content,
            assistant_message_content,
            messages,
            result_content,
            provider,
        )

    def _handle_standard_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_id: str,
        content: Any,
        assistant_message_content: List[Any],
        messages: List[Dict[str, Any]],
        final_text: List[str],
        user_id: str,
        provider: ModelProvider,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Handle standard tools using Arcade's tool execution flow."""
        try:
            # First authorize the tool
            auth_response = self.arcade_client.tools.authorize(
                tool_name=tool_name,
                user_id=user_id,
            )

            if auth_response.status != "completed":
                if not os.getenv("SKIP_CLI_AUTH"):
                    print(f"Authorization needed. URL: {auth_response.url}")
                    # Block and wait for the authorization to complete - only do this locally
                    self.arcade_client.auth.wait_for_completion(auth_response)
                else:
                    print("Skipping authorization. Marking tool call as failed.")
                    result_content = f"Unable to call {tool_name} because it requires authorization. Please authorize it manually outside of this program."

                    return self._create_tool_response(
                        tool_id,
                        content,
                        assistant_message_content,
                        messages,
                        result_content,
                        provider,
                    )

            # Convert tool_args to dict if it's a string
            tool_input = tool_args if isinstance(tool_args, dict) else eval(tool_args)

            # Execute the tool
            # NOTE: arcade has rate limits on the free plan
            response: ExecuteToolResponse = self.arcade_client.tools.execute(
                tool_name=tool_name,
                input=tool_input,
                user_id=user_id,
            )

            final_text.append(
                f"[Executing tool {tool_name} with args {tool_input} with id {tool_id}]"
            )

            # Handle the response
            if response.success:
                output = response.output
                if output is None:
                    raise ValueError("Successful tool execution without output")
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
        assistant_message_content: List[Any],
        messages: List[Dict[str, Any]],
        result_content: str,
        provider: ModelProvider,
    ) -> Tuple[List[Dict[str, Any]], str]:
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
