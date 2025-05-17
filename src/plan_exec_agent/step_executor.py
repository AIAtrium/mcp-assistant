import os
import json
from dotenv import load_dotenv
from typing import List, Dict
from anthropic import Anthropic
from openai import OpenAI
from arcadepy import Arcade
from langfuse.decorators import observe, langfuse_context
from .arcade_utils import get_tools_from_arcade, ModelProvider
from .tool_processor import ToolProcessor
from .llm_utils import LLMMessageCreator

load_dotenv()


class StepExecutor:
    def __init__(
        self,
        default_system_prompt: str = None,
        user_context: str = None,
        enabled_clients: List[str] = None,
    ):
        self.arcade_client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))
        self.tool_processor = ToolProcessor(arcade_client=self.arcade_client)

        # Initialize LLM clients
        anthropic_client = (
            Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            if os.getenv("ANTHROPIC_API_KEY")
            else None
        )
        openai_client = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if os.getenv("OPENAI_API_KEY")
            else None
        )

        # Initialize message creator with available clients
        self.message_creator = LLMMessageCreator(
            anthropic_client=anthropic_client, openai_client=openai_client
        )

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

    def _get_reference_tool(self, provider: ModelProvider):
        if provider == ModelProvider.ANTHROPIC:
            return {
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
        elif provider == ModelProvider.OPENAI:
            return {
                "type": "function",
                "function": {
                    "name": "reference_tool_output",
                    "description": "Reference the output of a previously called tool",
                    "parameters": {
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
                
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    # TODO: alter this to take into account enabled clients
    def get_all_tools(self, provider: ModelProvider):
        # Add a tool reference capability that allows the LLM to reference previous tool outputs
        reference_tool = self._get_reference_tool(provider)
        arcade_tools = get_tools_from_arcade(self.arcade_client, provider)
        return arcade_tools + [reference_tool]

    @observe()
    def process_input_with_agent_loop(
        self,
        input_action: str,
        provider: ModelProvider,
        user_id: str,
        system_prompt: str = None,
        langfuse_session_id: str = None,
        state: Dict = None,
    ):
        current_system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        # Set the observation name to include the current step if available
        if state and "current_plan" in state and state["current_plan"]:
            current_step = state["current_plan"][0]
            langfuse_context.update_current_observation(name=f"{current_step}")

        # Initialize conversation context
        tool_results_context = {}
        messages = [{"role": "user", "content": input_action}]

        # TODO: alter this to account for only the available tools the user wants to authorize
        available_tools = self.get_all_tools(provider)

        response = self.message_creator.create_message(
            provider=provider,
            messages=messages,
            available_tools=available_tools,
            system_prompt=current_system_prompt,
            langfuse_session_id=langfuse_session_id,
        )

        final_text = []

        # Continue processing until we have a complete response
        while True:
            assistant_message_content = []
            has_tool_calls = False

            # Handle different response formats based on provider
            if provider == ModelProvider.ANTHROPIC:
                response_contents = response.content
            elif provider == ModelProvider.OPENAI:
                # OpenAI returns a single choice with a message
                message = response.choices[0].message
                # Convert OpenAI format to match Anthropic's structure
                response_contents = []
                if message.content:
                    response_contents.append({"type": "text", "text": message.content})
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        response_contents.append({
                            "type": "tool_use",
                            "name": tool_call.function.name,
                            "input": json.loads(tool_call.function.arguments),
                            "id": tool_call.id
                        })

            for content in response_contents:
                if ('type' in content and content["type"] == "text") or (hasattr(content, "type") and content.type == "text"):
                    final_text.append(content["text"] if isinstance(content, dict) else content.text)
                    assistant_message_content.append(content)
                elif ('type' in content and content["type"] == "tool_use") or (hasattr(content, "type") and content.type == "tool_use"):
                    has_tool_calls = True
                    tool_name = content["name"] if isinstance(content, dict) else content.name
                    tool_args = content["input"] if isinstance(content, dict) else content.input
                    tool_id = content["id"] if isinstance(content, dict) else content.id

                    # Process the specific tool call
                    updated_messages, result_content = self.tool_processor.process_tool_call(
                        tool_name,
                        tool_args,
                        tool_id,
                        content,
                        assistant_message_content,
                        messages,
                        tool_results_context,
                        final_text,
                        user_id,
                        provider,
                        langfuse_session_id,
                    )

                    # Update conversation context
                    messages = updated_messages
                    if result_content:
                        tool_results_context[tool_id] = result_content
                    
                    # Get next response
                    response = self.message_creator.create_message(
                        provider=provider,
                        messages=messages,
                        available_tools=available_tools,
                        system_prompt=current_system_prompt,
                        langfuse_session_id=langfuse_session_id,
                    )

                    # Break the content loop to process the new response
                    break

            # If there are no more tool calls, add the final text and break the loop
            if not has_tool_calls:
                if provider == ModelProvider.ANTHROPIC and len(response.content) > 0:
                    if response.content[0].type == "text":
                        final_text.append(response.content[0].text)
                elif provider == ModelProvider.OPENAI and response.choices[0].message.content:
                    final_text.append(response.choices[0].message.content)
                break

        # Add a line at the end, before returning the result
        if state is not None and "tool_results" in state:
            state["tool_results"].update(tool_results_context)

        return "\n".join(final_text)




