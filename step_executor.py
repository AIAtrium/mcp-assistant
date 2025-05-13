import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict
from anthropic import Anthropic
from openai import OpenAI
from arcadepy import Arcade
from arcade_utils import get_tools_from_arcade, ModelProvider
from langfuse.decorators import observe, langfuse_context
from tool_processor import ToolProcessor
from llm_utils import LLMMessageCreator

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


def main():
    """
    This creates a sample daily briefing for today from my gmail and google calendar then writes it to a Notion database.
    Configuration can be customized in user_inputs.py, or will use defaults if not found.
    """
    # NOTE: the are Default values you can override in user_inputs.py
    DATE = datetime.today().strftime("%Y-%m-%d")
    NOTION_PAGE_TITLE = "Daily Briefings"
    LANGFUSE_SESSION_ID = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

    USER_CONTEXT = """
    I am David, the CTO / Co-Founder of a pre-seed startup based in San Francisco. 
    I handle all the coding and product development.
    We are a two person team, with my co-founder handling sales, marketing, and business development.
    
    When looking at my calendar, if you see anything titled 'b', that means it's a blocker.
    I often put blockers before or after calls that could go long.
    """

    BASE_SYSTEM_PROMPT = """
    You are a helpful assistant.
    """

    INPUT_ACTION = f"""
    Your goal is to create a daily briefing for today, {DATE}, from my gmail and google calendar.
    Do the following:
    1) check my gmail, look for unread emails and tell me if any are high priority
    2) check my google calendar, look for events from today and give me a summary of the events. 
    3) create a draft emails in my gmail inbox with a summary of the above information. DO NOT SEND THE EMAIL.
    """

    # Try to import user configurations, override defaults if found
    try:
        print("Loading values from user_inputs.py")
        import user_inputs

        # Override each value individually if it exists in user_inputs
        if hasattr(user_inputs, "INPUT_ACTION"):
            INPUT_ACTION = user_inputs.INPUT_ACTION
        if hasattr(user_inputs, "BASE_SYSTEM_PROMPT"):
            BASE_SYSTEM_PROMPT = user_inputs.BASE_SYSTEM_PROMPT
        if hasattr(user_inputs, "USER_CONTEXT"):
            USER_CONTEXT = user_inputs.USER_CONTEXT

        # TODO: implement after
        # if hasattr(user_inputs, "ENABLED_CLIENTS"):
        #     ENABLED_CLIENTS = user_inputs.ENABLED_CLIENTS
        #     print(
        #         f"System will run with only the following clients:\n{ENABLED_CLIENTS}\n\n"
        #     )
        # else:
        #     ENABLED_CLIENTS = DEFAULT_CLIENTS
    except ImportError:
        print("Unable to load values from user_inputs.py found, using default values")
        # ENABLED_CLIENTS = DEFAULT_CLIENTS

    print(f"INPUT_ACTION: {INPUT_ACTION}")

    # Initialize host with default system prompt and enabled clients
    executor = StepExecutor(
        default_system_prompt=BASE_SYSTEM_PROMPT,
        user_context=USER_CONTEXT,
        # enabled_clients=ENABLED_CLIENTS,
    )

    result = executor.process_input_with_agent_loop(
        input_action=INPUT_ACTION,
        provider=ModelProvider.OPENAI,
        user_id="david_test",
        langfuse_session_id=LANGFUSE_SESSION_ID,
    )
    print(result)


if __name__ == "__main__":
    main()
