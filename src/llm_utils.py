from typing import List, Dict, Any
from anthropic import Anthropic
from anthropic.types.message import Message
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from langfuse.decorators import observe, langfuse_context
from src.arcade_utils import ModelProvider


class LLMMessageCreator:
    def __init__(
        self, anthropic_client: Anthropic = None, openai_client: OpenAI = None
    ):
        self.anthropic = anthropic_client
        self.openai = openai_client

    @observe(as_type="generation")
    def create_message(
        self,
        provider: ModelProvider,
        messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        system_prompt: str,
        langfuse_session_id: str = None,
    ):
        """Wrapper method to route to the appropriate model provider."""
        if provider == ModelProvider.ANTHROPIC:
            return self._create_claude_message(
                messages, available_tools, system_prompt, langfuse_session_id
            )
        elif provider == ModelProvider.OPENAI:
            return self._create_openai_message(
                messages, available_tools, system_prompt, langfuse_session_id
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _create_claude_message(
        self, messages, available_tools, system_prompt, langfuse_session_id=None
    ):
        """Create a message using Claude API with the given messages and tools."""
        if not self.anthropic:
            raise ValueError("Anthropic client not initialized")

        # Add langfuse input tracking
        langfuse_context.update_current_observation(
            input=messages,
            model="claude-3-5-sonnet-20241022",
            session_id=langfuse_session_id,
        )

        response: Message = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=system_prompt,
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
                    "cache_read_input_tokens": response.usage.cache_read_input_tokens,
                }
            )

        return response

    def _create_openai_message(
        self, messages, available_tools, system_prompt, langfuse_session_id=None
    ):
        """Create a message using OpenAI API with the given messages and tools."""
        if not self.openai:
            raise ValueError("OpenAI client not initialized")

        # Add langfuse input tracking
        langfuse_context.update_current_observation(
            input=messages,
            model="gpt-4o",  # adjust as needed
            session_id=langfuse_session_id,
        )

        # Prepare messages with system prompt
        all_messages = [{"role": "system", "content": system_prompt}] + messages

        response: ChatCompletion = self.openai.chat.completions.create(
            model="gpt-4.1",
            messages=all_messages,
            tools=available_tools,
            tool_choice="auto",
        )

        if langfuse_session_id:
            langfuse_context.update_current_trace(session_id=langfuse_session_id)
            langfuse_context.flush()

            # Add cost tracking if available
            if hasattr(response, "usage"):
                langfuse_context.update_current_observation(
                    usage_details={
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                    }
                )

        return response
