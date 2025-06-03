from typing import Any

from anthropic import Anthropic
from anthropic.types.message import Message
from langfuse.decorators import langfuse_context, observe
from openai import NotGiven, OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from .arcade_utils import ModelProvider


class LLMMessageCreator:
    def __init__(
        self,
        anthropic_client: Anthropic | None = None,
        openai_client: OpenAI | None = None,
    ):
        self.anthropic = anthropic_client
        self.openai = openai_client

    @observe(as_type="generation")
    def create_message(
        self,
        provider: ModelProvider,
        messages: list[dict[str, Any]],
        available_tools: list[dict[str, Any]] | None,
        system_prompt: str,
        langfuse_data: dict[str, Any] | None = None,
        model: str | None = None,
    ):
        """Wrapper method to route to the appropriate model provider."""
        if provider == ModelProvider.ANTHROPIC:
            return self._create_claude_message(
                messages, available_tools, system_prompt, langfuse_data, model
            )
        elif provider == ModelProvider.OPENAI:
            return self._create_openai_message(
                messages, available_tools, system_prompt, langfuse_data, model
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _create_claude_message(
        self, messages, available_tools, system_prompt, langfuse_data=None, model=None
    ):
        """Create a message using Claude API with the given messages and tools."""
        if not self.anthropic:
            raise ValueError("Anthropic client not initialized")

        # Use provided model or default to claude-sonnet-4-20250514
        model_name = model or "claude-sonnet-4-20250514"

        # Add langfuse input tracking
        langfuse_context.update_current_observation(
            input=messages,
            model=model_name,
            session_id=langfuse_data["session_id"]
            if langfuse_data and "session_id" in langfuse_data
            else None,
        )

        # Prepare the API call parameters
        api_params = {
            "model": model_name,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
        }

        # Only add tools if they are provided and not None/empty
        if available_tools:
            api_params["tools"] = available_tools

        response: Message = self.anthropic.messages.create(**api_params)

        # if no session id is provided, doesn't flush to langfuse
        if (
            langfuse_data
            and "session_id" in langfuse_data
            and "user_id" in langfuse_data
        ):
            langfuse_context.update_current_trace(
                session_id=langfuse_data["session_id"], user_id=langfuse_data["user_id"]
            )
            langfuse_context.flush()

            if response.usage is not None:
                # Add cost tracking
                langfuse_context.update_current_observation(
                    usage_details={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                        "cache_read_input_tokens": response.usage.cache_read_input_tokens
                        or 0,
                    }
                )

        return response

    def _create_openai_message(
        self, messages, available_tools, system_prompt, langfuse_data=None, model=None
    ):
        """Create a message using OpenAI API with the given messages and tools."""
        if not self.openai:
            raise ValueError("OpenAI client not initialized")

        # Use provided model or default to gpt-4.1
        model_name = model or "gpt-4.1"

        # Add langfuse input tracking
        langfuse_context.update_current_observation(
            input=messages,
            model=model_name,
            session_id=langfuse_data["session_id"]
            if langfuse_data and "session_id" in langfuse_data
            else None,
        )

        # Prepare messages with system prompt
        all_messages = [{"role": "system", "content": system_prompt}] + messages

        response: ChatCompletion = self.openai.chat.completions.create(
            model=model_name,
            messages=all_messages,
            tools=available_tools,
            tool_choice="auto" if available_tools else NotGiven(),
        )

        if (
            langfuse_data
            and "session_id" in langfuse_data
            and "user_id" in langfuse_data
        ):
            langfuse_context.update_current_trace(
                session_id=langfuse_data["session_id"], user_id=langfuse_data["user_id"]
            )
            langfuse_context.flush()

            if hasattr(response, "usage") and response.usage is not None:
                langfuse_context.update_current_observation(
                    usage_details={
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                    }
                )

        return response

    def _parse_response_to_text(self, response, provider: ModelProvider) -> str:
        """
        Extract text content from an LLM response based on the provider.

        Args:
            response: The response from the LLM
            provider: The model provider (Anthropic or OpenAI)

        Returns:
            str: The extracted text or an error message if no text is found
        """
        try:
            if provider == ModelProvider.ANTHROPIC:
                # Check if response has content
                if response and hasattr(response, "content") and response.content:
                    # Look for text content in the response
                    for content in response.content:
                        if hasattr(content, "type") and content.type == "text":
                            return content.text
                        elif (
                            isinstance(content, dict) and content.get("type") == "text"
                        ):
                            return content.get("text", "")

                return "Error: No text content found in Anthropic response"

            elif provider == ModelProvider.OPENAI:
                if (
                    response
                    and hasattr(response, "choices")
                    and response.choices
                    and hasattr(response.choices[0], "message")
                ):
                    message = response.choices[0].message
                    if message.content:
                        return message.content

                return "Error: No text content found in OpenAI response"

            else:
                return f"Error: Unsupported provider {provider}"

        except Exception as e:
            return f"Error parsing response to text: {str(e)}"
