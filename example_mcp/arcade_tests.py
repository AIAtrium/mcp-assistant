import os

from anthropic import Anthropic
from arcadepy import Arcade
from arcadepy.types import ExecuteToolResponse
from openai import OpenAI

from mcp_assistant.errors import EmptyOutput
from mcp_assistant.utils import tool_input_from_tool_args


def test_tool_call_with_llm():
    client = OpenAI(
        base_url="https://api.arcade.dev/v1", api_key=os.getenv("ARCADE_API_KEY")
    )

    # match this to the user on our end
    # can be the user_id from our database. arcade will link this to the user on their end to load the credentials
    user_id = "fake@atriumlabs.dev"

    # Determine which tools will be available to the chatbot agent
    tools = ["Google.ListEmails"]

    # Ask the user for input
    prompt = input("Enter your prompt (type 'exit' to quit):")

    # Let the LLM choose a tool to use based on the user's prompt
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can interact with tools.",
            },
            {"role": "user", "content": prompt},
        ],
        model="claude-3-5-sonnet-20240620",
        user=user_id,
        # TODO: I am surprised this works
        # it should be a list of idcts with tools defintiions
        tools=tools,  # type: ignore
        # gives the actual JSON
        # TODO: I am surpirsed this worksit should be one of
        # Literal["none", "auto", "required"] though)
        tool_choice="execute",  # type: ignore
    )

    return response.choices[0].message.content


"""
The user will see the Arcade logo on the OAuth flow unless we set up our own OAuth clients
Arcade supports us using our own OAuth clients -> branded flows

They have integrations that work but not that many. 
Gmail, Calendar, Notion, Outlook, Slack, Github
Salesforce is in development.
Whatsapp is in development.
No Linear integration.

They also recommend that you not use MCP yet. They are building an MCP gateway.

'tool_choice': 'generate' -> this will give you a natural language description of the tool call rather than the JSON you need to parse

they support direct tool calling so you don't have to call the tool through the LLM
can use arcade's api just for the tool calling

The console will show you the redirect url that you click on to then authorize the application, like gmail, in the UI
"""


def test_direct_tool_call():
    """
    You need to call the LLM first to have it give you the tool call parameters
    then you can call the tool directly with the Arcade client

    https://docs.arcade.dev/home/use-tools/get-tool-definitions
    """
    client = Arcade(api_key=os.getenv("ARCADE_API_KEY"))

    user_input = "Who has emailed me today?"
    TOOL_NAME = "Google.ListEmails"

    # make sure to toggle the provider
    tool_definition = client.tools.formatted.get(name=TOOL_NAME, format="anthropic")
    print(tool_definition)

    def create_claude_message(messages, available_tools):
        """Create a message using Claude API with the given messages and tools."""
        system = "You are a helpful assistant that can interact with tools."

        anthropic = Anthropic()
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=available_tools,
        )
        return response

    llm_response = create_claude_message(
        messages=[{"role": "user", "content": user_input}],
        available_tools=[tool_definition],
    )
    print(llm_response)

    if llm_response.content[0].type == "text":
        print("LLM response is text")
        print(llm_response.content[0].text)
    if llm_response.content[1].type == "tool_use":
        print("LLM response is tool use")
        content = llm_response.content[1]
        extracted_tool_name = content.name
        tool_args = content.input
        tool_id = content.id
        print(
            f"Calling tool: {extracted_tool_name} with args: {tool_args} and id: {tool_id}"
        )

        # match this to the user on our end
        # can be the user_id from our database. arcade will link this to the user on their end to load the credentials
        user_id = "fake@atriumlabs.dev"

        auth_response = client.tools.authorize(
            tool_name=extracted_tool_name,
            user_id=user_id,
        )

        if auth_response.status != "completed":
            print(f"Click this link to authorize: {auth_response.url}")

        # Wait for the authorization to complete
        client.auth.wait_for_completion(auth_response)

        tool_input = tool_input_from_tool_args(tool_args)

        response: ExecuteToolResponse = client.tools.execute(
            tool_name=extracted_tool_name,
            input=tool_input,
            user_id=user_id,
        )

        print(f"Tool execution status: {response.status}")

        output = response.output
        if not output:
            raise EmptyOutput
        if output and output.error:
            print(f"Tool execution error: {output.error.message}")
        elif isinstance(output.value, dict):
            for key, value in output.value.items():
                print(f"{key}: {value}")
        else:
            print(f"Tool execution result: {output.value}")


if __name__ == "__main__":
    test_direct_tool_call()
