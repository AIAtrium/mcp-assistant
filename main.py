from datetime import datetime

from src.plan_exec_agent import ModelProvider, PlanExecAgent, StepExecutor

# deliebrately omit Github and Microsoft during testing
DEFAULT_TOOLKITS = ["google", "slack", "NotionToolkit", "Exa"]

# NOTE: these are Default values you can override in user_inputs.py
DATE = datetime.today().strftime("%Y-%m-%d")
NOTION_PAGE_TITLE = "Daily Briefings"

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
3) Go to the 'Development Board V2' Notion page, look for any 'Scheduled' or 'In progress' cards assigned to me and give me a quick summary of every ticket. 
4) Tell me if I have any unread messages on whatsapp. You should search at least the last 10 message threads. If not, say "No unread messages on whatsapp"
5) Write the output from the above steps into a new page in my Notion in the '{NOTION_PAGE_TITLE}' page. Title the entry '{DATE}', which is today's date. 
"""

# Try to import user configurations, override defaults if found
try:
    print("Loading values from user_inputs.py")
    import user_inputs  # pyright: ignore

    # Override each value individually if it exists in user_inputs
    if hasattr(user_inputs, "INPUT_ACTION"):
        INPUT_ACTION = user_inputs.INPUT_ACTION
    if hasattr(user_inputs, "BASE_SYSTEM_PROMPT"):
        BASE_SYSTEM_PROMPT = user_inputs.BASE_SYSTEM_PROMPT
    if hasattr(user_inputs, "USER_CONTEXT"):
        USER_CONTEXT = user_inputs.USER_CONTEXT
    if hasattr(user_inputs, "ENABLED_TOOLKITS"):
        ENABLED_TOOLKITS = user_inputs.ENABLED_TOOLKITS
        print(
            f"System will run with only the following toolkits:\n{ENABLED_TOOLKITS}\n\n"
        )
    else:
        ENABLED_TOOLKITS = DEFAULT_TOOLKITS
except ImportError:
    print("Unable to load values from user_inputs.py found, using default values")
    ENABLED_TOOLKITS = DEFAULT_TOOLKITS


def main():
    """
    This creates a sample daily briefing for today from my gmail and google calendar then writes it to a Notion database.
    Configuration can be customized in user_inputs.py, or will use defaults if not found.
    """
    # Initialize host with system prompt and user context
    host = PlanExecAgent(
        default_system_prompt=BASE_SYSTEM_PROMPT,
        user_context=USER_CONTEXT,
        enabled_toolkits=ENABLED_TOOLKITS,
    )

    print(f"INPUT_ACTION: {INPUT_ACTION}")

    result = host.execute_plan(
        INPUT_ACTION, provider=ModelProvider.OPENAI, task_id="testestesteststse"
    )
    print(result)


def step_executor():
    """
    This uses one agentic loop to execute the input action.
    This creates a sample daily briefing for today from my gmail and google calendar then writes it to a Notion database.
    Configuration can be customized in user_inputs.py, or will use defaults if not found.
    """
    langfuse_session_id = (
        f"step_executor_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print(f"INPUT_ACTION: {INPUT_ACTION}")

    # Initialize host with default system prompt and enabled clients
    executor = StepExecutor(
        default_system_prompt=BASE_SYSTEM_PROMPT,
        user_context=USER_CONTEXT,
        enabled_toolkits=ENABLED_TOOLKITS,
    )

    result = executor.process_input_with_agent_loop(
        input_action=INPUT_ACTION,
        provider=ModelProvider.OPENAI,
        user_id="david_test",
        langfuse_session_id=langfuse_session_id,
    )
    print(result)


if __name__ == "__main__":
    main()
