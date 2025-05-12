import asyncio
import operator
from datetime import datetime
from typing import Annotated, List, Tuple, Dict, Any, Union
from host import MCPHost
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langfuse.decorators import observe, langfuse_context

DEFAULT_CLIENTS = [
    "Google Calendar",
    "Gmail",
    "Notion",
    "Whatsapp",
    "Exa",
    "Outlook",
    "Slack",
]


class State(TypedDict):
    input: str
    inital_plan: List[str]
    current_plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    langfuse_session_id: str
    tool_results: Dict[str, Any]


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


class PlanExecAgent:
    def __init__(
        self,
        default_system_prompt: str = None,
        user_context: str = None,
        enabled_clients: List[str] = None,
    ):
        self.mcp_host = MCPHost(default_system_prompt, user_context, enabled_clients)

    async def initialize_mcp_clients(self):
        await self.mcp_host.initialize_mcp_clients()

    @observe()
    async def initial_plan(self, state: State) -> Plan:
        """Generate an initial plan based on the user's query."""

        # hybrid of langgraph's prompt and our own
        plan_system_prompt = f"""
        You are a planning agent. Your task is to break down a complex query into a sequence of steps.
        Create a plan in the form of a list of steps that need to be executed to fulfill the user's request.
        The plan should be detailed enough that each step can be executed by a tool-using agent.

        Do not add any superfluous steps.
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

        USER CONTEXT:
        {self.mcp_host.user_context}

        Use this context to inform your planning decisions. For example, prefer tools and approaches that align with the user's preferences and work environment.
        """

        plan_prompt = f"""
        Create a detailed plan to accomplish the following objective:
        
        {state["input"]}
        
        Use the submit_plan tool to provide your plan as a list of steps.
        Each step should be clear and actionable by an agent with access to tools.
        """

        # Get shared planning tools
        planning_tools = self.get_planning_tools()

        # Get available tools to inform the planning
        available_tools = await self.mcp_host.get_all_tools()
        tool_descriptions = "\n".join(
            [f"- {tool['name']}: {tool['description']}" for tool in available_tools]
        )
        plan_prompt += (
            f"\n\nYou can use the following tools in your plan:\n{tool_descriptions}"
        )

        messages = [{"role": "user", "content": plan_prompt}]

        # NOTE: we are not using the user-specific system prompt
        response = await self.mcp_host._create_claude_message(
            messages,
            [planning_tools["plan_tool"]],
            plan_system_prompt,
            state["langfuse_session_id"],
        )

        steps = []

        for content in response.content:
            if content.type == "tool_use" and content.name == "submit_plan":
                # Get the plan directly from the tool input
                steps = content.input.get("plan", [])
                break

        # Fallback if no tool call was made (should be rare with good prompting)
        if not steps and response.content and response.content[0].type == "text":
            print(
                "Warning: Plan was returned as text rather than tool call. Attempting to parse..."
            )
            response_text = response.content[0].text
            steps = self.extract_plan_from_response(response_text)

        # If something went wrong
        if not steps:
            steps = ["Error: Could not generate plan"]

        # Return a Plan object instead of a list
        return Plan(steps=steps)

    @observe()
    async def execute_step(self, state: State) -> str:
        """Execute the first step from the current plan using the MCP clients."""

        # Get the current step from the state
        step = state["current_plan"][0]

        executor_system_prompt = f"""
        You are an execution agent tasked with carrying out a specific step in a plan.
        Your current task is to execute the following step: "{step}"
        
        You have access to tools to help you accomplish this task. Use these tools to complete the step.
        Focus only on completing this specific step - do not attempt to execute other steps in the plan.
        
        IMPORTANT: If you retrieve data that will be needed by future steps (like email IDs, calendar events, etc.):
        1. Process the data completely in this step if that's what the step requires
        2. Provide a clear summary of what data you retrieved and how it's organized
        3. Begin your summary with "RESULT:" followed by what you accomplished

        For iterative tasks that process multiple items:
        - Make sure to process ALL items completely
        - Maintain a comprehensive summary that includes information from all items
        - Don't truncate or summarize too aggressively
        """

        # TODO: not clear if we want to pass this context in
        # Create context for the execution, including past steps
        past_steps_context = ""
        if "past_steps" in state and state["past_steps"]:
            past_steps_context = "## Previous steps that have been completed:\n"
            for i, (past_step, result) in enumerate(state["past_steps"]):
                past_steps_context += (
                    f"{i + 1}. Step: {past_step}\n   Result: {result}\n\n"
                )

        objective_context = f"## Your objective:\n{state['input']}\n\n"
        plan_context = (
            f"## Overall plan:\n"
            + "\n".join([f"{i + 1}. {s}" for i, s in enumerate(state["current_plan"])])
            + "\n\n"
        )

        execution_prompt = f"{objective_context}{plan_context}{past_steps_context}## Current step to execute:\n{step}\n\nPlease execute this step using the available tools."

        tool_results = state.get("tool_results")
        if tool_results is None:
            state["tool_results"] = {}

        result = await self.mcp_host.process_input_with_agent_loop(
            execution_prompt,
            system_prompt=executor_system_prompt,
            langfuse_session_id=state["langfuse_session_id"],
            state=state,
        )

        # Extract the most relevant part of the result
        # NOTE: want want to remove if the performance is poor - too many tokens removed
        processed_result = self.extract_final_result(result)
        return processed_result

    @observe()
    async def replan(self, state: State) -> Act:
        """
        Update the plan based on the results of previous steps.

        Args:
            state: The current state including input, initial plan, current plan, and past steps

        Returns:
            Act object with either a new Plan or a Response to the user
        """
        replan_system_prompt = f"""
        You are a planning agent. Your task is to revise an existing plan based on the results of steps that have already been executed.
        
        Evaluate whether:
        1. The plan needs to be modified based on new information
        2. Some steps should be removed or added
        3. The objective has been achieved
        
        If the objective has been achieved, use the submit_final_response tool.
        Otherwise, use the submit_plan tool with an updated plan of remaining steps.

        Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.

        USER CONTEXT:
        {self.mcp_host.user_context}

        Use this context to inform your planning decisions. For example, prefer tools and approaches that align with the user's preferences and work environment.
        """

        # Create context for replanning
        past_steps_context = "## Steps that have been completed:\n"
        for i, (past_step, result) in enumerate(state["past_steps"]):
            past_steps_context += f"{i + 1}. Step: {past_step}\n   Result: {result}\n\n"

        # only use current plan, not initial plan
        current_plan_context = (
            "## Current plan:\n"
            + "\n".join([f"{i + 1}. {s}" for i, s in enumerate(state["current_plan"])])
            + "\n\n"
        )

        tool_context = ""
        if "tool_results" in state and state["tool_results"]:
            tool_context = "## Data available from previous steps:\n"
            for key, value in state["tool_results"].items():
                if isinstance(value, list):
                    tool_context += f"- {key}: {len(value)} items\n"
                else:
                    tool_context += f"- {key}: Data available\n"

        # NOTE: the context window could get very large here
        replan_prompt = f"""
        ## Objective:
        {state["input"]}
        
        {current_plan_context}
        {past_steps_context}
        {tool_context}
        
        Based on the progress so far, decide whether to:
        1. Continue with an updated plan (use submit_plan tool if there are still steps needed)
        2. Provide a final response (use submit_final_response tool if the objective has been achieved)
        """

        # Get shared planning tools
        planning_tools = self.get_planning_tools()
        replan_tools = [planning_tools["plan_tool"], planning_tools["response_tool"]]

        messages = [{"role": "user", "content": replan_prompt}]

        response = await self.mcp_host._create_claude_message(
            messages, replan_tools, replan_system_prompt, state["langfuse_session_id"]
        )

        # Process the response to determine the action
        for content in response.content:
            if content.type == "tool_use":
                if content.name == "submit_plan":
                    new_steps = content.input.get("plan", [])
                    return Act(action=Plan(steps=new_steps))
                elif content.name == "submit_final_response":
                    final_response = content.input.get("response", "")
                    return Act(action=Response(response=final_response))

        # Fallback if no tool call was made
        if response.content and response.content[0].type == "text":
            response_text = response.content[0].text
            # Check if it seems like a final response
            if (
                "objective has been achieved" in response_text.lower()
                or "final response" in response_text.lower()
            ):
                return Act(action=Response(response=response_text))
            else:
                # Try to extract steps
                steps = self.extract_plan_from_response(response_text)
                return Act(action=Plan(steps=steps))

        # Default fallback
        return Act(action=Plan(steps=state["current_plan"]))

    async def cleanup(self):
        """Clean up resources."""
        await self.mcp_host.cleanup()

    def extract_plan_from_response(response_text: str) -> List[str]:
        """
        Extract a plan (list of steps) from Claude's response text.
        Handles various formats including JSON, markdown lists, and numbered lists.

        Args:
            response_text: The text response from Claude

        Returns:
            List[str]: A list of steps extracted from the response
        """
        import json
        import re

        # First try to find and parse JSON
        json_pattern = r"(\[[\s\S]*?\]|\{[\s\S]*?\})"
        json_matches = re.findall(json_pattern, response_text)

        for json_str in json_matches:
            try:
                # Try parsing as JSON array
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and all(
                    isinstance(item, str) for item in parsed
                ):
                    return parsed
                # Try parsing as JSON object with "steps" key
                elif (
                    isinstance(parsed, dict)
                    and "steps" in parsed
                    and isinstance(parsed["steps"], list)
                ):
                    return parsed["steps"]
            except json.JSONDecodeError:
                continue

        # If JSON parsing fails, look for markdown or numbered lists
        steps = []

        # Try to find markdown list items (- item or * item)
        markdown_pattern = r"[\-\*]\s*(.*?)(?=\n[\-\*]|\n\n|\n$|$)"
        markdown_matches = re.findall(markdown_pattern, response_text)
        if markdown_matches:
            return [step.strip() for step in markdown_matches if step.strip()]

        # Try to find numbered list items (1. item, 2. item, etc.)
        numbered_pattern = r"\d+\.\s*(.*?)(?=\n\d+\.|\n\n|\n$|$)"
        numbered_matches = re.findall(numbered_pattern, response_text)
        if numbered_matches:
            return [step.strip() for step in numbered_matches if step.strip()]

        # If all else fails, split by newlines and filter out empty lines
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        if lines:
            return lines

        # If nothing worked, return empty list
        return []

    def get_planning_tools(self):
        """Return common tools used for planning and replanning."""
        return {
            "plan_tool": {
                "name": "submit_plan",
                "description": "Submit a plan as a JSON array of strings, where each string is a step in the plan",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of strings where each string is a step in the plan",
                        }
                    },
                    "required": ["plan"],
                },
            },
            "response_tool": {
                "name": "submit_final_response",
                "description": "Submit a final response to the user when the objective is achieved",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "Final response to the user",
                        }
                    },
                    "required": ["response"],
                },
            },
        }

    def extract_final_result(self, text: str) -> str:
        """
        Extract the final RESULT section from the text if present, otherwise return the original text.
        This is to eliminate the repetition of intermediate results added to the `final_text` array
        inside the mcp_host.process_query function.
        We care only about storing the final result of the step in the past_steps array, not the intermediate results.
        """
        import re

        # Find all RESULT: sections
        result_sections = re.findall(r"RESULT:(.*?)(?=RESULT:|$)", text, re.DOTALL)

        if result_sections:
            # Return the last (most complete) RESULT section
            return result_sections[-1].strip()
        else:
            # If no RESULT sections found, return the original text
            # but try to clean up a bit by removing tool call logs
            cleaned_text = re.sub(r"\[Calling tool.*?\]", "", text)
            return cleaned_text.strip()

    @observe(as_type="trace")
    async def execute_plan(self, query: str, max_iterations: int = 25) -> str:
        """
        Execute a complete plan for the given query.

        This method orchestrates the entire plan-execute-replan cycle:
        1. Creates an initial plan based on the query
        2. Executes steps one by one
        3. Replans after each step
        4. Returns the final response when done

        Args:
            query: The user's query to process
            max_iterations: Maximum number of execution steps to run

        Returns:
            The final response to the user's query
        """
        # Initialize state with values needed for the entire lifecycle
        state = {
            "input": query,
            "langfuse_session_id": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
            "past_steps": [],
            "current_plan": [],
            "tool_results": {},
            "initial_plan": [],
            "response": "",
        }

        # Step 1: Generate the initial plan
        print(f"Generating initial plan for query: {query}")
        plan = await self.initial_plan(state)
        state["initial_plan"] = plan.steps
        state["current_plan"] = plan.steps.copy()
        print(f"Initial plan generated with {len(plan.steps)} steps")

        # Step 2-4: Execute steps, replan, and repeat
        iteration = 0

        while iteration < max_iterations and state["current_plan"]:
            iteration += 1
            print(f"\n==== Iteration {iteration}/{max_iterations} ====")

            # Execute the next step in the plan
            current_step = state["current_plan"][0]
            print(f"Executing step: {current_step}")

            result = await self.execute_step(state)
            print(f"Step execution completed")

            # Update past_steps with the completed step and its result
            state["past_steps"].append((current_step, result))

            # If we still have steps left or just completed the last one, replan
            print("Replanning based on execution results")
            replan_result = await self.replan(state)

            # Process the replanning result
            if isinstance(replan_result.action, Response):
                # We have a final response, we're done
                print("Plan completed with final response")
                state["response"] = replan_result.action.response
                break
            elif isinstance(replan_result.action, Plan):
                # Update the entire plan with the new plan (not just removing completed step)
                print(
                    f"Plan updated, new step count: {len(replan_result.action.steps)}"
                )
                state["current_plan"] = replan_result.action.steps

                # If the updated plan is empty, we're done
                if not state["current_plan"]:
                    print("Plan completed (no more steps)")
                    # Generate a final response based on results
                    final_response_prompt = f"""
                    ## Objective:
                    {query}
                    
                    ## Steps completed:
                    """
                    for i, (step, result) in enumerate(state["past_steps"]):
                        final_response_prompt += (
                            f"{i + 1}. {step}\n   Result: {result}\n\n"
                        )

                    final_response_prompt += (
                        "Please provide a final summary of what was accomplished."
                    )

                    state[
                        "response"
                    ] = await self.mcp_host.process_input_with_agent_loop(
                        final_response_prompt,
                        langfuse_session_id=state["langfuse_session_id"],
                    )
                    break

        # Check if we hit the iteration limit
        if iteration >= max_iterations and state["current_plan"]:
            print(
                f"⚠️ Max iterations ({max_iterations}) reached without completing the plan"
            )

            # Generate a partial response
            incomplete_response_prompt = f"""
            ## Objective:
            {query}
            
            ## Steps completed ({iteration}/{max_iterations}, plan not completed):
            """
            for i, (step, result) in enumerate(state["past_steps"]):
                incomplete_response_prompt += (
                    f"{i + 1}. {step}\n   Result: {result}\n\n"
                )

            incomplete_response_prompt += (
                f"## Remaining steps ({len(state['current_plan'])} steps):\n"
            )
            for i, step in enumerate(state["current_plan"]):
                incomplete_response_prompt += f"{i + 1}. {step}\n"

            incomplete_response_prompt += "\nPlease provide a summary of progress made and what remains to be done."

            state["response"] = await self.mcp_host.process_input_with_agent_loop(
                incomplete_response_prompt,
                langfuse_session_id=state["langfuse_session_id"],
            )

        # Return the final response
        return state["response"]


async def main():
    """
    This creates a sample daily briefing for today from my gmail and google calendar then writes it to a Notion database.
    Configuration can be customized in user_inputs.py, or will use defaults if not found.
    """
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

    QUERY = f"""
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
        import user_inputs

        # Override each value individually if it exists in user_inputs
        if hasattr(user_inputs, "QUERY"):
            QUERY = user_inputs.QUERY
        if hasattr(user_inputs, "BASE_SYSTEM_PROMPT"):
            BASE_SYSTEM_PROMPT = user_inputs.BASE_SYSTEM_PROMPT
        if hasattr(user_inputs, "USER_CONTEXT"):
            USER_CONTEXT = user_inputs.USER_CONTEXT
        if hasattr(user_inputs, "ENABLED_CLIENTS"):
            ENABLED_CLIENTS = user_inputs.ENABLED_CLIENTS
            print(
                f"System will run with only the following clients:\n{ENABLED_CLIENTS}\n\n"
            )
        else:
            ENABLED_CLIENTS = DEFAULT_CLIENTS
    except ImportError:
        print("Unable to load values from user_inputs.py found, using default values")
        ENABLED_CLIENTS = DEFAULT_CLIENTS
    # Initialize host with system prompt and user context
    host = PlanExecAgent(
        default_system_prompt=BASE_SYSTEM_PROMPT,
        user_context=USER_CONTEXT,
        enabled_clients=ENABLED_CLIENTS,
    )

    try:
        await host.initialize_mcp_clients()
        result = await host.execute_plan(QUERY)
        print(result)
    finally:
        await host.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
