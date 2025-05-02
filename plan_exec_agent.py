import asyncio
import operator
from datetime import datetime
from typing import Annotated, List, Tuple, Dict, Any, Union
from host import MCPHost
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class State(TypedDict):
    input: str
    inital_plan: List[str]
    current_plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    langfuse_session_id: str


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
    def __init__(self, default_system_prompt: str = None):
        self.mcp_host = MCPHost(default_system_prompt)

    async def initialize_mcp_clients(self):
        await self.mcp_host.initialize_mcp_clients()
        
        # NOTE: can remove this if we are recyling mcp_host.process_query
        await self.mcp_host.get_all_tools_from_servers()

    async def initial_plan(self, state: State) -> Plan:
        """Generate an initial plan based on the user's query."""
        
        # hybrid of langgraph's prompt and our own
        plan_system_prompt = """
        You are a planning agent. Your task is to break down a complex query into a sequence of steps.
        Create a plan in the form of a list of steps that need to be executed to fulfill the user's request.
        The plan should be detailed enough that each step can be executed by a tool-using agent.

        Do not add any superfluous steps.
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
        """
        
        plan_prompt = f"""
        Create a detailed plan to accomplish the following objective:
        
        {state['input']}
        
        Use the submit_plan tool to provide your plan as a list of steps.
        Each step should be clear and actionable by an agent with access to tools.
        """

        # Get shared planning tools
        planning_tools = self.get_planning_tools()
        
        # Get available tools to inform the planning
        available_tools = await self.mcp_host.get_all_tools()
        tool_descriptions = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in available_tools])
        plan_prompt += f"\n\nYou can use the following tools in your plan:\n{tool_descriptions}"
        
        messages = [{"role": "user", "content": plan_prompt}]

        # NOTE: we are not using the user-specific system prompt 
        response = await self.mcp_host._create_claude_message(
            messages, [planning_tools["plan_tool"]], plan_system_prompt, state['langfuse_session_id']
        )
        self.mcp_host._log_claude_response(response)

        steps = []
        
        for content in response.content:
            if content.type == "tool_use" and content.name == "submit_plan":
                # Get the plan directly from the tool input
                steps = content.input.get("plan", [])
                break
        
        # Fallback if no tool call was made (should be rare with good prompting)
        if not steps and response.content and response.content[0].type == "text":
            print("Warning: Plan was returned as text rather than tool call. Attempting to parse...")
            response_text = response.content[0].text
            steps = self.extract_plan_from_response(response_text)
        
        # If something went wrong
        if not steps:
            steps = ["Error: Could not generate plan"]
        
        # Return a Plan object instead of a list
        return Plan(steps=steps)

    async def execute_step(self, state: Dict[str, Any], step: str) -> str:
        pass

    async def replan(self, state: State) -> Act:
        """
        Update the plan based on the results of previous steps.
        
        Args:
            state: The current state including input, initial plan, current plan, and past steps
        
        Returns:
            Act object with either a new Plan or a Response to the user
        """
        replan_system_prompt = """
        You are a planning agent. Your task is to revise an existing plan based on the results of steps that have already been executed.
        
        Evaluate whether:
        1. The plan needs to be modified based on new information
        2. Some steps should be removed or added
        3. The objective has been achieved
        
        If the objective has been achieved, use the submit_final_response tool.
        Otherwise, use the submit_plan tool with an updated plan of remaining steps.

        Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.
        """
        
        # Create context for replanning
        past_steps_context = "## Steps that have been completed:\n"
        for i, (past_step, result) in enumerate(state["past_steps"]):
            past_steps_context += f"{i+1}. Step: {past_step}\n   Result: {result}\n\n"
        
        # only use current plan, not initial plan
        current_plan_context = "## Current plan:\n" + "\n".join([f"{i+1}. {s}" for i, s in enumerate(state["current_plan"])]) + "\n\n"
        
        replan_prompt = f"""
        ## Objective:
        {state['input']}
        
        {current_plan_context}
        {past_steps_context}
        
        Based on the progress so far, decide whether to:
        1. Continue with an updated plan (use submit_plan tool if there are still steps needed)
        2. Provide a final response (use submit_final_response tool if the objective has been achieved)
        """
        
        # Get shared planning tools
        planning_tools = self.get_planning_tools()
        replan_tools = [planning_tools["plan_tool"], planning_tools["response_tool"]]
        
        messages = [{"role": "user", "content": replan_prompt}]
        
        response = await self.mcp_host._create_claude_message(
            messages, replan_tools, replan_system_prompt, state['langfuse_session_id']
        )
        self.mcp_host._log_claude_response(response)
        
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
            if "objective has been achieved" in response_text.lower() or "final response" in response_text.lower():
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
        json_pattern = r'(\[[\s\S]*?\]|\{[\s\S]*?\})'
        json_matches = re.findall(json_pattern, response_text)
        
        for json_str in json_matches:
            try:
                # Try parsing as JSON array
                parsed = json.loads(json_str)
                if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                    return parsed
                # Try parsing as JSON object with "steps" key
                elif isinstance(parsed, dict) and "steps" in parsed and isinstance(parsed["steps"], list):
                    return parsed["steps"]
            except json.JSONDecodeError:
                continue
        
        # If JSON parsing fails, look for markdown or numbered lists
        steps = []
        
        # Try to find markdown list items (- item or * item)
        markdown_pattern = r'[\-\*]\s*(.*?)(?=\n[\-\*]|\n\n|\n$|$)'
        markdown_matches = re.findall(markdown_pattern, response_text)
        if markdown_matches:
            return [step.strip() for step in markdown_matches if step.strip()]
        
        # Try to find numbered list items (1. item, 2. item, etc.)
        numbered_pattern = r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|\n$|$)'
        numbered_matches = re.findall(numbered_pattern, response_text)
        if numbered_matches:
            return [step.strip() for step in numbered_matches if step.strip()]
        
        # If all else fails, split by newlines and filter out empty lines
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
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
                            "description": "Array of strings where each string is a step in the plan"
                        }
                    },
                    "required": ["plan"]
                }
            },
            "response_tool": {
                "name": "submit_final_response",
                "description": "Submit a final response to the user when the objective is achieved",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "Final response to the user"
                        }
                    },
                    "required": ["response"]
                }
            }
        }


async def main():
    """
    This creates a sample daily briefing for today from my gmail and google calendar then writes it to a Notion database.
    Override these variables OR the `query` to customize the daily briefing to your liking.
    """
    # variables
    DATE = datetime.today().strftime("%Y-%m-%d")
    NOTION_PAGE_TITLE = "Daily Briefings"

    # you can provide the model with background information about yourself to personalize its responses
    system_prompt = f"""
    You are a helpful assistant. 
    I am David, the CTO / Co-Founder of a pre-seed startup based in San Francisco. 
    I handle all the coding and product development.
    We are a two person team, with my co-founder handling sales, marketing, and business development.
    
    When looking at my calendar, if you see anything titled 'b', that means it's a blocker.
    I often put blockers before or after calls that could go long.  
    """

    # Initialize host with default system prompt
    host = PlanExecAgent(default_system_prompt=system_prompt)

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
        3) Go to the 'Development Board V2' Notion page, look for any 'Scheduled' or 'In progress' cards assigned to me and give me a quick summary of every ticket. 
        4) Tell me if I have any unread messages on whatsapp. You should search at least the last 10 message threads. If not, say "No unread messages on whatsapp"
        5) Write the output from the above steps into a new page in my Notion in the '{NOTION_PAGE_TITLE}' page. Title the entry '{DATE}', which is today's date. 
        """
        # result = await host.process_query(query)
        # print(result)

        ### FOR TESTING
        state = {
            'input': query,
            'langfuse_session_id': datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        }

        plan = await host.initial_plan(state)
        print("####### INITIALPLAN #######")
        print(plan)
        state['inital_plan'] = plan.steps
        state['current_plan'] = plan.steps

        past_step = (plan.steps[0], "Operation failed")
        state['past_steps'] = [past_step]

        replan = await host.replan(state)
        print("####### REPLAN #######")
        print(replan)
        
    finally:
        await host.cleanup()


if __name__ == "__main__":
    asyncio.run(main())