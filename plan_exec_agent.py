import asyncio
from datetime import datetime
from typing import List
from host import MCPHost
import operator
from typing import Annotated, List, Tuple, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class PlanExecAgent:
    def __init__(self, default_system_prompt: str = None):
        self.mcp_host = MCPHost(default_system_prompt)

    async def initialize_mcp_clients(self):
        await self.mcp_host.initialize_mcp_clients()
        
        # NOTE: can remove this if we are recyling mcp_host.process_query
        await self.mcp_host.get_all_tools_from_servers()

    async def generate_initial_plan(self, query: str, langfuse_session_id: str = None) -> List[str]:
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
        
        {query}
        
        Return ONLY a JSON array of strings, where each string is a step in the plan of the form:
        {{
            "steps": ["step1", "step2", "step3"]
        }}

        Return nothing else.
        """

        json_plan_tool = {
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
        }

        # Get available tools to inform the planning
        available_tools = await self.mcp_host.get_all_tools()
        tool_descriptions = "\n".join([f"- {tool['name']}: {tool['description']}" for tool in available_tools])
        plan_prompt += f"\n\nYou can use the following tools in your plan:\n{tool_descriptions}"
        
        messages = [{"role": "user", "content": plan_prompt}]

        # NOTE: we are not using the user-specific system prompt 
        response = await self.mcp_host._create_claude_message(
            messages, [json_plan_tool], plan_system_prompt, langfuse_session_id
        )
        self.mcp_host._log_claude_response(response)

        for content in response.content:
            if content.type == "tool_use" and content.name == "submit_plan":
                # Get the plan directly from the tool input
                return content.input.get("plan", [])
        
        # Fallback if no tool call was made (should be rare with good prompting)
        if response.content and response.content[0].type == "text":
            print("Warning: Plan was returned as text rather than tool call. Attempting to parse...")
            response_text = response.content[0].text
            return self.extract_plan_from_response(response_text)
        
        # If something went wrong
        return ["Error: Could not generate plan"]


    async def execute_step(self, state: Dict[str, Any], step: str) -> str:
        pass

    async def replan(self, plan: List[str]):
        pass
    
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
        plan = await host.generate_initial_plan(query)
        print(plan)
    finally:
        await host.cleanup()


if __name__ == "__main__":
    asyncio.run(main())