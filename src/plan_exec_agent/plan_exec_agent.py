import json
import operator
from datetime import datetime
from typing import Annotated, List, Tuple, Dict, Any, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langfuse.decorators import observe
from .step_executor import StepExecutor
from .arcade_utils import ModelProvider
from .redis_publisher import RedisPublisher


class State(TypedDict):
    """
    The state of the plan execution.

    past_steps are contain the step executed and a summarized result of the step.
    past_results are contain the step executed and the raw result of the step, with the exception of tool calls.
    tool_results are contain the raw results of the tool calls, with the ID mapped to the tool call.

    NOTE: we may have to consolidate tool_results and past_results into a single dict so that the model doesn't get confused
    """

    input: str
    provider: ModelProvider
    inital_plan: List[str]
    current_plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    langfuse_session_id: str
    tool_results: Dict[str, Any]
    past_results: Annotated[List[Tuple], operator.add]
    user_id: str
    task_id: str
    status: str


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
        enabled_toolkits: List[str] = None,
    ):
        self.step_executor = StepExecutor(
            default_system_prompt, user_context, enabled_toolkits
        )
        self.redis_publisher = RedisPublisher()

    @observe()
    def initial_plan(self, state: State) -> Plan:
        """Generate an initial plan based on the user's query."""

        # hybrid of langgraph's prompt and our own
        plan_system_prompt = f"""
        You are a planning agent. Your task is to break down a complex query into a sequence of steps.
        Create a plan in the form of a list of steps that need to be executed to fulfill the user's request.
        The plan should be detailed enough that each step can be executed by a tool-using agent.

        Do not add any superfluous steps.
        The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

        IMPORTANT: Do not include irreversible write actions (sending emails, creating documents, posting messages, etc.) unless explicitly requested by the user. Focus on research, analysis, and information gathering steps. 
        Only add write actions if the user specifically asks for them. If the user explicitly asks for them, assume you have permission.

        USER CONTEXT:
        {self.step_executor.user_context}

        Use this context to inform your planning decisions. For example, prefer tools and approaches that align with the user's preferences and work environment.
        """

        plan_prompt = f"""
        Create a detailed plan to accomplish the following objective:
        
        {state["input"]}
        
        Use the submit_plan tool to provide your plan as a list of steps.
        Each step should be clear and actionable by an agent with access to tools.
        """

        # Get shared planning tools
        planning_tools = self.get_planning_tools(state)

        # Get available tools to inform the planning
        available_tools = self.step_executor.get_all_tools(state["provider"])
        state["tools"] = available_tools

        tool_descriptions = "\n".join(
            [
                f"- {name}: {description}"
                for name, description in [
                    self._get_tool_description(tool, state["provider"])
                    for tool in available_tools
                ]
            ]
        )
        plan_prompt += (
            f"\n\nYou can use the following tools in your plan:\n{tool_descriptions}"
        )

        messages = [{"role": "user", "content": plan_prompt}]

        # NOTE: we are not using the user-specific system prompt
        response = self.step_executor.message_creator.create_message(
            state["provider"],
            messages,
            [planning_tools["plan_tool"]],
            plan_system_prompt,
            {"session_id": state["langfuse_session_id"], "user_id": state["user_id"]},
        )

        steps = self._extract_plan_from_response(response, state["provider"])

        # If something went wrong
        if not steps:
            steps = ["Error: Could not generate plan"]

        return Plan(steps=steps)

    def _extract_plan_from_response(
        self, response, provider: ModelProvider
    ) -> List[str]:
        """Extract plan steps from either Anthropic or OpenAI response."""
        if provider == ModelProvider.ANTHROPIC:
            return self._extract_plan_anthropic(response)
        elif provider == ModelProvider.OPENAI:
            return self._extract_plan_openai(response)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _extract_plan_anthropic(self, response) -> List[str]:
        """Extract plan steps from Anthropic response format."""
        steps = []
        for content in response.content:
            if content.type == "tool_use" and content.name == "submit_plan":
                steps = content.input.get("plan", [])
                break

        # Fallback if no tool call was made
        if not steps and response.content and response.content[0].type == "text":
            print(
                "Warning: Plan was returned as text rather than tool call. Attempting to parse..."
            )
            response_text = response.content[0].text
            steps = self.extract_plan_from_response(response_text)

        return steps

    def _extract_plan_openai(self, response) -> List[str]:
        """Extract plan steps from OpenAI response format."""
        steps = []
        message = response.choices[0].message

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "submit_plan":
                    args = json.loads(tool_call.function.arguments)
                    steps = args.get("plan", [])
                    break

        # Fallback if no tool call was made
        if not steps and message.content:
            print(
                "Warning: Plan was returned as text rather than tool call. Attempting to parse..."
            )
            steps = self.extract_plan_from_response(message.content)

        return steps

    @observe()
    def execute_step(self, state: State) -> str:
        """Execute the first step from the current plan using the MCP clients."""

        # Get the current step from the state
        step = state["current_plan"][0]

        executor_system_prompt = f"""
        You are an execution agent tasked with carrying out a specific step in a plan.
        Your current task is to execute the following step: "{step}"
        
        You have access to tools to help you accomplish this task. You can use these tools to complete the step.
        You have access to the results of previous tool calls performed earlier in the plan. You can use this information to complete the step.
        You also have access to the results of previous steps. You can use this information to complete the step.
        Focus only on completing *this specific step* - do not attempt to execute other steps in the plan.
        
        IMPORTANT: If you retrieve data that will be needed by future steps (like email IDs, calendar events, etc.):
        1. Process the data completely in this step if that's what the step requires
        2. Provide a clear summary of what data you retrieved and how it's organized
        3. Begin your summary with "RESULT:" followed by what you accomplished

        For iterative tasks that process multiple items:
        - Make sure to process ALL items completely
        - Maintain a comprehensive summary that includes information from all items
        - Don't truncate or summarize too aggressively

        When an action requires specific identifiers (email addresses, user IDs, channel names, etc.), do not guess or make assumptions. Use available tools or previous step results to obtain the correct identifiers. 
        If you cannot reliably access the required identifier after attempting to retrieve it, mark this step as failed rather than proceeding with incorrect information.

        DEPENDENCY REQUIREMENTS: 
        - Before executing this step, check if it requires data from previous steps (summaries, lists, IDs, etc.)
        - If this step mentions creating content "containing" or "including" information, you MUST retrieve that specific information first
        - Look for phrases like "summary of", "list of", "based on", "including data from" - these indicate dependencies
        - If you cannot locate required data from previous steps or tool results, state "Missing required data from previous steps" and mark the step as failed
        """

        # Create context for the execution, including past steps
        past_steps_context = ""
        if "past_steps" in state and state["past_steps"]:
            past_steps_context = "## Previous steps that have been completed:\n"
            for i, (past_step, result) in enumerate(state["past_steps"]):
                past_steps_context += (
                    f"{i + 1}. Step: {past_step}\n   Result: {result}\n\n"
                )

        tool_results = state.get("tool_results")
        if tool_results is None:
            state["tool_results"] = {}

        tool_context = ""
        if "tool_results" in state and state["tool_results"]:
            tool_context = "## Data available from tool calls in previous steps:\n"
            for key, (tool_name, value) in state["tool_results"].items():
                if isinstance(value, list):
                    tool_context += f"Tool name: {tool_name} - ID {key} (use this to reference the tool call): {len(value)} items\n"
                else:
                    tool_context += f"Tool name: {tool_name} - ID {key} (use this to reference the tool call): Data available\n"

        objective_context = f"## Your objective:\n{state['input']}\n\n"
        plan_context = (
            f"## Overall plan:\n"
            + "\n".join([f"{i + 1}. {s}" for i, s in enumerate(state["current_plan"])])
            + "\n\n"
        )

        final_instructions = """
        EXECUTION CHECKLIST:
        1. DEPENDENCY CHECK: Does this step require specific data from previous steps? If yes, locate and reference that data before proceeding.
        2. Execute this step using the available tools.
        3. If this step creates content that should "contain" or "include" information from previous steps, ensure you retrieve and incorporate that specific information.
        4. If you cannot access required dependencies, explicitly state what data is missing and mark the step as failed.
        """
        
        execution_prompt = f"{objective_context}{plan_context}{past_steps_context}{tool_context}## Current step to execute:\n{step}\n\n{final_instructions}"

        result: List[str] = self.step_executor.process_input_with_agent_loop(
            execution_prompt,
            state["provider"],
            user_id=state["user_id"],
            system_prompt=executor_system_prompt,
            langfuse_session_id=state["langfuse_session_id"],
            state=state,
        )

        # append the `final_text` from the executor agent to the past_results
        state["past_results"].append((step, result))

        processed_result = self._summarize_step_result(state)
        return processed_result

    @observe()
    def replan(self, state: State) -> Act:
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
        
        CRITICAL: You can ONLY mark the task as complete (use submit_final_response) if the LAST STEP of the current plan was performed correctly and successfully. 
        If the last step of the plan has not been completed yet, you MUST continue with an updated plan.

        If the objective has been achieved AND the last step of the plan was completed successfully, use the submit_final_response tool.
        Otherwise, use the submit_plan tool with an updated plan of remaining steps.

        Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.
        IMPORTANT: When adding new steps, do not include irreversible write actions (sending emails, creating documents, posting messages, etc.) unless explicitly requested by the user.
        If the user explicitly asks for them, assume you have permission.

        If a critical step fails 3 times in a row, determine that the task is failed and use the submit_final_response tool to respond to the user.

        USER CONTEXT:
        {self.step_executor.user_context}

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
            tool_context = "## Data available from tool calls in previous steps:\n"
            for key, (tool_name, value) in state["tool_results"].items():
                if isinstance(value, list):
                    tool_context += f"Tool name: {tool_name} - ID {key} (use this to reference the tool call): {len(value)} items\n"
                else:
                    tool_context += f"Tool name: {tool_name} - ID {key} (use this to reference the tool call): Data available\n"

        """
        Add explicit tracking of last step vs last completed step. 
        This to prevent the model from marking the task as complete if the last step of the plan has not been completed yet.
        """
        step_tracking_context = ""
        if state["current_plan"]:
            last_planned_step = state["current_plan"][-1]
            step_tracking_context += f"## CRITICAL STEP TRACKING:\n"
            step_tracking_context += (
                f'- The LAST STEP of the current plan is: "{last_planned_step}"\n'
            )

            if state["past_steps"]:
                last_completed_step, last_completed_result = state["past_steps"][-1]
                step_tracking_context += (
                    f'- The LAST COMPLETED STEP was: "{last_completed_step}"\n'
                )
                step_tracking_context += f"- The result of the last completed step was: {last_completed_result}\n"

                # Check if the last completed step matches the last planned step
                if last_completed_step.strip() == last_planned_step.strip():
                    step_tracking_context += (
                        f"- ✅ The last step of the plan WAS completed successfully\n"
                    )
                    step_tracking_context += f"- You can now mark the task as complete if the objective has been achieved\n"
                else:
                    step_tracking_context += (
                        f"- ❌ The last step of the plan has NOT been completed yet\n"
                    )
                    step_tracking_context += (
                        f"- You MUST continue with the plan - do NOT mark as complete\n"
                    )
            else:
                step_tracking_context += f"- No steps have been completed yet\n"
                step_tracking_context += (
                    f"- You MUST continue with the plan - do NOT mark as complete\n"
                )

            step_tracking_context += "\n"

        # NOTE: the context window could get very large here
        replan_prompt = f"""
        ## Objective:
        {state["input"]}
        
        {current_plan_context}
        {past_steps_context}
        {step_tracking_context}
        {tool_context}
        
        Based on the progress so far, decide whether to:
        1. Continue with an updated plan (use submit_plan tool if there are still steps needed)
        2. Provide a final response (use submit_final_response tool ONLY if the objective has been achieved AND the last step of the plan was completed successfully)
        
        REMEMBER: You can only mark the task as complete if the last step of the current plan was performed correctly.
        """

        # Get shared planning tools
        planning_tools = self.get_planning_tools(state)
        replan_tools = [planning_tools["plan_tool"], planning_tools["response_tool"]]

        messages = [{"role": "user", "content": replan_prompt}]

        response = self.step_executor.message_creator.create_message(
            state["provider"],
            messages,
            replan_tools,
            replan_system_prompt,
            {"session_id": state["langfuse_session_id"], "user_id": state["user_id"]},
        )

        return self._process_replan_response(response, state)

    def _process_replan_response(self, response, state: State) -> Act:
        """Process replan response based on provider."""
        if state["provider"] == ModelProvider.ANTHROPIC:
            return self._process_replan_anthropic(response, state)
        elif state["provider"] == ModelProvider.OPENAI:
            return self._process_replan_openai(response, state)
        else:
            raise ValueError(f"Unsupported provider: {state['provider']}")

    def _process_replan_anthropic(self, response, state: State) -> Act:
        """Process Anthropic replan response."""
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
            return self._handle_text_replan_response(response.content[0].text, state)

        return Act(action=Plan(steps=state["current_plan"]))

    def _process_replan_openai(self, response, state: State) -> Act:
        """Process OpenAI replan response."""
        message = response.choices[0].message

        if message.tool_calls:
            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "submit_plan":
                    return Act(action=Plan(steps=args.get("plan", [])))
                elif tool_call.function.name == "submit_final_response":
                    return Act(action=Response(response=args.get("response", "")))

        # Fallback if no tool call was made
        if message.content:
            return self._handle_text_replan_response(message.content, state)

        return Act(action=Plan(steps=state["current_plan"]))

    def _handle_text_replan_response(self, response_text: str, state: State) -> Act:
        """Handle text response for replan (shared between providers)."""
        if (
            "objective has been achieved" in response_text.lower()
            or "final response" in response_text.lower()
        ):
            return Act(action=Response(response=response_text))
        else:
            steps = self.extract_plan_from_response(response_text)
            return Act(action=Plan(steps=steps))

    def extract_plan_from_response(self, response_text: str) -> List[str]:
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

    def get_planning_tools(self, state: State):
        """Return common tools used for planning and replanning."""
        if state["provider"] == ModelProvider.OPENAI:
            return {
                "plan_tool": {
                    "type": "function",
                    "function": {
                        "name": "submit_plan",
                        "description": "Submit a plan as a JSON array of strings, where each string is a step in the plan",
                        "parameters": {
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
                },
                "response_tool": {
                    "type": "function",
                    "function": {
                        "name": "submit_final_response",
                        "description": "Submit a final response to the user when the objective is achieved",
                        "parameters": {
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
                },
            }
        elif state["provider"] == ModelProvider.ANTHROPIC:
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
        else:
            raise ValueError(f"Unsupported provider: {state['provider']}")

    def _extract_final_result(self, text: str) -> str:
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

    def _get_tool_description(self, tool, provider: ModelProvider) -> Tuple[str, str]:
        """Extract name and description from a tool based on provider format.

        Args:
            tool: The tool object from either OpenAI or Anthropic
            provider: The model provider

        Returns:
            Tuple[str, str]: (tool_name, tool_description)
        """
        if provider == ModelProvider.OPENAI:
            # OpenAI tools have a nested function structure
            if "function" in tool:
                return (tool["function"]["name"], tool["function"]["description"])
            # Handle reference tool which might be structured differently
            elif "name" in tool:
                return (tool["name"], tool["description"])
        elif provider == ModelProvider.ANTHROPIC:
            # Anthropic tools have a flat structure
            return (tool["name"], tool["description"])

        raise ValueError(f"Unsupported provider: {provider}")

    def _summarize_step_result(self, state: State) -> str:
        """
        Call the LLM to summarize this result into 1-2 information rich sentences
        """
        last_step, last_result = state["past_results"][-1]

        # NOTE: can add in the user context if needed
        summarize_step_system_prompt = """
        You are a planning agent. 
        You are responsible for taking an input action from a user and breaking it down into a plan consisting of a series of actionable steps.
        You are also responsible for determining the success or failure of each step and analyzing the results to help determine what the next step should be.
        """

        # NOTE: can add in the user input if needed
        summarize_step_prompt = f"""
        You are given a step from a plan and the result of that step.
        First determined in that step 'FAILED' or 'SUCCEEDED'.
        Then summarize the result into 1-2 information rich sentences. Do not exceed 2 sentences.
        Include as much detail as possible.
        Make sure to include the following if applicable:
        - if tools were called, state they were called, and how many times, i.e. 'XYZ tool was called 3 times' 
        - if anything like a summary written or analysis was performed that is the result of that step
        
        If the step failed, include the reason it failed.
        If the step succeeded, include the details of the success. 

        ## Step:
        {last_step}

        ## Result:
        {last_result}
        
        ## Summary:
        """

        messages = [{"role": "user", "content": summarize_step_prompt}]

        response = self.step_executor.message_creator.create_message(
            state["provider"],
            messages,
            None,
            summarize_step_system_prompt,
            {"session_id": state["langfuse_session_id"], "user_id": state["user_id"]},
        )

        step_summary = self.step_executor.message_creator._parse_response_to_text(
            response, state["provider"]
        )
        return step_summary

    def _get_categorization_tools(self, provider: ModelProvider):
        """Return tools for task result categorization."""
        if provider == ModelProvider.OPENAI:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "categorize_task_result",
                        "description": "Categorize the task execution result as either completed or failed",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": ["completed", "failed"],
                                    "description": "The status of the task execution - 'completed' if successful, 'failed' if unsuccessful",
                                },
                                "rationale": {
                                    "type": "string",
                                    "description": "1-2 sentence rationale explaining why the task was categorized this way",
                                },
                            },
                            "required": ["status", "rationale"],
                        },
                    },
                }
            ]
        elif provider == ModelProvider.ANTHROPIC:
            return [
                {
                    "name": "categorize_task_result",
                    "description": "Categorize the task execution result as either completed or failed",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["completed", "failed"],
                                "description": "The status of the task execution - 'completed' if successful, 'failed' if unsuccessful",
                            },
                            "rationale": {
                                "type": "string",
                                "description": "1-2 sentence rationale explaining why the task was categorized this way",
                            },
                        },
                        "required": ["status", "rationale"],
                    },
                }
            ]
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _extract_categorization_from_response(
        self, response, provider: ModelProvider
    ) -> str:
        """Extract categorization result from either Anthropic or OpenAI response."""
        if provider == ModelProvider.ANTHROPIC:
            return self._extract_categorization_anthropic(response)
        elif provider == ModelProvider.OPENAI:
            return self._extract_categorization_openai(response)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _extract_categorization_anthropic(self, response) -> str:
        """Extract categorization from Anthropic response format."""
        for content in response.content:
            if content.type == "tool_use" and content.name == "categorize_task_result":
                status = content.input.get("status", "failed")
                rationale = content.input.get("rationale", "No rationale provided")
                print(f"Task categorization rationale: {rationale}")
                return status

        # Fallback if no tool call was made
        print(
            "Warning: Categorization was returned as text rather than tool call. Defaulting to 'failed'."
        )
        return "failed"

    def _extract_categorization_openai(self, response) -> str:
        """Extract categorization from OpenAI response format."""
        message = response.choices[0].message

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.function.name == "categorize_task_result":
                    args = json.loads(tool_call.function.arguments)
                    status = args.get("status", "failed")
                    rationale = args.get("rationale", "No rationale provided")
                    print(f"Task categorization rationale: {rationale}")
                    return status

        # Fallback if no tool call was made
        print(
            "Warning: Categorization was returned as text rather than tool call. Defaulting to 'failed'."
        )
        return "failed"

    def _categorize_task_result(self, state: State) -> str:
        """
        Categorize the task result into a category of 'completed' OR 'failed'.
        """
        categorize_task_result_system_prompt = """
        You are a planning agent. 
        You are responsible for taking an input action from a user and breaking it down into a plan consisting of a series of actionable steps.
        You are also responsible for determining the success or failure of the entire task / plan once it is done running.
        """

        categorize_task_result_prompt = f"""
        You are given a task carried out by a plan-and-execute agent, the result of the task, and the final response to the user.
        Categorize the task result using the categorize_task_result tool.

        Use 'completed' if the task was completed successfully.
        Use 'failed' if the task failed to complete.

        Return a 1-2 sentence rationale for your categorization.

        ## Task:
        {state["input"]}

        ## Result:
        {state["past_steps"]}

        ## Final Response:
        {state["response"]}

        Please categorize this task result using the categorize_task_result tool.
        """

        # Get categorization tools
        categorization_tools = self._get_categorization_tools(state["provider"])

        messages = [{"role": "user", "content": categorize_task_result_prompt}]

        response = self.step_executor.message_creator.create_message(
            state["provider"],
            messages,
            categorization_tools,
            categorize_task_result_system_prompt,
            {"session_id": state["langfuse_session_id"], "user_id": state["user_id"]},
        )

        # Extract the status from the tool call response
        status = self._extract_categorization_from_response(response, state["provider"])
        return status

    @observe(as_type="trace")
    def execute_plan(
        self,
        input_action: str,
        provider: ModelProvider = ModelProvider.ANTHROPIC,
        max_iterations: int = 25,
        user_id: str = "david_test",
        langfuse_session_id: str = None,
        task_id: str = None,
    ) -> str:
        """
        Execute a complete plan for the given query.

        This method orchestrates the entire plan-execute-replan cycle:
        1. Creates an initial plan based on the query
        2. Executes steps one by one
        3. Replans after each step. If a critical step in the plan fails 3 times in a row, the task is marked as failed.
        4. Returns the final response when done

        Args:
            query: The user's query to process
            max_iterations: Maximum number of execution steps to run

        Returns:
            The final response to the user's query
        """
        langfuse_session_id = langfuse_session_id or datetime.today().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Initialize state with values needed for the entire lifecycle
        state = {
            "input": input_action,
            "langfuse_session_id": langfuse_session_id,
            "past_steps": [],
            "past_results": [],
            "current_plan": [],
            "tool_results": {},
            "initial_plan": [],
            "response": "",
            "provider": provider,
            "user_id": user_id,
            "task_id": task_id,
        }

        # Step 1: Generate the initial plan
        print(f"Generating initial plan for query: {input_action}")
        plan = self.initial_plan(state)
        state["initial_plan"] = plan.steps
        state["current_plan"] = plan.steps.copy()
        print(f"Initial plan generated with {len(plan.steps)} steps")

        # Publish initial plan to Redis if enabled\
        if self.redis_publisher.is_enabled():
            self.redis_publisher.publish_event("initial_plan", state)

        state = self.execute_plan_until_completion(state, max_iterations)

        # Categorize the task result
        task_result = self._categorize_task_result(state)
        state["status"] = task_result

        # Publish final result to Redis if enabled
        if self.redis_publisher.is_enabled():
            self.redis_publisher.publish_event("final_result", state)

        # Return the final response
        return state["response"]

    def execute_plan_until_completion(self, state: State, max_iterations: int = 25) -> str:
        # Step 2-4: Execute steps, replan, and repeat
        iteration = 0

        while iteration < max_iterations and state["current_plan"]:
            iteration += 1
            print(f"\n==== Iteration {iteration}/{max_iterations} ====")

            # Execute the next step in the plan
            current_step = state["current_plan"][0]
            print(f"Executing step: {current_step}")

            result = self.execute_step(state)
            print(f"Step execution completed")

            # Update past_steps with the completed step and its result
            state["past_steps"].append((current_step, result))

            # If we still have steps left or just completed the last one, replan
            print("Replanning based on execution results")
            replan_result = self.replan(state)

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
                    {state["input"]}
                    
                    ## Steps completed:
                    """
                    for i, (step, result) in enumerate(state["past_steps"]):
                        final_response_prompt += (
                            f"{i + 1}. {step}\n   Result: {result}\n\n"
                        )

                    final_response_prompt += (
                        "Please provide a final summary of what was accomplished."
                    )

                    state["response"] = (
                        self.step_executor.process_input_with_agent_loop(
                            final_response_prompt,
                            state["provider"],
                            user_id=state["user_id"],
                            langfuse_session_id=state["langfuse_session_id"],
                        )
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
            {state["input"]}
            
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

            state["response"] = self.step_executor.process_input_with_agent_loop(
                incomplete_response_prompt,
                state["provider"],
                user_id=state["user_id"],
                langfuse_session_id=state["langfuse_session_id"],
            )

        return state