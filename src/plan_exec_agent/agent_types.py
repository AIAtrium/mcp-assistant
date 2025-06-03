from typing import Annotated, Any, Dict, List, Union

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .arcade_utils import ModelProvider
import operator


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
    past_steps: Annotated[list[tuple], operator.add]
    response: str
    langfuse_session_id: str
    tool_results: Dict[str, Any]
    past_results: Annotated[list[tuple], operator.add]
    initial_plan: list
    user_id: str
    task_id: str
    status: str
    tools: list
    published_at: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class AgentUserResponse(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[AgentUserResponse, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )
