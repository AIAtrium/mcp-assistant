from typing import Any

from errors import InvalidToolArgsType


def tool_input_from_tool_args(tool_args: Any) -> Any:
    if isinstance(tool_args, dict):
        return tool_args
    elif isinstance(tool_args, str):
        return eval(tool_args)
    else:
        raise InvalidToolArgsType
