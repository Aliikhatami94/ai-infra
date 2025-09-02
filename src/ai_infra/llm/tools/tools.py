from typing import Callable, List, Any
from langchain_core.tools import StructuredTool


def tools_from_functions(functions: List[Callable[..., Any]]) -> List[StructuredTool]:
    """
    Given a list of functions (sync or async), return a list of StructuredTool objects.
    
    - Async functions -> passed with `coroutine=...`
    - Sync functions  -> passed with `func=...`
    """
    structured: List[StructuredTool] = []

    for fn in functions:
        if callable(fn):
            if callable(getattr(fn, "__call__", None)):
                # detect coroutine function
                if hasattr(fn, "__code__") and fn.__code__.co_flags & 0x80:
                    tool = StructuredTool.from_function(coroutine=fn)
                else:
                    tool = StructuredTool.from_function(func=fn)
                structured.append(tool)
            else:
                raise TypeError(f"Invalid tool {fn}: not callable")
        else:
            raise TypeError(f"Invalid tool {fn}: expected a function")

    return structured