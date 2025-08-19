import os
import inspect
from typing import List
from langchain_core.messages import SystemMessage


def resolve_arg_path(filename: str) -> str:
    for frame_info in inspect.stack():
        if os.path.abspath(frame_info.filename) != os.path.abspath(__file__):
            caller_file = frame_info.filename
            break
    else:
        caller_file = __file__
    caller_dir = os.path.dirname(os.path.abspath(caller_file))
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    rel_path = os.path.abspath(os.path.join(caller_dir, filename))
    if os.path.exists(rel_path):
        return rel_path
    for root, dirs, files in os.walk(caller_dir):
        if os.path.basename(filename) in files:
            return os.path.abspath(os.path.join(root, os.path.basename(filename)))
    raise FileNotFoundError(f"Could not find file: {filename} (checked as absolute, relative to {caller_dir}, and recursively)")