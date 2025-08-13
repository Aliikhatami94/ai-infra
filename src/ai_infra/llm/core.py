from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime
from langchain_core.messages import SystemMessage
from pydantic import BaseModel as PydanticModel  # for structured output
from dataclasses import is_dataclass, asdict

from ai_infra.llm.providers import Providers
from ai_infra.llm.settings import ModelSettings
from ai_infra.llm.utils import validate_provider_and_model, build_model_key, initialize_model

load_dotenv()

@dataclass
class ToolCallControls:
    tool_choice: Optional[Dict[str, Any]] = None     # e.g. {"type":"tool","name":"search"}
    parallel_tool_calls: bool = True                 # False = serialize tool calls
    force_once: bool = False                          # If True, only call tool once per run, even if not used

class CoreLLM:
    """
    Dynamic LLM + agent setup across providers with tool controls, streaming, structured output, and fallbacks.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {}   # cached chat models
        self.tools: List[Any] = []         # default tools

    # ---------- Model management ----------
    def set_model(self, provider: str, model_name: str, **kwargs):
        validate_provider_and_model(provider, model_name)
        key = build_model_key(provider, model_name)
        if key not in self.models:
            self.models[key] = initialize_model(key, provider, **(kwargs or {}))
        return self.models[key]

    def _get_or_create(self, provider: str, model_name: str, **kwargs):
        """Internal, used by fallbacks."""
        try:
            return self.set_model(provider, model_name, **kwargs)
        except Exception:
            # let caller handle fallback decision
            raise

    def _normalize_tool_controls(self, provider: str, controls: Any):
        # defaults
        tool_choice, parallel_tool_calls, force_once = None, True, False

        if controls is None:
            return tool_choice, parallel_tool_calls, force_once

        if is_dataclass(controls):
            controls = asdict(controls)

        if isinstance(controls, dict):
            tool_choice = controls.get("tool_choice")
            parallel_tool_calls = controls.get("parallel_tool_calls", True)
            force_once = bool(controls.get("force_once", False))

        # Allow simple strings everywhere ("none", "auto", "any")
        if isinstance(tool_choice, str):
            # passthrough; providers understand these tags
            return tool_choice, parallel_tool_calls, force_once

        # If caller passed {"name": "..."} normalize per provider
        if isinstance(tool_choice, dict):
            name = (
                    tool_choice.get("name")
                    or (tool_choice.get("function") or {}).get("name")
            )

            if provider in (Providers.openai, Providers.xai):
                # OpenAI expects {"type":"function","function":{"name": "<name>"}}
                if name:
                    tool_choice = {"type": "function", "function": {"name": name}}
                # also allow "none"/"auto" strings above

            elif provider == Providers.anthropic:
                # Anthropic expects {"type":"tool","name":"<name>"} or "none"/"auto"/"any"
                if name:
                    tool_choice = {"type": "tool", "name": name}
                # Convert any OpenAI-style input just in case
                if tool_choice.get("type") == "function" and "function" in tool_choice:
                    fn = (tool_choice["function"] or {}).get("name")
                    tool_choice = {"type": "tool", "name": fn} if fn else {"type": "any"}

            # Other providers: pass-through

        return tool_choice, parallel_tool_calls, force_once

    def _tool_used_already(self, state: Any) -> bool:
        """Heuristic: check if any AIMessage includes tool_calls OR any ToolMessage exists."""
        msgs = state.get("messages", []) if isinstance(state, dict) else getattr(state, "get", lambda *_: [])("messages", [])
        for m in reversed(msgs):
            # LangChain message objects
            if hasattr(m, "tool_calls") and m.tool_calls:
                return True
            if getattr(m, "type", None) == "tool":
                return True
            # dict-like form (defensive)
            if isinstance(m, dict):
                if m.get("tool_calls"):
                    return True
                if m.get("type") == "tool":
                    return True
        return False

    def _select_model(self, state: Any, runtime: Runtime[ModelSettings]) -> Any:
        ctx = runtime.context
        key = build_model_key(ctx.provider, ctx.model_name)
        if key not in self.models:
            self.set_model(ctx.provider, ctx.model_name, **(ctx.extra.get("model_kwargs", {}) if ctx.extra else {}))

        model = self.models[key]
        tools = ctx.tools if ctx.tools is not None else self.tools
        extra = ctx.extra or {}

        tool_choice, parallel_tool_calls, force_once = self._normalize_tool_controls(ctx.provider, extra.get("tool_controls"))

        # If forcing only once and we already used a tool, stop forcing
        if force_once and self._tool_used_already(state):
            tool_choice = None

        return model.bind_tools(
            tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

    # ---------- Agent builders ----------
    def create_agent(self, tools: Optional[List[Any]] = None) -> Any:
        agent_tools = tools if tools is not None else self.tools
        return create_react_agent(model=self._select_model, tools=agent_tools)

    def _prepare_agent(self, provider, model_name, tools=None, extra=None, model_kwargs=None):
        # ensure model exists and stash kwargs for _select_model
        model_kwargs = model_kwargs or {}
        self.set_model(provider, model_name, **model_kwargs)
        context = ModelSettings(
            provider=provider,
            model_name=model_name,
            tools=tools,
            extra={"model_kwargs": model_kwargs, **(extra or {})},
        )
        agent = self.create_agent(tools)
        return agent, context

    # ---------- Invocation ----------
    async def arun_agent(
            self,
            messages: List[Dict[str, Any]],
            provider: str,
            model_name: str,
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        agent, context = self._prepare_agent(provider, model_name, tools, extra, model_kwargs)
        return await agent.ainvoke({"messages": messages}, context=context)

    def run_agent(
            self,
            messages: List[Dict[str, Any]],
            provider: str,
            model_name: str,
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        agent, context = self._prepare_agent(provider, model_name, tools, extra, model_kwargs)
        return agent.invoke({"messages": messages}, context=context)

    # ---------- Streaming ----------
    async def arun_agent_stream(
            self,
            messages: List[Dict[str, Any]],
            provider: str,
            model_name: str,
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None,
            stream_mode: Union[str, List[str]] = ("updates", "values"),
    ):
        agent, context = self._prepare_agent(provider, model_name, tools, extra, model_kwargs)
        # normalize
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)

        # If caller wants LLM token streaming, pass-through (token, metadata) as per docs.
        # Otherwise, yield (mode, chunk) like your graph streaming.
        if modes == ["messages"] or (len(modes) == 1 and modes[0] == "messages"):
            async for token, metadata in agent.astream({"messages": messages}, context=context, stream_mode="messages"):
                yield token, metadata
            return

        async for mode, chunk in agent.astream({"messages": messages}, context=context, stream_mode=modes):
            yield mode, chunk

    # ---------- Structured output ----------
    def with_structured_output(self, provider: str, model_name: str, schema: Union[type[PydanticModel], Dict[str, Any]], **model_kwargs):
        """
        Return a *model instance* configured to emit structured JSON per schema.
        Use this to call .invoke directly (outside the prebuilt agent), or to plug into a custom node.
        """
        model = self._get_or_create(provider, model_name, **model_kwargs)
        try:
            # pydantic model or JSON schema dict
            return model.with_structured_output(schema)
        except Exception:
            # If provider/model doesn’t support it, return the base model (caller can validate post-hoc)
            return model

    # ---------- Helpers ----------
    @staticmethod
    def make_messages(user: str, system: Optional[str] = None, extras: Optional[List[Dict[str, Any]]] = None):
        """Quick helper to build message arrays with optional system prelude."""
        msgs: List[Dict[str, Any]] = []
        if system:
            msgs.append(SystemMessage(content=system).dict())
        msgs.append({"role": "user", "content": user})
        if extras:
            msgs.extend(extras)
        return msgs

    def run_with_fallbacks(
            self,
            messages: List[Dict[str, Any]],
            candidates: List[Tuple[str, str]],   # [(provider, model_name), ...] in priority order
            tools: Optional[List[Any]] = None,
            extra: Optional[Dict[str, Any]] = None,
            model_kwargs: Optional[Dict[str, Any]] = None
    ):
        last_err = None
        for provider, model_name in candidates:
            try:
                return self.run_agent(messages, provider, model_name, tools, extra, model_kwargs)
            except Exception as e:
                last_err = e
                continue
        raise last_err or RuntimeError("All fallbacks failed")