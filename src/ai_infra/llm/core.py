from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime
from langchain_core.messages import SystemMessage
from pydantic import BaseModel as PydanticModel  # for structured output

from ai_infra.llm.settings import ModelSettings
from ai_infra.llm.utils import validate_provider_and_model, build_model_key, initialize_model

load_dotenv()

@dataclass
class ToolCallControls:
    tool_choice: Optional[Dict[str, Any]] = None     # e.g. {"type":"tool","name":"search"}
    parallel_tool_calls: bool = True                 # False = serialize tool calls

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

    def _select_model(self, _state: Any, runtime: Runtime[ModelSettings]) -> Any:
        """Return a model bound with tools, using runtime.context (provider, model_name, tools, extra)."""
        ctx = runtime.context
        key = build_model_key(ctx.provider, ctx.model_name)
        if key not in self.models:
            self.set_model(ctx.provider, ctx.model_name, **(ctx.extra.get("model_kwargs", {}) if ctx.extra else {}))

        model = self.models[key]
        tools = ctx.tools if ctx.tools is not None else self.tools
        extra = ctx.extra or {}
        # If model doesn’t support tool calling, LangChain will no-op bind; that’s fine.  [oai_citation:6‡llms-full.md](file-service://file-Jbp6maycthGfk3mMHk2rQy)
        controls: ToolCallControls = extra.get("tool_controls") or ToolCallControls()
        return model.bind_tools(
            tools,
            tool_choice=controls.tool_choice,
            parallel_tool_calls=controls.parallel_tool_calls,
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