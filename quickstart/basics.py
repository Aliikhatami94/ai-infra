import asyncio
from langchain_core.tools import tool

from ai_infra.llm import CoreLLM, Providers, Models, CoreAgent, BaseLLMCore
from ai_infra.llm.tool_controls import ToolCallControls

base = BaseLLMCore()
llm = CoreLLM()
agent = CoreAgent()

def test_agent():
    res = agent.run_agent(
        messages=[{"role": "user", "content": "What is your name?"}],
        provider=Providers.google_genai,
        model_name=Models.google_genai.gemini_2_5_flash.value,
        model_kwargs={
            "temperature": 0.7,
        }
    )
    print(res)

def test_llm():
    res = llm.chat(
        user_msg="What is your name?",
        system="Your name is Alex and you are a helpful assistant.",
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
    )
    print(res)

def test_structured_output():
    from pydantic import BaseModel, Field
    class UserInfo(BaseModel):
        name: str = Field(..., description="The user's full name")
        age: int = Field(..., description="The user's age in years")
        email: str = Field(..., description="The user's email address")

    structured = llm.with_structured_output(
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
        schema=UserInfo
    )
    res = structured.invoke([{"role": "user", "content": "My name is John Doe, I am 30 years old, and my email is johndoe@gmail.com"}])
    print(res)

async def test_agent_stream():
    async for token, meta in agent.arun_agent_stream(
            messages=[{"role":"user","content":"Write me 3 paragraphs about AI."}],
            provider=Providers.openai,
            model_name=Models.openai.gpt_4_1_mini.value,
            model_kwargs={"temperature":0.7},
            stream_mode="messages",
    ):
        print(token.content, end="", flush=True)

def controlled_tool_agent():
    @tool()
    def dummy_weather_tool1(query: str):
        """A dummy weather tool that simulates a weather search."""
        return f"Search results for '{query}': It is always sunny and 85 degrees."

    @tool()
    def dummy_weather_tool2(query: str):
        """Another dummy weather tool that simulates a weather search."""
        return f"Weather in {query}: It is always rainy and 60 degrees."

    msgs = [{"role": "user", "content": "How is the weather in New York?"}]
    res = agent.run_agent(
        msgs,
        Providers.google_genai,
        Models.google_genai.gemini_2_5_flash.value,
        tools=[dummy_weather_tool1, dummy_weather_tool2],
        tool_controls=ToolCallControls(
            # can be {"name": "dummy_weather_tool1"}; normalizer will map to {"type":"tool","name":...}
            tool_choice={"name": "dummy_weather_tool1"},
            parallel_tool_calls=False,
            force_once=True,
        ),
        extra={"recursion_limit": 8},
    )
    print(res)

def human_in_the_loop():
    @tool
    def get_weather(city: str) -> str:
        """Return a short weather string for a city."""
        return f"Weather in {city}: sunny, 85°F"

    @tool
    def get_news(topic: str) -> str:
        """Return a headline for a topic."""
        return f"Top headline about {topic}: ..."

    def reviewer(name, args):
        print(f"\nAgent wants to call tool: {name} with args: {args}")
        ans = input("Approve? (y/n/m=modify): ").strip().lower()
        if ans == "y":
            return {"action": "pass"}
        if ans == "m":
            # quick edit flow
            new_args = dict(args or {})
            for k in list(new_args.keys()):
                nv = input(f"New value for {k} (blank=keep): ").strip()
                if nv:
                    new_args[k] = nv
            return {"action": "modify", "args": new_args}
        return {"action": "block", "replacement": "[blocked by reviewer]"}

    base.set_hitl(on_tool_call=reviewer)

    messages = [{"role": "user", "content": "What's the weather in New York?"}]
    resp = agent.run_agent(
        messages=messages,
        provider="openai",
        model_name="gpt-5-mini",
        tools=[get_weather, get_news],   # <- our wrapper will gate these
    )
    print("\nFINAL:", getattr(resp, "content", resp))

async def ask_with_retry():
    extra = {
        **CoreLLM.no_tools(),
        "retry": {"max_tries": 3, "base": 0.5, "jitter": 0.2},  # exponential backoff
    }
    res = await agent.arun_agent(
        messages=[{"role": "user", "content": "Give me one productivity tip."}],
        provider=Providers.openai,
        model_name=Models.openai.gpt_4_1_mini.value,
        extra=extra,
    )
    print(res)

async def hitl_stream():
    from langchain_core.tools import tool

    # --- HITL tool gate ---
    def make_tool_gate():
        """
        Always asks the human before executing a tool.
        If input() is unavailable (headless/CI), defaults to 'block'.
        """
        import json

        def _gate(tool_name: str, args: dict):
            print(f"\n[HITL] Tool request: {tool_name}({args})", flush=True)
            try:
                ans = input("Approve tool call? [y]es / [m]odify args / [b]lock: ").strip().lower()
            except EOFError:
                print("[HITL] Input unavailable; blocking tool call by default.\n", flush=True)
                return {"action": "block", "replacement": "[blocked by reviewer]"}

            if ans.startswith("b"):
                return {"action": "block", "replacement": "[blocked by reviewer]"}
            if ans.startswith("m"):
                raw = input("Enter JSON for modified args (empty to keep original): ").strip()
                try:
                    mod_args = json.loads(raw) if raw else args
                except Exception:
                    print("[HITL] Invalid JSON; keeping original args.", flush=True)
                    mod_args = args
                return {"action": "modify", "args": mod_args}

            return {"action": "pass"}  # default approve

        return _gate


    # --- A tiny tool so the agent actually uses tools ---
    @tool
    def get_weather(location: str) -> str:
        """Return a fake weather string for a location."""
        return f"It's sunny today in {location}."


    # enable tool HITL (calls your gate before executing any tool)
    base.set_hitl(on_tool_call=make_tool_gate())

    print(">>> Streaming agent token deltas only (no updates/values)\n")
    # NOTE: astream_agent_tokens is the helper that streams only LLM token deltas
    async for token, meta in agent.astream_agent_tokens(
            messages=[{"role": "user", "content": "Check Boston weather with a tool, then summarize in one line."}],
            provider=Providers.openai,
            model_name=Models.openai.gpt_4_1_mini.value,
            tools=[get_weather],
    ):
        # print tokens as they arrive
        print(token, end="", flush=True)

    print()  # newline at the end

async def chat_stream():
    res = llm.stream_tokens(
        user_msg="What is your name?",
        system="Your name is Alex and you are a helpful assistant.",
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
    )
    async for token, meta in res:
        print(token, end="", flush=True)

if __name__ == '__main__':
    # test_agent()
    # test_llm()
    # test_structured_output()
    # asyncio.run(test_agent_stream())
    controlled_tool_agent()
