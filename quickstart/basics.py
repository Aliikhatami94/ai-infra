from langchain_core.tools import tool

from ai_infra.llm import CoreLLM, Providers, Models
from ai_infra.llm.tool_controls import ToolCallControls, force_tool

core = CoreLLM()

def test_agent():
    res = core.run_agent(
        messages=[{"role": "user", "content": "What is your namea?"}],
        provider=Providers.google_genai,
        model_name=Models.google_genai.gemini_2_5_flash.value,
        model_kwargs={
            "temperature": 0.7,
        }
    )
    print(res)

def test_llm():
    llm = core.set_model(
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
    )
    res = llm.invoke("Hello, how are you?")
    print(res)

def test_sys_msg():
    model = core.set_model(
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
    )
    res = model.invoke([
        {"role": "system", "content": "Your name is Alex"},
        {"role": "user", "content": "Hey what is your name?"}
    ])
    print(res)

def test_structured_output():
    from pydantic import BaseModel, Field
    class UserInfo(BaseModel):
        name: str = Field(..., description="The user's full name")
        age: int = Field(..., description="The user's age in years")
        email: str = Field(..., description="The user's email address")

    llm = core.with_structured_output(
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
        schema=UserInfo
    )
    res = llm.invoke("My name is John Doe, I am 30 years old and my email is johndoe@gmail.com")
    print(res)

async def test_agent_stream():
    async for token, meta in core.arun_agent_stream(
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
    res = core.run_agent(
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

    core.set_hitl(on_tool_call=reviewer)

    messages = [{"role": "user", "content": "What's the weather in New York?"}]
    resp = core.run_agent(
        messages=messages,
        provider="openai",
        model_name="gpt-4.1-mini",
        tools=[get_weather, get_news],   # <- our wrapper will gate these
    )
    print("\nFINAL:", getattr(resp, "content", resp))
