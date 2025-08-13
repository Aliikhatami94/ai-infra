from ai_infra.llm import CoreLLM, Providers, Models
import asyncio

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

if __name__ == "__main__":
    asyncio.run(test_agent_stream())
