"""03_structured_output: Pydantic structured output example.
Usage: python -m quickstart.run llm_structured
Demonstrates forcing the model to return a validated schema.
"""
from pydantic import BaseModel, Field
from ai_infra.llm import CoreLLM, Providers, Models


class UserInfo(BaseModel):
    name: str = Field(..., description="Full name")
    age: int = Field(..., description="Age in years")
    email: str = Field(..., description="Email address")


def main():
    llm = CoreLLM()
    structured = llm.with_structured_output(
        provider=Providers.openai,
        model_name=Models.openai.gpt_4o.value,
        schema=UserInfo,
    )
    resp = structured.invoke([
        {"role": "user", "content": "I'm Jane Smith, 28 years old, email jane@example.com"}
    ])
    print("Raw response:", resp)
    # Access fields directly
    print("Name:", resp.name, "Age:", resp.age, "Email:", resp.email)