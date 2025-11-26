"""02_llm_chat_basic: Direct LLM chat example.
Usage: python -m quickstart.run llm_chat_basic
Shows simple system + user message interaction.
"""

from ai_infra.llm import LLM, Providers


def main():
    llm = LLM()
    resp = llm.chat(
        user_msg="What is one fun fact about the moon?",
        system="You are a concise assistant.",
        provider=Providers.openai,
        model_name="gpt-4o",
    )
    print("Response:\n", resp)


if __name__ == "__main__":
    main()
