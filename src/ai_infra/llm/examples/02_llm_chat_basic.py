"""02_llm_chat_basic: Direct LLM chat example.
Usage: python -m quickstart.run llm_chat_basic
Shows simple system + user message interaction.
"""
from ai_infra.llm import CoreLLM, Providers, Models


def main():
    llm = CoreLLM()
    resp = llm.chat(
        user_msg="What is one fun fact about the moon?",
        system="You are a concise assistant.",
        provider=Providers.mistralai,
        model_name=Models.mistralai.codestral_latest.value,
    )
    print("Response:\n", resp)

if __name__ == '__main__':
    main()