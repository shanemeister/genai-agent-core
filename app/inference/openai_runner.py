from langchain_openai import ChatOpenAI
from app.configs import OPENAI_API_KEY

def ask_openai(prompt, model_name="gpt-4o"):
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set. Cannot use OpenAI fallback.")
        return "", {}

    print(f"\n💬 Answer (OpenAI {model_name}):\n")
    llm = ChatOpenAI(model_name=model_name, temperature=0.3)
    response = llm.invoke(prompt, return_usage=True)
    print(response.content)
    print("📊 Token usage:", response.usage)
    return response.content, response.usage

