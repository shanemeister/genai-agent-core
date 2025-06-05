from app.llm_core import basic_query

if __name__ == "__main__":
    print("🔍 Running GenAI Agent Core")
    response = basic_query("Explain RAG in LLM-based systems.")
    print(response)
    