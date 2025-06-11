from app.llm_core import basic_query
from chat.postgres_history import init_chat_table

if __name__ == "__main__":
    init_chat_table()
    print("🔍 Running GenAI Agent Core")
    response = basic_query("Explain RAG in LLM-based systems.")
    print(response)
    