import argparse
from app.interface.query_handler import ask_question

def run_query():
    parser = argparse.ArgumentParser(description="Query your local or OpenAI LLM with optional filters.")
    parser.add_argument("question", type=str, help="Your question to ask.")
    parser.add_argument("--model", type=str, choices=["mixtral", "llama3", "gpt4o"], default="mixtral",
                        help="Choose the LLM to use: mixtral (GGUF), llama3 (Transformers), or gpt4o (OpenAI).")
    parser.add_argument("--filter-tag", type=str, help="Filter by semantic tag.")
    parser.add_argument("--filter-file", type=str, help="Filter by source filename.")
    parser.add_argument("--filter-all", action="store_true", help="Use all documents (no similarity filtering).")
    parser.add_argument("--stream", action="store_true", help="Stream response (for Mixtral GGUF).")
    parser.add_argument("--chat", action="store_true", help="Enable conversation memory (persistent across turns).")
    parser.add_argument("--session-id", type=str, help="Session ID for persistent chat history.")

    args = parser.parse_args()

    result = ask_question(
        question=args.question,
        model_choice=args.model,
        filter_tag=args.filter_tag,
        filter_filename=args.filter_file,
        filter_all=args.filter_all,
        stream=args.stream,
        session_id=args.session_id,
        chat_enabled=args.chat,
        history=[],
        rules=None
    )

    print("\n🧠 Answer:", result["answer"])
    print("\n📊 Meta:", result["meta"])


if __name__ == "__main__":
    run_query()