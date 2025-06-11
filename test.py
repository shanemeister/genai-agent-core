from chat.postgres_history import save_message, load_chat_history

session_id = "test-session-001"

# Save a user question and assistant reply
save_message(session_id, "user", "What is the capital of France?")
save_message(session_id, "assistant", "The capital of France is Paris.")

# Load and print chat history
history = load_chat_history(session_id)
for entry in history:
    print(f"[{entry['role']}] {entry['content']}")