
def generate_prompt(question: str, docs: list) -> str:
    context = "\n\n".join([f"[{doc.metadata.get('source')}] {doc.page_content}" for doc in docs])
    return (
        "You are an AI document analyst. Use the context below, which may include multiple files, to answer comprehensively.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    
def safe_token_len(text, tokenizer):
    tokens = tokenizer.encode(text)
    return len(tokens)