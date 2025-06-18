from app.llm_core import basic_query

def summarize_chunks(chunks, question="Summarize this information"):
    """Summarizes a list of document chunks using the default model."""
    full_text = "\n\n".join(chunks)
    prompt = f"{question}:\n\n{full_text}\n\nSummary:"
    return basic_query(prompt)