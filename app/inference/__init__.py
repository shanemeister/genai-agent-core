def run_inference(prompt: str, model_choice="mixtral", stream=False, history=None, **kwargs):
    """
    Unified dispatcher to run local or OpenAI models.

    Args:
        prompt (str): Input prompt
        model_choice (str): 'mixtral', 'llama3', 'gpt4o'
        stream (bool): Enable streaming if supported (Mixtral)
        history (list): Optional chat history
        kwargs: Additional args like temperature, etc.

    Returns:
        str | tuple: Model output string, or (string, usage) if OpenAI
    """
    if model_choice == "mixtral":
        from .mixtral_runner import ask_mistral_gguf
        return ask_mistral_gguf(prompt, stream=stream, history=history)

    elif model_choice == "llama3":
        from .llama3_runner import ask_llama3_hf
        return ask_llama3_hf(prompt)

    elif model_choice == "gpt4o":
        from .openai_runner import ask_openai
        return ask_openai(prompt)

    else:
        raise ValueError(f"Unsupported model: {model_choice}")