from app.configs import MISTRAL_GGUF_PATH, VECTORSTORE_PATH, LLAMA3_MODEL_PATH, DEVICE, HF_TOKEN, OPENAI_API_KEY
from llama_cpp import Llama
from app.inference.llama3_runner import ask_llama3_hf

def ask_mistral_gguf(prompt, stream=False, history=None, max_tokens=256, temperature=0.7):
    try:
        print("\n💡 Using Mistral GGUF via llama.cpp")
        model = Llama(
            model_path=MISTRAL_GGUF_PATH,
            n_ctx=4096,
            n_gpu_layers=100,
            seed=42
        )

        prompt = "Only summarize what is explicitly stated in the context. Do not make assumptions.\n\n" + prompt

        if stream:
            print("📤 Streaming response:")
            response_stream = model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            full_response = ""
            for chunk in response_stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)
                full_response += delta
            print()
            return full_response.strip()

        else:
            if history is None:
                history = []
            history = history.copy()  # don't mutate shared list
            history.append({"role": "user", "content": prompt})

            output = model.create_chat_completion(
                messages=history,
                max_tokens=max_tokens,
                temperature=temperature
            )
            response = output["choices"][0]["message"]["content"].strip()
            print(response)
            history.append({"role": "assistant", "content": response})
            return response

    except Exception as e:
        print(f"⚠️ Mistral GGUF failed to load or generate: {e}\nFalling back to LLaMA 3...")
        return ask_llama3_hf(prompt)