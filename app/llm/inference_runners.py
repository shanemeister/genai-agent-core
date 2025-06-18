from app.utils.vector_utils import get_vectorstore
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_openai import ChatOpenAI
import torch, os

# Configuration and constants
VECTORSTORE_PATH = "vectorstore"
MISTRAL_GGUF_PATH = "/home/exx/myCode/genai-agent-core/models/mixtral-gguf/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
LLAMA3_MODEL_PATH = "/home/exx/myCode/genai-agent-core/models/llama3-8b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

def ask_llama3_hf(prompt: str):
    try:
        print("\n🧠 Using LLaMA 3 8B (Hugging Face FP16)")
        tokenizer = AutoTokenizer.from_pretrained(LLAMA3_MODEL_PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA3_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        for k in inputs:
            if inputs[k].dtype == torch.float32:
                inputs[k] = inputs[k].half()

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            repetition_penalty=1.1,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        return response

    except Exception as e:
        print(f"⚠️ LLaMA 3 failed to load or generate: {e}\nFalling back to GPT-4o...")
        return ask_openai(prompt, [])
    
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