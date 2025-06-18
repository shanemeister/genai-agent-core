from llama_cpp import Llama
import torch
from app.inference.openai_runner import ask_openai
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.configs import LLAMA3_MODEL_PATH, DEVICE

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