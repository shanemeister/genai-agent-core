import torch 
import os

# Configuration and constants
VECTORSTORE_PATH = "vectorstore"
MISTRAL_GGUF_PATH = "/home/exx/myCode/genai-agent-core/models/mixtral-gguf/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
LLAMA3_MODEL_PATH = "/home/exx/myCode/genai-agent-core/models/llama3-8b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
