from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Validate environment variable
api_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACE_ACCESS_TOKEN not found in environment variables")

llm = HuggingFaceEndpoint(
    model="microsoft/DialoGPT-medium",
    task="text-generation",
    huggingfacehub_api_token="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",
    model_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.7,
        "do_sample": True
    }
)

print("Testing direct LLM endpoint...")
direct_result = llm.invoke("What is the capital of India?")
print(f"Direct result: {direct_result}")
