from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OpenAIEmbeddings(
    model= "text-embedding-3-large", dimensions=40
)


documens = [    "The capital of India is Delhi.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo.",
    "The capital of Germany is Berlin.",]

result = embedding.embed_documents(documens)

print(str(result))