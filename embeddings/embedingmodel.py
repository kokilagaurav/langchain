from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

embedding = OpenAIEmbeddings.load_dotenv(
    model= "text-embedding-3-large", dimension=40
)

result = embedding.query("capital of india is delhi")

print(str(result))