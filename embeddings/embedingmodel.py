from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OpenAIEmbeddings(
    model= "text-embedding-3-large", dimensions=40
)

result = embedding.embed_query("capital of india is delhi")

print(str(result))