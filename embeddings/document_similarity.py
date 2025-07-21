from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

load_dotenv()

document = ["The capital of India is Delhi.",
            "The capital of France is Paris.",
            "The capital of Japan is Tokyo.",
            "The capital of Germany is Berlin."]

query = "what is capital of India"

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large", dimensions=40
)

query_embedding = embedding.embed_query(query)
document_embeddings = embedding.embed_documents(document)

scores = cosine_similarity([query_embedding], document_embeddings)[0] # get similarity scores

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1] # sorting on the basis of similarity score

print(query)
print(document[index])
print("similarity score is:", score)

