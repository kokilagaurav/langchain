from langchain import OpenAI
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import os

OpenAI = ChatOpenAI(model = 'gpt-4')

result = OpenAI.invoke("What is the capital of France?")

print(result.content)  