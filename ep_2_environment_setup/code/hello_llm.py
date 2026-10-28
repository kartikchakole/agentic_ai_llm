import os 
from dotenv import find_dotenv, load_dotenv
from rich import print
from langchain.chat_models import init_chat_model

# Automatically locate and load the .env file even if executed from subdirectories
load_dotenv(find_dotenv())

MODEL_NAME = "gemini-3.5-flash"
MODEL_PROVIDER = "google_genai"
LLM = init_chat_model(model=MODEL_NAME, model_provider=MODEL_PROVIDER)

response = LLM.invoke("hello,whats going on")
print(response.text)