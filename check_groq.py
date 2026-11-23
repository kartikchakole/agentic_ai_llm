import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)
api_key = os.getenv("GROQ_API_KEY")

print("Checking Groq API Key...")
try:
    client = Groq(api_key=api_key)
    models = client.models.list()
    print("\n✅ Successfully connected to Groq! Available models on your account:")
    for m in models.data:
        print(f" - {m.id}")
except Exception as e:
    print(f"\n❌ Groq connection failed: {e}")
