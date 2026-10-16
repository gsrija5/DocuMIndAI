import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents="Python is used for artificial intelligence."
)

print("Embedding test successful!")

vector = response.embeddings[0].values

print("Number of values:", len(vector))
print("First 5 values:", vector[:5])