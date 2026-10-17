import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads GOOGLE_API_KEY from .env
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# 1. List the Flash models your free key can use
print("Available Flash models:")
for m in client.models.list():
    if "flash" in m.name:
        print(" ", m.name)

# 2. Send one test question
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="In one sentence, what is a Product Disclosure Sheet in Malaysian banking?",
)
print("\nTest answer:", response.text)