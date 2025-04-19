import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key loaded: {'Yes' if api_key else 'No'}")

# Initialize client
client = OpenAI(api_key=api_key)

# Test the API
try:
    response = client.embeddings.create(
        input="This is a test",
        model="text-embedding-ada-002"
    )
    print("API call successful!")
    print(f"Embedding length: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"Error: {str(e)}") 