import os
import sys

try:
    from google import genai
except Exception as exc:
    print("Missing dependency: google-genai. Install with: pip install google-genai")
    raise SystemExit(1) from exc

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Load .env if available
if load_dotenv is not None:
    load_dotenv()

# Configure your API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Missing GOOGLE_API_KEY. Add it to your environment or .env file.")
    sys.exit(1)

client = genai.Client(api_key=api_key)

print("Available Gemini models:")
for model in client.models.list():
    print(f"- {model.name}")
