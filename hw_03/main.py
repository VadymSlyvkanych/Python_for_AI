import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Set it in .env file or as an environment variable."
    )

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="google/gemini-2.5-flash-lite",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": "Расскажи интересный факт о языке Python.",
        },
    ],
)

print(response.choices[0].message.content)
