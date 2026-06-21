import os

from dotenv import load_dotenv
from openai import (
    APITimeoutError,
    RateLimitError,
    OpenAI,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

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
    timeout=30.0,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
)
def ask_gemini(prompt: str) -> str:
    response = client.chat.completions.create(
        model="google/gemini-2.5-flash-lite",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    try:
        result = ask_gemini("Расскажи интересный факт о Python.")
        print("Ответ Gemini:")
        print(result)
    except Exception as e:
        print(f"Ошибка после всех попыток: {e}")

# v run python hw_04/main.py

