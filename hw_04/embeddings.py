import os

import numpy as np
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


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search_similar(query: str, texts: list[str], top_n: int = 3) -> list[tuple[str, float]]:
    query_emb = get_embedding(query)
    text_embs = [get_embedding(t) for t in texts]

    similarities = [(texts[i], cosine_similarity(query_emb, text_embs[i])) for i in range(len(texts))]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


if __name__ == "__main__":
    text1 = "Python — это язык программирования, который любят за простоту."
    text2 = "JavaScript широко используется для веб-разработки."
    text3 = "Монти Пайтон — это британская комедийная группа."

    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    emb3 = get_embedding(text3)

    print(f"Размерность эмбеддинга: {len(emb1)}")
    print(f"text1 <-> text2: {cosine_similarity(emb1, emb2):.4f}")
    print(f"text1 <-> text3: {cosine_similarity(emb1, emb3):.4f}")
    print(f"text2 <-> text3: {cosine_similarity(emb2, emb3):.4f}")

    print("\n--- Поиск похожих текстов ---")
    corpus = [
        "Python — отличный язык для анализа данных",
        "JavaScript — король фронтенда",
        "NumPy используется для научных вычислений",
        "React — библиотека для интерфейсов",
        "Pandas удобен для работы с таблицами",
    ]
    query = "Наука о данных на Python"
    results = search_similar(query, corpus)
    print(f"Запрос: {query}")
    for text, score in results:
        print(f"  {score:.4f} — {text}")

# uv run python hw_04/embeddings.py
