import os

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found")

llm = ChatOpenAI(
    model="google/gemini-2.5-flash-lite",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    max_tokens=500,
)

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,
)

loader = TextLoader("hw_05/sample.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(chunks, embeddings)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
)

questions = [
    "Что такое искусственный интеллект?",
    "Чем машинное обучение отличается от ИИ?",
    "Как работают нейронные сети?",
    "Что такое трансформеры в контексте ИИ?",
    "Какие примеры крупных языковых моделей существуют?",
]

for question in questions:
    print(f"\nВопрос: {question}")
    answer = qa.invoke(question)
    print(f"Ответ: {answer['result']}")

# uv run python hw_05/qa.py