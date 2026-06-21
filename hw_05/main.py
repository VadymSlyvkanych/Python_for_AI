import os

from dotenv import load_dotenv
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY not found. "
        "Set it in .env file or as an environment variable."
    )

llm = ChatOpenAI(
    model="google/gemini-2.5-flash-lite",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    max_tokens=1000,
)

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)

chain = load_summarize_chain(llm, chain_type="stuff")
summary = chain.invoke(chunks)

print(summary["output_text"])
