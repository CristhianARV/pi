# models.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import FastEmbedSparse

def make_embeddings(model="Qwen/Qwen3-Embedding-0.6B"):
    return HuggingFaceEmbeddings(model_name=model)

def make_sparse_embeddings(model="Qdrant/bm25"):
    return FastEmbedSparse(model_name=model)

def make_llm(model="qwen3.5:9b", base_url="http://157.26.83.15/ollama/"):
    return ChatOllama(model=model, base_url=base_url)