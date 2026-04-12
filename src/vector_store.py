import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

PERSIST_DIR = "chroma_db"

def get_huggingface_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(chunks: List[Document]) -> Chroma:
    embeddings = get_huggingface_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return vector_store

def load_vector_store() -> Chroma:
    embeddings = get_huggingface_embeddings()
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vector_store