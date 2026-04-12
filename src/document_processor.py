from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, PythonLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

def load_and_chunk_folder(folder_path: str) -> List[Document]:
    all_documents = []
    files_loaded = 0
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == ".py":
                    loader = PythonLoader(file_path)
                elif ext in [".md", ".txt"]:
                    loader = TextLoader(file_path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(file_path)
                else:
                    continue
                
                docs = loader.load()
                all_documents.extend(docs)
                files_loaded += 1
            except Exception:
                pass
                
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = splitter.split_documents(all_documents)
    
    print(f"Total files loaded: {files_loaded}")
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks

def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Chunk_size and overlap required by the prompt
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = splitter.split_documents(pages)
    
    print(f"Total pages loaded: {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")
    
    return chunks