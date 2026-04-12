from langchain_community.vectorstores import Chroma
from src.document_processor import load_and_chunk_pdf, load_and_chunk_folder
from src.vector_store import create_vector_store
from src.llm_engine import generate_test_cases

def build_pipeline(pdf_path: str) -> Chroma:
    chunks = load_and_chunk_pdf(pdf_path)
    vector_store = create_vector_store(chunks)
    return vector_store

def build_pipeline_from_folder(folder_path: str) -> Chroma:
    chunks = load_and_chunk_folder(folder_path)
    vector_store = create_vector_store(chunks)
    return vector_store

def run_query(vector_store: Chroma, feature: str, erp_system: str = "Generic", prompt_style: str = "Detailed") -> tuple:
    # similarity search with k=3
    results = vector_store.similarity_search_with_score(feature, k=3)
    
    # joins retrieved chunks as context
    context = "\n".join([r[0].page_content for r in results])
    scores_list = [round(float(r[1]), 3) for r in results]
    
    # calls generate_test_cases
    output = generate_test_cases(feature, context, erp_system, prompt_style)
    
    return output, scores_list
