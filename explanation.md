# AutoTestGen Explanation

This document serves to structurally explain the architecture, design choices, and flow of the AutoTestGen application. As an intelligent test case generator, AutoTestGen leverages Retrieval-Augmented Generation (RAG) atop large language models to construct reliable and relevant software test cases based purely on the given context (Requirements Documents).

## What is RAG? (Retrieval-Augmented Generation)
Large language models (LLMs) are frozen in time and have limits on context sizes, meaning we cannot always feed them huge 500-page requirement PDFs in a single prompt. 

RAG solves this by implementing a search engine step before querying the LLM:
1. Break down large documents into smaller chunks.
2. Find the top most relevant chunks for a specific query.
3. Pass only those relevant chunks alongside the user query to the LLM. 

By applying RAG to AutoTestGen, we ensure that test cases are generated solely from the provided product specs without hallucinatory features being added.

---

## Architectural Breakdown

### 1. `document_processor.py` (Ingestion Phase)
**Purpose:** Handles the loading and chunking of user-uploaded PDF files.
**Concepts used:**
- `PyPDFLoader`: A LangChain construct that seamlessly reads content from `.pdf` files page by page.
- `RecursiveCharacterTextSplitter`: Splits large pages horizontally into chunks. We used `chunk_size=1000` to group meaningful paragraphs together, and `chunk_overlap=200` to prevent abruptly slicing sentences in half, assuring context naturally bridges between chunk N and N+1.

### 2. `vector_store.py` (Database & Embeddings Phase)
**Purpose:** Converts raw text chunks into numbers and stores them for semantic querying.
**Concepts used:**
- **HuggingFace Embeddings (`all-MiniLM-L6-v2`)**: Transforms textual strings into thousands of multi-dimensional arrays (embeddings). Meaningfully similar strings will map to similar numeric spaces. 
- **ChromaDB**: An AI-native vector database strictly designed to store embeddings and index them allowing us to retrieve the nearest neighbor points quickly.
  - `persist_directory="chroma_db"`: Instructs ChromaDB to save its index to the hard drive inside the `/chroma_db` folder. This means the DB won't wipe across Application restarts.

### 3. `llm_engine.py` (The Generation Engine)
**Purpose:** Interfaces dynamically with ChatGroq to leverage LLMs and strictly defines how the model generates outputs.
**Concepts used:**
- **ChatGroq (`llama3-8b-8192`)**: Groq uses blazing fast LPU inference engines to run the Llama-3 8B model. With `temperature=0.3` set, the generated outputs will be highly deterministic and logical (as creativity should be minimal for QA).
- **PromptTemplate**: By predefining instructions, constraints, and parsing requirements, we enforce that our LLM will only behave as a "Senior QA Engineer".

### 4. `rag_pipeline.py` (The Orchestrator)
**Purpose:** Acts as the traffic controller that unifies the components.
**Concepts used:**
- Abstracting pipeline concerns: Streamlit should never chunk documents directly. `build_pipeline` acts as an endpoint handling PDF parsing, embedding creations, and DB saving seamlessly.
- `vector_store.similarity_search(feature, k=3)`: Retreives the top 3 highest scored context pieces directly matching the requested Feature.

### 5. `app.py` (Streamlit UI Layer)
**Purpose:** Gives the user a frontend client to interact with the backend infrastructure intuitively.
**Concepts used:**
- **State Management**: `st.session_state` preserves information between interactions. Streamlit normally re-runs from top to bottom on any button click. Storing `vector_store` and our `test_cases` inside session state ensures we aren't regenerating embeddings pointlessly or resetting output screens on every interaction.
- **Dynamic CSS Manipulation**: `st.markdown(..., unsafe_allow_html=True)` allows us to cleanly map CSS variables rendering highly interactive "Positive", "Negative", and "Edge Case" cards aesthetically.

## Workflow

1. The user launches `app.py` in the terminal via `streamlit run app.py`
2. Through the Sidebar, the User uploads a requirement PDF.
3. The uploaded PDF is saved manually to disk and passed into `build_pipeline()`.
4. `document_processor.py` processes the file. `vector_store.py` builds the DB.
5. The App state sets `Pipeline Ready`.
6. User enters "Login form". 
7. The `run_query` acts on ChromaDB sending relevant pieces representing "Login form" to Llama-3.
8. Text returns, is split via python dictionary parsers, and converted into visually styled Streamlit Cards.
