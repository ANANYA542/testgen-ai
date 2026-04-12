import os
import re
import pandas as pd
import streamlit as st
from typing import List, Dict, Any

from src.rag_pipeline import build_pipeline, build_pipeline_from_folder, run_query

# Configure Streamlit page
st.set_page_config(
    page_title="AutoTestGen",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enforce light theme CSS matching modern frontend design
st.markdown("""
<style>
    /* Global Background Elements */
    .stApp {
        background-color: #FDFBF7;
    }
    
    [data-testid="stSidebar"] {
        background-color: #F4EFE6;
        border-right: 1px solid #EAE3D6;
    }
    
    /* Badges */
    .positive-badge {
        background-color: #bdf2e8;
        color: #0b7562;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.65rem;
        font-weight: 800;
        text-transform: uppercase;
        display: inline-block;
    }
    .negative-badge {
        background-color: #ffd8d6;
        color: #a51d1d;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.65rem;
        font-weight: 800;
        text-transform: uppercase;
        display: inline-block;
    }
    .edge-badge {
        background-color: #f0e1f9;
        color: #7b29a2;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.65rem;
        font-weight: 800;
        text-transform: uppercase;
        display: inline-block;
    }
    
    /* Cards */
    .card {
        border: 1px solid #EAE3D6;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #ffffff;
        color: #4A443A;
        box-shadow: 0 4px 6px -1px rgba(80,70,50,0.05), 0 2px 4px -1px rgba(80,70,50,0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .card h4 {
        color: #0f172a;
        font-size: 1.05rem;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 700;
        line-height: 1.4;
    }
    
    .card p {
        font-size: 0.85rem;
        color: #475569;
        margin-bottom: 12px;
        line-height: 1.5;
    }

    .scenario-label {
        font-size: 0.65rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        display: block;
    }
    
    /* Details block for View Steps */
    details.view-steps {
        margin-top: auto;
        background-color: #FDFBF7;
        padding: 10px;
        border-radius: 8px;
        cursor: pointer;
        border: 1px solid #EAE3D6;
    }
    details.view-steps summary {
        font-size: 0.8rem;
        font-weight: 700;
        color: #8C7D64;
        text-align: center;
        list-style: none; /* remove default arrow */
        display: block;
    }
    details.view-steps summary::-webkit-details-marker {
        display: none;
    }
    details.view-steps p {
        margin-top: 10px;
        font-size: 0.8rem;
        color: #475569;
    }

    /* History item */
    .history-item {
        background-color: #ffffff;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 13px;
        color: #4A443A;
        border: 1px solid #EAE3D6;
        box-shadow: 0 1px 2px rgba(80,70,50,0.03);
    }
    
    /* Metrics Card */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #EAE3D6;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(80,70,50,0.03);
        margin-bottom: 20px;
    }
    .metric-label {
        font-size: 0.65rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6B5C4B;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "test_cases" not in st.session_state:
    st.session_state.test_cases = []
if "pipeline_stage" not in st.session_state:
    st.session_state.pipeline_stage = 0
if "retrieval_scores" not in st.session_state:
    st.session_state.retrieval_scores = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []

def parse_test_cases(raw_text: str) -> List[Dict[str, str]]:
    cases = []
    blocks = raw_text.split("Test ID:")
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split("\\n")
        tc_data = {
            "test_id": "",
            "scenario": "",
            "steps": "",
            "expected_result": "",
            "type": ""
        }
        
        # Add back the split key prefix
        tc_data["test_id"] = lines[0].strip()
        
        current_section = None
        steps_lines = []
        
        for line in lines[1:]:
            line_str = line.strip()
            if line_str.startswith("Scenario:"):
                tc_data["scenario"] = line_str.replace("Scenario:", "").strip()
            elif line_str.startswith("Expected Result:"):
                tc_data["expected_result"] = line_str.replace("Expected Result:", "").strip()
                current_section = "expected"
            elif line_str.startswith("Type:"):
                tc_data["type"] = line_str.replace("Type:", "").strip()
                current_section = "type"
            elif line_str.startswith("Steps:"):
                current_section = "steps"
            else:
                if current_section == "steps" and line_str:
                    steps_lines.append(line_str)
                    
        tc_data["steps"] = "\\n".join(steps_lines)
        if tc_data["test_id"]:
            cases.append(tc_data)
            
    return cases

# Sidebar Section
with st.sidebar:
    st.markdown('<h3 style="color: #6B5C4B; margin-top: 0; padding-top: 0;">⚙️ Test Pipeline</h3>', unsafe_allow_html=True)
    st.markdown('<div style="color: #A39682; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">RAG-QA Context</div>', unsafe_allow_html=True)
    
    st.write("---")
    st.write("**DOCUMENT INGESTION**")
    
    input_type = st.radio(
        "Input Type",
        options=["Single PDF", "Project Folder"],
        label_visibility="collapsed"
    )
    
    if input_type == "Single PDF":
        uploaded_file = st.file_uploader("Upload Requirements Document (PDF)", type=["pdf"])
    elif input_type == "Project Folder":
        uploaded_files = st.file_uploader(
            "Upload Project Folder",
            type=["pdf", "py", "md", "txt", "docx"],
            accept_multiple_files=True
        )
    
    st.write("**TARGET PLATFORM**")
    erp_system = st.selectbox(
        "ERP Platform", 
        options=["Generic", "SAP S/4HANA", "Oracle Fusion Cloud", "Workday", "Salesforce"],
        label_visibility="collapsed"
    )
    
    st.write("**PROMPT STYLE**")
    prompt_style = st.selectbox(
        "Prompt Style",
        options=["Detailed", "Concise", "Gherkin"],
        label_visibility="collapsed"
    )
    
    if st.button("Process Document", type="primary", use_container_width=True):
        if input_type == "Single PDF":
            if uploaded_file is not None:
                temp_path = os.path.join("data", "uploaded_temp.pdf")
                os.makedirs("data", exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                with st.spinner("Processing document..."):
                    try:
                        st.session_state.pipeline_stage = 1
                        vector_store = build_pipeline(temp_path)
                        st.session_state.vector_store = vector_store
                        st.session_state.pipeline_stage = 4
                        st.success("Document processed successfully.")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
            else:
                st.warning("Please upload a PDF document first.")
        elif input_type == "Project Folder":
            if uploaded_files:
                temp_folder = os.path.join("data", "uploaded_folder")
                os.makedirs(temp_folder, exist_ok=True)
                
                for uf in uploaded_files:
                    file_path = os.path.join(temp_folder, uf.name)
                    with open(file_path, "wb") as f:
                        f.write(uf.getbuffer())
                        
                with st.spinner("Processing folder..."):
                    try:
                        st.session_state.pipeline_stage = 1
                        vector_store = build_pipeline_from_folder(temp_folder)
                        st.session_state.vector_store = vector_store
                        st.session_state.pipeline_stage = 4
                        st.success("Folder processed successfully.")
                    except Exception as e:
                        st.error(f"Error processing folder: {str(e)}")
            else:
                st.warning("Please upload folder files first.")
            
    st.write("---")
    st.write("**Pipeline Status**")
    
    stages = [
        ("Document Loaded", 1),
        ("Chunks Created", 2),
        ("Embeddings Stored", 3),
        ("Pipeline Ready", 4)
    ]
    
    for stage_name, stage_num in stages:
        if st.session_state.pipeline_stage >= stage_num:
            color = "green"
        else:
            color = "grey"
        st.markdown(f'<span style="color:{color}">●</span> {stage_name}', unsafe_allow_html=True)

    if st.session_state.query_history:
        st.write("---")
        st.write("**Query History**")
        for q in reversed(st.session_state.query_history[-5:]):
            st.markdown(f'<div class="history-item">{q}</div>', unsafe_allow_html=True)


# Main Application Section
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown('<h2 style="color: #6B5C4B; font-weight: 800; margin-top: 0; padding-top: 0;">AutoTestGen</h2>', unsafe_allow_html=True)
with col2:
    if st.session_state.vector_store is None:
        feature_input = st.text_input("Feature Name", disabled=True, placeholder="🔍 Upload document to begin search...", label_visibility="collapsed")
    else:
        feature_input = st.text_input("Feature Name", placeholder="🔍 Feature Name (e.g. User Login)", label_visibility="collapsed")

st.write("") # Spacer

if st.session_state.vector_store is None:
    st.info("Please complete the Document Ingestion setup in the sidebar to get started.")
else:
    if st.button("Generate Suite", type="primary"):
        if feature_input.strip() == "":
            st.warning("Please enter a feature name.")
        else:
            with st.spinner("Generating test cases..."):
                try:
                    raw_cases, scores = run_query(
                        st.session_state.vector_store, 
                        feature_input,
                        erp_system=erp_system,
                        prompt_style=prompt_style
                    )
                    st.session_state.retrieval_scores = scores
                    if feature_input not in st.session_state.query_history:
                        st.session_state.query_history.append(feature_input)
                        
                    parsed_cases = parse_test_cases(raw_cases)
                    
                    if parsed_cases:
                        st.session_state.test_cases = parsed_cases
                    else:
                        st.error("Model did not return test cases in the expected format.")
                        st.text(raw_cases) # Fallback to show raw output
                        
                except Exception as e:
                    st.error(f"Failed to generate test cases: {str(e)}")

# Display results if available
if st.session_state.test_cases:
    if st.session_state.retrieval_scores:
        st.markdown('<h4>📊 Confidence Metrics</h4>', unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.retrieval_scores))
        for i, score in enumerate(st.session_state.retrieval_scores):
            with cols[i]:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-label">CHUNK {i+1} DISTANCE</div>
                    <div class="metric-value">{score:.3f}</div>
                </div>
                ''', unsafe_allow_html=True)
        st.write("---")
        
    st.markdown('<h4>Generated Test Cases</h4>', unsafe_allow_html=True)
    
    cases = st.session_state.test_cases
    for i in range(0, len(cases), 3):
        cols = st.columns(3)
        for j, tc in enumerate(cases[i:i+3]):
            with cols[j]:
                test_type = tc["type"].lower()
                badge_class = "edge-badge"
                if "positive" in test_type:
                    badge_class = "positive-badge"
                elif "negative" in test_type:
                    badge_class = "negative-badge"
                    
                badge_html = f'<span class="{badge_class}">{tc["type"]}</span>'
                formatted_steps = tc['steps'].replace('\\n', '<br/>')
                
                card_html = f"""
                <div class="card">
                    <h4>{tc['test_id']} {badge_html}</h4>
                    <span class="scenario-label">SCENARIO</span>
                    <p>{tc['scenario']}</p>
                    <span class="scenario-label">EXPECTED</span>
                    <p>{tc['expected_result']}</p>
                    <details class="view-steps">
                        <summary>View Steps  ></summary>
                        <p>{formatted_steps}</p>
                    </details>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
        
    # Download as CSV functionality
    df = pd.DataFrame(st.session_state.test_cases)
    csv = df.to_csv(index=False).encode('utf-8')
    json_data = df.to_json(orient="records", indent=2).encode('utf-8')
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name='test_cases.csv',
            mime='text/csv'
        )
    with col2:
        st.download_button(
            label="Download as JSON",
            data=json_data,
            file_name='test_cases.json',
            mime='application/json'
        )
