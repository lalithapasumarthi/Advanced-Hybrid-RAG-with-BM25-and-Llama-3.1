import streamlit as st
import os
import base64
from PyPDF2 import PdfReader
from indexer import QdrantIndexing
from retriever import retriver
from generate import generate
from qdrant_client import QdrantClient

# Streamlit layout
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Hybrid RAG using BM25</h1>", unsafe_allow_html=True)

# Initialize session state variables
if 'question' not in st.session_state:
    st.session_state['question'] = ''
if 'answer' not in st.session_state:
    st.session_state['answer'] = ''
if 'pdf_uploaded' not in st.session_state:
    st.session_state['pdf_uploaded'] = False
if 'pdf_file_name' not in st.session_state:
    st.session_state['pdf_file_name'] = None
if 'refresh_page' not in st.session_state:
    st.session_state['refresh_page'] = False
if 'indexing_complete' not in st.session_state:
    st.session_state['indexing_complete'] = False

# Function to clear data and reset state
def clear_data():
    qdrant_client = QdrantClient(url="http://localhost:6333")
    
    if qdrant_client.collection_exists("collection_bm25"):
        qdrant_client.delete_collection("collection_bm25")
    
    if st.session_state['pdf_file_name'] and os.path.exists(f"temp/{st.session_state['pdf_file_name']}"):
        os.remove(f"temp/{st.session_state['pdf_file_name']}")
    
    st.session_state['question'] = ''
    st.session_state['answer'] = ''
    st.session_state['pdf_uploaded'] = False
    st.session_state['pdf_file_name'] = None
    st.session_state['indexing_complete'] = False
    st.session_state.pop('file_uploader', None)
    st.session_state['refresh_page'] = True

# Function to display PDF
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Sidebar for PDF Upload
st.sidebar.header("Upload your PDF")
pdf_file = st.sidebar.file_uploader("Drag and drop your PDF here", type="pdf", key="file_uploader")

# Create two columns for layout (swapped order)
col1, col2 = st.columns([0.6, 0.4])

# Text input and Answer Display (left column)
with col1:
    st.header("Please ask any questions you have.")
    st.session_state['question'] = st.text_input("Enter your question here:", st.session_state['question'])
    answer_placeholder = st.empty()

    if st.session_state['question'] and st.session_state['indexing_complete']:
        with st.spinner("Searching for relevant information..."):
            search = retriver()
            retrieved_docs = search.hybrid_search(query=st.session_state['question'])
            context = " ".join(retrieved_docs)

        with st.spinner("Generating answer..."):
            llm = generate()
            st.session_state['answer'] = llm.llm_query(question=st.session_state['question'], context=context)

        answer_placeholder.write(f"Answer:\n{st.session_state['answer']}")
    elif st.session_state['question'] and not st.session_state['indexing_complete']:
        st.warning("Please wait for the document to be indexed before asking questions.")

    if st.button("Clear"):
        clear_data()

# PDF processing and display (right column)
with col2:
    if pdf_file is not None and not st.session_state['pdf_uploaded']:
        st.session_state['pdf_uploaded'] = True
        st.session_state['pdf_file_name'] = pdf_file.name
        
        with open(f"temp/{pdf_file.name}", "wb") as f:
            f.write(pdf_file.getbuffer())

        with st.spinner("Indexing the document... This may take a moment."):
            collection_name = "collection_bm25"
            indexing = QdrantIndexing(pdf_path=f"temp/{pdf_file.name}")
            indexing.read_pdf()
            indexing.client_collection()
            indexing.document_insertion()
            st.session_state['indexing_complete'] = True
        st.sidebar.success("Document indexed successfully!")

    if st.session_state['pdf_uploaded']:
        st.write(f"Displaying PDF: {st.session_state['pdf_file_name']}")
        display_pdf(f"temp/{st.session_state['pdf_file_name']}")

# Check if page refresh is needed
if st.session_state['refresh_page']:
    st.session_state['refresh_page'] = False
    st.rerun()