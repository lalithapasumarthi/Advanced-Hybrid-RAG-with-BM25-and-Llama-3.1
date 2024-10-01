# Advanced Hybrid RAG with BM25 and Llama 3.1

This project is an advanced hybrid Retrieval-Augmented Generation (RAG) system that combines the power of BM25 (for sparse vector search) and Llama 3.1 model (for natural language generation). The system is built using Qdrant, a vector database, for indexing and retrieval, and includes dense embeddings using `SentenceTransformers` to enable a hybrid retrieval system.
## Table of Contents
1. [Key Components](#key-components)
   - [Indexer (indexer.py)](#indexer-indexerpy)
   - [Retriever (retriever.py)](#retriever-retrieverpy)
   - [Generator (generate.py)](#generator-generatepy)
   - [Streamlit App (app.py)](#streamlit-app-apppy)
2. [Requirements](#requirements)
3. [Setup and Usage](#setup-and-usage)
   - [Qdrant Setup](#qdrant-setup)
   - [Running the Indexer](#running-the-indexer)
   - [Running the Retriever](#running-the-retriever)
   - [Running the Generator](#running-the-generator)
   - [Streamlit Web App](#streamlit-web-app)
   - [PDF Upload and Question Asking](#pdf-upload-and-question-asking)
4. [Features](#features)
5. [Future Work](#future-work)
6. [Acknowledgments](#acknowledgments)
7. [License](#license)

## Key Components

1. **Indexer (indexer.py)**:
   - Reads PDF documents, extracts text, and splits it into manageable chunks.
   - Uses BM25 and dense embeddings from `SentenceTransformers` to represent each document chunk.
   - Stores both sparse and dense vectors in Qdrant for efficient retrieval.

2. **Retriever (retriever.py)**:
   - Provides hybrid retrieval using both sparse and dense embeddings.
   - Combines results from BM25 and dense embeddings using Rank Reciprocal Fusion (RRF) to return the most relevant documents.

3. **Generator (generate.py)**:
   - Uses Llama 3.1 to generate human-like, informative answers based on the retrieved document context.
   - Allows for contextual LLM queries and provides concise answers.

4. **Streamlit App (app.py)**:
   - A user interface for uploading PDF documents, asking questions, and receiving generated answers.
   - Displays uploaded PDFs, processes questions, and provides RAG-powered answers.

## Requirements

- Python 3.9 or higher
- `torch`
- `transformers`
- `qdrant-client`
- `sentence-transformers`
- `PyPDF2`
- `fastembed`
- `sklearn`
- `numpy`
- `streamlit`

Install the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Setup and Usage

1. **Qdrant Setup**:
   - Ensure Qdrant is running locally on `http://localhost:6333`. You can use Docker to set up Qdrant:

     ```bash
     docker run -p 6333:6333 qdrant/qdrant
     ```

2. **Running the Indexer**:
   - To index a PDF document, run the `indexer.py` script:

     ```bash
     python indexer.py
     ```

   - This will extract text from the PDF, create sparse and dense vectors, and store them in the Qdrant collection.

3. **Running the Retriever**:
   - To retrieve relevant document chunks based on a query, run the `retriever.py` script:

     ```bash
     python retriever.py
     ```

4. **Running the Generator**:
   - Use the `generate.py` script to generate answers using the Llama 3.1 model:

     ```bash
     python generate.py
     ```

5. **Streamlit Web App**:
   - Launch the web app to interact with the system via a browser:

     ```bash
     streamlit run app.py
     ```

6. **PDF Upload and Question Asking**:
   - Use the sidebar to upload a PDF document.
   - Enter your query in the text input box and click on "Submit".
   - The system will retrieve relevant chunks from the document and generate an answer using the Llama model.

## Features

- **Hybrid Retrieval**: Combines sparse (BM25) and dense (transformer-based) embeddings for accurate document retrieval.
- **Retrieval-Augmented Generation**: Uses Llama 3.1 to generate answers based on the retrieved document context.
- **Interactive UI**: Streamlit app for easy interaction, document upload, and question-answering.

## Future Work

- **Support for Multi-PDF Indexing**: Extend the system to handle multiple PDFs and enhance search across documents.
- **Query Expansion**: Improve the retrieval process by incorporating query expansion techniques for better document matching.
- **Scaling**: Explore distributed setups for indexing and retrieval across large document collections.

## Acknowledgments

- Qdrant
- Llama 3.1
- BM25

## License

This project is licensed under the MIT License.
