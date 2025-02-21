# QueryPDF

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Introduction

**QueryPDF** is an advanced RAG-powered chatbot that allows users to query PDF documents using OpenAI's GPT models and Google's Gemini models. By leveraging a robust pipeline that includes document ingestion, data extraction, vector storage, auto-merging retrieval, and reranking, QueryPDF provides highly relevant and contextual responses based on uploaded PDFs. The chatbot is optimized for handling conversations, ensuring continuity and context-aware interactions.

Key technologies used in QueryPDF:
- Streamlit for an intuitive web interface.
- OpenAI for LLM-powered query answering.
- Gemini 2.0 Flash for accurate data extraction.
- TimescaleDB as a scalable vector store with Timescale Vector.
- Alibaba-NLP reranker for reranking auto-merging retrieval results.
- Docker for containerization and deployment.
- NVIDIA GPU acceleration for optimized inference.

## Features

### 🔍 PDF Querying with RAG Pipeline
- Upload a PDF and ask questions about its contents.
- Uses Retrieval-Augmented Generation (RAG) to fetch the most relevant information.

### ⚡ High-Performance Embedding & Retrieval
- **OpenAI's embedding models** power semantic search.
- **PostgreSQL with TimescaleDB** for efficient vector storage and retrieval.
- **Auto-merging retrieval with reranking** for improved document ranking and relevance.

### 🎭 Multi-Tone Response Generation
- Choose from multiple response styles, such as **business lawyer, economist, critical journalist, theologist, elementary school teacher, preacher, or YouTuber**.
- Tailored responses for different perspectives and communication styles.

### 🚀 Fast and Scalable Deployment
- **Containerized with Docker** for easy deployment.
- **Supports NVIDIA GPUs** for accelerated inference.
- **Runs locally or in the cloud** with minimal setup.

### 🔧 Configurable and Extensible
- Fine-tune settings via `settings.py` to adjust retrieval, reranking, and LLM behavior.
- Easily swap or extend models for different tasks.
- API-key-based authentication for OpenAI and Gemini.

## Installation

### Steps

1. **Clone the Repository**
   ```sh
   git clone https://github.com/woutman/querypdf.git
   cd querypdf
   ```

2. **Set Up Environment Variables**
   Create a `.env` file in `querypdf/` by copying `example.env` and set up the environment variables.

3. **Build the Docker container**
   Run ```sh docker-compose build``` to build the Docker container.

4. **Run the Docker container with docker-compose**
   Run ```sh docker-compose up``` to run the Docker container.

## Usage

- Once the server is running, open a browser and navigate to `http://127.0.0.1:8501/`.
- Upload a PDF file by dragging it from your file explorer to the upload field or by clicking the "Browse files" button.
- **Optional:** Choose a a tone of voice for the chatbot's response in the dropdown menu.
- Enter your question about the PDF in the input field and click "Send".

## Project Structure

```
woutman-querypdf/
├── Dockerfile
├── docker-compose.yml
├── entrypoint.sh
├── example.env
├── requirements.txt
├── .dockerignore
└── main/
    ├── app.py
    ├── settings.py
    ├── database/
    │   ├── context_store.py
    │   ├── models.py
    │   └── vector_store.py
    ├── llm/
    │   ├── gemini_interface.py
    │   └── openai_interface.py
    └── rag/
        ├── extraction.py
        ├── ingestion.py
        ├── instructions.py
        ├── rag.py
        └── types.py
```