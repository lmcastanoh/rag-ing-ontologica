# RAG Application (LangChain + LangGraph + FastAPI + Streamlit)

This project implements a Retrieval-Augmented Generation (RAG) system
using:

-   LangChain
-   LangGraph
-   FastAPI (backend)
-   Streamlit (frontend)
-   ChromaDB (vector database)
-   OpenAI (LLM + embeddings)

------------------------------------------------------------------------

## Project Structure

    rag-ing-ontologica/
    │
    ├── backend/
    │   ├── app.py
    │   ├── rag_graph.py
    │   ├── rag_store.py
    │   ├── data/              #PDFs here
    │   └── .env              
    │
    ├── frontend/
    │   ├── streamlit_app.py
    │   └── .streamlit/
    │
    ├── requeriments.txt
    └── README.md

------------------------------------------------------------------------

## Requirements

-   Python 3.12

------------------------------------------------------------------------

## Initial Setup (Run Once)

### Install Python 3.12

    download from 

### Create Virtual Environment

    cd path-to-rag/rag-ing-ontologica
    py -3.12 -m venv .venv  
    .\.venv\Scripts\Activate


### Install Dependencies

    pip install -r requeriments.txt

------------------------------------------------------------------------

## Add Documents for RAG

Place your PDFs inside:

backend/data/

Example:

    backend/data/
      file1.pdf
      file2.pdf

------------------------------------------------------------------------

## Run Backend (FastAPI)

Terminal 1:

    cd path-to/rag-ing-ontologica
    .\.venv\Scripts\Activate
    cd backend
    uvicorn app:app --reload --port 8001

Optional debug:

    uvicorn app:app --reload --port 8001 --log-level debug

Verify:

http://localhost:8001/docs

------------------------------------------------------------------------

## Run Frontend (Streamlit)

Terminal 2:

    cd path-to/rag-ing-ontologica
    .\.venv\Scripts\Activate
    cd frontend
    streamlit run streamlit_app.py

Streamlit runs at:

http://localhost:8501

------------------------------------------------------------------------


## Ingest Documents

Go to:

http://localhost:8001/docs

POST /ingest

Body:

    {
      "data_dir": "./data"
    }

Expected response:

    {
      "files_dir": "./data",
      "raw_docs": 5,
      "chunks": 42,
      "ids_added": 42
    }

------------------------------------------------------------------------

## Test Chat

POST /chat or /chat/stream

Example:

    {
      "question": "What is the main topic of the documents?"
    }

------------------------------------------------------------------------

## View LangGraph Structure

In rag_graph.py add:

    print(graph.get_graph().draw_ascii())

Restart backend to see:

START → retrieve → generate → END

For Mermaid:

    print(graph.get_graph().draw_mermaid())

Paste into:

https://mermaid.live

------------------------------------------------------------------------

## Troubleshooting

Port already in use:

    pkill -f uvicorn

Or change port:

    uvicorn app:app --reload --port 8002

Then update API_BASE accordingly.

Streamlit not found:

    pip install streamlit

OpenAI 401 error:

    echo $OPENAI_API_KEY

OpenAI 429 error (Quota exceeded):

Check billing: https://platform.openai.com/account/billing

Or switch embedding model:

    OpenAIEmbeddings(model="text-embedding-3-small")

------------------------------------------------------------------------

## Quick Start Commands

Backend:

    cd rag-ing-ontologica
    source .venv/bin/activate
    cd backend
    uvicorn app:app --reload --port 8001

Frontend:

    cd rag-ing-ontologica
    source .venv/bin/activate
    cd frontend
    streamlit run streamlit_app.py

------------------------------------------------------------------------

## Architecture Overview

User (Streamlit)\
↓\
FastAPI Backend\
↓\
LangGraph\
↓\
Retriever (Chroma)\
↓\
OpenAI Embeddings\
↓\
OpenAI Chat Model

------------------------------------------------------------------------

Your RAG system is now fully operational.
