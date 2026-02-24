# backend/rag_store.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader, TextLoader

PERSIST_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_collection")

def get_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-large"))
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

def load_files(data_dir: str) -> List[Document]:
    p = Path(data_dir)
    if not p.exists():
        raise ValueError(f"data_dir does not exist: {data_dir}")

    docs: List[Document] = []

    # PDFs
    for fp in p.rglob("*.pdf"):
        loader = PyPDFLoader(str(fp))
        docs.extend(loader.load())

    # (Opcional) TXT también
    for fp in p.rglob("*.txt"):
        docs.extend(TextLoader(str(fp), encoding="utf-8").load())
    print("DATA DIR:", p.resolve())
    print("PDFs found:", len(list(p.rglob("*.pdf"))))
    return docs

def ingest(data_dir: str) -> dict:
    raw_docs = load_files(data_dir)
    if not raw_docs:
        raise ValueError(
            f"No documents loaded from data_dir={data_dir}. "
            "Put .pdf (or .txt) files there."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(raw_docs)
    if not splits:
        raise ValueError("Loaded documents but produced 0 chunks. Are files empty?")

    vs = get_vector_store()
    ids = vs.add_documents(documents=splits)
    return {"files_dir": data_dir, "raw_docs": len(raw_docs), "chunks": len(splits), "ids_added": len(ids)}