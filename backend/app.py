# backend/app.py
# ==============================================================================
# API FastAPI para el sistema RAG de fichas tecnicas vehiculares.
#
# Endpoints:
#   POST /ingest       — Ingesta PDFs desde un directorio a ChromaDB
#   POST /chat/stream  — Chat con streaming SSE (Server-Sent Events)
#
# El grafo LangGraph se construye una sola vez al iniciar la aplicacion.
# Cada sesion de chat se identifica por session_id para mantener historial.
# ==============================================================================
from __future__ import annotations

import json
import sys
from pathlib import Path

# Asegura que backend/ esté en el path para imports relativos,
# funciona tanto desde la raíz (uvicorn backend.app:app) como desde backend/
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from dotenv import load_dotenv

# IMPORTANTE: cargar .env ANTES de importar rag_graph para que las variables
# de LangSmith (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT)
# esten disponibles cuando LangChain inicialice el cliente de tracing.
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from rag_graph import build_rag_graph
from rag_store import ingest, ingest_semantic, get_vector_store, get_semantic_vector_store

# Reportar estado de LangSmith al arrancar
import os
if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and os.getenv("LANGCHAIN_API_KEY"):
    print(f"[LangSmith] Tracing ACTIVO — proyecto: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
else:
    print("[LangSmith] Tracing inactivo (configura LANGCHAIN_TRACING_V2 y LANGCHAIN_API_KEY en .env)")

app = FastAPI(title="LangGraph RAG API")

# Construir el grafo LangGraph una sola vez al arrancar.
# Incluye: vector store, LLMs, tools, nodos y edges del grafo.
graph = build_rag_graph()


class ChatRequest(BaseModel):
    """Modelo de request para el endpoint de chat.

    Campos:
        question:   Pregunta del usuario en lenguaje natural.
        session_id: Identificador de sesion para mantener historial conversacional.
                    Permite follow-ups como "y cuanto pesa?" heredando el modelo previo.
    """

    question: str
    session_id: str = "default"


class IngestRequest(BaseModel):
    """Modelo de request para el endpoint de ingestion.

    Campos:
        data_dir: Ruta al directorio con PDFs organizados por marca.
                  Ejemplo: "./data" (relativo a backend/)
    """

    data_dir: str


@app.post("/ingest")
def ingest_route(req: IngestRequest):
    """Ingesta documentos PDF a ChromaDB (chunking fijo).

    Limpia datos existentes y re-ingesta desde el directorio indicado.
    """
    try:
        vs = get_vector_store()
        existing = vs._collection.count()
        if existing > 0:
            all_ids = vs._collection.get()["ids"]
            if all_ids:
                vs._collection.delete(ids=all_ids)
        result = ingest(req.data_dir)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/ingest/semantic")
def ingest_semantic_route(req: IngestRequest):
    """Ingesta documentos PDF a ChromaDB (chunking semantico).

    Limpia datos existentes y re-ingesta desde el directorio indicado.
    """
    try:
        vs = get_semantic_vector_store()
        existing = vs._collection.count()
        if existing > 0:
            all_ids = vs._collection.get()["ids"]
            if all_ids:
                vs._collection.delete(ids=all_ids)
        result = ingest_semantic(req.data_dir)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.delete("/ingest")
def delete_ingest():
    """Limpia la coleccion fija para permitir re-ingesta."""
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    client.delete_collection("rag_collection")
    return JSONResponse({"status": "deleted", "collection": "rag_collection"})


@app.delete("/ingest/semantic")
def delete_ingest_semantic():
    """Limpia la coleccion semantica para permitir re-ingesta."""
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db_semantic")
    client.delete_collection("rag_collection_semantic")
    return JSONResponse({"status": "deleted", "collection": "rag_collection_semantic"})


@app.post("/kg_chat")
async def kg_chat(req: ChatRequest):
    """Chat usando el agente KG-RAG (Knowledge Graph + Vector Store).

    Combina consultas SPARQL sobre GraphDB con búsqueda semántica en PDFs.
    Retorna una respuesta estructurada con datos exactos del Knowledge Graph.
    """
    import asyncio
    from kg_rag_agent import responder_con_kg
    try:
        respuesta = await asyncio.to_thread(responder_con_kg, req.question)
        return JSONResponse({"answer": respuesta, "fuente": "kg_rag"})
    except Exception as e:
        return JSONResponse({"answer": f"Error en KG-RAG: {e}", "fuente": "kg_rag"}, status_code=500)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Chat con streaming SSE.

    Ejecuta el grafo RAG completo y emite 3 tipos de eventos SSE:
    - "token":         Respuesta final del RAG (texto)
    - "trazabilidad":  JSON con ruta completa, decisiones, chunks, evaluacion
    - "done":          Senial de fin del stream

    Nota: emite solo la respuesta final (no tokens intermedios) para evitar
    texto duplicado cuando el grafo reintenta la generacion internamente.
    """

    async def event_gen():
        # Estado inicial del grafo: pregunta, listas vacias, flags en false
        inputs = {
            "question": req.question,
            "docs": [],
            "answer": "",
            "messages": [HumanMessage(content=req.question)],
        }
        # thread_id vincula esta invocacion a una sesion persistente (MemorySaver)
        config = {"configurable": {"thread_id": req.session_id}}

        # Ejecutar el grafo completo y emitir solo la respuesta final
        try:
            final = await graph.ainvoke(inputs, config=config)
            answer = final.get("answer", "")
            if isinstance(answer, str) and answer.strip():
                yield {"event": "token", "data": answer}
        except Exception as exc:
            yield {"event": "token", "data": f"Error interno: {exc}"}

        # Emitir trazabilidad desde el estado final del grafo
        try:
            final_state = await graph.aget_state(config)
            traza = final_state.values.get("trazabilidad", {})
            if traza:
                yield {"event": "trazabilidad", "data": json.dumps(traza, ensure_ascii=False)}
        except Exception:
            pass

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_gen())
