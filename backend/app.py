# backend/app.py
from __future__ import annotations

import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from rag_graph import build_rag_graph
from rag_store import ingest

from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root (two levels up), not from inside backend/
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="LangGraph RAG API")

graph = build_rag_graph()

class ChatRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    data_dir: str

@app.post("/ingest")
def ingest_route(req: IngestRequest):
    result = ingest(req.data_dir)
    return JSONResponse(result)

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streams tokens using LangGraph stream_mode="messages" :contentReference[oaicite:12]{index=12}
    """
    async def event_gen():
        inputs = {"question": req.question, "docs": [], "answer": ""}

        # "messages" streams (token, metadata) tuples from LLM-invoking nodes :contentReference[oaicite:13]{index=13}
        async for chunk in graph.astream(inputs, stream_mode="messages"):
            # chunk is typically: (token, metadata) or structured tuples depending on runtime
            # We'll normalize to a "token" string for the UI.
            try:
                token, meta = chunk  # many providers yield (token, metadata)
                yield {"event": "token", "data": token}
            except Exception:
                # fallback: just dump whatever arrived (useful for debugging)
                yield {"event": "debug", "data": json.dumps(chunk, default=str)}

        # Also send final answer via a non-stream call (optional, but handy)
        final = graph.invoke(inputs)
        yield {"event": "final", "data": final.get("answer", "")}

    return EventSourceResponse(event_gen())