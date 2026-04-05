# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) system for automotive technical manuals. It answers questions about vehicle specs using PDFs organized by brand (Toyota, Mazda, Volkswagen, Peugeot, Opel, MG Emotor, Seat) with anti-hallucination grounding, multi-intent routing, and conversational memory.

**Stack**: FastAPI + LangGraph + OpenAI (`gpt-5-nano`) + ChromaDB + HuggingFace embeddings (`all-MiniLM-L6-v2`) + Streamlit + EasyOCR

## Setup

```bash
# Python 3.12 required
py -3.12 -m venv .venv
.\.venv\Scripts\Activate
pip install -r requeriments.txt   # note: misspelled filename intentional (don't rename)
```

Create `.env` at the project root:
```
OPENAI_API_KEY=sk-...
HF_TOKEN=hf-...
```

## Running

```bash
# Backend (from repo root or backend/)
uvicorn backend.app:app --reload --port 8001
# Swagger UI: http://localhost:8001/docs

# Frontend (separate terminal)
cd frontend
streamlit run streamlit_app.py
# http://localhost:8501

# Ingest PDFs (must run before first chat)
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "./data"}'

# Visualize graph structure
cd backend && python draw_graph.py  # outputs grafo.png
```

## Testing

```bash
# Run intent routing tests
cd backend
python test_intent_routes.py
```

No broader test suite exists. Manual testing is primarily done via the Streamlit UI.

## Architecture

### LangGraph State Machine (`backend/rag_graph.py`)

The core is an 8-node LangGraph graph with 4 routing paths:

```
classify_intent → (GENERAL) → answer_general → END
               → (needs_retrieval) → retrieve → decide_tools
                                                    → (tools=True) → call_tools → tools → generate_grounded
                                                    → (tools=False) → generate_grounded
                                      generate_grounded → evaluate_grounding
                                                            → (approved or max retries) → END
                                                            → (retry) → generate_grounded
```

**Intent types**: `Búsqueda` (specific lookup), `Resumen` (model summary), `Comparación` (model comparison), `GENERAL` (no retrieval needed)

**RAGState** key fields:
- `messages`: Full conversational history (LangGraph `add_messages` reducer)
- `last_model` / `last_make`: Persist across turns via `_keep_latest` reducer (enables follow-up questions without restating context)
- `trazabilidad`: Accumulates the full decision log, retrieved chunks, prompts, and grounding scores per turn
- `retry_count` / `critic_feedback`: Drive the regeneration loop (max 1 retry)

### Key Design Patterns

**Anti-hallucination**: Generated answers must include inline citations `[doc_id=<filename>; página=<N>]`. A critic LLM scores each answer (0–1) on grounding, citation presence, and completeness. Answers scoring < 0.5 are regenerated once with the critic's feedback injected.

**Dynamic retrieval k**: The classifier LLM suggests `suggested_k` (4–12). A fallback `K_POR_INTENCION` dict handles cases where classification fails. Comparison queries split k evenly between both models.

**Keyword fallback**: Regex patterns detect `Resumen`/`Comparación` intent if the LLM classifier miscategorizes — avoids wrong-path routing.

**Tool usage**: `Resumen` and `Comparación` intents trigger 5 LangGraph tools (`buscar_especificacion`, `buscar_por_marca`, `comparar_modelos`, `resumir_ficha`, `listar_modelos_disponibles`). Other intents use direct similarity search only.

### Module Responsibilities

| File | Responsibility |
|------|----------------|
| `backend/rag_graph.py` | LangGraph nodes, routing conditions, state schema |
| `backend/rag_store.py` | ChromaDB init, PDF ingestion (pdfplumber + EasyOCR fallback), chunking |
| `backend/schemas.py` | Pydantic models for structured LLM outputs (`IntentClassification`, `GroundingEvaluation`) |
| `backend/prompts.py` | System prompts for classifier, generator, and critic LLMs |
| `backend/tools.py` | 5 LangGraph tools for structured retrieval operations |
| `backend/app.py` | FastAPI endpoints: `POST /ingest`, `POST /chat/stream` (SSE) |
| `frontend/streamlit_app.py` | Chat UI, session management, SSE consumption, traceability panel |

### Ingestion Pipeline (`backend/rag_store.py`)

PDFs under `backend/data/<Make>/<model>.pdf` → `pdfplumber` text extraction → EasyOCR fallback for scanned pages → `_fix_doubled_text` corruption repair → `RecursiveCharacterTextSplitter` (1000 chars, 150 overlap) → ChromaDB with metadata: `{source, page, marca, modelo, doc_id, chunk_id, ocr}`.

## Knowledge Graph (Parte C)

### Archivos
| Archivo | Propósito |
|---------|-----------|
| `backend/ontologia/vehiculos.ttl` | Ontología OWL en Turtle (cargar en GraphDB) |
| `backend/ontologia/sparql_queries.py` | Consultas SPARQL: SELECT, FILTER, ORDER BY, LIMIT, UPDATE |
| `backend/ontologia/inferences.py` | 5 casos de inferencia documentados |
| `backend/kg_retriever.py` | Funciones de consulta SPARQL contra GraphDB |
| `backend/kg_rag_agent.py` | Agente LangGraph KG-RAG (SPARQL + vector + LLM) |

### Setup GraphDB
1. Descargar GraphDB Free desde `graphdb.ontotext.com`
2. Crear repositorio `vehiculos` con Ruleset = **OWL2-RL** (para inferencias)
3. Importar `backend/ontologia/vehiculos.ttl` vía `Import > RDF Files`
4. GraphDB corre en `http://localhost:7200`

### Ejecutar consultas SPARQL
```bash
cd backend
python ontologia/sparql_queries.py   # SELECT, FILTER, ORDER BY, LIMIT, UPDATE
python ontologia/inferences.py       # 5 casos de inferencia
python kg_rag_agent.py               # Prueba del agente combinado
```

### API Events (`/chat/stream`)

The SSE stream emits three event types:
- `token` — streaming answer text
- `trazabilidad` — JSON with full decision trace (intent, chunks, scores, retries)
- `done` — signals end of stream
